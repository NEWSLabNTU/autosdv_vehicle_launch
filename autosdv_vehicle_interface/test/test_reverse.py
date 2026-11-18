# Copyright 2026 NEWSLab, National Taiwan University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Deliberate reverse in the longitudinal shell (``reverse_enabled``), no ROS.

The car is the measured ESC state machine (docs/design/esc-pwm-model.md) in
front of ``motor_plant`` with ``allow_reverse``, read through the hall-sensor
model (magnitude only), and signed the way ``velocity_report`` signs it: by
the direction the shell publishes.
"""
from autosdv_vehicle_interface.actuator_core import make_shell_config, normalize_params
from autosdv_vehicle_interface.longitudinal import (
    DIRECTION_FORWARD,
    DIRECTION_REVERSE,
    EscState,
    LongitudinalShell,
    Mode,
    OVERRIDE_EMERGENCY,
    OVERRIDE_VELOCITY_TIMEOUT,
    reverse_unavailable_reason,
    ShellConfig,
)
from autosdv_vehicle_interface.motor_plant import HallSpeedSensor, MotorPlant, MotorPlantParams
from conftest import RecordingLogger
import pytest

# Car C, measured 2026-10-19 (docs/design/esc-pwm-model.md)
CAR_C = {'init_pwm': 385, 'min_pwm': 360, 'max_pwm': 470, 'brake_pwm': 375,
         'full_stop_threshold': 0.1, 'brake_threshold': 0.2, 'velocity_deadband': 0.05,
         'forward_start_pwm': 400, 'forward_hold_pwm': 390, 'reverse_stop_pwm': 380,
         'reverse_start_pwm': 375, 'reverse_arm_pwm': 340}
DT = 0.01           # the actuator's 100 Hz tick
REPORT_EVERY = 5    # velocity_report's 20 Hz


def cfg(**kw):
    """Return car C's shell config, reverse on, with ``kw`` changed."""
    return ShellConfig(**{**CAR_C, 'reverse_enabled': True, **kw})


class ForwardP:
    """A forward-only plug-in: break-away plus P; would brake for a negative target."""

    def __init__(self, init=385, start=400):
        self.init, self.start = init, start
        self.calls = 0
        self.resets = 0

    def reset(self):
        """Count resets."""
        self.resets += 1

    def update(self, target, measured, dt):
        """Return a forward PWM for a positive target, the legacy PID's sign otherwise."""
        self.calls += 1
        if target > 0:
            return self.start + int(40 * (target - measured))
        return self.init + int(60 * (target - measured))


class EscCar:
    """
    The real ESC, ahead of the motor model; truth for the shell's EscModel.

    ``arm`` is where the ESC really arms (a profile can be off by its 5-count
    step); ``state`` is its power-on state.
    """

    def __init__(self, c, state=EscState.LOCKED, arm=None):
        self.fs, self.fh = c.forward_start_pwm, c.forward_hold_pwm or c.init_pwm
        self.rs, self.rst = c.reverse_stop_pwm, c.reverse_start_pwm
        self.arm = c.reverse_arm_pwm if arm is None else arm
        self.arm_return = c.reverse_stop_pwm
        self.init = c.init_pwm
        self.state = state
        self.plant = MotorPlant(MotorPlantParams(init_pwm=c.init_pwm, deadband=0, gain=0.03,
                                                 allow_reverse=True, coast_decel=0.3))
        self.sensor = HallSpeedSensor()

    def step(self, u, dt):
        s = self.state
        if u >= self.fs:
            s = EscState.FORWARD
        elif s == EscState.ARMED:
            pass        # armed, anything below the stop reverses: a pull is a full reverse
        elif u <= self.arm:
            s = EscState.ARMING
        elif s == EscState.FORWARD and u <= self.fh:
            s = EscState.LOCKED
        elif s == EscState.ARMING and u >= self.arm_return:
            s = EscState.ARMED
        self.state = s
        v = self.plant.speed
        p = self.plant.params
        if s == EscState.FORWARD:
            p.allow_reverse, eff = False, max(u, self.init + 1)
        elif s == EscState.ARMED and (u <= self.rst or (u < self.rs and v < -0.01)):
            p.allow_reverse, eff = True, min(u, self.init - 1)
        elif u < self.rs:
            p.allow_reverse, eff = False, self.init - 10     # brakes, never reverses
        else:
            p.allow_reverse, eff = False, self.init          # coasts
        self.plant.step(eff, dt)


class Run:
    """Close the loop: shell -> ESC car -> hall sensor -> signed report -> shell."""

    def __init__(self, shell, car):
        self.shell, self.car = shell, car
        self.t = 0.0
        self.i = 0
        self.report = 0.0
        self.report_t = 0.0
        self.log = []   # (t, target, pwm, true speed, reported, mode, model state, true state)
        self.before = []    # the real ESC's state as each PWM reached it

    def go(self, seconds, target, override=None, stale=False, gear='auto'):
        """
        Run ``seconds``; ``gear`` is Autoware's gear command.

        'auto' follows the target's sign as the shift decider does (and keeps it
        at zero); True/False hold REVERSE / DRIVE.
        """
        n = int(round(seconds / DT))
        for _ in range(n):
            self.i += 1
            self.t = round(self.i * DT, 6)
            tgt = target(self.t) if callable(target) else target
            if gear == 'auto':
                if tgt is not None and tgt != 0.0:
                    self.shell.set_gear_reverse(tgt < 0.0)
            else:
                self.shell.set_gear_reverse(gear)
            self.before.append(self.car.state)
            if self.i % REPORT_EVERY == 0 and not stale:
                mag = self.car.sensor.report(self.t)
                sign = -1.0 if self.shell.drive_direction == DIRECTION_REVERSE else 1.0
                self.report = sign * mag
                self.report_t = self.t
            speed = None if stale else self.report
            pwm = self.shell.step(self.t, tgt, speed, self.report_t, override)
            self.car.step(pwm, DT)
            self.car.sensor.update(self.t, self.car.plant.distance)
            self.log.append((self.t, tgt, pwm, self.car.plant.speed, self.report,
                             self.shell.mode, self.shell.esc_state, self.car.state))
        return self

    @property
    def speed(self):
        return self.car.plant.speed


def make(state=EscState.LOCKED, arm=None, controller=None, logger=None, **kw):
    c = cfg(**kw)
    shell = LongitudinalShell(c, controller or ForwardP(), logger or RecordingLogger())
    return Run(shell, EscCar(c, state=state, arm=arm))


def assert_no_pull_reaches_an_armed_esc(run):
    """Check the arming pull (<= reverse_arm_pwm) never goes to an ESC that is already armed."""
    arm = run.shell.config.reverse_arm_pwm
    for r, before in zip(run.log, run.before):
        assert not (r[2] <= arm and before == EscState.ARMED), \
            f't={r[0]}: pull {r[2]} to an armed ESC (a full-strength reverse)'


def assert_never_unasked_reverse(run):
    """No tick sends a sub-stop PWM to an armed ESC unless the shell is reversing on purpose."""
    c = run.shell.config
    for t, tgt, pwm, v, rep, mode, model, truth in run.log:
        if truth == EscState.ARMED and pwm < c.reverse_stop_pwm:
            assert mode == Mode.REVERSE and tgt is not None and tgt < -c.full_stop_threshold, \
                f't={t}: {pwm} to an armed ESC in {mode} for target {tgt}'


# ------------------------------------------------------------ off by default

def test_reverse_is_off_by_default():
    assert ShellConfig(**CAR_C).reverse_enabled is False
    params = normalize_params({})
    assert params['reverse_enabled'] is False


def test_a_negative_target_with_reverse_off_never_reverses():
    run = make(reverse_enabled=False).go(1.0, 0.0).go(5.0, -0.4)
    assert min(r[3] for r in run.log) >= 0.0
    assert min(r[2] for r in run.log) > CAR_C['reverse_arm_pwm']   # never pulled to ARM
    assert run.car.state != EscState.ARMED
    assert run.shell.drive_direction == DIRECTION_FORWARD


@pytest.mark.parametrize('target', [0.0, 0.5, -0.05, -0.09])
def test_reverse_on_and_a_target_at_or_above_minus_threshold_never_reverses(target):
    run = make().go(1.0, 0.0).go(4.0, target)
    assert min(r[3] for r in run.log) >= 0.0
    assert min(r[2] for r in run.log) > CAR_C['reverse_arm_pwm']
    assert all(r[5] not in (Mode.REVERSE, Mode.REVERSE_ARM) for r in run.log)


@pytest.mark.parametrize('missing', ['reverse_start_pwm', 'reverse_stop_pwm',
                                     'reverse_arm_pwm', 'forward_start_pwm'])
def test_unmeasured_thresholds_refuse_reverse(missing):
    log = RecordingLogger()
    kw = {missing: 0}
    if missing == 'reverse_stop_pwm':
        kw['reverse_start_pwm'] = 375
    run = make(logger=log, **kw)
    assert run.shell.reverse_enabled is False
    assert any(missing in m and 'reverse' in m for m in log.at('error'))
    assert reverse_unavailable_reason(cfg(**kw)) is not None
    run.go(1.0, 0.0).go(4.0, -0.4)
    assert min(r[3] for r in run.log) >= 0.0
    assert min(r[2] for r in run.log) > CAR_C['reverse_arm_pwm']


def test_a_reverse_floor_at_or_below_arm_refuses_reverse():
    log = RecordingLogger()
    run = make(logger=log, reverse_floor_pwm=340)
    assert run.shell.reverse_enabled is False and log.at('error')


# ---------------------------------------------------------------- reversing

def test_from_rest_it_arms_then_reverses_at_the_target_speed():
    c = cfg()
    run = make().go(1.0, 0.0).go(5.0, -0.3)
    pwms = [r[2] for r in run.log[100:]]
    # the arming sequence: pull to ARM or below, return to the stop, then reverse
    first_arm = next(i for i, p in enumerate(pwms) if p <= c.reverse_arm_pwm)
    first_back = next(i for i, p in enumerate(pwms) if p < c.reverse_stop_pwm
                      and i > first_arm and pwms[i - 1] >= c.reverse_stop_pwm)
    assert all(p <= c.reverse_arm_pwm for p in pwms[first_arm:first_arm + 20])   # held 0.2 s+
    between = pwms[first_arm:first_back]
    ret = [p for p in between if p > c.reverse_arm_pwm]
    assert ret and all(c.reverse_stop_pwm <= p < c.forward_start_pwm for p in ret)
    assert pwms[first_back] <= c.reverse_start_pwm             # from rest: at or below REV_START
    # the car went backward at the target, within tolerance, over the last 1.5 s
    tail = [r[3] for r in run.log[-150:]]
    assert max(tail) < 0.0
    assert sum(tail) / len(tail) == pytest.approx(-0.3, abs=0.05)
    assert run.car.state == EscState.ARMED
    assert run.shell.esc_state == EscState.ARMED
    assert run.shell.drive_direction == DIRECTION_REVERSE
    assert run.shell.mode == Mode.REVERSE
    assert_never_unasked_reverse(run)


def test_the_reverse_command_stays_within_its_floor_and_below_the_stop():
    c = cfg()
    run = make().go(1.0, 0.0).go(5.0, -0.3)
    for r in run.log:
        if r[5] == Mode.REVERSE:
            assert c.min_pwm <= r[2] < c.reverse_stop_pwm


def test_reverse_speed_is_capped():
    run = make(reverse_max_speed=0.4).go(1.0, 0.0).go(6.0, -2.0)
    tail = [r[3] for r in run.log[-200:]]
    assert min(tail) > -0.4 - 0.06
    assert sum(tail) / len(tail) == pytest.approx(-0.4, abs=0.06)
    assert min(r[2] for r in run.log if r[5] == Mode.REVERSE) >= CAR_C['min_pwm']


def test_reverse_floor_bounds_the_command():
    run = make(reverse_floor_pwm=370).go(1.0, 0.0).go(5.0, -2.0)
    assert min(r[2] for r in run.log if r[5] == Mode.REVERSE) >= 370


def test_the_esc_model_tracks_the_real_esc_through_arm_reverse_stop_forward():
    run = make().go(1.0, 0.0).go(5.0, -0.3).go(4.0, 0.0).go(4.0, 0.5).go(3.0, 0.0)
    armed_at = next(i for i, r in enumerate(run.log) if r[7] == EscState.ARMING)
    for t, tgt, pwm, v, rep, mode, model, truth in run.log[armed_at:]:
        assert model == truth, f't={t}: model {model}, ESC {truth} after {pwm}'
    assert run.car.state != EscState.ARMED     # forward drive disarmed it


def test_moving_forward_a_reverse_target_brakes_first_and_arms_only_once_stopped():
    c = cfg()
    run = make().go(1.0, 0.0).go(4.0, 0.6)
    assert run.speed > 0.3
    run.go(6.0, -0.3)
    tail = run.log[500:]
    arm_ticks = [r for r in tail if r[2] <= c.reverse_arm_pwm]
    assert arm_ticks
    assert all(abs(r[3]) < 0.02 for r in arm_ticks)            # only at a standstill
    before = tail[:tail.index(arm_ticks[0])]
    assert all(r[2] > c.reverse_arm_pwm for r in before)       # the stopping brake stays above ARM
    assert run.speed < -0.2
    assert_never_unasked_reverse(run)


# -------------------------------------------------------------- leaving reverse

def test_stopping_from_reverse_waits_for_a_standstill_before_forward():
    c = cfg()
    run = make().go(1.0, 0.0).go(5.0, -0.4)
    assert run.speed < -0.3
    n = len(run.log)
    run.go(6.0, 0.5)
    after = run.log[n:]
    for t, tgt, pwm, v, rep, mode, model, truth in after:
        if v < -0.005:          # still rolling backward
            assert c.reverse_stop_pwm <= pwm < c.forward_start_pwm, \
                f't={t}: {pwm} while moving backward at {v}'
            assert mode == Mode.REVERSE_STOP
    first_fwd = next(i for i, r in enumerate(after) if r[2] >= c.forward_start_pwm)
    assert all(r[3] > -0.005 for r in after[first_fwd:])
    assert run.speed > 0.3
    assert run.car.state == EscState.FORWARD
    assert run.shell.drive_direction == DIRECTION_FORWARD
    assert_never_unasked_reverse(run)


def test_stop_from_reverse_leaves_the_esc_armed_at_the_stop():
    c = cfg()
    run = make().go(1.0, 0.0).go(4.0, -0.4).go(4.0, 0.0)
    assert abs(run.speed) < 1e-3
    assert run.log[-1][2] == c.init_pwm
    assert run.car.state == EscState.ARMED and run.shell.esc_state == EscState.ARMED
    assert_never_unasked_reverse(run)


def test_a_second_reverse_leg_does_not_re_arm():
    c = cfg()
    run = make().go(1.0, 0.0).go(4.0, -0.4).go(3.0, 0.0)
    n = len(run.log)
    run.go(4.0, -0.3)
    assert all(r[2] > c.reverse_arm_pwm for r in run.log[n:])
    assert run.speed < -0.2


def test_the_direction_latches_backward_until_the_car_stops():
    run = make().go(1.0, 0.0)
    assert run.shell.drive_direction == DIRECTION_FORWARD
    run.go(4.0, -0.4)
    assert run.shell.drive_direction == DIRECTION_REVERSE and run.report < -0.2
    n = len(run.log)
    run.go(4.0, 0.0)
    for r in run.log[n:]:
        if r[3] < -0.05:
            assert r[4] <= 0.0      # the report stays signed backward while it rolls
    assert run.shell.drive_direction == DIRECTION_FORWARD
    run.go(3.0, 0.5)
    assert run.shell.drive_direction == DIRECTION_FORWARD and run.report > 0.2


# ----------------------------------------------------------------- overrides

@pytest.mark.parametrize('override', [OVERRIDE_EMERGENCY, 'command_timeout', 'controller_fault'])
def test_an_override_while_reversing_coasts_to_a_stop_at_the_stop_pwm(override):
    c = cfg()
    run = make().go(1.0, 0.0).go(4.0, -0.4)
    assert run.speed < -0.3
    n = len(run.log)
    run.go(3.0, -0.4, override=override)
    after = run.log[n:]
    assert all(r[2] == c.init_pwm for r in after)       # never brake_pwm: armed, it reverses
    assert all(r[5] == Mode.OVERRIDE_NEUTRAL for r in after)
    assert abs(run.speed) < 1e-3
    assert run.car.state == EscState.ARMED
    assert_never_unasked_reverse(run)
    # override cleared, still asked backward: reverses again, already armed
    m = len(run.log)
    run.go(3.0, -0.4)
    assert all(r[2] > c.reverse_arm_pwm for r in run.log[m:])
    assert run.speed < -0.2


def test_a_stale_velocity_while_reversing_coasts_at_the_stop_pwm():
    c = cfg()
    run = make().go(1.0, 0.0).go(4.0, -0.4)
    n = len(run.log)
    run.go(3.0, -0.4, override=OVERRIDE_VELOCITY_TIMEOUT, stale=True)
    assert all(r[2] == c.init_pwm for r in run.log[n:])
    assert abs(run.speed) < 1e-3
    assert_never_unasked_reverse(run)


def test_an_esc_that_does_not_arm_trips_the_wheel_speed_check():
    """The profile's ARM is 10 counts above this ESC's (beyond the margin): never arms."""
    c = cfg()
    log = RecordingLogger()
    run = make(arm=c.reverse_arm_pwm - 10, logger=log, wheel_speed_check_enabled=True)
    run.go(1.0, 0.0).go(6.0, -0.4)
    assert run.shell.wheel_fault_latched
    assert run.log[-1][2] == c.init_pwm
    assert abs(run.speed) < 1e-6
    assert any('Wheel speed fault' in m for m in log.at('error'))


def test_a_reverse_wheel_fault_re_arms_on_the_next_attempt():
    c = cfg()
    run = make(arm=c.reverse_arm_pwm - 10, wheel_speed_check_enabled=True)
    run.go(1.0, 0.0).go(6.0, -0.4)
    assert run.shell.wheel_fault_latched
    run.car.arm = c.reverse_arm_pwm           # the next pull does arm
    run.go(2.0, 0.0)                          # a stop command clears the fault
    assert not run.shell.wheel_fault_latched
    n = len(run.log)
    run.go(5.0, -0.4)
    assert any(r[2] <= c.reverse_arm_pwm for r in run.log[n:])
    assert run.speed < -0.2


def test_a_shell_reverse_never_reaches_the_plug_in():
    ctl = ForwardP()
    run = make(controller=ctl).go(1.0, 0.0)
    calls = ctl.calls
    run.go(4.0, -0.4)
    assert ctl.calls == calls


# --------------------------------------------------------------- plumbing

def test_actuator_params_reach_the_shell():
    raw = {'init_pwm': 385, 'min_pwm': 360, 'max_pwm': 470, 'brake_pwm': 375,
           'full_stop_threshold': 0.1, 'brake_threshold': 0.2, 'velocity_deadband': 0.05,
           'reverse_enabled': True, 'reverse_max_speed': 0.4, 'reverse_floor_pwm': 365}
    c = make_shell_config(normalize_params(raw))
    assert c.reverse_enabled is True
    assert c.reverse_max_speed == 0.4 and c.reverse_floor_pwm == 365
    off = make_shell_config(normalize_params({k: v for k, v in raw.items()
                                              if not k.startswith('reverse_')}))
    assert off.reverse_enabled is False


# ------------------------------------------------- review fixes (2026-11-18)

def test_the_pull_goes_below_arm_by_the_margin():
    c = cfg()
    run = make().go(1.0, 0.5).go(3.0, 0.0).go(3.0, -0.3)   # forward first: the ESC is locked
    pulls = [r[2] for r in run.log if r[5] == Mode.REVERSE_ARM and r[2] < c.reverse_stop_pwm]
    assert pulls and set(pulls) == {c.reverse_arm_pwm - 5}


def test_an_esc_left_armed_at_start_reverses_without_a_pull():
    """Restarted while the ESC is armed: the model says armed, so no pull."""
    c = cfg()
    run = make(state=EscState.ARMED).go(1.0, 0.0).go(4.0, -0.3)
    assert all(r[2] > c.reverse_arm_pwm for r in run.log)
    first = next(r for r in run.log if r[5] == Mode.REVERSE)
    assert first[2] <= c.reverse_start_pwm
    assert run.speed < -0.2
    assert_no_pull_reaches_an_armed_esc(run)


def test_a_locked_esc_at_start_is_found_out_and_then_armed():
    """Fresh power-on (really locked, modelled armed): reverse fails to move it, then it arms."""
    c = cfg()
    log = RecordingLogger()
    run = make(logger=log).go(1.0, 0.0).go(6.0, -0.3)
    assert any(r[2] <= c.reverse_arm_pwm for r in run.log)
    assert run.speed < -0.2
    assert not run.shell.wheel_fault_latched
    assert_no_pull_reaches_an_armed_esc(run)


@pytest.mark.parametrize('interrupt', ['override', 'flicker'])
def test_an_abort_during_arm_return_does_not_pull_again(interrupt):
    c = cfg()
    run = make().go(1.0, 0.5).go(3.0, 0.0)          # locked
    run.go(0.45, -0.3)                               # pull 0.3 s, then into the return
    assert run.shell.reverse_phase.value == 'arm_return'
    if interrupt == 'override':
        run.go(0.1, -0.3, override=OVERRIDE_EMERGENCY)
    else:
        run.go(0.05, -0.05)
    n = len(run.log)
    run.go(3.0, -0.3)
    assert all(r[2] > c.reverse_arm_pwm for r in run.log[n:])
    assert run.speed < -0.2
    assert_no_pull_reaches_an_armed_esc(run)


CAR_A = {'init_pwm': 380, 'min_pwm': 360, 'max_pwm': 470, 'brake_pwm': 360,
         'forward_start_pwm': 385, 'forward_hold_pwm': 0, 'reverse_stop_pwm': 375,
         'reverse_start_pwm': 365, 'reverse_arm_pwm': 345}


@pytest.mark.parametrize('car', [CAR_A, {}], ids=['car_a', 'car_c'])
def test_a_dead_hall_sensor_while_reversing_latches_the_wheel_fault(car):
    log = RecordingLogger()
    shell = LongitudinalShell(cfg(wheel_speed_check_enabled=True, **car), ForwardP(), log)
    shell.set_gear_reverse(True)
    t = 0.0
    for _ in range(800):        # 8 s at rest, reading 0, asked backward
        t = round(t + DT, 6)
        shell.step(t, -0.3, 0.0, t)
    assert shell.wheel_fault_latched
    assert shell.last_output_pwm == shell.config.init_pwm
    assert any('Wheel speed fault' in m for m in log.at('error'))


def test_without_the_reverse_gear_command_a_negative_target_never_reverses():
    c = cfg()
    run = make().go(1.0, 0.0).go(5.0, -0.4, gear=False)
    assert min(r[3] for r in run.log) >= 0.0
    assert min(r[2] for r in run.log) > c.reverse_arm_pwm
    assert run.shell.drive_direction == DIRECTION_FORWARD


def test_gear_drive_while_reversing_stops_the_car():
    c = cfg()
    run = make().go(1.0, 0.0).go(4.0, -0.4)
    assert run.speed < -0.3
    n = len(run.log)
    run.go(3.0, -0.4, gear=False)
    assert all(r[2] >= c.reverse_stop_pwm for r in run.log[n:])
    assert abs(run.speed) < 1e-3


def test_motion_during_arming_is_not_a_reverse_episode():
    shell = LongitudinalShell(cfg(reverse_standstill_time=0.0), ForwardP(), RecordingLogger())
    shell.esc.state = EscState.LOCKED
    shell.set_gear_reverse(True)
    shell.step(0.01, -0.3, 0.0, 0.01)
    assert shell.mode == Mode.REVERSE_ARM
    shell.step(0.02, -0.3, 0.3, 0.02)      # the car moves (pushed): abort
    assert shell.drive_direction == DIRECTION_FORWARD


def test_a_stale_velocity_gives_no_standstill_credit():
    shell = LongitudinalShell(cfg(), ForwardP(), RecordingLogger())
    shell.esc.state = EscState.LOCKED
    shell.set_gear_reverse(True)
    t, v = 0.0, 0.0
    for i in range(300):            # stand, arm, reverse (fed a moving reading)
        t = round(t + DT, 6)
        v = 0.0 if shell.mode != Mode.REVERSE else -0.3
        shell.step(t, -0.3, v, t)
    assert shell.mode == Mode.REVERSE
    shell.step(t + DT, -0.3, -0.05, t + DT)        # slowing: last reading below the threshold
    for i in range(100):
        t = round(t + DT, 6)
        shell.step(t + DT, -0.3, None, t, OVERRIDE_VELOCITY_TIMEOUT)
    assert shell.drive_direction == DIRECTION_REVERSE    # still unknown: still backward


@pytest.mark.parametrize('vehicle_id', ['car_a', 'car_b', 'car_c', 'sim'])
def test_every_instance_can_reverse(vehicle_id):
    """The profiles in individual_params give a reverse the shell accepts (floor included)."""
    from autosdv_vehicle_interface import car_profile
    c = make_shell_config(normalize_params({**car_profile.load(vehicle_id),
                                            'reverse_enabled': True}))
    assert reverse_unavailable_reason(c) is None
