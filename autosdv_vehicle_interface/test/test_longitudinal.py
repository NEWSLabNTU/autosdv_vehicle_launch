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
"""Safety shell: overrides, plug-in supervision, clamping. No ROS, no hardware."""
from autosdv_vehicle_interface.longitudinal import (
    LongitudinalShell,
    Mode,
    OVERRIDE_COMMAND_TIMEOUT,
    OVERRIDE_CONTROLLER_FAULT,
    OVERRIDE_EMERGENCY,
    OVERRIDE_WHEEL_SPEED,
    ShellConfig,
)
import pytest

INIT, MIN, MAX, BRAKE = 370, 360, 470, 340


def config(**kw):
    base = {
        "init_pwm": INIT, "min_pwm": MIN, "max_pwm": MAX, "brake_pwm": BRAKE,
        "full_stop_threshold": 0.1, "brake_threshold": 0.2, "velocity_deadband": 0.0,
        "override_brake_duration": 1.0, "controller_fault_holdoff": 0.5,
        "controller_max_faults": 3,
    }
    base.update(kw)
    return ShellConfig(**base)


class Scripted:
    """A plug-in returning a fixed value (or raising), recording its calls."""

    def __init__(self, value=400):
        self.value = value
        self.calls = []
        self.resets = 0
        self.reset_error = None

    def reset(self):
        self.resets += 1
        if self.reset_error is not None:
            raise self.reset_error

    def update(self, target_speed, measured_speed, dt):
        self.calls.append((target_speed, measured_speed, dt))
        if isinstance(self.value, BaseException):
            raise self.value
        return self.value


def make(controller=None, logger=None, **kw):
    controller = controller if controller is not None else Scripted()
    return LongitudinalShell(config(**kw), controller, logger), controller


# ---------------------------------------------------------- safety modes win

def test_waiting_until_command_and_speed():
    shell, ctl = make()
    assert shell.step(0.0, None, 0.0, 0.0) == INIT
    assert shell.step(0.0, 1.0, None, 0.0) == INIT
    assert shell.mode == Mode.WAITING
    assert ctl.calls == []


def test_brake_and_full_stop_override_the_plugin():
    shell, ctl = make(Scripted(MAX))
    assert shell.step(0.0, 0.0, 1.0, 0.0) == BRAKE
    assert shell.mode == Mode.BRAKE
    assert shell.step(0.01, 0.0, 0.05, 0.0) == INIT
    assert shell.mode == Mode.FULL_STOP
    assert ctl.calls == []


def test_plugin_output_is_clamped():
    shell, ctl = make(Scripted(10_000))
    assert shell.step(0.0, 1.0, 0.5, 0.0) == MAX
    ctl.value = -10_000
    assert shell.step(0.01, 1.0, 0.5, 0.0) == MIN
    ctl.value = 401.9
    assert shell.step(0.02, 1.0, 0.5, 0.0) == 401
    assert shell.mode == Mode.ACTIVE


def test_plugin_receives_filtered_speed_and_dt_since_report():
    shell, ctl = make()
    shell.step(1.00, 1.0, 0.5, 0.98)
    shell.step(1.01, 1.0, 1.5, 0.98)
    (t0, m0, dt0), (t1, m1, dt1) = ctl.calls
    assert m0 == 0.5
    assert m1 == pytest.approx(0.3 * 1.5 + 0.7 * 0.5)
    assert dt0 == pytest.approx(0.02) and dt1 == pytest.approx(0.03)


def test_dt_source_loop():
    shell, ctl = make(dt_source="loop")
    for k in range(3):
        shell.step(1.0 + 0.01 * k, 1.0, 0.5, 0.0)
    assert [c[2] for c in ctl.calls] == pytest.approx([0.0, 0.01, 0.01])


def test_bad_dt_source_rejected():
    with pytest.raises(ValueError):
        make(dt_source="bogus")


# ------------------------------------------------------------------ overrides

def test_override_brakes_while_moving_then_neutral(logger):
    shell, ctl = make(logger=logger)
    shell.step(0.0, 1.0, 1.0, 0.0)
    out = [shell.step(0.01 * k, 1.0, 1.0, 0.0, OVERRIDE_COMMAND_TIMEOUT) for k in range(1, 150)]
    # brake for override_brake_duration (1.0 s), then neutral even if still "moving"
    assert set(out[:99]) == {BRAKE}
    assert set(out[101:]) == {INIT}
    assert len(logger.at("warning")) == 1


def test_override_is_neutral_at_once_when_stopped():
    shell, _ = make()
    assert shell.step(0.0, 1.0, 0.0, 0.0, OVERRIDE_EMERGENCY) == INIT
    assert shell.mode == Mode.OVERRIDE_NEUTRAL


def test_override_without_speed_uses_last_reading():
    shell, _ = make(Scripted(420))
    shell.step(0.0, 1.0, 0.5, 0.0)            # last reading: moving
    assert shell.step(0.01, 1.0, None, 0.0, OVERRIDE_COMMAND_TIMEOUT) == BRAKE
    assert shell.step(1.2, 1.0, None, 0.0, OVERRIDE_COMMAND_TIMEOUT) == INIT

    shell, _ = make(Scripted(420))
    shell.step(0.0, 1.0, 0.15, 0.0)           # last reading: crawling, below brake_threshold
    assert shell.step(0.01, 1.0, None, 0.0, OVERRIDE_COMMAND_TIMEOUT) == INIT


def test_override_without_any_reading_brakes_only_if_driving_forward():
    shell, _ = make()
    shell.last_output_pwm = 420               # driven forward, speed never reported
    assert shell.step(0.0, 1.0, None, 0.0, OVERRIDE_COMMAND_TIMEOUT) == BRAKE

    shell, _ = make()
    assert shell.step(0.0, 1.0, None, 0.0, OVERRIDE_COMMAND_TIMEOUT) == INIT


def test_velocity_loss_mid_override_does_not_reapply_brake():
    """Seen live: neutral at a crawl, then the report lapses -- stay neutral."""
    shell, _ = make(Scripted(420))
    shell.step(0.0, 1.0, 0.15, 0.0)
    assert shell.step(0.01, 1.0, 0.15, 0.0, OVERRIDE_COMMAND_TIMEOUT) == INIT
    assert shell.step(0.5, 1.0, None, 0.0, OVERRIDE_COMMAND_TIMEOUT) == INIT


def test_override_resets_plugin_on_recovery(logger):
    shell, ctl = make(logger=logger)
    shell.step(0.0, 1.0, 0.5, 0.0)
    shell.step(0.01, 1.0, 0.5, 0.0, OVERRIDE_EMERGENCY)
    n_calls = len(ctl.calls)
    shell.step(0.02, 1.0, 0.5, 0.0, OVERRIDE_EMERGENCY)
    assert len(ctl.calls) == n_calls and ctl.resets == 0
    assert shell.step(0.03, 1.0, 0.5, 0.0) == 400
    assert ctl.resets == 1
    assert shell.override_reason is None
    assert any("cleared" in m for m in logger.at("info"))


def test_override_beats_everything_including_full_stop_targets():
    shell, ctl = make(Scripted(MAX))
    for reason in (OVERRIDE_EMERGENCY, OVERRIDE_COMMAND_TIMEOUT):
        assert shell.step(0.0, 2.0, 1.5, 0.0, reason) == BRAKE
    assert ctl.calls == []


# ---------------------------------------------------------- plug-in faults

def test_plugin_exception_goes_neutral_and_logs(logger):
    shell, ctl = make(Scripted(RuntimeError("student bug")), logger=logger)
    assert shell.step(0.0, 1.0, 0.05, 0.0) == INIT          # stopped: neutral
    assert shell.override_reason == OVERRIDE_CONTROLLER_FAULT
    errors = logger.at("error")
    assert len(errors) == 1 and "student bug" in errors[0] and "Traceback" in errors[0]


def test_plugin_exception_while_moving_brakes_first(logger):
    shell, ctl = make(Scripted(400), logger=logger)
    shell.step(0.0, 1.0, 1.0, 0.0)
    ctl.value = ZeroDivisionError("oops")
    assert shell.step(0.01, 1.0, 1.0, 0.0) == BRAKE


def test_plugin_fault_holdoff_then_retry_then_latch(logger):
    shell, ctl = make(Scripted(RuntimeError("bad")), logger=logger)
    t = 0.0
    shell.step(t, 1.0, 0.5, 0.0)
    assert shell.fault_count == 1
    # During the holdoff the plug-in is not called
    calls = len(ctl.calls)
    for k in range(1, 40):
        shell.step(t + 0.01 * k, 1.0, 0.5, 0.0)
    assert len(ctl.calls) == calls
    # After the holdoff: reset() and a retry, which fails again
    shell.step(0.6, 1.0, 0.5, 0.0)
    assert ctl.resets == 1 and shell.fault_count == 2
    shell.step(1.2, 1.0, 0.5, 0.0)
    assert shell.fault_count == 3 and shell.fault_latched
    # Latched: never called again, neutral forever, no new logs
    calls, n_err = len(ctl.calls), len(logger.at("error"))
    for k in range(200):
        assert shell.step(2.0 + k * 0.01, 1.0, 0.0, 0.0) == INIT
    assert len(ctl.calls) == calls and len(logger.at("error")) == n_err == 3
    assert "disabled" in logger.at("error")[-1]


def test_plugin_recovers_after_transient_fault():
    ctl = Scripted(RuntimeError("once"))
    shell, _ = make(ctl)
    shell.step(0.0, 1.0, 0.15, 0.0)
    ctl.value = 410
    assert shell.step(0.1, 1.0, 0.15, 0.0) == INIT           # holdoff
    assert shell.step(0.6, 1.0, 0.15, 0.0) == 410            # resumed
    assert not shell.fault_latched and ctl.resets == 1


def test_reset_exception_counts_as_fault():
    ctl = Scripted(RuntimeError("x"))
    shell, _ = make(ctl)
    shell.step(0.0, 1.0, 0.15, 0.0)
    ctl.value = 410
    ctl.reset_error = ValueError("reset broken")
    assert shell.step(0.6, 1.0, 0.15, 0.0) == INIT
    assert shell.fault_count == 2


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), "400", True, [400]])
def test_non_numeric_output_is_a_fault(bad):
    shell, _ = make(Scripted(bad))
    assert shell.step(0.0, 1.0, 0.15, 0.0) == INIT
    assert shell.fault_count == 1


def test_max_faults_zero_never_latches():
    shell, ctl = make(Scripted(RuntimeError("x")), controller_max_faults=0)
    for k in range(10):
        shell.step(k * 0.6, 1.0, 0.5, 0.0)
    assert shell.fault_count == 10 and not shell.fault_latched


# ---------------------------------------------------------- wheel speed check

FWD = 395   # forward_start_pwm; the check's threshold is FWD + 10


def wheel_shell(value=420, logger=None, **kw):
    kw.setdefault("wheel_speed_check_enabled", True)
    return make(Scripted(value), logger=logger, forward_start_pwm=FWD, **kw)


def drive(shell, t0, t1, speed, target=0.8, dt=0.01):
    """Step from t0 to t1; return the PWMs."""
    out = []
    for k in range(int(round((t1 - t0) / dt))):
        out.append(shell.step(t0 + k * dt, target, speed, t0 + k * dt))
    return out


def test_wheel_speed_check_trips_on_a_dead_sensor(logger):
    shell, _ = wheel_shell(logger=logger)
    out = drive(shell, 0.0, 0.99, 0.0)
    assert set(out) == {420}                       # within wheel_speed_check_time
    out = drive(shell, 0.99, 1.2, 0.0)
    assert shell.wheel_fault_latched
    assert shell.override_reason == OVERRIDE_WHEEL_SPEED
    assert out[-1] == INIT                          # neutral: the reading says stopped
    assert len(logger.at("error")) == 1


def test_wheel_speed_check_quiet_while_the_car_moves():
    shell, _ = wheel_shell()
    out = drive(shell, 0.0, 5.0, 0.6)
    assert set(out) == {420} and not shell.wheel_fault_latched


def test_wheel_speed_check_ignores_pwm_below_the_threshold():
    # a slow start sitting at forward_start + 9 never trips
    shell, _ = wheel_shell(value=FWD + 9)
    drive(shell, 0.0, 5.0, 0.0)
    assert not shell.wheel_fault_latched


def test_wheel_speed_check_needs_the_whole_window():
    # motion for one tick restarts the window
    shell, _ = wheel_shell()
    drive(shell, 0.0, 0.8, 0.0)
    drive(shell, 0.8, 0.81, 0.3)
    drive(shell, 0.81, 1.6, 0.0)
    assert not shell.wheel_fault_latched


def test_wheel_speed_fault_holds_until_a_stop_command():
    shell, ctl = wheel_shell()
    drive(shell, 0.0, 1.2, 0.0)
    assert set(drive(shell, 1.2, 10.0, 0.0)) == {INIT}      # still commanded 0.8: held
    drive(shell, 10.0, 10.9, 0.0, target=0.0)               # stop, not long enough
    assert shell.wheel_fault_latched
    drive(shell, 10.9, 11.2, 0.0, target=0.0)
    assert not shell.wheel_fault_latched
    assert shell.step(11.3, 0.8, 0.5, 11.3) == 420           # re-engaged: control resumes
    assert shell.override_reason is None and ctl.resets == 1


def test_wheel_speed_check_off_by_parameter():
    shell, _ = wheel_shell(wheel_speed_check_enabled=False)
    out = drive(shell, 0.0, 5.0, 0.0)
    assert set(out) == {420} and not shell.wheel_fault_latched


def test_wheel_speed_check_off_without_forward_start(logger):
    shell, _ = make(Scripted(420), logger=logger, wheel_speed_check_enabled=True)
    drive(shell, 0.0, 5.0, 0.0)
    assert not shell.wheel_fault_latched
    assert any("forward_start_pwm" in m for m in logger.at("warning"))


def test_wheel_speed_check_turned_off_at_run_time_clears_the_fault():
    shell, _ = wheel_shell()
    drive(shell, 0.0, 1.2, 0.0)
    assert shell.wheel_fault_latched
    shell.set_wheel_speed_check(False)
    assert not shell.wheel_fault_latched
    out = drive(shell, 1.2, 5.0, 0.0)
    assert out[-1] == 420 and shell.override_reason is None
    shell.set_wheel_speed_check(True)
    drive(shell, 5.0, 6.2, 0.0)
    assert shell.wheel_fault_latched
