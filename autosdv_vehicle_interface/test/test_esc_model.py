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

"""The ESC model in longitudinal.py: state machine, profile checks, reverse guard."""
from autosdv_vehicle_interface.longitudinal import (
    check_esc_config,
    EscModel,
    EscState,
    LongitudinalShell,
    Mode,
    ShellConfig,
)
import pytest

# Car C, measured 2026-10-19 (docs/design/esc-pwm-model.md)
CAR_C = dict(init_pwm=385, min_pwm=360, max_pwm=470, brake_pwm=375,
             full_stop_threshold=0.1, brake_threshold=0.2, velocity_deadband=0.05,
             forward_start_pwm=400, forward_hold_pwm=390, reverse_stop_pwm=380,
             reverse_start_pwm=375, reverse_arm_pwm=340)


def cfg(**kw):
    """Return car C's shell config with ``kw`` changed."""
    return ShellConfig(**{**CAR_C, **kw})


class Const:
    """A speed controller that always answers one PWM."""

    def __init__(self, pwm):
        self.pwm = pwm

    def reset(self):
        """Nothing to reset."""

    def update(self, target, measured, dt):
        """Return the fixed PWM."""
        return self.pwm


def test_the_state_machine_follows_car_cs_measurements():
    """Lock, forward, arm, reverse, disarm: the sequence the car showed."""
    m = EscModel(cfg())
    assert m.state == EscState.ARMED            # unknown at start: assume armed
    assert m.update(400) == EscState.FORWARD    # forward disarms
    assert m.update(395) == EscState.FORWARD    # 395 holds once moving
    assert m.update(390) == EscState.LOCKED     # released
    assert m.update(375) == EscState.LOCKED     # below the stop only brakes
    assert m.update(340) == EscState.ARMING     # the pull that arms
    assert m.update(385) == EscState.ARMED      # the return
    assert m.update(375) == EscState.ARMED      # reverses
    assert m.update(400) == EscState.FORWARD    # disarmed


def test_unmeasured_thresholds_leave_the_model_off():
    """A profile without thresholds: no model, no checks, the shell as before."""
    m = EscModel(cfg(forward_start_pwm=0, reverse_arm_pwm=0))
    assert m.state == EscState.UNMODELLED and m.update(300) == EscState.UNMODELLED
    check_esc_config(cfg(brake_pwm=340, forward_start_pwm=0, reverse_arm_pwm=0,
                         reverse_stop_pwm=0))


@pytest.mark.parametrize('kw, why', [
    (dict(brake_pwm=340), 'brake_pwm 340 is at or below reverse_arm_pwm'),
    (dict(min_pwm=340), 'min_pwm 340 is at or below reverse_arm_pwm'),
    (dict(init_pwm=400), 'not below forward_start_pwm'),
    (dict(init_pwm=378), 'below reverse_stop_pwm'),
    (dict(forward_hold_pwm=402), 'forward_hold_pwm 402'),
    (dict(reverse_start_pwm=380), 'reverse_start_pwm 380'),
])
def test_a_profile_that_could_reverse_or_drive_unasked_is_refused(kw, why):
    """Each unsafe threshold combination raises, naming the cause."""
    with pytest.raises(ValueError, match=why):
        LongitudinalShell(cfg(**kw), Const(385))


def test_armed_the_shell_never_reverses_the_car_unasked():
    """A sub-stop command while armed becomes the stop; forward disarms."""
    shell = LongitudinalShell(cfg(), Const(370))     # the plug-in asks for 370
    pwm = shell.step(0.0, 0.5, 0.3, 0.0)             # forward target, moving
    assert shell.mode == Mode.ACTIVE and pwm == 385   # armed at start: 370 refused
    shell.controller = Const(405)
    assert shell.step(0.1, 0.5, 0.3, 0.1) == 405 and shell.esc_state == EscState.FORWARD
    shell.controller = Const(370)
    assert shell.step(0.2, 0.5, 0.3, 0.2) == 370      # locked now: 370 only brakes


def test_a_backward_target_may_go_below_the_stop():
    """The guard lets a sub-stop command through when reverse is asked for."""
    shell = LongitudinalShell(cfg(), Const(370))
    assert shell.step(0.0, -0.5, -0.3, 0.0) == 370
