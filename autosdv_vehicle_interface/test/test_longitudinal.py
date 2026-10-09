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
