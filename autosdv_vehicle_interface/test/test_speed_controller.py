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
"""Speed-controller interface and loader."""
import sys
import types

from autosdv_vehicle_interface.speed_controller import (
    AckermannSpeedPID,
    DEFAULT_SPEED_CONTROLLER,
    load_speed_controller,
    parse_class_path,
    SpeedPID,
)
import pytest

PARAMS = {
    "init_pwm": 370, "min_pwm": 360, "max_pwm": 470, "brake_pwm": 340, "rate": 100.0,
    "kp_speed": 12.0, "ki_speed": 10.0, "kd_speed": 0.001, "integral_limit": 50.0,
    "enable_conditional_integration": True, "derivative_filter_alpha": 0.15,
}


@pytest.fixture
def student_module():
    """Provide a throwaway module standing in for a student's speed_pid.py."""
    mod = types.ModuleType("fake_student_pkg.speed_pid")

    class StudentPID:
        def __init__(self, params):
            self.params = params
            self.kp = params.get("kp", 1.0)

        def reset(self):
            pass

        def update(self, target_speed, measured_speed, dt):
            return self.params["init_pwm"] + int(self.kp * (target_speed - measured_speed))

    class NoUpdate:
        def __init__(self, params):
            pass

        def reset(self):
            pass

    mod.StudentPID = StudentPID
    mod.NoUpdate = NoUpdate
    pkg = types.ModuleType("fake_student_pkg")
    pkg.__path__ = []
    sys.modules["fake_student_pkg"] = pkg
    sys.modules["fake_student_pkg.speed_pid"] = mod
    yield mod
    del sys.modules["fake_student_pkg.speed_pid"]
    del sys.modules["fake_student_pkg"]


@pytest.mark.parametrize("spec,expected", [
    ("pkg.module:Class", ("pkg.module", "Class")),
    (" pkg.module:Class ", ("pkg.module", "Class")),
    ("pkg.module.Class", ("pkg.module", "Class")),
])
def test_parse_class_path(spec, expected):
    assert parse_class_path(spec) == expected


@pytest.mark.parametrize("spec", ["Class", ":Class", "pkg.module:"])
def test_parse_class_path_rejects(spec):
    with pytest.raises(ValueError):
        parse_class_path(spec)


@pytest.mark.parametrize("spec", ["", None, DEFAULT_SPEED_CONTROLLER])
def test_default_controller(spec):
    ctl = load_speed_controller(spec, PARAMS)
    assert isinstance(ctl, AckermannSpeedPID)
    assert isinstance(ctl, SpeedPID)


def test_load_student_controller_with_params(student_module):
    params = dict(PARAMS, kp=40.0)
    ctl = load_speed_controller("fake_student_pkg.speed_pid:StudentPID", params)
    assert ctl.kp == 40.0
    assert ctl.update(1.0, 0.5, 0.01) == 390
    params["kp"] = 0.0  # the controller holds its own copy
    assert ctl.params["kp"] == 40.0


def test_load_errors(student_module):
    with pytest.raises(ImportError):
        load_speed_controller("no_such_pkg.speed_pid:SpeedPID", PARAMS)
    with pytest.raises(AttributeError):
        load_speed_controller("fake_student_pkg.speed_pid:Missing", PARAMS)
    with pytest.raises(TypeError):
        load_speed_controller("fake_student_pkg.speed_pid:NoUpdate", PARAMS)


def test_default_controller_output_and_reset():
    ctl = AckermannSpeedPID(PARAMS)
    pwm = ctl.update(1.0, 0.0, 0.01)
    # P = 12, I = 10 * 1 * 0.01 = 0.1, D ~ 0 -> 370 + int(12.1)
    assert pwm == 382
    assert ctl.last_output_offset == pytest.approx(12.1)
    p, i, d = ctl.debug_terms()
    assert (p, i) == (pytest.approx(12.0), pytest.approx(0.1))
    ctl.reset()
    assert ctl.pid.integral == 0.0 and ctl.last_output_offset == 0.0


def test_default_controller_limits_offset_to_half_range():
    ctl = AckermannSpeedPID(PARAMS)
    assert ctl.update(100.0, 0.0, 0.01) == 370 + 55
    ctl.reset()
    assert ctl.update(-100.0, 0.0, 0.01) == 370 - 55


def test_interface_base_class():
    base = SpeedPID({"a": 1})
    base.reset()
    with pytest.raises(NotImplementedError):
        base.update(1.0, 0.0, 0.01)


def test_max_pwm_offset_unset_keeps_today_s_limit():
    c = AckermannSpeedPID(dict(PARAMS))
    assert c.max_pwm_offset == 55
    assert (c.pid.min_output, c.pid.max_output) == (-55, 55)
    assert c.update(20.0, 0.0, 0.01) == 425   # saturated: 370 + 55
    c.reset()
    assert c.update(-20.0, 5.0, 0.01) == 315


def test_max_pwm_offset_100_reaches_full_throttle():
    c = AckermannSpeedPID(dict(PARAMS, max_pwm_offset=100))
    assert c.update(20.0, 0.0, 0.01) == 470


def test_max_pwm_offset_capped_at_the_actuator_clamp():
    c = AckermannSpeedPID(dict(PARAMS, max_pwm_offset=150))
    assert c.max_pwm_offset == 100   # max_pwm - init_pwm
    assert c.update(20.0, 0.0, 0.01) == 470


@pytest.mark.parametrize('bad', [0, -5])
def test_max_pwm_offset_must_be_positive(bad):
    with pytest.raises(ValueError, match='max_pwm_offset'):
        AckermannSpeedPID(dict(PARAMS, max_pwm_offset=bad))


def test_max_pwm_offset_through_the_loader():
    c = load_speed_controller(None, dict(PARAMS, max_pwm_offset=80))
    assert c.max_pwm_offset == 80


# Car C's measured ESC: neutral 385, the band 385..400, held from 391
COMP = dict(PARAMS, init_pwm=385, forward_start_pwm=400, forward_hold_pwm=390,
            full_stop_threshold=0.1, esc_deadzone_compensation=True)


def test_deadzone_compensation_off_by_default_is_the_legacy_controller():
    legacy = AckermannSpeedPID(PARAMS)
    unset = AckermannSpeedPID(dict(PARAMS, forward_start_pwm=400, forward_hold_pwm=390,
                                   full_stop_threshold=0.1))
    for measured in (0.0, 0.0, 0.2, 0.6, 0.3):
        assert unset.update(0.3, measured, 0.02) == legacy.update(0.3, measured, 0.02)


def test_deadzone_compensation_breaks_away_from_rest():
    ctl = AckermannSpeedPID(COMP)
    assert ctl.update(0.3, 0.0, 0.02) >= 400


def test_deadzone_compensation_never_drops_into_the_band_while_moving():
    ctl = AckermannSpeedPID(COMP)
    ctl.update(0.3, 0.0, 0.02)
    # far over target: the legacy PID would command well below 390
    for _ in range(50):
        assert ctl.update(0.3, 0.8, 0.02) == 391


def test_deadzone_compensation_does_not_wind_down_on_the_floor():
    ctl = AckermannSpeedPID(COMP)
    for _ in range(100):
        ctl.update(0.3, 0.8, 0.02)
    # back under target: it leaves the floor at once, not after unwinding
    assert ctl.update(0.3, 0.2, 0.02) > 391


def test_deadzone_compensation_zero_target_is_legacy():
    ctl = AckermannSpeedPID(COMP)
    legacy = AckermannSpeedPID(dict(PARAMS, init_pwm=385))
    ctl.update(0.3, 0.4, 0.02)
    for measured in (0.4, 0.2, 0.0):
        assert ctl.update(0.0, measured, 0.02) == legacy.update(0.0, measured, 0.02)


@pytest.mark.parametrize("start,hold", [(0, 390), (400, 0), (390, 390), (390, 400)])
def test_deadzone_compensation_needs_the_measured_band(start, hold):
    with pytest.raises(ValueError):
        AckermannSpeedPID(dict(COMP, forward_start_pwm=start, forward_hold_pwm=hold))


def test_drive_floor_pwm_raises_the_floor():
    ctl = AckermannSpeedPID(dict(COMP, drive_floor_pwm=397))
    ctl.update(0.3, 0.0, 0.02)
    assert ctl.update(0.3, 0.8, 0.02) == 397


@pytest.mark.parametrize("floor", [390, 400, 380])
def test_drive_floor_pwm_must_lie_inside_the_band(floor):
    with pytest.raises(ValueError):
        AckermannSpeedPID(dict(COMP, drive_floor_pwm=floor))
