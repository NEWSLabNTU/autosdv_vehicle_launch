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
