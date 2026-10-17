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
"""ActuatorCore without ROS: the actuator's tick on explicit clocks."""
import os

import yaml

from autosdv_vehicle_interface.actuator_core import (
    ActuatorCore, build_controller_params, normalize_params)
from autosdv_vehicle_interface.longitudinal import Mode
from autosdv_vehicle_interface.speed_controller import load_speed_controller

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
ACTUATOR_YAML = os.path.join(PKG, 'params', 'actuator.yaml')
VEHICLE_INFO_YAML = os.path.join(
    PKG, '..', 'autosdv_vehicle_description', 'config', 'vehicle_info.param.yaml')


def file_params(**overrides):
    raw = {}
    for path in (ACTUATOR_YAML, VEHICLE_INFO_YAML):
        with open(path) as f:
            raw.update(yaml.safe_load(f)['/**']['ros__parameters'])
    raw.update(overrides)
    return normalize_params(raw)


def make_core(**overrides):
    params = file_params(**overrides)
    controller = load_speed_controller(None, build_controller_params(params))
    return ActuatorCore(params, controller, start_time=0.0)


def test_normalize_renames_vehicle_info_keys_and_fills_defaults():
    p = file_params()
    assert p['wheelbase'] == 0.319 and p['track_width'] == 0.263
    assert p['max_steering_angle'] == 0.349
    assert p['control_cmd_timeout'] == 0.3 and p['speed_controller_dt_source'] == 'velocity_report'


def test_waiting_until_both_inputs_arrive():
    core = make_core()
    assert core.tick(0.01, 0.01) == (370, 489)
    assert core.longitudinal.mode == Mode.WAITING


def test_drives_forward_then_brakes_on_a_stop_command():
    core = make_core(speed_controller_dt_source='loop')
    t = 0.0
    for k in range(50):
        t = 0.01 * k
        if k % 5 == 0:
            core.on_velocity(t, t, 0.5)
        if k % 3 == 0:
            core.on_command(t, 1.5, 0.0)
        motor, steer = core.tick(t, t)
    assert motor > 375 and steer == 489
    core.on_command(t + 0.01, 0.0, 0.0)
    assert core.tick(t + 0.02, t + 0.02)[0] == 340   # BRAKE: target 0, moving at 0.5 m/s


def test_command_timeout_brakes_then_neutral_and_centres_steering():
    core = make_core()
    core.on_velocity(0.0, 0.0, 1.0)
    core.on_command(0.0, 1.0, 0.1)
    assert core.tick(0.5, 0.5) == (340, 489)   # 0.5 s > control_cmd_timeout 0.3: brake, centred
    assert core.longitudinal.override_reason == 'command_timeout'


def test_gate_emergency_overrides_the_controller():
    core = make_core()
    core.on_velocity(0.0, 0.0, 2.0)
    core.on_command(0.0, 2.0, 0.0)
    core.gate_emergency = True
    assert core.tick(0.01, 0.01)[0] == 340
    assert core.longitudinal.override_reason == 'emergency'


def test_steering_clamps_to_the_lock_and_maps_through_the_ratio():
    core = make_core(tire_angle_to_steer_ratio=-143.0)
    core.on_velocity(0.0, 0.0, 0.0)
    for k in range(200):                               # let the 0.3 EMA settle
        core.on_command(0.01 * k, 0.0, 1.0)             # far beyond 0.349
        steer = core.tick(0.01 * k, 0.01 * k)[1]
    assert core.state.target_tire_angle == 0.349
    assert steer == 439                                 # min_steer: the lock, clamped
