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
individual_params/config/<vehicle_id>/autosdv_vehicle/actuator.param.yaml.

Each car's instance file only overrides actuator.yaml.
"""
import os

import pytest
import yaml

from autosdv_vehicle_interface import car_profile
from autosdv_vehicle_interface.longitudinal import check_esc_config, ShellConfig

PARAMS = os.path.join(os.path.dirname(__file__), '..', 'params')
CONFIG = car_profile.config_dir()
CARS = [car_profile.instance_file(v, CONFIG) for v in car_profile.names(CONFIG)]


def ros_params(path):
    """Return a param file's ros__parameters for every node."""
    with open(path) as f:
        return yaml.safe_load(f)['/**']['ros__parameters'] or {}


def vehicle_id(path):
    """Return the vehicle_id a .../<vehicle_id>/autosdv_vehicle/actuator.param.yaml is for."""
    return os.path.basename(os.path.dirname(os.path.dirname(path)))


BASE = ros_params(os.path.join(PARAMS, 'actuator.yaml'))


def test_the_fleet_is_there():
    """The cars lab2 names, and sim (the RC sim's and the benches'), have an instance file."""
    assert {'sim', 'car_a', 'car_b', 'car_c'} <= set(car_profile.names(CONFIG))


def test_sim_is_actuator_yaml():
    """sim, the simulated nominal car, restates actuator.yaml, which the plants model.

    car_a used to be this car; it is measured now, so only sim is pinned to the file."""
    sim = ros_params(car_profile.instance_file('sim', CONFIG))
    pwm = {k: v for k, v in sim.items() if k not in ESC_POINTS}
    assert pwm == {k: BASE[k] for k in pwm}
    # unmodelled: actuator.yaml's ratio (or a launch's stand-in)
    assert 'tire_angle_to_steer_ratio' not in sim
    # the simulated ESC's own thresholds (the RC sim's plant reads them), all
    # set, and a profile the shell accepts
    assert ESC_POINTS <= set(sim)
    p = {**BASE, **sim}
    check_esc_config(ShellConfig(init_pwm=p['init_pwm'], min_pwm=p['min_pwm'],
                                 max_pwm=p['max_pwm'], brake_pwm=p['brake_pwm'],
                                 full_stop_threshold=p['full_stop_threshold'],
                                 brake_threshold=p['brake_threshold'],
                                 velocity_deadband=p['velocity_deadband'],
                                 **{k: p[k] for k in ESC_POINTS}))


# What every car sets. A file left partial would inherit car A's stop PWM.
# tire_angle_to_steer_ratio is set only once measured: unset, actuator.yaml's
# or a launch's stand-in (Lab 2's actuator_param_overrides) applies.
REQUIRED = {'init_pwm', 'brake_pwm', 'init_steer', 'min_steer', 'max_steer'}
# The ESC's measured points (actuator.yaml: 0 = not measured) likewise.
ESC_POINTS = {'forward_start_pwm', 'forward_hold_pwm', 'reverse_stop_pwm',
              'reverse_start_pwm', 'reverse_arm_pwm'}
OPTIONAL = {'tire_angle_to_steer_ratio'} | ESC_POINTS


@pytest.mark.parametrize('path', CARS, ids=vehicle_id)
def test_every_car_sets_its_pwm_points(path):
    """Each instance sets the PWM points, and only those plus a measured ratio."""
    keys = set(ros_params(path))
    assert REQUIRED <= keys, REQUIRED - keys
    assert keys <= REQUIRED | OPTIONAL, keys - REQUIRED - OPTIONAL


@pytest.mark.parametrize('path', CARS, ids=vehicle_id)
def test_a_car_file_only_overrides_actuator_yaml(path):
    """A car file sets only actuator.yaml keys, to a usable PWM layout."""
    car = ros_params(path)
    # a misspelt key would load as a new parameter and change nothing
    assert set(car) <= set(BASE), set(car) - set(BASE)
    p = {**BASE, **car}
    assert p['min_steer'] < p['init_steer'] < p['max_steer']
    assert p['min_pwm'] <= p['init_pwm'] < p['max_pwm']
    # below neutral; it may sit above min_pwm (car C's provisional brake 375, min 360)
    assert p['brake_pwm'] < p['init_pwm']


def tuning_file(path):
    """Return the car's speed_controller.param.yaml beside its actuator file."""
    return os.path.join(os.path.dirname(path), 'speed_controller.param.yaml')


# What a car's tuning file may set: the default controller's gains and its ESC
# dead-band compensation. PWM points stay in actuator.param.yaml.
TUNING = {'kp_speed', 'ki_speed', 'kd_speed', 'integral_limit',
          'enable_conditional_integration', 'derivative_filter_alpha',
          'esc_deadzone_compensation', 'drive_floor_pwm'}


@pytest.mark.parametrize('path', CARS, ids=vehicle_id)
def test_every_car_has_a_tuning_file_of_tuning_only(path):
    """The launch loads it for every car, so it must exist; only TUNING keys."""
    keys = set(ros_params(tuning_file(path)))
    assert keys <= TUNING, keys - TUNING
    assert keys <= set(BASE), keys - set(BASE)


@pytest.mark.parametrize('path', CARS, ids=vehicle_id)
def test_a_compensating_car_has_its_band_measured(path):
    """esc_deadzone_compensation needs forward_hold_pwm < floor < forward_start_pwm."""
    p = car_profile.load(vehicle_id(path), PARAMS, CONFIG)
    if p.get('esc_deadzone_compensation'):
        floor = p['drive_floor_pwm'] or p['forward_hold_pwm'] + 1
        assert 0 < p['forward_hold_pwm'] < floor < p['forward_start_pwm']


def test_the_loader_merges_like_the_launch():
    """car_profile.load is what the actuator sees: actuator.yaml, the car, its tuning."""
    for path in CARS:
        assert car_profile.load(vehicle_id(path), PARAMS, CONFIG) == \
            {**BASE, **ros_params(path), **ros_params(tuning_file(path))}
    for bad in ('car_z', 'default', ''):
        with pytest.raises(ValueError, match='unknown vehicle_id'):
            car_profile.load(bad, PARAMS, CONFIG)


def test_the_launch_loads_the_same_file():
    """vehicle_interface.launch.xml loads the instance file car_profile reads."""
    launch = os.path.join(os.path.dirname(__file__), '..', '..', 'autosdv_vehicle_launch',
                          'launch', 'vehicle_interface.launch.xml')
    with open(launch) as f:
        text = f.read()
    for name in (car_profile.INSTANCE_FILE, car_profile.TUNING_FILE):
        assert '$(find-pkg-share individual_params)/config/$(var vehicle_id)/' + name in text
    assert 'car_profile' not in text
