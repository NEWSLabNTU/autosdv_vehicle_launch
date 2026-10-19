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
    """The cars the RC sim, the benches and lab2 name all have an instance file."""
    assert {'car_a', 'car_b', 'car_c'} <= set(car_profile.names(CONFIG))


def test_car_a_is_actuator_yaml():
    """car_a restates actuator.yaml, the Velodyne car's values; they must agree."""
    car_a = ros_params(car_profile.instance_file('car_a', CONFIG))
    assert car_a == {k: BASE[k] for k in car_a}


# What every car sets. A file left partial would inherit car A's stop PWM.
# tire_angle_to_steer_ratio is set only once measured: unset, actuator.yaml's
# or a launch's stand-in (Lab 2's actuator_param_overrides) applies.
REQUIRED = {'init_pwm', 'brake_pwm', 'init_steer', 'min_steer', 'max_steer'}
OPTIONAL = {'tire_angle_to_steer_ratio'}


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
    assert p['brake_pwm'] < p['init_pwm']


def test_the_loader_merges_like_the_launch():
    """car_profile.load is what the actuator sees: actuator.yaml, then the car."""
    for path in CARS:
        assert car_profile.load(vehicle_id(path), PARAMS, CONFIG) == {**BASE, **ros_params(path)}
    for bad in ('car_z', 'default', ''):
        with pytest.raises(ValueError, match='unknown vehicle_id'):
            car_profile.load(bad, PARAMS, CONFIG)


def test_the_launch_loads_the_same_file():
    """vehicle_interface.launch.xml loads the instance file car_profile reads."""
    launch = os.path.join(os.path.dirname(__file__), '..', '..', 'autosdv_vehicle_launch',
                          'launch', 'vehicle_interface.launch.xml')
    with open(launch) as f:
        text = f.read()
    assert ('$(find-pkg-share individual_params)/config/$(var vehicle_id)/'
            + car_profile.INSTANCE_FILE) in text
    assert 'car_profile' not in text
