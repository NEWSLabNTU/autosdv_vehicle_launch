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

"""params/cars/*.param.yaml: each car file only overrides actuator.yaml."""
import glob
import os

import pytest
import yaml

PARAMS = os.path.join(os.path.dirname(__file__), '..', 'params')
CARS = sorted(glob.glob(os.path.join(PARAMS, 'cars', '*.param.yaml')))


def ros_params(path):
    """Return a param file's ros__parameters for every node."""
    with open(path) as f:
        return yaml.safe_load(f)['/**']['ros__parameters'] or {}


BASE = ros_params(os.path.join(PARAMS, 'actuator.yaml'))


def test_car_a_is_actuator_yaml():
    """car_a restates actuator.yaml, the Velodyne car's values; they must agree."""
    car_a = ros_params(os.path.join(PARAMS, 'cars', 'car_a.param.yaml'))
    assert car_a == {k: BASE[k] for k in car_a}


# What every car sets. A profile left partial would inherit car A's stop PWM.
# tire_angle_to_steer_ratio is set only once measured: unset, actuator.yaml's
# or a launch's stand-in (Lab 2's actuator_param_overrides) applies.
REQUIRED = {'init_pwm', 'brake_pwm', 'init_steer', 'min_steer', 'max_steer'}
OPTIONAL = {'tire_angle_to_steer_ratio'}


@pytest.mark.parametrize('path', CARS, ids=os.path.basename)
def test_every_car_sets_its_pwm_points(path):
    """Each profile sets the PWM points, and only those plus a measured ratio."""
    keys = set(ros_params(path))
    assert REQUIRED <= keys, REQUIRED - keys
    assert keys <= REQUIRED | OPTIONAL, keys - REQUIRED - OPTIONAL


@pytest.mark.parametrize('path', CARS, ids=os.path.basename)
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
    from autosdv_vehicle_interface import car_profile
    assert car_profile.names(PARAMS) == [os.path.basename(c)[:-len('.param.yaml')] for c in CARS]
    for path in CARS:
        name = os.path.basename(path)[:-len('.param.yaml')]
        assert car_profile.load(name, PARAMS) == {**BASE, **ros_params(path)}
    with pytest.raises(ValueError, match='unknown car profile'):
        car_profile.load('car_z', PARAMS)
    with pytest.raises(ValueError):
        car_profile.load('', PARAMS)

