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
A car's PWM profile without ROS: actuator.yaml merged with params/cars/<car>.

The vehicle interface loads the same two files through its launch file
(car_profile:=); this is for tools that drive the PCA9685 themselves or model
the car offline, so they read the numbers from the one place instead of
keeping a copy::

    from autosdv_vehicle_interface import car_profile
    p = car_profile.load('car_b')       # {'init_pwm': 395, 'init_steer': 400, ...}
    car_profile.names()                 # ['car_a', 'car_b', 'car_c']
"""
import argparse
import os
from typing import Any, Dict, List, Optional

import yaml

PACKAGE = 'autosdv_vehicle_interface'


def params_dir() -> str:
    """Return the params/ directory: the installed share, else the source tree."""
    try:
        from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
    except ImportError:     # no ROS environment
        PackageNotFoundError = None
    if PackageNotFoundError is not None:
        try:
            return os.path.join(get_package_share_directory(PACKAGE), 'params')
        except PackageNotFoundError:     # not installed: the source tree
            pass
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'params')


def _ros_params(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)['/**']['ros__parameters'] or {}


def names(directory: Optional[str] = None) -> List[str]:
    """List the car profiles, e.g. ['car_a', 'car_b', 'car_c']."""
    cars = os.path.join(directory or params_dir(), 'cars')
    return sorted(f[:-len('.param.yaml')] for f in os.listdir(cars) if f.endswith('.param.yaml'))


def load(name: str, directory: Optional[str] = None) -> Dict[str, Any]:
    """Return actuator.yaml's parameters under the car's profile, as the actuator sees them."""
    directory = directory or params_dir()
    path = os.path.join(directory, 'cars', f'{name}.param.yaml')
    if not name or not os.path.isfile(path):
        raise ValueError(f'unknown car profile {name!r}: one of {", ".join(names(directory))}')
    return {**_ros_params(os.path.join(directory, 'actuator.yaml')), **_ros_params(path)}


def add_argument(parser: argparse.ArgumentParser, *flags: str, **kwargs) -> None:
    """Add --car-profile car_a|car_b|car_c, required unless a default is given."""
    kwargs.setdefault('required', 'default' not in kwargs)
    kwargs.setdefault('help', 'the car: ' + ', '.join(names()) + ' (params/cars/)')
    parser.add_argument(*(flags or ('--car-profile',)), choices=names(), **kwargs)
