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
A car's PWM points without ROS: actuator.yaml merged with the car's instance file.

The instance file is individual_params/config/<vehicle_id>/autosdv_vehicle/
actuator.param.yaml. The vehicle interface loads the same two files through its
launch file (vehicle_id:=); this is for tools that drive the PCA9685 themselves
or model the car offline, so they read the numbers from the one place instead
of keeping a copy::

    from autosdv_vehicle_interface import car_profile
    p = car_profile.load('car_b')       # {'init_pwm': 395, 'init_steer': 400, ...}
    car_profile.names()                 # ['car_a', 'car_b', 'car_c']
"""
import argparse
import os
from typing import Any, Dict, List, Optional

import yaml

PACKAGE = 'autosdv_vehicle_interface'
INSTANCES_PACKAGE = 'individual_params'
INSTANCE_FILE = os.path.join('autosdv_vehicle', 'actuator.param.yaml')
# individual_params' config/ in this workspace's source tree, relative to src/
_SOURCE_CONFIG = os.path.join('param', 'autoware_individual_params', 'individual_params', 'config')


def _share(package: str) -> Optional[str]:
    """Return a package's installed share directory, or None without ROS or the package."""
    try:
        from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
    except ImportError:     # no ROS environment
        return None
    try:
        return get_package_share_directory(package)
    except (PackageNotFoundError, OSError):     # not installed, or no AMENT_PREFIX_PATH
        return None


def params_dir() -> str:
    """Return the params/ directory: the installed share, else the source tree."""
    share = _share(PACKAGE)
    if share is not None:
        return os.path.join(share, 'params')
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'params')


def config_dir() -> str:
    """Return individual_params' config/: the installed share, else the source tree."""
    share = _share(INSTANCES_PACKAGE)
    if share is not None:
        return os.path.join(share, 'config')
    here = os.path.dirname(os.path.realpath(__file__))
    while os.path.dirname(here) != here:
        candidate = os.path.join(here, 'src', _SOURCE_CONFIG)
        if os.path.isdir(candidate):
            return candidate
        here = os.path.dirname(here)
    raise FileNotFoundError(f'{INSTANCES_PACKAGE} is neither installed nor in the source tree')


def _ros_params(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)['/**']['ros__parameters'] or {}


def instance_file(vehicle_id: str, config: Optional[str] = None) -> str:
    """Return config/<vehicle_id>/autosdv_vehicle/actuator.param.yaml's path."""
    return os.path.join(config or config_dir(), vehicle_id, INSTANCE_FILE)


def names(config: Optional[str] = None) -> List[str]:
    """List the vehicle_ids with an actuator file, e.g. ['car_a', 'car_b', 'car_c']."""
    config = config or config_dir()
    return sorted(d for d in os.listdir(config) if os.path.isfile(instance_file(d, config)))


def load(vehicle_id: str, directory: Optional[str] = None,
         config: Optional[str] = None) -> Dict[str, Any]:
    """Return actuator.yaml's parameters under the car's instance file, as the actuator sees them."""
    directory = directory or params_dir()
    config = config or config_dir()
    path = instance_file(vehicle_id, config) if vehicle_id else ''
    if not path or not os.path.isfile(path):
        raise ValueError(f'unknown vehicle_id {vehicle_id!r}: one of {", ".join(names(config))} '
                         f'(individual_params/config/<vehicle_id>/{INSTANCE_FILE})')
    return {**_ros_params(os.path.join(directory, 'actuator.yaml')), **_ros_params(path)}


def add_argument(parser: argparse.ArgumentParser, *flags: str, **kwargs) -> None:
    """Add --vehicle-id car_a|car_b|car_c (dest vehicle_id), required unless a default is given."""
    kwargs.setdefault('required', 'default' not in kwargs)
    kwargs.setdefault('dest', 'vehicle_id')
    kwargs.setdefault('help', 'the car: ' + ', '.join(names())
                      + ' (individual_params/config/<vehicle_id>/)')
    parser.add_argument(*(flags or ('--vehicle-id',)), choices=names(), **kwargs)
