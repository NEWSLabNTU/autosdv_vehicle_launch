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
"""The actuator node against ActuatorCore and its own sim backend; its clock under sim time.

Needs rclpy and the Autoware messages (skipped without). Run under
ROS_DOMAIN_ID=73 and flock tmp/bench.lock.
"""
import os
import time

import pytest

os.environ.setdefault('ROS_DOMAIN_ID', '73')
rclpy = pytest.importorskip('rclpy')
actuator = pytest.importorskip('autosdv_vehicle_interface.actuator')

from autosdv_vehicle_interface import hal  # noqa: E402
from autosdv_vehicle_interface.actuator_core import (  # noqa: E402
    ActuatorCore,
    build_controller_params,
)
from autosdv_vehicle_interface.safety import InputState  # noqa: E402
from autosdv_vehicle_interface.speed_controller import load_speed_controller  # noqa: E402
from autoware_control_msgs.msg import Control  # noqa: E402
from autoware_vehicle_msgs.msg import VelocityReport  # noqa: E402
from rosgraph_msgs.msg import Clock  # noqa: E402
from test_actuator_core import file_params  # noqa: E402
from test_actuator_node import load_params  # noqa: E402


@pytest.fixture(scope='module')
def ros():
    rclpy.init()
    yield
    rclpy.try_shutdown()


@pytest.fixture
def nodes(ros, fake_pca, monkeypatch):
    monkeypatch.setattr(actuator, 'PCA9685', fake_pca)
    made = []

    def _make(**overrides):
        overrides.setdefault('enable_debug_publishing', False)
        node = actuator.AutoSdvActuator(parameter_overrides=load_params(**overrides))
        made.append(node)
        return node

    yield _make
    for node in made:
        node._neutral_written = True
        node.destroy_node()


def script():
    """(tick, velocity report or None, (target, tire angle) or None) for 400 ticks of 10 ms."""
    out = []
    for k in range(400):
        velocity = None if k % 5 else min(2.0, 0.01 * k)
        command = None
        if k % 3 == 0:
            command = (1.5 if k < 250 else 0.0, 0.02 if 100 <= k < 200 else 0.0)
        out.append((k, velocity, command))
    return out


def drive_node(node, t0=1000.0):
    clock = {'t': t0}
    node._now = lambda: clock['t']
    node._monotonic = lambda: clock['t']
    node.core.state.last_update_time = t0
    for k, velocity, command in script():
        clock['t'] = t0 + 0.01 * k
        if velocity is not None:
            msg = VelocityReport()
            msg.longitudinal_velocity = velocity
            node.velocity_callback(msg)
        if command is not None:
            msg = Control()
            msg.longitudinal.velocity, msg.lateral.steering_tire_angle = command
            node.control_callback(msg)
        node.timer_callback()


def test_node_and_core_write_identical_pwm(nodes):
    node = nodes(speed_controller_dt_source='loop')
    drive_node(node)
    written = [value for _, value in node.driver.writes]       # motor, steer, motor, steer, ...
    params = file_params(speed_controller_dt_source='loop')
    core = ActuatorCore(params, load_speed_controller(None, build_controller_params(params)),
                        start_time=1000.0)
    expected = []
    for k, velocity, command in script():
        t = 1000.0 + 0.01 * k
        if velocity is not None:
            core.on_velocity(t, t, velocity)
        if command is not None:
            core.on_command(t, *command)
        expected.extend(core.tick(t, t))
    assert written == expected
    assert max(expected[0::2]) > 375                            # it did drive


def test_real_and_sim_backends_see_identical_writes(nodes):
    real = nodes(speed_controller_dt_source='loop')
    sim = nodes(speed_controller_dt_source='loop', hardware='sim')
    assert isinstance(sim.driver, hal.SimPCA9685)
    recorded = []
    sim.driver._publish = lambda channel, value: recorded.append((channel, value))
    drive_node(real)
    drive_node(sim)
    assert recorded == real.driver.writes


def test_unknown_hardware_refuses_to_start(nodes):
    with pytest.raises(ValueError, match='hardware'):
        nodes(hardware='simulator')


def test_wall_clock_without_sim_time(nodes):
    node = nodes()
    assert abs(node._now() - time.time()) < 0.5


def test_sim_time_drives_dt_and_watchdogs(nodes):
    node = nodes(use_sim_time=True)
    pub_node = rclpy.create_node('rc_sim_clock_pub')
    pub = pub_node.create_publisher(Clock, '/clock', 10)

    def set_clock(seconds):
        msg = Clock()
        msg.clock.sec = int(seconds)
        msg.clock.nanosec = int(round((seconds - int(seconds)) * 1e9))
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            pub.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.05)
            if abs(node._now() - seconds) < 1e-6:
                return
        raise AssertionError(f"the node's clock never reached {seconds}")

    try:
        set_clock(100.0)
        assert node._monotonic() == pytest.approx(100.0)
        msg = VelocityReport()
        node.velocity_callback(msg)
        assert node.core.state.last_update_time == pytest.approx(100.0)
        time.sleep(0.6)                      # wall time passes, sim time does not
        assert node.core.velocity_watchdog.check(node._monotonic()) == InputState.FRESH
        set_clock(100.6)                     # 0.6 s of sim time > velocity_timeout
        assert node.core.velocity_watchdog.check(node._monotonic()) == InputState.TIMED_OUT
    finally:
        pub_node.destroy_node()


GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'golden_real_pwm.json')


def run_scenario(node):
    """rc_scenario through the node's callbacks on a fake clock."""
    import rc_scenario as sc
    clock = {'t': sc.T0}
    node._now = lambda: clock['t']
    node._monotonic = lambda: clock['t']
    node.core.state.last_update_time = sc.T0
    for k, kind, args in sc.events():
        clock['t'] = sc.T0 + sc.TICK * k
        if kind == 'vel':
            msg = VelocityReport()
            msg.longitudinal_velocity = args[0]
            node.velocity_callback(msg)
        elif kind == 'cmd':
            msg = Control()
            msg.longitudinal.velocity, msg.lateral.steering_tire_angle = args
            node.control_callback(msg)
        elif kind == 'gate':
            node.gate_emergency = args[0]
        else:
            node.timer_callback()


def test_real_path_matches_the_pre_refactor_set_pwm_sequence(nodes):
    """golden_real_pwm.json was recorded from the node before ActuatorCore existed."""
    import json
    node = nodes(speed_controller_dt_source='loop')
    run_scenario(node)
    with open(GOLDEN) as f:
        golden = [tuple(w) for w in json.load(f)]
    assert len(golden) == 1200
    assert node.driver.writes == golden


def test_dry_run_writes_nothing_and_still_ticks(nodes):
    node = nodes(speed_controller_dt_source='loop', dry_run=True)
    assert node.driver is None
    run_scenario(node)
    assert node.core.longitudinal.mode is not None and node.state.current_speed is not None
