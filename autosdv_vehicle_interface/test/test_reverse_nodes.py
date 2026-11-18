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
reverse_enabled in the nodes: the actuator's direction, the signed speed, the gear report.

Needs rclpy and the Autoware message packages (skipped without them); never
touches I2C or GPIO.
"""
import os

import pytest

os.environ.setdefault('ROS_DOMAIN_ID', '76')
rclpy = pytest.importorskip('rclpy')
actuator = pytest.importorskip('autosdv_vehicle_interface.actuator')

from autosdv_vehicle_interface import gear_manager, velocity_report  # noqa: E402
from autoware_control_msgs.msg import Control  # noqa: E402
from autoware_internal_debug_msgs.msg import Float32MultiArrayStamped  # noqa: E402
from autoware_vehicle_msgs.msg import GearCommand, GearReport, VelocityReport  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from std_msgs.msg import Int8  # noqa: E402
from test_actuator_node import load_params  # noqa: E402

CAR_C_ESC = {'init_pwm': 385, 'brake_pwm': 375, 'forward_start_pwm': 400,
             'forward_hold_pwm': 390, 'reverse_stop_pwm': 380, 'reverse_start_pwm': 375,
             'reverse_arm_pwm': 340}
VR_PARAMS = {'pin': 15, 'rate': 20.0, 'wheel_diameter': 10.5, 'markers_per_rotation': 12,
             'frame_id': 'base_link', 'hardware': 'sim'}
GM_PARAMS = {'init_pwm': 385, 'min_pwm': 360, 'max_pwm': 470, 'brake_pwm': 375}


@pytest.fixture(scope='module')
def ros():
    rclpy.init()
    yield
    rclpy.try_shutdown()


@pytest.fixture
def nodes(ros):
    made = []
    yield made
    for n in made:
        if hasattr(n, '_neutral_written'):
            n._neutral_written = True
        n.destroy_node()


# ------------------------------------------------------------------- actuator

def make_actuator(nodes, monkeypatch, fake_pca, **overrides):
    monkeypatch.setattr(actuator, 'PCA9685', fake_pca)
    overrides.setdefault('enable_debug_publishing', False)
    node = actuator.AutoSdvActuator(parameter_overrides=load_params(**overrides))
    nodes.append(node)
    return node


def command(node, v):
    msg = Control()
    msg.longitudinal.velocity = float(v)
    now_ns = node.get_clock().now().nanoseconds
    msg.stamp.sec, msg.stamp.nanosec = divmod(now_ns, 10**9)
    node.control_callback(msg)


def speed(node, v):
    msg = VelocityReport()
    msg.longitudinal_velocity = float(v)
    node.velocity_callback(msg)


def test_actuator_reverse_is_off_by_default(nodes, monkeypatch, fake_pca):
    node = make_actuator(nodes, monkeypatch, fake_pca, **CAR_C_ESC)
    assert node.get_parameter('reverse_enabled').value is False
    assert node.longitudinal.reverse_enabled is False


def published_direction(node, monkeypatch):
    sent = []
    monkeypatch.setattr(node.direction_publisher, 'publish', lambda m: sent.append(m.data))
    speed(node, 0.0)
    command(node, 0.0)
    node.timer_callback()
    return sent


def test_actuator_refuses_reverse_without_the_cars_thresholds(nodes, monkeypatch, fake_pca):
    # actuator.yaml alone: no ESC thresholds measured
    node = make_actuator(nodes, monkeypatch, fake_pca, reverse_enabled=True)
    assert node.longitudinal.reverse_enabled is False
    assert published_direction(node, monkeypatch) == [0]     # the verdict: unavailable


def test_actuator_with_reverse_off_publishes_unavailable(nodes, monkeypatch, fake_pca):
    node = make_actuator(nodes, monkeypatch, fake_pca, **CAR_C_ESC)
    assert published_direction(node, monkeypatch) == [0]


def test_a_refused_actuator_keeps_the_gear_manager_off_reverse(nodes, monkeypatch, fake_pca):
    """The floor is an actuator-only check: the gear manager learns the verdict from the topic."""
    act = make_actuator(nodes, monkeypatch, fake_pca, reverse_enabled=True,
                        reverse_floor_pwm=340, **CAR_C_ESC)
    assert act.longitudinal.reverse_enabled is False
    (verdict,) = published_direction(act, monkeypatch)
    gm = make_gear(nodes, reverse_enabled=True, **CAR_C_ESC)
    assert gear(gm, cmd=GearCommand.REVERSE, v=0.0, pwm=385, direction=verdict) \
        != GearReport.REVERSE


def test_actuator_needs_the_reverse_gear_command(nodes, monkeypatch, fake_pca):
    node = make_actuator(nodes, monkeypatch, fake_pca, reverse_enabled=True,
                         reverse_standstill_time=0.0, **CAR_C_ESC)
    node.longitudinal.esc.state = esc_locked()
    speed(node, 0.0)
    command(node, -0.3)
    node.timer_callback()
    # no gear command: the forward modes, as with reverse off (never a pull, never a reverse)
    assert node.driver.last(0) > CAR_C_ESC['reverse_arm_pwm']
    assert node.longitudinal.drive_direction == 1
    node.gear_cmd_callback(GearCommand(command=GearCommand.REVERSE))
    command(node, -0.3)
    speed(node, 0.0)
    node.timer_callback()
    assert node.driver.last(0) == CAR_C_ESC['reverse_arm_pwm'] - 5


def test_actuator_arms_and_publishes_its_direction(nodes, monkeypatch, fake_pca):
    # standstill time 0: the first tick at rest may arm (no wall-clock wait in a unit test)
    node = make_actuator(nodes, monkeypatch, fake_pca, reverse_enabled=True,
                         reverse_standstill_time=0.0, **CAR_C_ESC)
    assert node.longitudinal.reverse_enabled is True
    node.longitudinal.esc.state = esc_locked()     # a locked ESC: it pulls
    node.gear_cmd_callback(GearCommand(command=GearCommand.REVERSE))
    sent = []
    monkeypatch.setattr(node.direction_publisher, 'publish', lambda m: sent.append(m.data))
    speed(node, 0.0)
    command(node, 0.0)
    node.timer_callback()
    assert sent == [1]
    command(node, -0.3)
    speed(node, 0.0)
    node.timer_callback()
    assert node.driver.last(0) == CAR_C_ESC['reverse_arm_pwm'] - 5
    assert sent == [1, -1]
    assert node.mode_publisher is not None and node._published_mode == 'reverse_arm'


def esc_locked():
    from autosdv_vehicle_interface.longitudinal import EscState
    return EscState.LOCKED


# ------------------------------------------------------------- velocity report

def make_report(nodes, monkeypatch, value, **overrides):
    params = {**VR_PARAMS, **overrides}
    node = velocity_report.AutoSdvVelocityReportNode(
        parameter_overrides=[Parameter(k, value=v) for k, v in params.items()])
    nodes.append(node)
    monkeypatch.setattr(node.estimator, 'report', lambda now_ns: value)
    sent = []
    monkeypatch.setattr(node.publisher, 'publish', lambda m: sent.append(m.longitudinal_velocity))
    return node, sent


def test_velocity_is_signed_by_the_drive_direction_when_reverse_is_on(nodes, monkeypatch):
    node, sent = make_report(nodes, monkeypatch, 0.3, reverse_enabled=True)
    node.publish_report()
    node.direction_callback(Int8(data=-1))
    node.publish_report()
    node.direction_callback(Int8(data=1))
    node.publish_report()
    assert sent == [0.3, -0.3, 0.3]


def test_a_standstill_is_zero_not_minus_zero(nodes, monkeypatch):
    node, sent = make_report(nodes, monkeypatch, 0.0, reverse_enabled=True)
    node.direction_callback(Int8(data=-1))
    node.publish_report()
    assert sent == [0.0] and str(sent[0]) == '0.0'


def test_an_unavailable_verdict_reads_as_forward(nodes, monkeypatch):
    node, sent = make_report(nodes, monkeypatch, 0.3, reverse_enabled=True)
    node.direction_callback(Int8(data=0))
    node.publish_report()
    assert sent == [0.3]


def test_velocity_never_negative_with_reverse_off(nodes, monkeypatch):
    node, sent = make_report(nodes, monkeypatch, 0.3)
    assert node.get_parameter('reverse_enabled').value is False
    node.direction_callback(Int8(data=-1))
    node.publish_report()
    assert sent == [0.3]


# ---------------------------------------------------------------- gear manager

def make_gear(nodes, **overrides):
    params = {**GM_PARAMS, **overrides}
    node = gear_manager.GearManager(
        parameter_overrides=[Parameter(k, value=v) for k, v in params.items()])
    nodes.append(node)
    return node


def gear(node, cmd=None, v=None, pwm=None, direction=None):
    if cmd is not None:
        node.gear_command_callback(GearCommand(command=cmd))
    if v is not None:
        node.velocity_callback(VelocityReport(longitudinal_velocity=float(v)))
    if pwm is not None:
        node.pwm_debug_callback(Float32MultiArrayStamped(data=[float(pwm), 0.0]))
    if direction is not None:
        node.direction_callback(Int8(data=direction))
    node.publish_gear_status()
    return node.current_gear


def test_gear_reverse_is_reported_at_a_standstill_when_asked_so_autoware_can_command_it(nodes):
    node = make_gear(nodes, reverse_enabled=True, **CAR_C_ESC)
    assert gear(node, cmd=GearCommand.REVERSE, v=0.0, pwm=385, direction=1) == GearReport.REVERSE


def test_gear_stays_reverse_through_arming_driving_and_stopping(nodes):
    node = make_gear(nodes, reverse_enabled=True, **CAR_C_ESC)
    gear(node, cmd=GearCommand.REVERSE, v=0.0, direction=1)
    assert gear(node, pwm=340, direction=-1) == GearReport.REVERSE     # the arming pull
    assert gear(node, pwm=385) == GearReport.REVERSE                   # the return
    assert gear(node, pwm=372, v=-0.3) == GearReport.REVERSE           # reversing
    # Autoware asks DRIVE: still REVERSE while the car rolls back, DRIVE once stopped
    assert gear(node, cmd=GearCommand.DRIVE, pwm=385, v=-0.2) == GearReport.REVERSE
    assert gear(node, v=0.0, direction=1) == GearReport.DRIVE
    assert gear(node, pwm=410, v=0.4) == GearReport.DRIVE


def test_gear_is_drive_while_moving_forward_even_if_reverse_is_asked(nodes):
    node = make_gear(nodes, reverse_enabled=True, **CAR_C_ESC)
    assert gear(node, cmd=GearCommand.DRIVE, v=0.5, pwm=410, direction=1) == GearReport.DRIVE
    assert gear(node, cmd=GearCommand.REVERSE, v=0.4, pwm=375) == GearReport.DRIVE
    assert gear(node, v=0.0, pwm=385) == GearReport.REVERSE


def test_gear_ignores_the_direction_with_reverse_off(nodes):
    node = make_gear(nodes)
    assert node.get_parameter('reverse_enabled').value is False
    gear(node, cmd=GearCommand.REVERSE, v=0.0, pwm=385)
    # legacy: the arming PWM at a standstill reads as a locked brake -> PARK
    assert gear(node, pwm=340, direction=-1) == GearReport.PARK


@pytest.mark.parametrize('verdict', [None, 0])
def test_gear_manager_never_reports_reverse_without_a_willing_actuator(nodes, verdict):
    """Nothing heard from the actuator, or 0 (reverse refused): never REVERSE."""
    node = make_gear(nodes, reverse_enabled=True, **CAR_C_ESC)
    assert gear(node, cmd=GearCommand.REVERSE, v=0.0, pwm=385, direction=verdict) \
        != GearReport.REVERSE
