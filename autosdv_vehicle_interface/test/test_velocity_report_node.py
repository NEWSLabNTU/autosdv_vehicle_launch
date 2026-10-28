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
"""velocity_report with hardware:=sim: the node and a bare VelocityEstimator agree on edges.

Needs rclpy (skipped without). Run under ROS_DOMAIN_ID=73 and flock tmp/bench.lock.
"""
import os
import sys
import time

import pytest

os.environ.setdefault('ROS_DOMAIN_ID', '73')
rclpy = pytest.importorskip('rclpy')

# Imported plainly, not skipped: before Step 6 this import is the red phase on
# any machine without Jetson.GPIO
from autosdv_vehicle_interface import hal  # noqa: E402
from autosdv_vehicle_interface import velocity_report  # noqa: E402
from autosdv_vehicle_interface.hal import EDGE_TOPIC  # noqa: E402
from autosdv_vehicle_interface.velocity_estimator import VelocityEstimator  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from std_msgs.msg import Int64MultiArray  # noqa: E402

PARAMS = {'pin': 15, 'rate': 20.0, 'wheel_diameter': 10.5, 'markers_per_rotation': 12,
          'frame_id': 'base_link', 'hardware': 'sim'}


@pytest.fixture(scope='module')
def ros():
    rclpy.init()
    yield
    rclpy.try_shutdown()


def spin_until(node, predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        rclpy.spin_once(node, timeout_sec=0.05)
    return predicate()


def test_sim_edges_give_the_estimators_value(ros):
    node = velocity_report.AutoSdvVelocityReportNode(
        parameter_overrides=[Parameter(k, value=v) for k, v in PARAMS.items()])
    pub_node = rclpy.create_node('rc_sim_edge_pub')
    pub = pub_node.create_publisher(Int64MultiArray, EDGE_TOPIC, 100)
    try:
        assert spin_until(node, lambda: pub.get_subscription_count() > 0)
        d = node.distance_per_marker
        t, edges = 1_000_000_000, []
        for i in range(300):
            t += int(round(d / (0.3 + 0.01 * i) * 1e9))
            edges.append(t)
        ref = VelocityEstimator(d)
        for i in range(0, len(edges), 10):
            batch = edges[i:i + 10]
            pub.publish(Int64MultiArray(data=batch))
            for e in batch:
                ref.on_edge(e)
        assert spin_until(node, lambda: node.estimator.last_edge_ns == edges[-1])
        assert node.estimator.current_velocity == ref.current_velocity
        later = edges[-1] + 30_000_000
        assert node.estimator.report(later) == ref.report(later)
    finally:
        pub_node.destroy_node()
        node.destroy_node()


def test_imports_without_jetson_gpio(ros, monkeypatch):
    """The module imports with Jetson.GPIO unavailable, and hardware:=sim selects SimGPIO."""
    import importlib
    monkeypatch.setitem(sys.modules, 'Jetson', None)       # importing Jetson.GPIO now raises
    monkeypatch.delitem(sys.modules, 'Jetson.GPIO', raising=False)
    module = importlib.reload(velocity_report)
    assert module.AutoSdvVelocityReportNode.__name__ == 'AutoSdvVelocityReportNode'
    assert 'Jetson.GPIO' not in sys.modules
    node = rclpy.create_node('rc_sim_gpio_probe')
    try:
        assert isinstance(module.make_gpio('sim', node=node), hal.SimGPIO)
    finally:
        node.destroy_node()


def test_sim_acknowledges_each_report_after_publishing_it(ros):
    """hardware:=sim: each report is followed by a lockstep ack for its timer deadline."""
    from std_msgs.msg import String
    node = velocity_report.AutoSdvVelocityReportNode(
        parameter_overrides=[Parameter(k, value=v) for k, v in PARAMS.items()])
    order = []
    node.publisher.publish = lambda msg, inner=node.publisher.publish: (
        order.append(('report', msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec)),
        inner(msg))
    probe = rclpy.create_node('rc_sim_ack_probe')
    acks = []
    probe.create_subscription(String, hal.LOCKSTEP_ACK_TOPIC, lambda m: acks.append(m.data), 100)
    try:
        assert spin_until(node, lambda: len(order) >= 3)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and len(acks) < 3:
            rclpy.spin_once(probe, timeout_sec=0.05)
        name, deadline_ns, period_ns, extra = hal.parse_ack(acks[0])
        assert name == hal.VELOCITY_REPORT and period_ns == 50_000_000
        assert extra == {'late_edges': 0}
        stamps = [t for _, t in order]
        # the deadline is the timer's, at or just before the report's stamp
        assert any(0 <= t - deadline_ns < period_ns for t in stamps)
        node.on_gpio_rising(15, node.last_report_ns - 1)     # an edge the last report covered
        assert node.late_edges == 1
    finally:
        probe.destroy_node()
        node.destroy_node()


def test_real_hardware_publishes_no_ack(ros, monkeypatch):
    made = {}

    class FakeReader:
        description = 'fake pin'

        def __init__(self, pin, on_marker, clock_ns, burst_gap, active_level):
            made.update(pin=pin, on_marker=on_marker, burst_gap=burst_gap,
                        active_level=active_level)

        def close(self):
            pass

    monkeypatch.setattr(velocity_report, 'make_burst_reader', FakeReader)
    params = dict(PARAMS, hardware='real')
    node = velocity_report.AutoSdvVelocityReportNode(
        parameter_overrides=[Parameter(k, value=v) for k, v in params.items()])
    try:
        assert node.lockstep is None
        assert not [t for t, _ in node.get_publisher_names_and_types_by_node(
            node.get_name(), node.get_namespace()) if t == hal.LOCKSTEP_ACK_TOPIC]
        # one marker per reported burst, at the burst's own time
        assert made['burst_gap'] == 0.001 and made['active_level'] == 0
        made['on_marker'](1_000_000_000)
        made['on_marker'](1_100_000_000)
        assert node.estimator.current_velocity == pytest.approx(
            node.distance_per_marker / 0.1)
    finally:
        node.destroy_node()


def test_sim_report_waits_for_its_edges(ros, monkeypatch):
    """Under lockstep a report is held until the plant marks the edges complete up to it."""
    node = velocity_report.AutoSdvVelocityReportNode(
        parameter_overrides=[Parameter(k, value=v) for k, v in PARAMS.items()])
    acks, reports = [], []
    node.lockstep.ack = lambda deadline, period, **extra: acks.append((deadline, extra))
    node.publisher.publish = lambda msg: reports.append(msg.longitudinal_velocity)
    monkeypatch.setattr(velocity_report, 'timer_deadline_ns', lambda timer, now: 2_000_000_000)
    try:
        node.lockstep.active = True
        node.gpio.until_ns = 1_995_000_000            # edges in up to 5 ms before the deadline
        node.publish_callback()
        assert reports == [] and acks == [] and node._held_report_ns == 2_000_000_000
        node.gpio.until_ns = 2_000_000_000
        node._on_edges_until(2_000_000_000)
        assert len(reports) == 1 and acks == [(2_000_000_000, {'late_edges': 0})]
        node.publish_callback()                       # edges already complete: no hold
        assert len(reports) == 2
    finally:
        node.destroy_node()
