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
