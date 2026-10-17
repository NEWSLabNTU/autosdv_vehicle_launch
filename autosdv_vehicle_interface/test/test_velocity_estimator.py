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
"""VelocityEstimator against the bench's independent copy of the node's arithmetic."""
import math

from autosdv_vehicle_interface.motor_plant import HallSpeedSensor
from autosdv_vehicle_interface.velocity_estimator import VelocityEstimator
import pytest

D = 0.105 * math.pi / 12


def edge_train():
    """Edges of a wheel accelerating to 3 m/s, then braking to a crawl and stopping."""
    t_ns, v, out = 1_000_000_000, 0.2, []
    for i in range(400):
        v = min(3.0, v + 0.02) if i < 300 else max(0.05, v - 0.2)
        t_ns += int(round(D / v * 1e9))
        out.append(t_ns)
    return out


def test_matches_the_bench_hall_model():
    est, ref = VelocityEstimator(D), HallSpeedSensor()
    edges = edge_train()
    i = 0
    for now in range(1_000_000_000, edges[-1] + 2_000_000_000, 50_000_000):
        while i < len(edges) and edges[i] <= now:
            est.on_edge(edges[i])
            ref._on_edge(edges[i] / 1e9)
            i += 1
        assert est.report(now) == pytest.approx(ref.report(now / 1e9), rel=1e-9, abs=1e-12)


def test_zero_before_any_edge_and_after_the_timeout():
    est = VelocityEstimator(D)
    assert est.report(10**9) == 0.0
    est.on_edge(10**9)
    est.on_edge(10**9 + 20_000_000)
    assert est.report(10**9 + 25_000_000) > 0.0
    assert est.report(10**9 + 20_000_000 + 600_000_000) == 0.0


def test_edges_closer_than_1ms_are_ignored():
    est = VelocityEstimator(D)
    est.on_edge(0)
    est.on_edge(10_000_000)
    v = est.current_velocity
    est.on_edge(10_500_000)
    assert est.current_velocity == v
    assert est.last_edge_ns == 10_500_000
