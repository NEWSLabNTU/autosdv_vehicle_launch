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


def _edges(est, gaps_s, t0_ns=0):
    t = t0_ns
    est.on_edge(t)
    for g in gaps_s:
        t += int(g * 1e9)
        est.on_edge(t)
    return t


def test_gap_median_1_is_the_legacy_estimator():
    gaps = [0.046, 0.046, 0.060, 0.032, 0.046, 0.046]
    legacy, one = VelocityEstimator(D), VelocityEstimator(D, gap_median=1)
    for est in (legacy, one):
        _edges(est, gaps)
    assert one.current_velocity == legacy.current_velocity


def test_gap_median_rejects_a_late_callback():
    # steady 0.5 m/s, then one edge timed 25 ms late: a long gap, a short one
    # (car C read 0.5 -> 0.88 m/s at a steady PWM, about a 30 ms late edge)
    steady = D / 0.5
    gaps = [steady] * 6 + [steady + 0.025, steady - 0.025] + [steady] * 2
    single, median = VelocityEstimator(D), VelocityEstimator(D, gap_median=3)
    peak_single = peak_median = 0.0
    t = 0
    single.on_edge(t)
    median.on_edge(t)
    for g in gaps:
        t += int(g * 1e9)
        single.on_edge(t)
        median.on_edge(t)
        peak_single = max(peak_single, single.current_velocity)
        peak_median = max(peak_median, median.current_velocity)
    assert peak_single > 0.58
    assert peak_median < 0.505


def test_gap_median_still_follows_a_real_change():
    est = VelocityEstimator(D, gap_median=3)
    _edges(est, [D / 0.3] * 15 + [D / 0.6] * 15)
    assert abs(est.current_velocity - 0.6) < 0.01


def test_gap_median_restarts_after_a_stop():
    est = VelocityEstimator(D, gap_median=3)
    t = _edges(est, [D / 0.6] * 5)
    assert est.report(t + int(1e9)) == 0.0
    assert len(est.gaps) == 0


def test_gap_median_must_be_positive():
    import pytest
    with pytest.raises(ValueError):
        VelocityEstimator(D, gap_median=0)
