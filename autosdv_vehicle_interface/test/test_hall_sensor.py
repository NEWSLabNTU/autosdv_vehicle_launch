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
Test hall_sensor.BurstDetector: one count per magnet from chattering edges.

No GPIO.
"""
from autosdv_vehicle_interface.hall_sensor import BurstDetector

import pytest

MS = 1_000_000


def burst(d, t0, n=5, spacing=20_000):
    """
    Feed n edges ``spacing`` ns apart from t0.

    Like the bounce measured on car A.
    """
    for i in range(n):
        d.edge(t0 + i * spacing)


def test_a_magnet_pass_counts_once_at_its_arrival():
    """A magnet pass is counted once, at its arrival."""
    d = BurstDetector(burst_gap=0.001, active_level=0)
    burst(d, 100 * MS)  # the magnet arrives
    assert d.settle(100 * MS + 500_000, 0) is None  # still inside the burst
    assert d.settle(102 * MS, 0) == 100 * MS  # quiet, settled LOW: counted
    burst(d, 140 * MS)  # the magnet leaves
    assert d.settle(142 * MS, 1) is None  # settled HIGH: not counted
    assert (d.bursts, d.markers) == (2, 1)


def test_edges_closer_than_the_gap_stay_one_burst():
    """Edges closer together than the gap stay one burst."""
    d = BurstDetector(burst_gap=0.001)
    burst(d, 0, n=40, spacing=900_000)  # 0.9 ms apart: one long burst
    assert d.settle(40 * MS, 0) == 0
    assert d.bursts == 1


def test_twelve_magnets_one_revolution():
    """Twelve magnets make twelve counts in one revolution."""
    d = BurstDetector()
    counted = []
    for k in range(12):
        t = k * 50 * MS
        for start, level in ((t, 0), (t + 20 * MS, 1)):
            burst(d, start)
            got = d.settle(start + 2 * MS, level)
            if got is not None:
                counted.append(got)
    assert counted == [k * 50 * MS for k in range(12)]
    assert (d.bursts, d.markers) == (24, 12)


def test_active_high_counts_the_other_switch():
    """Active level 1 counts the other switch."""
    d = BurstDetector(active_level=1)
    burst(d, 0)
    assert d.settle(2 * MS, 0) is None
    burst(d, 10 * MS)
    assert d.settle(12 * MS, 1) == 10 * MS


def test_nothing_open_nothing_settles():
    """With no open burst nothing settles, and a zero gap is refused."""
    assert BurstDetector().settle(5 * MS, 0) is None
    with pytest.raises(ValueError):
        BurstDetector(burst_gap=0.0)
