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
import time

from autosdv_vehicle_interface import safety
from autosdv_vehicle_interface.safety import (
    classify_stamp_age,
    InputState,
    InputWatchdog,
    mrm_requires_stop,
    StampVerdict,
    write_neutral,
)
import pytest


def test_watchdog_waits_until_first_message(logger):
    wd = InputWatchdog("control command", 0.3, logger)
    assert wd.check(100.0) == InputState.WAITING
    assert logger.lines == []


def test_watchdog_trips_once_per_episode_and_recovers(logger):
    wd = InputWatchdog("control command", 0.3, logger)
    wd.feed(0.0)
    assert wd.check(0.29) == InputState.FRESH
    # Lapse: one warning however many ticks it lasts
    for k in range(100):
        assert wd.check(0.31 + k * 0.01) == InputState.TIMED_OUT
    assert len(logger.at("warning")) == 1
    assert "control command" in logger.at("warning")[0]
    # Fresh command: recovers, logs once
    wd.feed(2.0)
    assert wd.check(2.0) == InputState.FRESH
    assert wd.check(2.1) == InputState.FRESH
    assert len(logger.at("info")) == 1
    # Second episode logs again
    assert wd.check(2.5) == InputState.TIMED_OUT
    assert len(logger.at("warning")) == 2
    assert wd.episodes == 2


def test_watchdog_disabled_with_nonpositive_timeout(logger):
    wd = InputWatchdog("velocity report", 0.0, logger)
    wd.feed(0.0)
    assert wd.check(1e6) == InputState.FRESH


@pytest.mark.parametrize("age,expected", [
    (None, StampVerdict.OK),
    (0.0, StampVerdict.OK),
    (0.49, StampVerdict.OK),
    (-0.2, StampVerdict.OK),
    (0.6, StampVerdict.STALE),
    (-0.6, StampVerdict.FUTURE),
    (3600.0, StampVerdict.CLOCK_MISMATCH),
    (-1.7e9, StampVerdict.CLOCK_MISMATCH),
    (float("nan"), StampVerdict.CLOCK_MISMATCH),
])
def test_classify_stamp_age(age, expected):
    assert classify_stamp_age(age, 0.5) == expected


def test_classify_stamp_age_disabled():
    assert classify_stamp_age(3600.0, 0.0) == StampVerdict.OK


@pytest.mark.parametrize("state,behavior,any_mrm,expected", [
    (safety.MRM_STATE_UNKNOWN, 0, True, False),
    (safety.MRM_STATE_NORMAL, 1, True, False),
    (safety.MRM_STATE_OPERATING, safety.MRM_BEHAVIOR_EMERGENCY_STOP, False, True),
    (safety.MRM_STATE_OPERATING, 3, False, False),  # comfortable stop, left to the planner
    (safety.MRM_STATE_OPERATING, 3, True, True),
    (safety.MRM_STATE_SUCCEEDED, safety.MRM_BEHAVIOR_EMERGENCY_STOP, False, True),
    (safety.MRM_STATE_FAILED, 1, False, True),
])
def test_mrm_requires_stop(state, behavior, any_mrm, expected):
    assert mrm_requires_stop(state, behavior, any_mrm) == expected


def test_mrm_constants_match_autoware_message():
    msgs = pytest.importorskip("autoware_adapi_v1_msgs.msg")
    m = msgs.MrmState
    assert (m.UNKNOWN, m.NORMAL, m.MRM_OPERATING, m.MRM_SUCCEEDED, m.MRM_FAILED) == (
        safety.MRM_STATE_UNKNOWN, safety.MRM_STATE_NORMAL, safety.MRM_STATE_OPERATING,
        safety.MRM_STATE_SUCCEEDED, safety.MRM_STATE_FAILED)
    assert m.EMERGENCY_STOP == safety.MRM_BEHAVIOR_EMERGENCY_STOP


def test_write_neutral_writes_both_channels(fake_pca, logger):
    driver = fake_pca()
    assert write_neutral(driver, [(0, 370), (1, 489)], 0.5, logger)
    assert driver.writes == [(0, 370), (1, 489)]


def test_write_neutral_survives_i2c_error(fake_pca, logger):
    driver = fake_pca()
    driver.fail = True
    assert not write_neutral(driver, [(0, 370), (1, 489)], 0.5, logger)
    assert logger.at("error")


def test_write_neutral_does_not_hang_on_wedged_bus(fake_pca, logger, hang_event):
    driver = fake_pca()
    driver.hang = hang_event
    start = time.monotonic()
    assert not write_neutral(driver, [(0, 370)], 0.2, logger)
    assert time.monotonic() - start < 1.0
    assert any("did not finish" in m for m in logger.at("error"))


def test_write_neutral_without_driver(logger):
    assert not write_neutral(None, [(0, 370)], 0.2, logger)
