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
"""Shared test helpers. Nothing here touches I2C or GPIO."""
import threading

import pytest


class RecordingLogger:
    """Collects log lines by level; stands in for rclpy and logging loggers."""

    def __init__(self):
        self.lines = []

    def _log(self, level, msg, *args, **kwargs):
        self.lines.append((level, str(msg)))

    def debug(self, msg, *a, **k):
        self._log("debug", msg)

    def info(self, msg, *a, **k):
        self._log("info", msg)

    def warning(self, msg, *a, **k):
        self._log("warning", msg)

    warn = warning

    def error(self, msg, *a, **k):
        self._log("error", msg)

    def fatal(self, msg, *a, **k):
        self._log("fatal", msg)

    def at(self, level):
        return [m for lv, m in self.lines if lv == level]


class FakePCA9685:
    """In-memory PCA9685: records every set_pwm, optionally fails or hangs."""

    instances = []

    def __init__(self, address=0x40, busnum=None, **kwargs):
        self.address = address
        self.busnum = busnum
        self.freq = None
        self.writes = []
        self.fail = False
        self.hang = None  # threading.Event to block on
        FakePCA9685.instances.append(self)

    def set_pwm_freq(self, freq):
        self.freq = freq

    def set_pwm(self, channel, on, off):
        if self.hang is not None:
            self.hang.wait()
        if self.fail:
            raise OSError(121, "Remote I/O error")
        self.writes.append((channel, off))

    def last(self, channel):
        for ch, value in reversed(self.writes):
            if ch == channel:
                return value
        return None


@pytest.fixture
def logger():
    return RecordingLogger()


@pytest.fixture
def fake_pca():
    FakePCA9685.instances.clear()
    yield FakePCA9685
    for inst in FakePCA9685.instances:
        if inst.hang is not None:
            inst.hang.set()


@pytest.fixture
def hang_event():
    event = threading.Event()
    yield event
    event.set()
