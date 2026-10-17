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
"""A fixed 6 s drive for the actuator tests: reports, commands, a gap, an emergency.

Pure Python. ``events()`` yields (tick, kind, args) in the order the callbacks
run within a 10 ms tick; the golden file ``golden_real_pwm.json`` was recorded
by running exactly this through the pre-refactor node.
"""
TICK = 0.01
TICKS = 600
T0 = 1000.0


def events():
    """(tick, kind, args): kind is 'vel' (speed), 'cmd' (target, tire angle) or 'gate' (bool)."""
    out = []
    for k in range(TICKS):
        if k == 540:
            out.append((k, 'gate', (True,)))
        if k == 560:
            out.append((k, 'gate', (False,)))
        if k % 5 == 0 and not 380 <= k < 440:                  # velocity reports, with an outage
            out.append((k, 'vel', (min(2.0, 0.01 * k),)))
        if k % 3 == 0 and not 450 <= k < 520:                  # commands, with a gap
            if k < 150:
                target = 1.5
            elif k < 250:
                target = -0.5
            elif k < 350:
                target = 0.8
            else:
                target = 0.0
            angle = 0.5 if 100 <= k < 200 else (-0.2 if 200 <= k < 300 else 0.0)
            out.append((k, 'cmd', (target, angle)))
        out.append((k, 'tick', ()))
    return out
