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
The plug-in split must not change what the motor receives.

Drives the frozen pre-split controller (legacy_reference.py) and the new
LongitudinalShell + default AckermannSpeedPID through the same scripted
sequences -- the node's real timing (100 Hz loop, 20 Hz velocity reports with
jitter, dt = time since the last report), starts, stops, braking, saturation,
reverse requests, deadband -- and requires identical PWM on every tick, plus
identical PID internals.
"""
import math
import random

from autosdv_vehicle_interface.longitudinal import LongitudinalShell, ShellConfig
from autosdv_vehicle_interface.motor_plant import MotorPlant
from autosdv_vehicle_interface.speed_controller import load_speed_controller
from legacy_reference import LegacyLongitudinal
import pytest

BASE = {
    "rate": 100.0,
    "init_pwm": 370,
    "min_pwm": 360,
    "max_pwm": 470,
    "brake_pwm": 340,
    "brake_threshold": 0.2,
    "kp_speed": 12.0,
    "ki_speed": 10.0,
    "kd_speed": 0.001,
    "integral_limit": 50.0,
    "enable_conditional_integration": True,
    "velocity_deadband": 0.0,
    "full_stop_threshold": 0.1,
    "derivative_filter_alpha": 0.15,
}

VARIANTS = {
    "actuator.yaml": {},
    "deadband": {"velocity_deadband": 0.05},
    "no_conditional_integration": {"enable_conditional_integration": False},
    "high_gain": {"kp_speed": 60.0, "ki_speed": 40.0, "kd_speed": 0.5, "integral_limit": 20.0},
    "odd_range": {"min_pwm": 361, "max_pwm": 452},
}


def target_profile(t):
    """Target speed: idle, accelerate, cruise, saturate, stop, reverse request, restart."""
    if t < 0.3:
        return None  # no command yet
    if t < 1.0:
        return 0.0
    if t < 3.0:
        return min(1.0, (t - 1.0) * 0.8)
    if t < 5.0:
        return 1.0
    if t < 6.0:
        return 3.0  # beyond what the PWM range can reach: saturation
    if t < 8.0:
        return 0.0  # brake, then full stop
    if t < 9.0:
        return -0.5  # reverse request
    if t < 9.5:
        return 0.05  # inside full_stop_threshold
    if t < 11.0:
        return 0.15 + 0.05 * math.sin(3.0 * t)  # small speeds around the thresholds
    return 0.8


def run_sequence(params, seed, duration=12.0):
    """Return per-tick tuples from the legacy and the new implementation."""
    rng = random.Random(seed)

    legacy = LegacyLongitudinal(params)
    controller = load_speed_controller("", dict(params))  # default path
    shell = LongitudinalShell(
        ShellConfig(
            init_pwm=params["init_pwm"],
            min_pwm=params["min_pwm"],
            max_pwm=params["max_pwm"],
            brake_pwm=params["brake_pwm"],
            full_stop_threshold=params["full_stop_threshold"],
            brake_threshold=params["brake_threshold"],
            velocity_deadband=params["velocity_deadband"],
        ),
        controller,
    )

    plant = MotorPlant()
    pwm = params["init_pwm"]
    current_speed = None
    last_velocity_time = 0.0
    next_report = 0.25 + rng.uniform(0.0, 0.05)  # first report after start
    t = 0.0
    rows = []

    while t < duration:
        # Loop jitter, as a ROS timer has
        tick = 0.01 + rng.uniform(-0.002, 0.002)
        plant.step(pwm, tick)
        t += tick

        if t >= next_report:
            # Hall sensor: unsigned, noisy; occasionally a dropped report
            current_speed = abs(plant.speed) + rng.gauss(0.0, 0.02)
            current_speed = max(0.0, current_speed)
            last_velocity_time = t
            next_report += 0.05 + rng.uniform(-0.005, 0.005)
            if rng.random() < 0.05:
                next_report += 0.05

        target = target_profile(t)
        delta_time = t - last_velocity_time

        # Legacy: node state, then run_longitudinal_control(delta_time)
        legacy.state.target_speed = target
        legacy.state.current_speed = current_speed
        legacy_pwm = legacy.run_longitudinal_control(delta_time)

        new_pwm = shell.step(t, target, current_speed, last_velocity_time)
        assert shell.override_reason is None

        lp = legacy.speed_controller
        np_ = controller.pid
        rows.append((
            round(t, 6),
            legacy_pwm, new_pwm,
            legacy.state.last_pwm_value, shell.last_pwm_value,
            legacy.state.in_reverse, shell.in_reverse,
            (lp.proportional, lp.integral, lp.derivative, lp.derivative_filtered,
             lp.is_saturated, lp.last_input),
            (np_.proportional, np_.integral, np_.derivative, np_.derivative_filtered,
             np_.is_saturated, np_.last_input),
            legacy.state.speed_control_pwm_offset, shell.speed_control_pwm_offset,
        ))
        pwm = new_pwm

    return rows


@pytest.mark.parametrize("variant", sorted(VARIANTS))
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_new_shell_matches_legacy(variant, seed):
    params = dict(BASE)
    params.update(VARIANTS[variant])
    rows = run_sequence(params, seed)

    for row in rows:
        t, lpwm, npwm, lhold, nhold, lrev, nrev, lpid, npid, loff, noff = row
        assert npwm == lpwm, f"t={t}: PWM {npwm} != legacy {lpwm}"
        assert nhold == lhold, f"t={t}: held PWM differs"
        assert nrev == lrev, f"t={t}: in_reverse differs"
        assert npid == lpid, f"t={t}: PID state differs"
        assert noff == loff, f"t={t}: PWM offset differs"


def test_sequence_exercises_every_legacy_mode():
    """Guard against a vacuous comparison: all four legacy modes must occur."""
    params = dict(BASE)
    params.update(VARIANTS["deadband"])
    rows = run_sequence(params, seed=0)
    pwms = [r[1] for r in rows]

    assert params["brake_pwm"] in pwms                      # BRAKE
    assert pwms.count(params["init_pwm"]) > 50              # WAITING / FULL_STOP
    assert any(r[7][4] for r in rows)                        # ACTIVE, PID saturated
    high_gain = dict(BASE)
    high_gain.update(VARIANTS["high_gain"])
    assert high_gain["min_pwm"] in [r[1] for r in run_sequence(high_gain, seed=0)]  # clamped
    assert len(set(pwms)) > 30                               # ACTIVE, modulating
    held = sum(1 for a, b in zip(rows, rows[1:]) if b[1] == a[3] and b[1] != params["init_pwm"])
    assert held > 0                                          # DEADBAND_HOLD
    assert any(r[5] for r in rows)                           # reverse requested
