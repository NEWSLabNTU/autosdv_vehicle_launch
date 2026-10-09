#!/usr/bin/env python3
"""
Offline model of the AutoSDV drive train and wheel-speed sensor. No ROS.

Two pieces, used by ``speed_bench.py`` to exercise a speed controller without
the car:

- :class:`MotorPlant` -- motor PWM -> vehicle speed. A first-order response
  toward a steady-state speed that is linear in the PWM beyond a deadband
  around ``init_pwm``, braking below it, and coasting inside it.
- :class:`HallSpeedSensor` -- vehicle speed -> the ``velocity_report`` node's
  output: edge-to-edge timing on 12 wheel markers, the node's EMA (alpha 0.3),
  published at 20 Hz, zero after 0.5 s without an edge, and no sign.

PROVISIONAL: :class:`MotorPlantParams` are guesses consistent with
``params/actuator.yaml`` (neutral 370, PWM range 360..470), not values fitted
from the vehicle. Before the bench is used to grade gains, fit ``deadband``,
``gain``, ``time_constant``, ``coast_time_constant`` and ``brake_decel`` from
bags of ``/autosdv/actuator_node/debug/pwm_values`` against
``/vehicle/status/velocity_status`` (PWM steps from standstill and to neutral),
and record the fit and the vehicle here.
"""
from dataclasses import dataclass
import math
from typing import Optional


@dataclass
class MotorPlantParams:
    """
    Drive-train parameters. PROVISIONAL -- not yet fitted from bags.

    Attributes:
        init_pwm: Neutral PWM (actuator.yaml ``init_pwm``)
        deadband: PWM counts either side of ``init_pwm`` that produce no drive
        gain: Steady-state speed per PWM count beyond the deadband [m/s/count]
        time_constant: First-order time constant while driven [s]
        coast_time_constant: Speed decay time constant in the deadband [s]
        brake_decel: Deceleration below the deadband while moving [m/s^2]
        max_speed: Speed limit of the model [m/s]
        allow_reverse: Drive backwards below the deadband once stopped. The
            ESC needs a neutral-then-reverse sequence the model does not
            represent, so this is off by default.
    """

    init_pwm: int = 370
    deadband: int = 5
    gain: float = 0.03
    time_constant: float = 0.4
    coast_time_constant: float = 1.5
    brake_decel: float = 3.0
    max_speed: float = 3.0
    allow_reverse: bool = False


class MotorPlant:
    """Integrate vehicle speed and distance under a held motor PWM."""

    def __init__(self, params: Optional[MotorPlantParams] = None, speed: float = 0.0):
        self.params = params if params is not None else MotorPlantParams()
        self.speed = float(speed)
        self.distance = 0.0  # odometer, unsigned [m]

    def steady_state_speed(self, pwm: int) -> float:
        """Speed the model settles at under a constant ``pwm`` from rest."""
        p = self.params
        drive = pwm - p.init_pwm
        if drive > p.deadband:
            return min(p.max_speed, p.gain * (drive - p.deadband))
        if drive < -p.deadband and p.allow_reverse:
            return max(-p.max_speed, p.gain * (drive + p.deadband))
        return 0.0

    def step(self, pwm: int, dt: float) -> float:
        """Advance by ``dt`` seconds with ``pwm`` held; return the new speed."""
        p = self.params
        drive = pwm - p.init_pwm
        v = self.speed

        if drive > p.deadband or (drive < -p.deadband and p.allow_reverse and v <= 0.0):
            target = self.steady_state_speed(pwm)
            v += (target - v) * (1.0 - math.exp(-dt / p.time_constant))
        elif drive < -p.deadband:
            # Brake toward zero from either direction, never through it
            if v > 0.0:
                v = max(0.0, v - p.brake_decel * dt)
            elif v < 0.0:
                v = min(0.0, v + p.brake_decel * dt)
        else:
            v *= math.exp(-dt / p.coast_time_constant)

        v = max(-p.max_speed, min(p.max_speed, v))
        self.distance += abs(0.5 * (self.speed + v) * dt)
        self.speed = v
        return v


class HallSpeedSensor:
    """
    Reproduce ``velocity_report.py`` from the distance the wheel has rolled.

    Call :meth:`update` with the odometer whenever the plant advances; it emits
    marker edges, and :meth:`report` returns what the node would publish.
    """

    def __init__(
        self,
        wheel_diameter: float = 0.105,
        markers_per_rotation: int = 12,
        alpha: float = 0.3,
        timeout: float = 0.5,
    ):
        self.distance_per_marker = wheel_diameter * math.pi / markers_per_rotation
        self.alpha = alpha
        self.timeout = timeout

        self.current_velocity = 0.0
        self.last_edge_time: Optional[float] = None
        self._next_edge_distance = self.distance_per_marker
        self._last_time = 0.0
        self._last_distance = 0.0

    def update(self, now: float, distance: float) -> None:
        """Register the odometer reading at ``now``; emit any edges since the last call."""
        while distance >= self._next_edge_distance:
            # Interpolate the crossing time within the last interval
            span = distance - self._last_distance
            frac = (self._next_edge_distance - self._last_distance) / span if span > 0 else 1.0
            edge_time = self._last_time + frac * (now - self._last_time)
            self._on_edge(edge_time)
            self._next_edge_distance += self.distance_per_marker
        self._last_time = now
        self._last_distance = distance

    def _on_edge(self, t: float) -> None:
        if self.last_edge_time is not None:
            dt = t - self.last_edge_time
            if dt > 0.001:
                instantaneous = self.distance_per_marker / dt
                if self.current_velocity == 0.0:
                    self.current_velocity = instantaneous
                else:
                    self.current_velocity = (
                        self.alpha * instantaneous + (1.0 - self.alpha) * self.current_velocity
                    )
        self.last_edge_time = t

    def report(self, now: float) -> float:
        """Return the speed the node publishes at ``now``."""
        if self.last_edge_time is not None and now - self.last_edge_time > self.timeout:
            self.current_velocity = 0.0
        return self.current_velocity
