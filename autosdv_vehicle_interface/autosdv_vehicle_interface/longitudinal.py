#!/usr/bin/env python3
"""
Longitudinal safety shell for the AutoSDV actuator: target speed -> motor PWM.

The shell decides, once per control tick, which mode drives the motor. Every
mode except ACTIVE is decided here, and the plug-in speed controller (see
``speed_controller.py``) is consulted only in ACTIVE:

=================  ==========================================  ==============
mode               when                                        motor PWM
=================  ==========================================  ==============
OVERRIDE_BRAKE     emergency / command timeout / stale         ``brake_pwm``
                   velocity / controller fault, still moving
OVERRIDE_NEUTRAL   the same, once stopped or the brake time    ``init_pwm``
                   is used up
WAITING            no command or no velocity received yet      ``init_pwm``
BRAKE              target ~ 0 and measured > brake_threshold   ``brake_pwm``
FULL_STOP          target ~ 0 and measured ~ 0                 ``init_pwm``
DEADBAND_HOLD      ``|error| < velocity_deadband``             last PWM
ACTIVE             otherwise                                   plug-in, clamped
=================  ==========================================  ==============

BRAKE, FULL_STOP, DEADBAND_HOLD and ACTIVE are the modes the actuator has
always had, reproduced exactly; ``test/test_speed_controller_equivalence.py``
proves it against a frozen copy of the pre-split code.

This module has no ROS dependency, so the motor bench (``speed_bench.py``) runs
the same shell the vehicle does.
"""
from dataclasses import dataclass
from enum import Enum
import logging
import math
import numbers
import traceback
from typing import Optional


class Mode(Enum):
    """Longitudinal control mode, published on ``~/debug/mode``."""

    WAITING = "waiting"
    BRAKE = "brake"
    FULL_STOP = "full_stop"
    DEADBAND_HOLD = "deadband_hold"
    ACTIVE = "active"
    OVERRIDE_BRAKE = "override_brake"
    OVERRIDE_NEUTRAL = "override_neutral"


# Override reasons, in the order the node checks them.
OVERRIDE_EMERGENCY = "emergency"
OVERRIDE_COMMAND_TIMEOUT = "command_timeout"
OVERRIDE_VELOCITY_TIMEOUT = "velocity_timeout"
OVERRIDE_CONTROLLER_FAULT = "controller_fault"

DT_SOURCE_VELOCITY_REPORT = "velocity_report"
DT_SOURCE_LOOP = "loop"


@dataclass
class ShellConfig:
    """
    Parameters of the safety shell; names follow ``params/actuator.yaml``.

    Attributes:
        init_pwm: Neutral motor PWM
        min_pwm: Lowest PWM the controller may command
        max_pwm: Highest PWM the controller may command
        brake_pwm: PWM for active braking
        full_stop_threshold: |speed| below which target/measured count as zero [m/s]
        brake_threshold: Measured speed above which a zero target brakes [m/s]
        velocity_deadband: |error| below which the last PWM is held [m/s]
        override_brake_duration: Longest brake at the start of an override [s]
        controller_fault_holdoff: Neutral time after a controller exception [s]
        controller_max_faults: Faults before the controller is disabled for
            good (until restart); zero or less never disables it
        dt_source: ``"velocity_report"`` -- dt is the time since the last
            velocity report, the legacy behaviour -- or ``"loop"``, the time
            since the previous control tick
    """

    init_pwm: int
    min_pwm: int
    max_pwm: int
    brake_pwm: int
    full_stop_threshold: float
    brake_threshold: float
    velocity_deadband: float
    override_brake_duration: float = 1.5
    controller_fault_holdoff: float = 1.0
    controller_max_faults: int = 3
    dt_source: str = DT_SOURCE_VELOCITY_REPORT


class LongitudinalShell:
    """
    Mode selection, safety overrides and plug-in supervision for the motor.

    Call :meth:`step` once per control tick. All times are in seconds on one
    clock chosen by the caller (the node uses wall time, the bench uses
    simulated time).
    """

    # Low-pass filter on the measured speed before the controller sees it
    # (EMA). Hard-coded in the original actuator as well.
    MEASURED_FILTER_ALPHA = 0.3

    # Speed below which the vehicle may change direction [m/s]
    STANDING_STILL_EPSILON = 0.1

    def __init__(self, config: ShellConfig, controller, logger=None):
        if config.dt_source not in (DT_SOURCE_VELOCITY_REPORT, DT_SOURCE_LOOP):
            raise ValueError(
                f"speed_controller_dt_source must be '{DT_SOURCE_VELOCITY_REPORT}' "
                f"or '{DT_SOURCE_LOOP}', got '{config.dt_source}'"
            )
        self.config = config
        self.controller = controller
        self.logger = logger if logger is not None else logging.getLogger(__name__)

        # Controller-side state, as in the original actuator
        self.filtered_measured: Optional[float] = None
        self.last_pwm_value = config.init_pwm  # held in DEADBAND_HOLD
        self.speed_control_pwm_offset = 0.0
        self.in_reverse = False

        # What the shell returned last tick, whatever the mode
        self.last_output_pwm = config.init_pwm
        # Last speed reading seen, for overrides that run without a fresh one
        self.last_known_speed: Optional[float] = None
        self.mode = Mode.WAITING
        self.last_dt = 0.0
        self._last_step_time: Optional[float] = None

        # Override episode
        self.override_reason: Optional[str] = None
        self._override_start = 0.0
        self._override_start_pwm = config.init_pwm

        # Controller faults
        self.fault_count = 0
        self.fault_latched = False
        self._fault_until: Optional[float] = None

    # ------------------------------------------------------------------ step

    def step(
        self,
        now: float,
        target_speed: Optional[float],
        current_speed: Optional[float],
        last_velocity_time: float,
        override: Optional[str] = None,
    ) -> int:
        """
        Run one control tick.

        Args:
            now: Current time [s].
            target_speed: Commanded speed, None until the first command.
            current_speed: Measured speed, None until the first report -- or,
                during an override, when the report is stale.
            last_velocity_time: Arrival time of the last velocity report [s].
            override: A reason to stop regardless of the command, or None.

        Returns:
            int: Motor PWM, always within the hardware's range.
        """
        dt = self._delta_time(now, last_velocity_time)
        self._last_step_time = now
        self.last_dt = dt
        if current_speed is not None:
            self.last_known_speed = current_speed

        if override is None and self._controller_faulted(now):
            override = OVERRIDE_CONTROLLER_FAULT

        if override is None and self.override_reason is not None:
            if not self._resume(now):
                override = OVERRIDE_CONTROLLER_FAULT

        if override is not None:
            pwm = self._run_override(now, override, current_speed)
        else:
            pwm = self._run_control(now, target_speed, current_speed, dt)

        self.last_output_pwm = pwm
        return pwm

    def _delta_time(self, now: float, last_velocity_time: float) -> float:
        if self.config.dt_source == DT_SOURCE_LOOP:
            if self._last_step_time is None:
                return 0.0
            return now - self._last_step_time
        # Legacy: time since the last velocity report, not since the last tick
        return now - last_velocity_time

    # -------------------------------------------------------------- control

    def _run_control(
        self,
        now: float,
        target_speed: Optional[float],
        current_speed: Optional[float],
        dt: float,
    ) -> int:
        """Reproduce the original run_longitudinal_control, plug-in in the middle."""
        cfg = self.config

        # Return init PWM if we don't have necessary state information
        if target_speed is None or current_speed is None:
            self.mode = Mode.WAITING
            return cfg.init_pwm

        # Check if we need to apply brake when coming to a stop
        if (abs(target_speed) < cfg.full_stop_threshold and
                abs(current_speed) > cfg.brake_threshold):
            self.mode = Mode.BRAKE
            return cfg.brake_pwm

        # Check for full stop condition
        if (abs(current_speed) < cfg.full_stop_threshold and
                abs(target_speed) < cfg.full_stop_threshold):
            self.mode = Mode.FULL_STOP
            return cfg.init_pwm

        # Low-pass filter on measured velocity
        if self.filtered_measured is None:
            self.filtered_measured = current_speed
        else:
            alpha = self.MEASURED_FILTER_ALPHA
            self.filtered_measured = (
                alpha * current_speed + (1.0 - alpha) * self.filtered_measured
            )
        measured_velocity = self.filtered_measured

        # Apply velocity deadband
        if abs(target_speed - measured_velocity) < cfg.velocity_deadband:
            self.mode = Mode.DEADBAND_HOLD
            return self.last_pwm_value

        # Handle reverse logic (bookkeeping only; published for debugging)
        self._update_reverse(target_speed, current_speed)

        try:
            pwm = self._validate_output(
                self.controller.update(target_speed, measured_velocity, dt)
            )
        except Exception as e:  # noqa: B902 - a plug-in may raise anything
            self._on_fault(now, "update()", e)
            return self._run_override(now, OVERRIDE_CONTROLLER_FAULT, current_speed)

        self.speed_control_pwm_offset = getattr(
            self.controller, "last_output_offset", float(pwm - cfg.init_pwm)
        )

        clamped_pwm = max(cfg.min_pwm, min(cfg.max_pwm, pwm))
        self.last_pwm_value = clamped_pwm
        self.mode = Mode.ACTIVE
        return clamped_pwm

    @staticmethod
    def _validate_output(value) -> int:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"update() returned {value!r}, not a PWM number")
        if not math.isfinite(value):
            raise ValueError(f"update() returned {value!r}")
        return int(value)

    def _update_reverse(self, target_speed: float, current_speed: float) -> None:
        """Track the forward/reverse intent, as run_control_reverse did."""
        if abs(current_speed) < self.STANDING_STILL_EPSILON:
            if target_speed < -0.1:
                self.in_reverse = True
            elif target_speed > 0.1:
                self.in_reverse = False
        elif (current_speed * target_speed) < 0 and abs(current_speed) <= 0.5:
            if target_speed < 0 and not self.in_reverse:
                self.in_reverse = True
            elif target_speed > 0 and self.in_reverse:
                self.in_reverse = False

    # ------------------------------------------------------------- override

    def _run_override(self, now: float, reason: str, current_speed: Optional[float]) -> int:
        """
        Brake, then neutral.

        Brakes while the vehicle is moving faster than ``brake_threshold`` --
        judged by the current reading, else the last reading seen, else (no
        reading ever) by whether the motor was driven forward when the override
        began -- for at most ``override_brake_duration``. Neutral after that, and at once if the
        vehicle is already stopped: holding ``brake_pwm`` on a stopped car can
        make the ESC reverse.
        """
        cfg = self.config
        if self.override_reason is None:
            self._override_start = now
            self._override_start_pwm = self.last_output_pwm
            self.logger.warning(f"Motor override ({reason}): braking, then neutral")
        elif self.override_reason != reason:
            self.logger.warning(f"Motor override: {self.override_reason} -> {reason}")
        self.override_reason = reason

        if current_speed is None:
            current_speed = self.last_known_speed
        if current_speed is not None:
            moving = abs(current_speed) > cfg.brake_threshold
        else:
            moving = self._override_start_pwm > cfg.init_pwm

        if moving and now - self._override_start < cfg.override_brake_duration:
            self.mode = Mode.OVERRIDE_BRAKE
            return cfg.brake_pwm

        self.mode = Mode.OVERRIDE_NEUTRAL
        return cfg.init_pwm

    def _resume(self, now: float) -> bool:
        """End an override episode: reset the controller. False if reset() raised."""
        try:
            self.controller.reset()
        except Exception as e:  # noqa: B902
            self._on_fault(now, "reset()", e)
            return False

        self.logger.info(f"Motor override ({self.override_reason}) cleared: speed control resumed")
        self.override_reason = None
        self.filtered_measured = None
        self.last_pwm_value = self.config.init_pwm
        self.speed_control_pwm_offset = 0.0
        return True

    # --------------------------------------------------------------- faults

    def _controller_faulted(self, now: float) -> bool:
        if self.fault_latched:
            return True
        if self._fault_until is None:
            return False
        if now < self._fault_until:
            return True
        self._fault_until = None
        return False

    def _on_fault(self, now: float, where: str, error: Exception) -> None:
        cfg = self.config
        self.fault_count += 1
        detail = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        if cfg.controller_max_faults > 0 and self.fault_count >= cfg.controller_max_faults:
            self.fault_latched = True
            self.logger.error(
                f"Speed controller {where} failed ({self.fault_count} faults): "
                f"disabled until the actuator restarts, motor held in neutral\n{detail}"
            )
        else:
            self._fault_until = now + cfg.controller_fault_holdoff
            self.logger.error(
                f"Speed controller {where} failed (fault {self.fault_count}): motor to "
                f"neutral, retrying in {cfg.controller_fault_holdoff:.1f} s\n{detail}"
            )
