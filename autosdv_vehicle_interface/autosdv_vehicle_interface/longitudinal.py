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
                   velocity / controller fault / wheel speed
                   fault, still moving
OVERRIDE_NEUTRAL   the same, once stopped or the brake time    ``init_pwm``
                   is used up
WAITING            no command or no velocity received yet      ``init_pwm``
BRAKE              target ~ 0 and measured > brake_threshold   ``brake_pwm``
FULL_STOP          target ~ 0 and measured ~ 0                 ``init_pwm``
COAST              target ~ 0 and measured between the two     ``init_pwm``
                   thresholds above
DEADBAND_HOLD      ``|error| < velocity_deadband``             last PWM
ACTIVE             otherwise                                   plug-in, clamped
=================  ==========================================  ==============

The wheel speed fault (``wheel_speed_check_enabled``) is the one override the
shell raises from what it observes rather than from an input. A velocity
report can be fresh and still wrong: a hall sensor that is unplugged, on the
wrong pin or out of its magnet's reach reports 0.0 at its usual rate, and the
plug-in, seeing the car stopped, raises the throttle to ``max_pwm`` (Lab 2,
car B, 2026-10-27: 404 to 445 in six seconds, into a wall). So the shell
watches its own output: driven at ``forward_start_pwm +
wheel_speed_check_pwm_margin`` or above for ``wheel_speed_check_time`` while
the measured speed stays below ``wheel_speed_check_min_speed``, it latches the
override. The latch clears once the command has been a stop for
``WHEEL_FAULT_CLEAR_TIME`` (a disengage), or when the check is turned off.
Turn it off (the actuator's ``wheel_speed_check_enabled``, settable while it
runs) for work that drives the motor while the wheel speed legitimately stays
near zero, such as a lifted car under inspection.

BRAKE, FULL_STOP, DEADBAND_HOLD and ACTIVE are the modes the actuator has
always had, reproduced exactly; ``test/test_speed_controller_equivalence.py``
proves it against a frozen copy of the pre-split code.

COAST is new (``coast_on_stop``, on by default; the equivalence test turns
it off). Before it, a zero target with the measured speed between
``full_stop_threshold`` and ``brake_threshold`` (0.1..0.2 m/s) fell through
to ACTIVE, and the plug-in, its integral still holding what cruising needed,
drove the motor forward on a stop command: a car slowing through that band
re-accelerated into BRAKE, and cycled BRAKE/ACTIVE/FULL_STOP while creeping
forward. A stop command now never reaches the plug-in.

ESC model (docs/design/esc-pwm-model.md). The cars' ESCs share one state
machine with per-car thresholds: forward from rest needs ``forward_start_pwm``
and, once moving, holds above ``forward_hold_pwm``; below the stop the ESC only
brakes until reverse is armed, by a pull to ``reverse_arm_pwm`` or lower and a
return; armed, it reverses at or below ``reverse_start_pwm`` and keeps
reversing below ``reverse_stop_pwm``, and only ``forward_start_pwm`` disarms it.
When a car's profile sets these thresholds (0 = not measured), the shell
refuses a configuration that could arm reverse by itself (``brake_pwm`` or
``min_pwm`` at or below ``reverse_arm_pwm``) or whose stop is not inside the
band where nothing moves (``reverse_stop_pwm <= init_pwm < forward_start_pwm``),
tracks the ESC state from the commands it sends (:class:`EscModel`), and, should
the ESC ever be armed, never sends a command that would reverse the car
unasked. The shell does not drive backward on purpose.

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
    COAST = "coast"
    DEADBAND_HOLD = "deadband_hold"
    ACTIVE = "active"
    OVERRIDE_BRAKE = "override_brake"
    OVERRIDE_NEUTRAL = "override_neutral"


# Override reasons, in the order the node checks them.
OVERRIDE_EMERGENCY = "emergency"
OVERRIDE_COMMAND_TIMEOUT = "command_timeout"
OVERRIDE_VELOCITY_TIMEOUT = "velocity_timeout"
OVERRIDE_CONTROLLER_FAULT = "controller_fault"
OVERRIDE_WHEEL_SPEED = "wheel_speed"


class EscState(Enum):
    """The ESC's state, as the shell's own commands imply it."""

    UNMODELLED = "unmodelled"   # the car's profile has no thresholds
    LOCKED = "locked"           # power-on, or after forward: below the stop brakes
    FORWARD = "forward"         # driving forward
    ARMING = "arming"           # pulled to reverse_arm_pwm or lower
    ARMED = "armed"             # below reverse_start_pwm reverses


# The ESC thresholds a car profile may set; 0 means not measured.
ESC_PWM_KEYS = ("forward_start_pwm", "forward_hold_pwm", "reverse_stop_pwm",
                "reverse_start_pwm", "reverse_arm_pwm")


class EscModel:
    """
    The ESC state machine, driven by the commanded PWM alone.

    Conservative where a command alone cannot tell: any return above
    ``reverse_arm_pwm`` after arming counts as armed (the measured return
    points were at or below the stop), and forward motion ends at
    ``forward_hold_pwm`` (``init_pwm`` when that is not measured).
    """

    def __init__(self, config: "ShellConfig"):
        self.enabled = config.reverse_arm_pwm > 0 and config.forward_start_pwm > 0
        self.forward_start = config.forward_start_pwm
        self.forward_hold = config.forward_hold_pwm or config.init_pwm
        self.arm = config.reverse_arm_pwm
        # Unknown at start (a manual test may have left reverse armed): assume
        # armed until a forward command disarms it. It costs nothing -- a
        # stopped car needs no brake -- and a wrong "locked" could reverse it.
        self.state = EscState.ARMED if self.enabled else EscState.UNMODELLED

    def update(self, pwm: int) -> EscState:
        if not self.enabled:
            return self.state
        s = self.state
        if pwm >= self.forward_start:
            s = EscState.FORWARD
        elif pwm <= self.arm:
            s = EscState.ARMING
        elif s == EscState.FORWARD and pwm <= self.forward_hold:
            s = EscState.LOCKED
        elif s == EscState.ARMING:
            s = EscState.ARMED
        self.state = s
        return s


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
        coast_on_stop: COAST a zero target below ``brake_threshold`` instead
            of running the plug-in (see the module docstring). Off only to
            compare against the legacy code.
        forward_start_pwm, forward_hold_pwm, reverse_stop_pwm,
        reverse_start_pwm, reverse_arm_pwm: the car's ESC thresholds (see the
            module docstring); 0 = not measured, which leaves the shell as it
            always was.
        wheel_speed_check_enabled: Stop the motor when it is driven forward
            but the wheel speed stays at zero (see :meth:`LongitudinalShell.step`).
            Needs ``forward_start_pwm``; off here, on in the actuator node.
        wheel_speed_check_pwm_margin: Counts above ``forward_start_pwm`` that
            count as driving the motor
        wheel_speed_check_time: How long the motor may be driven with the wheel
            speed below ``wheel_speed_check_min_speed`` [s]
        wheel_speed_check_min_speed: Measured |speed| that counts as motion [m/s]
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
    coast_on_stop: bool = True
    forward_start_pwm: int = 0
    forward_hold_pwm: int = 0
    reverse_stop_pwm: int = 0
    reverse_start_pwm: int = 0
    reverse_arm_pwm: int = 0
    wheel_speed_check_enabled: bool = False
    wheel_speed_check_pwm_margin: int = 10
    wheel_speed_check_time: float = 1.0
    wheel_speed_check_min_speed: float = 0.05


def check_esc_config(cfg: ShellConfig) -> None:
    """Refuse a car profile whose own values could reverse the car unasked."""
    problems = []
    if cfg.reverse_arm_pwm > 0:
        for name in ("brake_pwm", "min_pwm"):
            if getattr(cfg, name) <= cfg.reverse_arm_pwm:
                problems.append(
                    f"{name} {getattr(cfg, name)} is at or below reverse_arm_pwm "
                    f"{cfg.reverse_arm_pwm}: every use of it arms the ESC's reverse")
    if cfg.forward_start_pwm > 0 and not cfg.init_pwm < cfg.forward_start_pwm:
        problems.append(f"init_pwm {cfg.init_pwm} is not below forward_start_pwm "
                        f"{cfg.forward_start_pwm}: the stop would drive forward")
    if cfg.reverse_stop_pwm > 0 and not cfg.reverse_stop_pwm <= cfg.init_pwm:
        problems.append(f"init_pwm {cfg.init_pwm} is below reverse_stop_pwm "
                        f"{cfg.reverse_stop_pwm}: armed, the stop would reverse")
    if cfg.forward_hold_pwm > 0 and cfg.forward_start_pwm > 0 and \
            not cfg.init_pwm <= cfg.forward_hold_pwm < cfg.forward_start_pwm:
        problems.append(f"forward_hold_pwm {cfg.forward_hold_pwm} must lie in "
                        f"init_pwm..forward_start_pwm ({cfg.init_pwm}..{cfg.forward_start_pwm})")
    if cfg.reverse_start_pwm > 0 and cfg.reverse_stop_pwm > 0 and \
            not cfg.reverse_start_pwm < cfg.reverse_stop_pwm:
        problems.append(f"reverse_start_pwm {cfg.reverse_start_pwm} must be below "
                        f"reverse_stop_pwm {cfg.reverse_stop_pwm}")
    if problems:
        raise ValueError("car profile (ESC thresholds): " + "; ".join(problems))


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

    # How long a stop command must last before a wheel speed fault clears [s]
    WHEEL_FAULT_CLEAR_TIME = 1.0

    def __init__(self, config: ShellConfig, controller, logger=None):
        if config.dt_source not in (DT_SOURCE_VELOCITY_REPORT, DT_SOURCE_LOOP):
            raise ValueError(
                f"speed_controller_dt_source must be '{DT_SOURCE_VELOCITY_REPORT}' "
                f"or '{DT_SOURCE_LOOP}', got '{config.dt_source}'"
            )
        check_esc_config(config)
        self.config = config
        self.esc = EscModel(config)
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

        # Wheel speed check: when the motor began to be driven without motion,
        # whether the fault is latched, and when the stop that clears it began
        self.wheel_fault_latched = False
        self._driven_still_since: Optional[float] = None
        self._stop_command_since: Optional[float] = None
        if config.wheel_speed_check_enabled and config.forward_start_pwm <= 0:
            self.logger.warning(
                "Wheel speed check is off: it needs the car's forward_start_pwm, "
                "which is not measured (0)")

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

        if override is None and self._wheel_speed_faulted(now, target_speed):
            override = OVERRIDE_WHEEL_SPEED

        if override is None and self.override_reason is not None:
            if not self._resume(now):
                override = OVERRIDE_CONTROLLER_FAULT

        if override is not None:
            pwm = self._run_override(now, override, current_speed)
        else:
            pwm = self._run_control(now, target_speed, current_speed, dt)

        pwm = self._guard_reverse(pwm, target_speed if override is None else None)
        self._check_wheel_speed(now, pwm, current_speed)
        self.esc.update(pwm)
        self.last_output_pwm = pwm
        return pwm

    def set_wheel_speed_check(self, enabled: bool) -> None:
        """Turn the wheel speed check on or off at run time; off clears a latched fault."""
        self.config.wheel_speed_check_enabled = bool(enabled)
        self._driven_still_since = None
        self._stop_command_since = None
        if not enabled and self.wheel_fault_latched:
            self.wheel_fault_latched = False
            self.logger.warning("Wheel speed check turned off: the latched fault is cleared")
        else:
            self.logger.info(f"Wheel speed check {'on' if enabled else 'off'}")

    @property
    def esc_state(self) -> EscState:
        return self.esc.state

    def _guard_reverse(self, pwm: int, target_speed: Optional[float]) -> int:
        """Armed, a command below reverse_stop_pwm reverses: never send it unasked."""
        cfg = self.config
        if self.esc.state != EscState.ARMED or cfg.reverse_stop_pwm <= 0:
            return pwm
        if pwm >= cfg.reverse_stop_pwm:
            return pwm
        if target_speed is not None and target_speed < -cfg.full_stop_threshold:
            return pwm   # a backward target: the reverse is asked for
        self.logger.warning(
            f"ESC is armed for reverse: {pwm} would reverse the car; sending "
            f"init_pwm {cfg.init_pwm} instead")
        return cfg.init_pwm

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

        # A stop command between the two thresholds: neutral, never the
        # plug-in, whose integral would drive the car forward
        if cfg.coast_on_stop and abs(target_speed) < cfg.full_stop_threshold:
            self.mode = Mode.COAST
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

    def _wheel_check_active(self) -> bool:
        cfg = self.config
        return cfg.wheel_speed_check_enabled and cfg.forward_start_pwm > 0

    def _check_wheel_speed(self, now: float, pwm: int, current_speed: Optional[float]) -> None:
        """Latch the wheel speed fault when this tick's PWM drives a car that does not move."""
        cfg = self.config
        if not self._wheel_check_active() or self.wheel_fault_latched:
            self._driven_still_since = None
            return
        driven = pwm >= cfg.forward_start_pwm + cfg.wheel_speed_check_pwm_margin
        still = current_speed is not None and abs(current_speed) < cfg.wheel_speed_check_min_speed
        if not (driven and still):
            self._driven_still_since = None
            return
        if self._driven_still_since is None:
            self._driven_still_since = now
            return
        if now - self._driven_still_since >= cfg.wheel_speed_check_time:
            self.wheel_fault_latched = True
            self._driven_still_since = None
            self._stop_command_since = None
            self.logger.error(
                f"Wheel speed fault: motor at {pwm} (>= "
                f"{cfg.forward_start_pwm + cfg.wheel_speed_check_pwm_margin}) for "
                f"{cfg.wheel_speed_check_time:.1f} s while the wheel speed reads "
                f"{current_speed:.3f} m/s. Is the hall sensor connected, and on the "
                f"configured pin? Motor held stopped until the command is a stop for "
                f"{self.WHEEL_FAULT_CLEAR_TIME:.1f} s (disengage), or "
                f"wheel_speed_check_enabled is set false")

    def _wheel_speed_faulted(self, now: float, target_speed: Optional[float]) -> bool:
        """Report whether the fault is latched; clear it after a long enough stop command."""
        if not self.wheel_fault_latched:
            return False
        stop = target_speed is None or abs(target_speed) < self.config.full_stop_threshold
        if not stop:
            self._stop_command_since = None
            return True
        if self._stop_command_since is None:
            self._stop_command_since = now
        if now - self._stop_command_since < self.WHEEL_FAULT_CLEAR_TIME:
            return True
        self.wheel_fault_latched = False
        self._stop_command_since = None
        self.logger.info("Wheel speed fault cleared by a stop command")
        return False

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
