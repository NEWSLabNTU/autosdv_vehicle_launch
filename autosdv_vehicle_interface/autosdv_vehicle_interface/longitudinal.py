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
unasked.

Reverse (``reverse_enabled``, off by default). Off, the shell does not drive
backward on purpose: a backward target runs the modes above, which only brake
or stop a car whose ESC is locked. On -- and only if the car's
``forward_start_pwm``, ``reverse_stop_pwm``, ``reverse_start_pwm`` and
``reverse_arm_pwm`` are measured and the floor and cap are sane, else it logs
an error and stays off -- reverse needs two keys: a target below
``-full_stop_threshold`` AND Autoware's REVERSE gear command
(:meth:`LongitudinalShell.set_gear_reverse`). A negative target alone runs the
modes above. With both, the reverse sequence (:class:`ReversePhase`) runs in
their place:

============  ===============================================  =====================
mode          when                                             motor PWM
============  ===============================================  =====================
BRAKE/COAST   asked backward while still rolling forward       as above
FULL_STOP     then until still for ``reverse_standstill_time``  ``init_pwm``
REVERSE_ARM   ESC not modelled armed: the pull                 ``reverse_arm_pwm -
                                                               reverse_arm_margin``
REVERSE_ARM   then the return (from here the ESC counts armed)  ``init_pwm``
REVERSE       ESC modelled armed: speed loop                   floor..``reverse_stop_pwm - 1``
REVERSE_STOP  target or gear no longer backward, until still   ``init_pwm``
============  ===============================================  =====================

**The pull goes only to an ESC the model does not call armed**: on an armed
ESC ``reverse_arm_pwm`` (below ``reverse_start_pwm``) is a full-strength
reverse. So an ESC the model calls armed -- the start state, after an earlier
pull, after a reverse -- is reversed directly, at most ``reverse_start_pwm``
from rest. If that does not move the car within ``wheel_speed_check_time``
(below ``wheel_speed_check_min_speed`` at any command at or below
``reverse_start_pwm``), the ESC was only assumed armed: the model goes LOCKED
and the shell arms it. If an ESC this shell armed, or one that has reversed,
does not move, the wheel speed fault latches (when the check is on) and the
model goes LOCKED, so the next attempt arms afresh. Motion while arming aborts
the sequence without a reverse episode (the ESC only brakes while it arms).

The speed loop is the shell's own (feed-forward + PI on |speed|, the target
capped at ``reverse_max_speed``), not the plug-in, whose contract is forward.
There is no brake while reversing -- any PWM below the stop reverses an armed
ESC -- so the overrides coast at ``init_pwm``, and the guard lets a sub-stop
PWM through only when it is this sequence's own. Forward resumes only after the
standstill (a stale speed gives no credit), and its first ``forward_start_pwm``
disarms the ESC. :attr:`LongitudinalShell.drive_direction` is -1 from the
arming until that standstill: the actuator publishes it (0 when reverse is off
or refused), the velocity report signs the hall speed by it and the gear
manager reports REVERSE from it.

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
    # Deliberate reverse (``reverse_enabled``); never entered when it is off
    REVERSE_ARM = "reverse_arm"
    REVERSE = "reverse"
    REVERSE_STOP = "reverse_stop"


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

# The thresholds deliberate reverse needs, all measured (non-zero)
REVERSE_PWM_KEYS = ("forward_start_pwm", "reverse_stop_pwm", "reverse_start_pwm",
                    "reverse_arm_pwm")

# The direction the shell drives the car, published by the actuator for
# velocity_report's sign and the gear manager: REVERSE from the start of a
# reverse request's arming until the car is at a standstill after it.
DIRECTION_FORWARD = 1
DIRECTION_REVERSE = -1
# What the actuator publishes when reverse is off or refused
DIRECTION_UNAVAILABLE = 0


class ReversePhase(Enum):
    """Where the shell is in a deliberate reverse; IDLE whenever it is off."""

    IDLE = "idle"                # not reversing: the forward modes above
    WAIT_STOP = "wait_stop"      # asked backward while rolling forward: stop first
    ARM_PULL = "arm_pull"        # at reverse_arm_pwm, stopped
    ARM_RETURN = "arm_return"    # back at init_pwm (>= ARM_RETURN), stopped
    DRIVE = "drive"              # reversing under the shell's own speed loop
    STOPPING = "stopping"        # at init_pwm until the car stands still


# Phases in which the car is, or is about to be, driven backward
_REVERSE_EPISODE = (ReversePhase.ARM_PULL, ReversePhase.ARM_RETURN, ReversePhase.DRIVE,
                    ReversePhase.STOPPING)


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
        elif s == EscState.ARMED:
            pass    # armed, a pull to arm is just a stronger reverse; only forward disarms
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
        reverse_enabled: Drive backward on a target below
            ``-full_stop_threshold`` (see the module docstring). Off by default;
            refused when the car's reverse thresholds are not measured.
        reverse_max_speed: Reverse speed cap [m/s]; a faster backward target
            is driven at this speed
        reverse_floor_pwm: Lowest PWM the reverse loop may command; 0 means
            ``min_pwm``. Must lie above ``reverse_arm_pwm``.
        reverse_ff_gain: Reverse loop feed-forward [counts per m/s of target]
        reverse_kp: Reverse loop proportional gain [counts per m/s of error]
        reverse_ki: Reverse loop integral gain [counts per m of error]
        reverse_integral_limit: Reverse loop integral bound [counts]
        reverse_arm_time: How long the arming pull is held [s]
        reverse_arm_margin: Counts below ``reverse_arm_pwm`` the pull goes to
            (the threshold is measured to +-5)
        reverse_arm_return_time: How long the return to ``init_pwm`` is held
            before reversing [s]
        reverse_standstill_time: How long the speed must stay below
            ``full_stop_threshold`` before the car counts as stopped after a
            reverse, and may drive forward [s]
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
    reverse_enabled: bool = False
    reverse_max_speed: float = 0.5
    reverse_floor_pwm: int = 0
    reverse_ff_gain: float = 20.0
    reverse_kp: float = 20.0
    reverse_ki: float = 20.0
    reverse_integral_limit: float = 15.0
    reverse_arm_time: float = 0.3
    reverse_arm_margin: int = 5
    reverse_arm_return_time: float = 0.3
    reverse_standstill_time: float = 0.5


def reverse_unavailable_reason(cfg: ShellConfig) -> Optional[str]:
    """Return why this car cannot reverse on purpose, or None if it can."""
    missing = [k for k in REVERSE_PWM_KEYS if getattr(cfg, k) <= 0]
    if missing:
        return (f"the car's {', '.join(missing)} not measured (0); reverse needs the "
                "ESC's arming and reverse thresholds (docs/design/esc-pwm-model.md)")
    floor = cfg.reverse_floor_pwm or cfg.min_pwm
    if not cfg.reverse_arm_pwm < floor <= cfg.reverse_start_pwm:
        return (f"reverse_floor_pwm {floor} must lie in reverse_arm_pwm + 1 .. "
                f"reverse_start_pwm ({cfg.reverse_arm_pwm + 1}..{cfg.reverse_start_pwm}): "
                "at or below reverse_arm_pwm the reverse command re-arms the ESC")
    if not 0 <= cfg.reverse_arm_margin < cfg.reverse_arm_pwm:
        return (f"reverse_arm_margin {cfg.reverse_arm_margin} must lie in "
                f"0..{cfg.reverse_arm_pwm - 1}: the pull is reverse_arm_pwm - margin")
    if cfg.reverse_max_speed <= 0.0:
        return f"reverse_max_speed {cfg.reverse_max_speed} must be positive"
    return None


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

    # Measured reverse speed above reverse_max_speed at which the loop coasts [m/s]
    REVERSE_OVERSPEED = 0.2

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

        # Deliberate reverse (off unless reverse_enabled and the car can)
        self.reverse_enabled = bool(config.reverse_enabled)
        if self.reverse_enabled:
            why = reverse_unavailable_reason(config)
            if why is not None:
                self.reverse_enabled = False
                self.logger.error(f"Reverse refused, kept disabled: {why}")
            else:
                self.logger.info(
                    f"Reverse enabled: arms at {config.reverse_arm_pwm}, reverses at "
                    f"<= {config.reverse_start_pwm} from rest, up to "
                    f"{config.reverse_max_speed:.2f} m/s")
        self.reverse_phase = ReversePhase.IDLE
        self._phase_since = 0.0
        self._rev_armed = False        # known armed: this shell armed it, or it reversed
        self._gear_reverse = False     # the latest gear command is REVERSE
        self._rev_still_since: Optional[float] = None   # reverse commanded, no motion
        self._rev_integral = 0.0
        self._rev_filtered: Optional[float] = None
        self._still_since: Optional[float] = None
        self._reverse_output = False   # this tick's PWM is a deliberate reverse command
        self._loop_dt = 0.0

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
        self._loop_dt = 0.0 if self._last_step_time is None else now - self._last_step_time
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

        self._reverse_output = False
        if self.reverse_enabled:
            self._track_standstill(now, current_speed)

        if override is not None:
            if self.reverse_enabled:
                self._interrupt_reverse(now)
            pwm = self._run_override(now, override, current_speed)
        elif self.reverse_enabled and self._reverse_wanted(target_speed, current_speed):
            pwm = self._run_reverse(now, target_speed, current_speed)
        else:
            pwm = self._run_control(now, target_speed, current_speed, dt)

        pwm = self._guard_reverse(pwm, target_speed if override is None else None)
        self._check_wheel_speed(now, pwm, current_speed)
        self.esc.update(pwm)
        if self.esc.state != EscState.ARMED:
            self._rev_armed = False
        self.last_output_pwm = pwm
        return pwm

    def set_gear_reverse(self, reverse: bool) -> None:
        """Take Autoware's gear command: reverse only while it is REVERSE (a second key)."""
        self._gear_reverse = bool(reverse)

    @property
    def drive_direction(self) -> int:
        """DIRECTION_REVERSE from a reverse request's arming until the car stops after it."""
        if self.reverse_phase in _REVERSE_EPISODE:
            return DIRECTION_REVERSE
        return DIRECTION_FORWARD

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
        if self.reverse_enabled:
            # Only the shell's own reverse sequence may (the arming pull, the
            # reverse loop); every other mode's sub-stop PWM becomes the stop
            if self._reverse_output:
                return pwm
        elif target_speed is not None and target_speed < -cfg.full_stop_threshold:
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

    # -------------------------------------------------------------- reverse

    def _set_phase(self, now: float, phase: ReversePhase) -> None:
        if phase != self.reverse_phase:
            self.logger.info(f"Reverse: {self.reverse_phase.value} -> {phase.value}")
            self.reverse_phase = phase
            self._phase_since = now

    def _track_standstill(self, now: float, current_speed: Optional[float]) -> None:
        """Note since when the speed has been below full_stop_threshold; unknown is not still."""
        speed = current_speed
        if speed is None or abs(speed) >= self.config.full_stop_threshold:
            self._still_since = None
        elif self._still_since is None:
            self._still_since = now

    def _stood_still(self, now: float) -> bool:
        return (self._still_since is not None and
                now - self._still_since >= self.config.reverse_standstill_time)

    def _reverse_wanted(self, target_speed: Optional[float],
                        current_speed: Optional[float]) -> bool:
        """Whether this tick belongs to the reverse logic rather than the forward modes."""
        if self.reverse_phase != ReversePhase.IDLE:
            return True
        return self._backward_asked(target_speed, current_speed)

    def _backward_asked(self, target_speed: Optional[float],
                        current_speed: Optional[float]) -> bool:
        """Whether backward is asked: a backward target and the REVERSE gear command."""
        return (self._gear_reverse and target_speed is not None and current_speed is not None
                and target_speed < -self.config.full_stop_threshold)

    def _interrupt_reverse(self, now: float) -> None:
        """Drop the reverse request for an override; a car that may be rolling back stops."""
        if self.reverse_phase in (ReversePhase.WAIT_STOP, ReversePhase.ARM_PULL,
                                  ReversePhase.ARM_RETURN):
            # Nothing has driven the car backward: no reverse episode to finish
            self._set_phase(now, ReversePhase.IDLE)
        elif self.reverse_phase in (ReversePhase.DRIVE, ReversePhase.STOPPING):
            if self.reverse_phase != ReversePhase.STOPPING:
                self._set_phase(now, ReversePhase.STOPPING)
            if self._stood_still(now):
                self._end_reverse(now)
        self._rev_still_since = None

    def _end_reverse(self, now: float) -> None:
        """Back to the forward modes, with the plug-in and its filter fresh."""
        self._set_phase(now, ReversePhase.IDLE)
        self._rev_integral = 0.0
        self._rev_filtered = None
        self._rev_still_since = None
        self.filtered_measured = None
        self.last_pwm_value = self.config.init_pwm
        self.speed_control_pwm_offset = 0.0
        try:
            self.controller.reset()
        except Exception as e:  # noqa: B902
            self._on_fault(now, "reset()", e)

    def _run_reverse(self, now: float, target_speed: Optional[float],
                     current_speed: Optional[float]) -> int:
        """
        Run one tick of the reverse sequence (docs/design/esc-pwm-model.md).

        Stop if rolling forward; at a standstill, arm the ESC unless the model
        says it is (possibly) armed already -- a pull to an armed ESC is a
        full-strength reverse -- then reverse under :meth:`_reverse_loop`. A
        target that is no longer backward stops the car at ``init_pwm`` (the
        ESC stays armed) and only a standstill hands back to the forward modes.
        """
        cfg = self.config
        want = self._backward_asked(target_speed, current_speed)
        speed = abs(current_speed) if current_speed is not None else None
        stopped = speed is not None and speed < cfg.full_stop_threshold
        phase = self.reverse_phase

        if phase == ReversePhase.IDLE:
            phase = ReversePhase.WAIT_STOP
            self._set_phase(now, phase)

        if phase in (ReversePhase.ARM_PULL, ReversePhase.ARM_RETURN) and \
                (not want or not stopped):
            # Asked forward again, or the car moved (it was not reversed: the ESC
            # only brakes while it arms): drop the request. A pull already made
            # leaves the ESC armed, and the model says so, so it is not repeated.
            self._set_phase(now, ReversePhase.IDLE)
            return self._run_control(now, target_speed, current_speed, self.last_dt)

        if phase == ReversePhase.WAIT_STOP:
            if not want:
                self._set_phase(now, ReversePhase.IDLE)
                return self._run_control(now, target_speed, current_speed, self.last_dt)
            if not stopped:
                # Rolling forward: an ordinary stop first. brake_pwm stays above
                # reverse_arm_pwm (check_esc_config), so it cannot arm the ESC.
                self.mode = Mode.BRAKE if speed > cfg.brake_threshold else Mode.COAST
                return cfg.brake_pwm if speed > cfg.brake_threshold else cfg.init_pwm
            if not self._stood_still(now):
                # Below full_stop_threshold is not yet still (the hall reading
                # lags): the arming pull on a creeping car is a hard brake
                self.mode = Mode.FULL_STOP
                return cfg.init_pwm
            # Possibly armed (the model's start state, an earlier pull, or a
            # reverse): no pull, reverse from rest at reverse_start_pwm. An ESC
            # that turns out locked does not move, and is then armed (below).
            phase = ReversePhase.DRIVE if self.esc.state == EscState.ARMED \
                else ReversePhase.ARM_PULL
            self._set_phase(now, phase)

        if phase == ReversePhase.ARM_PULL:
            if now - self._phase_since < cfg.reverse_arm_time:
                self.mode = Mode.REVERSE_ARM
                self._reverse_output = True
                return cfg.reverse_arm_pwm - cfg.reverse_arm_margin
            phase = ReversePhase.ARM_RETURN
            self._set_phase(now, phase)
            self._rev_armed = True      # from the return on, the ESC is armed

        if phase == ReversePhase.ARM_RETURN:
            if now - self._phase_since < cfg.reverse_arm_return_time:
                self.mode = Mode.REVERSE_ARM
                return cfg.init_pwm
            phase = ReversePhase.DRIVE
            self._set_phase(now, phase)

        if phase == ReversePhase.STOPPING and want and self.esc.state == EscState.ARMED:
            phase = ReversePhase.DRIVE      # backward again, still armed
            self._set_phase(now, phase)

        if phase == ReversePhase.DRIVE:
            if want and self.esc.state == EscState.ARMED:
                pwm = self._reverse_loop(target_speed, speed, stopped)
                if self._reverse_did_not_move(now, pwm, speed):
                    return self._run_reverse(now, target_speed, current_speed)
                self.mode = Mode.REVERSE
                self._reverse_output = True
                return pwm
            phase = ReversePhase.STOPPING
            self._set_phase(now, phase)

        # STOPPING: the stop, inside the band where nothing moves; the ESC stays
        # armed. No brake exists: anything below the stop reverses an armed ESC,
        # anything at forward_start_pwm drives forward.
        self._rev_integral = 0.0
        self._rev_filtered = None
        self._rev_still_since = None
        if self._stood_still(now):
            self._end_reverse(now)
            if want:
                return self._run_reverse(now, target_speed, current_speed)
            return self._run_control(now, target_speed, current_speed, self.last_dt)
        self.mode = Mode.REVERSE_STOP
        return cfg.init_pwm

    def _reverse_did_not_move(self, now: float, pwm: int, speed: float) -> bool:
        """
        Watch a reverse command for motion; True when the sequence was changed.

        Commanded at or below ``reverse_start_pwm`` with the wheel speed below
        ``wheel_speed_check_min_speed`` for ``wheel_speed_check_time``: an ESC
        only assumed armed (the start state) was locked after all, so the
        model is set LOCKED and the shell arms it. One this shell armed, or
        that has reversed, means the wheel speed is not read: the wheel speed
        fault latches (when the check is on), the model goes LOCKED and the
        next attempt arms afresh.
        """
        cfg = self.config
        if speed >= cfg.wheel_speed_check_min_speed:
            self._rev_armed = True          # it reversed: armed for certain
            self._rev_still_since = None
            return False
        if pwm > cfg.reverse_start_pwm:
            self._rev_still_since = None
            return False
        if self._rev_still_since is None:
            self._rev_still_since = now
            return False
        if now - self._rev_still_since < cfg.wheel_speed_check_time:
            return False
        self._rev_still_since = None
        self._rev_integral = 0.0
        self._rev_filtered = None
        if not self._rev_armed:
            self.logger.warning(
                f"Reverse at {pwm} did not move the car in {cfg.wheel_speed_check_time:.1f} s: "
                "the ESC was not armed after all; arming it")
            self.esc.state = EscState.LOCKED
            self._set_phase(now, ReversePhase.ARM_PULL)
            return True
        self.esc.state = EscState.LOCKED    # whatever it is, arm properly next time
        self._rev_armed = False
        if self._wheel_check_active():
            self.wheel_fault_latched = True
            self._stop_command_since = None
            self.logger.error(
                f"Wheel speed fault: reversing at {pwm} (<= reverse_start_pwm "
                f"{cfg.reverse_start_pwm}) for {cfg.wheel_speed_check_time:.1f} s while the "
                f"wheel speed reads {speed:.3f} m/s on an ESC this shell armed. Is the hall "
                f"sensor connected? Motor held stopped until the command is a stop for "
                f"{self.WHEEL_FAULT_CLEAR_TIME:.1f} s")
            self._set_phase(now, ReversePhase.STOPPING)
            return True
        return False

    def _reverse_loop(self, target_speed: float, speed: float, stopped: bool) -> int:
        """
        Reverse speed loop: feed-forward + PI on |speed|, below the stop.

        The shell's own loop, not the plug-in: a plug-in's contract is a
        forward one (``measured_speed`` never negative, PWM around
        ``init_pwm``), and the ESC's reverse band starts below
        ``reverse_stop_pwm``, not at ``init_pwm``. Counts grow downward from
        ``reverse_stop_pwm - 1`` (the slowest reverse), clamped to the floor;
        from rest the command is at most ``reverse_start_pwm``, the break-away.
        """
        cfg = self.config
        dt = self._loop_dt     # the tick period, whatever dt_source says
        if self._rev_filtered is None:
            self._rev_filtered = speed
        else:
            a = self.MEASURED_FILTER_ALPHA
            self._rev_filtered = a * speed + (1.0 - a) * self._rev_filtered
        target = min(-target_speed, cfg.reverse_max_speed)
        error = target - self._rev_filtered
        top = cfg.reverse_stop_pwm - 1
        floor = cfg.reverse_floor_pwm or cfg.min_pwm
        counts = cfg.reverse_ff_gain * target + cfg.reverse_kp * error + self._rev_integral
        pwm = top - int(round(counts))
        saturated = (pwm <= floor and error > 0) or (pwm >= top and error < 0)
        if not saturated:
            lim = cfg.reverse_integral_limit
            self._rev_integral = max(
                -lim, min(lim, self._rev_integral + cfg.reverse_ki * error * dt))
        if self._rev_filtered > cfg.reverse_max_speed + self.REVERSE_OVERSPEED:
            return cfg.init_pwm      # well over the cap: coast (the ESC stays armed)
        pwm = max(floor, min(top, pwm))
        if stopped:
            pwm = min(pwm, cfg.reverse_start_pwm)
        self.last_pwm_value = pwm
        self.speed_control_pwm_offset = float(pwm - cfg.init_pwm)
        return pwm

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

        if self.reverse_phase in _REVERSE_EPISODE:
            # Rolling backward with the ESC armed: brake_pwm is below the stop
            # and would drive it backward harder. The stop coasts it.
            moving = False

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
