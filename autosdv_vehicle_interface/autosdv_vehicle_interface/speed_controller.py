#!/usr/bin/env python3
"""
Speed-controller plug-in interface for the AutoSDV actuator.

The actuator node turns a target speed into a motor PWM count in two layers:

- A **safety shell** (``longitudinal.py``) that it owns and that always wins:
  emergency stop, command watchdog, e-brake, full stop, deadband hold, output
  clamping and fault handling.
- A **speed controller** that the shell calls only in active control, and that
  is selected by import path with the ``speed_controller_class`` parameter.

A speed controller is any class with this shape (inheriting from
:class:`SpeedPID` is optional, the shell only duck-types it)::

    class SpeedPID:
        def __init__(self, params: dict): ...
        def reset(self) -> None: ...
        def update(self, target_speed: float, measured_speed: float, dt: float) -> int:
            '''Return the motor PWM count. The actuator clamps and overrides it.'''

``params`` holds the actuator's motor parameters (``init_pwm``, ``min_pwm``,
``max_pwm``, ``brake_pwm``, ``rate``, the ``*_speed`` PID gains, ...) merged
with the mapping in the ``speed_controller_params`` parameter, which wins on a
key collision.

:class:`AckermannSpeedPID` is the default and reproduces the controller this
node has always run, bit for bit.
"""
import importlib
from typing import Any, Dict, Optional, Tuple

DEFAULT_SPEED_CONTROLLER = "autosdv_vehicle_interface.speed_controller:AckermannSpeedPID"


class AckermannPID:
    """
    Custom PID controller for Ackermann control with anti-windup protection.

    Similar to CARLA's PID implementation but adapted for Python and ROS 2.
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        min_output: float = -1.0,
        max_output: float = 1.0,
        integral_limit: float = None,
        enable_conditional_integration: bool = False,
    ):
        """
        Initialize a new PID controller.

        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            min_output: Minimum output value
            max_output: Maximum output value
            integral_limit: Maximum integral accumulation (None = use max_output)
            enable_conditional_integration: Stop integrating when output saturated
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.min_output = min_output
        self.max_output = max_output
        self.integral_limit = integral_limit if integral_limit is not None else max_output
        self.enable_conditional_integration = enable_conditional_integration

        self.set_point = 0.0

        # Internal state
        self.proportional = 0.0
        self.integral = 0.0
        self.derivative = 0.0
        self.derivative_filtered = 0.0  # Filtered derivative for smoother control
        self.is_saturated = False  # Track saturation state for conditional integration

        self.last_error = 0.0
        self.last_input = 0.0

        # Derivative filtering parameter (EMA)
        self.derivative_filter_alpha = 0.15  # Lower = more filtering (heavy filtering for stability)

    def set_target_point(self, point: float) -> None:
        """
        Set the target point for the controller.

        Args:
            point: Target value
        """
        self.set_point = point

    def run(self, input_value: float, delta_time: float) -> float:
        """
        Run the PID controller for one step.

        Args:
            input_value: Current value
            delta_time: Time since last update in seconds

        Returns:
            float: Controller output
        """
        error = self.set_point - input_value

        self.proportional = self.kp * error

        # Conditional integration - only integrate if not saturated
        if self.enable_conditional_integration:
            if not self.is_saturated:
                self.integral += self.ki * error * delta_time
        else:
            # Always integrate (existing behavior)
            self.integral += self.ki * error * delta_time

        # Clamp integral to separate limit (anti-windup)
        self.integral = max(-self.integral_limit, min(self.integral, self.integral_limit))

        # Derivative on measurement (not error) to reduce derivative kick
        # Negate to provide damping: when measurement increases rapidly, reduce output
        if delta_time > 0:
            raw_derivative = -(self.kd * (input_value - self.last_input)) / delta_time
            # Apply low-pass filter to derivative to reduce noise sensitivity
            self.derivative_filtered = (
                self.derivative_filter_alpha * raw_derivative +
                (1.0 - self.derivative_filter_alpha) * self.derivative_filtered
            )
            self.derivative = self.derivative_filtered
        else:
            self.derivative = 0.0

        # Calculate output
        output = self.proportional + self.integral + self.derivative

        # Clamp output and track saturation state
        output = max(self.min_output, min(output, self.max_output))
        self.is_saturated = (output >= self.max_output or output <= self.min_output)

        # Keep track of state
        self.last_error = error
        self.last_input = input_value

        return output

    def reset(self) -> None:
        """Reset the controller state."""
        self.proportional = 0.0
        self.integral = 0.0
        self.derivative = 0.0
        self.is_saturated = False

        self.last_error = 0.0
        self.last_input = 0.0


class SpeedPID:
    """
    The speed-controller interface: target speed in, motor PWM count out.

    The shell calls :meth:`update` once per control tick, and only when no
    safety mode is active. It calls :meth:`reset` after every episode in which
    the controller was bypassed by an override (watchdog, emergency, stale
    velocity, controller fault), so integrators do not carry a stale state into
    the next approach.

    Whatever :meth:`update` returns is clamped to ``[min_pwm, max_pwm]``. An
    exception, or a value that is not a finite number, puts the motor in
    neutral (braking first if the vehicle is moving) and is logged; it does not
    stop the node.
    """

    def __init__(self, params: Dict[str, Any]):
        self.params = dict(params)

    def reset(self) -> None:
        """Forget all internal state (integrators, filters, last values)."""

    def update(self, target_speed: float, measured_speed: float, dt: float) -> int:
        """
        Compute the motor command for one control tick.

        Args:
            target_speed: Commanded longitudinal speed [m/s].
            measured_speed: Measured speed [m/s], after the shell's low-pass
                filter (EMA, alpha 0.3). The hall sensor has no direction, so
                this is never negative on the real vehicle.
            dt: Seconds since the previous sample (see ``speed_controller_dt_source``).

        Returns:
            int: Motor PWM count; ``params["init_pwm"]`` is neutral.
        """
        raise NotImplementedError


class AckermannSpeedPID(SpeedPID):
    """
    The default speed controller: :class:`AckermannPID` around ``init_pwm``.

    Output is ``init_pwm + int(pid_output)``, the PID output limited to
    ``+-(max_pwm - min_pwm) // 2``. These are the exact semantics of the
    controller that was inlined in ``actuator.py`` before the plug-in split.

    Reads ``kp_speed``, ``ki_speed``, ``kd_speed``, ``integral_limit``,
    ``enable_conditional_integration``, ``derivative_filter_alpha``,
    ``init_pwm``, ``min_pwm`` and ``max_pwm`` from ``params``.
    """

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        pwm_range = int(params["max_pwm"]) - int(params["min_pwm"])
        max_pwm_offset = pwm_range // 2

        self.init_pwm = int(params["init_pwm"])
        self.pid = AckermannPID(
            kp=params["kp_speed"],
            ki=params["ki_speed"],
            kd=params["kd_speed"],
            min_output=-max_pwm_offset,
            max_output=max_pwm_offset,
            integral_limit=params["integral_limit"],
            enable_conditional_integration=params["enable_conditional_integration"],
        )
        self.pid.derivative_filter_alpha = params["derivative_filter_alpha"]

        # Last unrounded PID output, published on ~/debug/control_values
        self.last_output_offset = 0.0

    def reset(self) -> None:
        self.pid.reset()
        self.last_output_offset = 0.0

    def update(self, target_speed: float, measured_speed: float, dt: float) -> int:
        self.pid.set_target_point(target_speed)
        pwm_correction = self.pid.run(measured_speed, dt)
        self.last_output_offset = pwm_correction
        return self.init_pwm + int(pwm_correction)

    def debug_terms(self) -> Tuple[float, float, float]:
        """Return the (P, I, D) terms of the last update, for ~/debug/pid_values."""
        return (self.pid.proportional, self.pid.integral, self.pid.derivative)


def parse_class_path(spec: str) -> Tuple[str, str]:
    """
    Split ``"pkg.module:Class"`` into ``("pkg.module", "Class")``.

    A dotted ``"pkg.module.Class"`` is accepted too, split at the last dot.
    """
    spec = spec.strip()
    if ":" in spec:
        module_name, _, class_name = spec.partition(":")
    else:
        module_name, _, class_name = spec.rpartition(".")
    if not module_name or not class_name:
        raise ValueError(
            f"speed controller '{spec}' is not an import path of the form 'pkg.module:Class'"
        )
    return module_name, class_name


def load_speed_controller(spec: Optional[str], params: Dict[str, Any]):
    """
    Import and instantiate the speed controller named by ``spec``.

    Args:
        spec: ``"pkg.module:Class"``; empty or None selects the default.
        params: Passed to the class's constructor.

    Raises:
        ImportError, AttributeError, TypeError, ValueError: The class cannot be
            loaded or does not implement the interface. The actuator refuses to
            start rather than silently fall back to a different controller.
    """
    if not spec:
        spec = DEFAULT_SPEED_CONTROLLER
    module_name, class_name = parse_class_path(spec)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    for method in ("reset", "update"):
        if not callable(getattr(cls, method, None)):
            raise TypeError(f"speed controller '{spec}' has no {method}() method")
    return cls(dict(params))
