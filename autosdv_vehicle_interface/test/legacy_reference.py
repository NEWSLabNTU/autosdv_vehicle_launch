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
# flake8: noqa
"""
FROZEN copy of the longitudinal controller as it was before the plug-in split.

Do not edit. This is the reference test_speed_controller_equivalence.py holds
the refactored code to. Everything below the marker is copied verbatim from
autosdv_vehicle_interface/actuator.py at autosdv_vehicle_launch commit 046230a
(AckermannPID, and the AutoSdvActuator methods run_longitudinal_control,
run_control_full_stop and run_control_reverse); only the enclosing class
LegacyLongitudinal and its __init__ are new.
"""
from enum import Enum
from types import SimpleNamespace


class MotorState(Enum):
    STOPPED = "stopped"
    FORWARD = "forward"
    BACKWARD = "backward"
    BRAKE_LOCKED = "brake_locked"
    TRANSITIONING = "transitioning"


class _NullLogger:
    def debug(self, *args, **kwargs):
        pass


# ---- verbatim from 046230a below ----
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

class LegacyLongitudinal:
    """The pre-split AutoSdvActuator, reduced to its longitudinal path."""

    def __init__(self, params):
        # As AutoSdvActuator.initialize_pid_controllers / initialize_state did
        pwm_range = params["max_pwm"] - params["min_pwm"]
        max_pwm_offset = pwm_range // 2
        self.speed_controller = AckermannPID(
            kp=params["kp_speed"],
            ki=params["ki_speed"],
            kd=params["kd_speed"],
            min_output=-max_pwm_offset,
            max_output=max_pwm_offset,
            integral_limit=params["integral_limit"],
            enable_conditional_integration=params["enable_conditional_integration"],
        )
        self.velocity_deadband = params["velocity_deadband"]
        self.full_stop_threshold = params["full_stop_threshold"]
        self.brake_threshold = params["brake_threshold"]
        self.speed_controller.derivative_filter_alpha = params["derivative_filter_alpha"]
        self.config = SimpleNamespace(
            min_pwm=params["min_pwm"],
            init_pwm=params["init_pwm"],
            max_pwm=params["max_pwm"],
            brake_pwm=params["brake_pwm"],
        )
        self.state = SimpleNamespace(
            target_speed=None,
            current_speed=None,
            speed_control_pwm_offset=0.0,
            in_reverse=False,
            motor_state=MotorState.STOPPED,
            last_pwm_value=params["init_pwm"],
        )

    def get_logger(self):
        return _NullLogger()

    # Update run_longitudinal_control to match diagram's PID flow:
    def run_longitudinal_control(self, delta_time: float) -> int:
        """
        Implements longitudinal control with low-pass filtering as shown in diagram.

        Flow:
        1. Low-pass filter on measured velocity (EMA α=0.3)
        2. Calculate error (target - measured)
        3. PID controller
        4. Anti-windup integral limit
        5. PWM correction and clamping

        Args:
            delta_time: Time since last update in seconds

        Returns:
            int: PWM value for motor
        """
        # Return init PWM if we don't have necessary state information
        if self.state.target_speed is None or self.state.current_speed is None:
            return self.config.init_pwm

        # Check if we need to apply brake when coming to a stop
        if (abs(self.state.target_speed) < self.full_stop_threshold and
            abs(self.state.current_speed) > self.brake_threshold):
            self.get_logger().debug(f"Applying brake")
            return self.config.brake_pwm

        # Check for full stop condition
        if self.run_control_full_stop():
            return self.config.init_pwm

        # Low-pass filter on measured velocity (from diagram: EMA α=0.3)
        raw_measured = self.state.current_speed
        if not hasattr(self.state, 'filtered_yaw_rate'):
            self.state.filtered_yaw_rate = raw_measured
        else:
            alpha_measured = 0.3  # From diagram
            self.state.filtered_yaw_rate = (
                alpha_measured * raw_measured +
                (1.0 - alpha_measured) * self.state.filtered_yaw_rate
            )

        # Use target speed directly (diagram shows no filter on target)
        target_velocity = self.state.target_speed
        measured_velocity = self.state.filtered_yaw_rate

        # Calculate error
        yaw_rate_error = target_velocity - measured_velocity

        # Apply velocity deadband
        if abs(yaw_rate_error) < self.velocity_deadband:
            return self.state.last_pwm_value

        # Handle reverse logic
        self.run_control_reverse()

        # PID controller
        self.speed_controller.set_target_point(target_velocity)
        pwm_correction = self.speed_controller.run(measured_velocity, delta_time)

        # Store for debug
        self.state.speed_control_pwm_offset = pwm_correction

        # Combine: pwm = init + correction
        raw_pwm = self.config.init_pwm + int(pwm_correction)

        # Clamp to min/max PWM
        clamped_pwm = max(self.config.min_pwm, min(self.config.max_pwm, raw_pwm))

        # Store for next iteration
        self.state.last_pwm_value = clamped_pwm

        return clamped_pwm

    def run_control_full_stop(self) -> bool:
        """
        Check if vehicle should be in full stop mode.

        Returns:
            bool: True if full stop should be applied
        """
        # Use configurable threshold for considering vehicle stopped
        # Check if both current and target speeds are near zero
        if (
            abs(self.state.current_speed) < self.full_stop_threshold
            and abs(self.state.target_speed) < self.full_stop_threshold
        ):
            return True

        return False

    def run_control_reverse(self) -> None:
        """
        Handle the logic for changing between forward and reverse gears.
        Works with the motor state machine to respect PWM braking constraints.
        """
        # Epsilon threshold for considering vehicle stopped
        standing_still_epsilon = 0.1  # [m/s]

        # Only process if we have valid speed data
        if self.state.current_speed is None or self.state.target_speed is None:
            return

        current_speed = self.state.current_speed
        target_speed = self.state.target_speed

        # Check if vehicle is essentially stopped
        if abs(current_speed) < standing_still_epsilon:
            # Vehicle is standing still, safe to change direction
            if target_speed < -0.1:  # Requesting reverse
                self.state.in_reverse = True
            elif target_speed > 0.1:  # Requesting forward
                self.state.in_reverse = False
            # else: target near zero, maintain current direction mode

        else:
            # Vehicle is moving, handle direction change requests carefully
            direction_change_requested = (current_speed * target_speed) < 0

            if direction_change_requested:
                # Direction change requested while moving
                # Check motor state to determine strategy
                if self.state.motor_state == MotorState.BRAKE_LOCKED:
                    # Already braking, wait for stop before changing direction
                    pass  # Keep current in_reverse state
                elif abs(current_speed) > 0.5:  # Moving fast
                    # Force stop first by setting target to zero
                    # Don't change in_reverse state yet
                    pass  # Let the controller bring speed to zero first
                else:
                    # Moving slowly, can start preparing for direction change
                    if target_speed < 0 and not self.state.in_reverse:
                        # Prepare for reverse
                        self.state.in_reverse = True
                    elif target_speed > 0 and self.state.in_reverse:
                        # Prepare for forward
                        self.state.in_reverse = False

