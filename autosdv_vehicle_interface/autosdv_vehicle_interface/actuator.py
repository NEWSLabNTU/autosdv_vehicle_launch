#!/usr/bin/env python3
"""
Actuator control node for AutoSDV using Ackermann control.
This module implements a cascaded PID controller for vehicle's motor and steering using Ackermann steering model.
"""
import sys
import math
import time
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from Adafruit_PCA9685 import PCA9685
from simple_pid import PID

import rclpy
from rclpy.node import Node
from rclpy import Parameter
from geometry_msgs.msg import TwistWithCovarianceStamped
from autoware_control_msgs.msg import Control
from autoware_vehicle_msgs.msg import VelocityReport
from std_msgs.msg import MultiArrayDimension, MultiArrayLayout
from autoware_internal_debug_msgs.msg import Float32MultiArrayStamped


class MotorState(Enum):
    """
    Motor state enum for tracking RC car motor behavior.

    The RC car has special PWM behavior:
    - PWM = init_pwm: STOPPED
    - PWM > init_pwm: FORWARD
    - PWM < init_pwm: BACKWARD or BRAKE_LOCKED
    - Special case: When moving forward and PWM drops below (init_pwm - threshold),
      motor locks for braking and cannot go backward until returning to init_pwm first
    """
    STOPPED = "stopped"
    FORWARD = "forward"
    BACKWARD = "backward"
    BRAKE_LOCKED = "brake_locked"
    TRANSITIONING = "transitioning"


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
        self.is_saturated = False  # Track saturation state for conditional integration

        self.last_error = 0.0
        self.last_input = 0.0

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
            self.derivative = -(self.kd * (input_value - self.last_input)) / delta_time
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


class AutoSdvActuator(Node):
    """
    Node that controls the vehicle's motor and steering servo using an Ackermann control model.

    Subscribes to:
    - Control commands from Autoware
    - Velocity reports for current speed

    Uses a cascaded PID controller similar to CARLA's Ackermann implementation:
    - Outer loop: Speed controller that produces target acceleration
    - Inner loop: Acceleration controller that produces throttle/brake commands
    """

    def __init__(self):
        super().__init__("autosdv_actuator_node")

        # Declare all parameters
        self.declare_all_parameters()

        # Get parameter values
        parameter_values = self.get_all_parameter_values()

        # Create configuration
        self.config = self.create_config(parameter_values)

        # Initialize PID controllers
        self.initialize_pid_controllers(parameter_values)

        # Initialize controller state
        self.state = self.initialize_state()

        # Set initial PWM value from config
        self.state.last_pwm_value = parameter_values["init_pwm"]

        # Initialize hardware
        self.driver = self.initialize_pwm_driver(parameter_values)

        # Set up subscriptions
        self.setup_subscriptions()

        # Create timer
        publication_period = 1.0 / parameter_values["rate"]
        self.timer = self.create_timer(publication_period, self.timer_callback)

        # Debug log
        self.get_logger().info("Ackermann controller initialized")

    def declare_all_parameters(self):
        """Declare all parameters used by the node."""
        # Hardware parameters
        self.declare_parameter("i2c_address", Parameter.Type.INTEGER)
        self.declare_parameter("i2c_busnum", Parameter.Type.INTEGER)
        self.declare_parameter("pwm_freq", Parameter.Type.INTEGER)

        # Publication rate
        self.declare_parameter("rate", Parameter.Type.DOUBLE)

        # Debugging options
        self.declare_parameter("enable_debug_publishing", Parameter.Type.BOOL)

        # Dry-run mode for simulation
        self.declare_parameter("dry_run", Parameter.Type.BOOL)

        # Longitudinal control parameters
        self.declare_parameter("kp_speed", Parameter.Type.DOUBLE)
        self.declare_parameter("ki_speed", Parameter.Type.DOUBLE)
        self.declare_parameter("kd_speed", Parameter.Type.DOUBLE)

        # Anti-windup configuration
        self.declare_parameter("integral_limit", Parameter.Type.DOUBLE)
        self.declare_parameter("enable_conditional_integration", Parameter.Type.BOOL)

        # Deadbands and thresholds
        self.declare_parameter("velocity_deadband", Parameter.Type.DOUBLE)
        self.declare_parameter("full_stop_threshold", Parameter.Type.DOUBLE)

        # Velocity filtering
        self.declare_parameter("velocity_measurement_filter_alpha", Parameter.Type.DOUBLE)
        self.declare_parameter("velocity_command_filter_alpha", Parameter.Type.DOUBLE)

        self.declare_parameter("min_pwm", Parameter.Type.INTEGER)
        self.declare_parameter("init_pwm", Parameter.Type.INTEGER)
        self.declare_parameter("max_pwm", Parameter.Type.INTEGER)
        self.declare_parameter("brake_pwm", Parameter.Type.INTEGER)
        self.declare_parameter("brake_threshold", Parameter.Type.DOUBLE)

        # Steering control parameters
        self.declare_parameter("steering_speed", Parameter.Type.DOUBLE)
        self.declare_parameter("min_steer", Parameter.Type.INTEGER)
        self.declare_parameter("init_steer", Parameter.Type.INTEGER)
        self.declare_parameter("max_steer", Parameter.Type.INTEGER)
        self.declare_parameter("max_steering_angle", Parameter.Type.DOUBLE)

        self.declare_parameter("tire_angle_to_steer_ratio", Parameter.Type.DOUBLE)

    def get_all_parameter_values(self):
        """
        Get all parameter values.

        Returns:
            dict: Dictionary with parameter names as keys and their values
        """
        params = {}

        # Hardware parameters
        params["i2c_address"] = (
            self.get_parameter("i2c_address").get_parameter_value().integer_value
        )
        params["i2c_busnum"] = (
            self.get_parameter("i2c_busnum").get_parameter_value().integer_value
        )
        params["pwm_freq"] = (
            self.get_parameter("pwm_freq").get_parameter_value().integer_value
        )

        # Publication rate
        params["rate"] = self.get_parameter("rate").get_parameter_value().double_value

        # Debug settings
        params["enable_debug_publishing"] = (
            self.get_parameter("enable_debug_publishing")
            .get_parameter_value()
            .bool_value
        )

        # Dry-run mode
        params["dry_run"] = (
            self.get_parameter("dry_run")
            .get_parameter_value()
            .bool_value
        )

        # PWM parameters
        params["min_pwm"] = (
            self.get_parameter("min_pwm").get_parameter_value().integer_value
        )
        params["init_pwm"] = (
            self.get_parameter("init_pwm").get_parameter_value().integer_value
        )
        params["max_pwm"] = (
            self.get_parameter("max_pwm").get_parameter_value().integer_value
        )
        params["brake_pwm"] = (
            self.get_parameter("brake_pwm").get_parameter_value().integer_value
        )
        params["brake_threshold"] = (
            self.get_parameter("brake_threshold").get_parameter_value().double_value
        )

        # Steering parameters
        params["min_steer"] = (
            self.get_parameter("min_steer").get_parameter_value().integer_value
        )
        params["init_steer"] = (
            self.get_parameter("init_steer").get_parameter_value().integer_value
        )
        params["max_steer"] = (
            self.get_parameter("max_steer").get_parameter_value().integer_value
        )
        params["tire_angle_to_steer_ratio"] = (
            self.get_parameter("tire_angle_to_steer_ratio")
            .get_parameter_value()
            .double_value
        )

        # Ackermann parameters
        params["steering_speed"] = (
            self.get_parameter("steering_speed").get_parameter_value().double_value
        )
        params["max_steering_angle"] = (
            self.get_parameter("max_steering_angle").get_parameter_value().double_value
        )

        # PID controller parameters - speed
        params["kp_speed"] = (
            self.get_parameter("kp_speed").get_parameter_value().double_value
        )
        params["ki_speed"] = (
            self.get_parameter("ki_speed").get_parameter_value().double_value
        )
        params["kd_speed"] = (
            self.get_parameter("kd_speed").get_parameter_value().double_value
        )

        # Anti-windup configuration
        params["integral_limit"] = (
            self.get_parameter("integral_limit").get_parameter_value().double_value
        )
        params["enable_conditional_integration"] = (
            self.get_parameter("enable_conditional_integration")
            .get_parameter_value()
            .bool_value
        )

        # Deadbands and thresholds
        params["velocity_deadband"] = (
            self.get_parameter("velocity_deadband").get_parameter_value().double_value
        )
        params["full_stop_threshold"] = (
            self.get_parameter("full_stop_threshold").get_parameter_value().double_value
        )

        # Velocity filtering
        params["velocity_measurement_filter_alpha"] = (
            self.get_parameter("velocity_measurement_filter_alpha")
            .get_parameter_value()
            .double_value
        )
        params["velocity_command_filter_alpha"] = (
            self.get_parameter("velocity_command_filter_alpha")
            .get_parameter_value()
            .double_value
        )

        return params

    def create_config(self, params):
        """
        Create the configuration object.

        Args:
            params: Dictionary with parameter values

        Returns:
            Config: Configuration object
        """
        return Config(
            min_pwm=params["min_pwm"],
            init_pwm=params["init_pwm"],
            max_pwm=params["max_pwm"],
            brake_pwm=params["brake_pwm"],
            min_steer=params["min_steer"],
            init_steer=params["init_steer"],
            max_steer=params["max_steer"],
            tire_angle_to_steer_ratio=params["tire_angle_to_steer_ratio"],
            steering_speed=params["steering_speed"],
            max_steering_angle=params["max_steering_angle"],
        )

    def initialize_pid_controllers(self, params):
        """
        Initialize the PID controller for speed control.

        Args:
            params: Dictionary with parameter values
        """
        # Speed controller - outputs PWM offset directly
        # Output range: -90 to +90 (for PWM range 370±90 = 280-460)
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

        # Store filtering and deadband parameters as instance variables
        self.velocity_deadband = params["velocity_deadband"]
        self.full_stop_threshold = params["full_stop_threshold"]
        self.brake_threshold = params["brake_threshold"]
        self.vel_meas_alpha = params["velocity_measurement_filter_alpha"]
        self.vel_cmd_alpha = params["velocity_command_filter_alpha"]

    def initialize_state(self):
        """
        Initialize the controller state.

        Returns:
            State: State object
        """
        return State(
            target_speed=None,
            current_speed=None,
            target_tire_angle=None,
            current_tire_angle=None,
            # Control state
            speed_control_pwm_offset=0.0,
            last_speed=0.0,
            in_reverse=False,
            last_update_time=time.time(),
            # Motor state machine
            motor_state=MotorState.STOPPED,
            last_pwm_value=307,  # Will be set from config
            transition_target=None,
            brake_threshold=20,  # Default threshold
        )

    def initialize_pwm_driver(self, params):
        """
        Initialize the PCA9685 PWM driver.

        Args:
            params: Dictionary with parameter values

        Returns:
            PCA9685: PWM driver object or None if in dry-run mode
        """
        # Store dry-run mode
        self.dry_run = params.get("dry_run", False)

        if self.dry_run:
            self.get_logger().warn("DRY-RUN MODE ENABLED - No PWM output will be sent to hardware")
            return None

        try:
            driver = PCA9685(address=params["i2c_address"], busnum=params["i2c_busnum"])
            driver.set_pwm_freq(params["pwm_freq"])
            self.get_logger().info(f"PCA9685 initialized successfully on I2C bus {params['i2c_busnum']}, address 0x{params['i2c_address']:02x}")
            return driver
        except Exception as e:
            self.get_logger().error(f"Failed to initialize PCA9685: {e}")
            self.get_logger().error("Motor control will not work! Check I2C connections.")
            return None

    def setup_subscriptions(self):
        """Set up subscriptions to ROS topics."""
        # Subscribe to control commands
        self.control_cmd_subscription = self.create_subscription(
            Control,
            "~/input/control_cmd",
            self.control_callback,
            1,
        )

        # Subscribe to speed data
        self.speed_subscription = self.create_subscription(
            VelocityReport,
            "~/input/velocity_status",
            self.velocity_callback,
            1,
        )

        # Create debug publishers
        self.debug_control_publisher = self.create_publisher(
            Float32MultiArrayStamped, "~/debug/control_values", 10
        )

        self.debug_pwm_publisher = self.create_publisher(
            Float32MultiArrayStamped, "~/debug/pwm_values", 10
        )

        self.debug_pid_publisher = self.create_publisher(
            Float32MultiArrayStamped, "~/debug/pid_values", 10
        )

    def velocity_callback(self, msg):
        """
        Callback for velocity report messages.
        Updates the current speed.

        Args:
            msg: VelocityReport message containing current vehicle speed
        """
        # Store the last speed before updating
        self.state.last_speed = (
            self.state.current_speed if self.state.current_speed is not None else 0.0
        )

        # Update current speed
        speed = msg.longitudinal_velocity
        self.state.current_speed = speed

        # Update timestamp
        current_time = time.time()
        self.state.last_update_time = current_time

    def control_callback(self, msg):
        """
        Callback for control command messages.
        Updates the target speed and steering angle state.

        Args:
            msg: Control message containing target velocity and steering angle
        """
        self.state.target_speed = msg.longitudinal.velocity
        self.state.target_tire_angle = msg.lateral.steering_tire_angle

        # Clamp the target steering angle to the vehicle's maximum steering capability
        max_angle = self.config.max_steering_angle
        self.state.target_tire_angle = max(
            -max_angle, min(self.state.target_tire_angle, max_angle)
        )

    def timer_callback(self):
        """
        Timer callback that runs at the configured rate.
        Implements the Ackermann control loop similar to CARLA.
        """
        # Calculate delta time
        current_time = time.time()
        delta_time = current_time - self.state.last_update_time

        # Apply velocity filtering (exponential moving average)
        if self.state.current_speed is not None:
            raw_measured_velocity = self.state.current_speed
            if self.state.filtered_measured_velocity == 0.0:
                # Initialize on first call
                self.state.filtered_measured_velocity = raw_measured_velocity
            else:
                self.state.filtered_measured_velocity = (
                    self.vel_meas_alpha * raw_measured_velocity +
                    (1.0 - self.vel_meas_alpha) * self.state.filtered_measured_velocity
                )

        if self.state.target_speed is not None:
            raw_target_velocity = self.state.target_speed
            if self.state.filtered_target_velocity == 0.0:
                # Initialize on first call
                self.state.filtered_target_velocity = raw_target_velocity
            else:
                self.state.filtered_target_velocity = (
                    self.vel_cmd_alpha * raw_target_velocity +
                    (1.0 - self.vel_cmd_alpha) * self.state.filtered_target_velocity
                )

        # Lateral Control (Steering)
        steer_value = self.run_steering_control(delta_time)

        # Longitudinal Control (Direct PWM Output)
        pwm_value = self.run_longitudinal_control(delta_time)

        # Set the power of the DC motor
        if self.dry_run:
            self.get_logger().debug(f"DRY-RUN: Would set motor PWM to {pwm_value}")
        else:
            try:
                if self.driver is not None:
                    self.driver.set_pwm(0, 0, pwm_value)
                    self.get_logger().debug(f"Motor PWM set to {pwm_value}")
                else:
                    self.get_logger().error("PWM driver is None - hardware not initialized!")
            except Exception as e:
                self.get_logger().error(f"Failed to set motor PWM: {e}")

        # Set angle of the steering servo
        if self.dry_run:
            self.get_logger().debug(f"DRY-RUN: Would set steering PWM to {steer_value}")
        else:
            try:
                if self.driver is not None:
                    self.driver.set_pwm(1, 0, steer_value)
                else:
                    self.get_logger().error("PWM driver is None - hardware not initialized!")
            except Exception as e:
                self.get_logger().error(f"Failed to set steering PWM: {e}")

        # Publish debug information if enabled
        if (
            self.get_parameter("enable_debug_publishing")
            .get_parameter_value()
            .bool_value
        ):
            self.publish_debug_control_values()
            self.publish_debug_pwm_values(pwm_value, steer_value)
            self.publish_debug_pid_values()

    def run_steering_control(self, delta_time: float) -> int:
        """
        Implements the steering control logic using the Ackermann model.

        Args:
            delta_time: Time since last update in seconds

        Returns:
            int: PWM value for the steering servo
        """
        # If no target tire angle available, return initial steering value (straight)
        if self.state.target_tire_angle is None:
            return self.config.init_steer

        # Instant steering
        target_angle = self.state.target_tire_angle
        steer_angle = target_angle

        # Convert the steering angle to PWM value
        steer = steer_angle * self.config.tire_angle_to_steer_ratio
        steer_pwm = self.config.init_steer + int(steer)

        # Limit steering PWM value to min/max range
        steer_pwm = max(self.config.min_steer, min(self.config.max_steer, steer_pwm))

        return steer_pwm

    def run_longitudinal_control(self, delta_time: float) -> int:
        """
        Implements the longitudinal control logic using direct PWM output.

        Uses a single PID controller that maps speed error directly to PWM offset,
        which is then added to the init_pwm value to get the final PWM output.

        Args:
            delta_time: Time since last update in seconds

        Returns:
            int: PWM value for motor (clamped to min_pwm/max_pwm range)
        """
        # Return init PWM if we don't have necessary state information
        if self.state.target_speed is None or self.state.current_speed is None:
            return self.config.init_pwm

        # Check if we need to apply brake when coming to a stop
        if (abs(self.state.target_speed) < self.full_stop_threshold and
            abs(self.state.current_speed) > self.brake_threshold):
            # Target is near zero but vehicle is moving - apply brake
            self.get_logger().debug(f"Applying brake: target={self.state.target_speed:.2f}, current={self.state.current_speed:.2f}")
            return self.config.brake_pwm

        # Check for full stop condition
        if self.run_control_full_stop():
            return self.config.init_pwm

        # Use filtered velocities for control
        target_velocity = self.state.filtered_target_velocity
        current_velocity = self.state.filtered_measured_velocity

        # Apply velocity deadband - ignore small errors to prevent jitter
        velocity_error = abs(target_velocity - current_velocity)
        if velocity_error < self.velocity_deadband:
            # Within deadband - maintain current PWM to avoid jitter
            return self.state.last_pwm_value

        # Handle reverse control logic
        self.run_control_reverse()

        # Run speed PID controller to get PWM offset using filtered velocities
        self.speed_controller.set_target_point(target_velocity)
        pwm_offset = self.speed_controller.run(current_velocity, delta_time)

        # Store pwm_offset for debug publishing
        self.state.speed_control_pwm_offset = pwm_offset

        # Calculate final PWM value based on direction
        if self.state.in_reverse:
            # Reverse: subtract offset from init_pwm
            pwm_value = int(self.config.init_pwm - pwm_offset)
        else:
            # Forward: add offset to init_pwm
            pwm_value = int(self.config.init_pwm + pwm_offset)

        # Clamp to valid PWM range
        pwm_value = max(self.config.min_pwm, min(self.config.max_pwm, pwm_value))

        # Store for next iteration
        self.state.last_pwm_value = pwm_value

        return pwm_value

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

    def reset_controllers(self) -> None:
        """
        Reset PID controller and motor state to their initial state.
        """
        self.speed_controller.reset()

        # Reset state variables
        self.state.speed_control_pwm_offset = 0.0
        self.state.in_reverse = False

        # Reset motor state machine
        self.state.motor_state = MotorState.STOPPED
        self.state.last_pwm_value = self.config.init_pwm
        self.state.transition_target = None

    def publish_debug_control_values(self):
        """
        Publish debug information about control values.

        Publishes PWM offset and direction information.
        """
        import math

        msg = Float32MultiArrayStamped()

        # Add timestamp
        msg.stamp = self.get_clock().now().to_msg()

        # Create descriptive layout
        msg.layout.dim.append(MultiArrayDimension())
        msg.layout.dim[0].label = "control_values"
        msg.layout.dim[0].size = 2
        msg.layout.dim[0].stride = 2

        # Sanitize values to prevent NaN/Inf
        pwm_offset = self.state.speed_control_pwm_offset
        if not math.isfinite(pwm_offset):
            pwm_offset = 0.0

        # Add data: [pwm_offset, in_reverse]
        msg.data = [
            float(pwm_offset),
            1.0 if self.state.in_reverse else 0.0
        ]

        # Publish message
        self.debug_control_publisher.publish(msg)

    def publish_debug_pwm_values(self, motor_pwm: int, steer_pwm: int):
        """
        Publish debug information about PWM values.

        Args:
            motor_pwm: Current motor PWM value
            steer_pwm: Current steering servo PWM value
        """
        msg = Float32MultiArrayStamped()

        # Add timestamp
        msg.stamp = self.get_clock().now().to_msg()

        # Create descriptive layout
        msg.layout.dim.append(MultiArrayDimension())
        msg.layout.dim[0].label = "pwm_values"
        msg.layout.dim[0].size = 2
        msg.layout.dim[0].stride = 2

        # Add data: [motor_pwm, steer_pwm]
        msg.data = [float(motor_pwm), float(steer_pwm)]

        # Publish message
        self.debug_pwm_publisher.publish(msg)

    def publish_debug_pid_values(self):
        """
        Publish debug information about PID controller values.
        
        Publishes speed PID values and PWM offset.
        """
        msg = Float32MultiArrayStamped()

        # Add timestamp
        msg.stamp = self.get_clock().now().to_msg()

        # Create descriptive layout
        msg.layout.dim.append(MultiArrayDimension())
        msg.layout.dim[0].label = "pid_values"
        msg.layout.dim[0].size = 8
        msg.layout.dim[0].stride = 8

        # Add data with the following format:
        # [target_speed, current_speed, target_tire_angle, current_tire_angle,
        #  speed_p, speed_i, speed_d, pwm_offset]
        target_speed = (
            self.state.target_speed if self.state.target_speed is not None else 0.0
        )
        current_speed = (
            self.state.current_speed if self.state.current_speed is not None else 0.0
        )
        target_tire_angle = (
            self.state.target_tire_angle
            if self.state.target_tire_angle is not None
            else 0.0
        )
        current_tire_angle = (
            self.state.current_tire_angle
            if self.state.current_tire_angle is not None
            else 0.0
        )

        # Sanitize PID values to prevent NaN/Inf
        import math
        def sanitize(value):
            return float(value) if math.isfinite(value) else 0.0

        msg.data = [
            sanitize(target_speed),
            sanitize(current_speed),
            sanitize(target_tire_angle),
            sanitize(current_tire_angle),
            sanitize(self.speed_controller.proportional),
            sanitize(self.speed_controller.integral),
            sanitize(self.speed_controller.derivative),
            sanitize(self.state.speed_control_pwm_offset),
        ]

        # Publish message
        self.debug_pid_publisher.publish(msg)


@dataclass
class State:
    """
    Dataclass to hold the current state of the controller.

    Attributes:
        target_speed: Target longitudinal velocity from control commands
        current_speed: Current measured vehicle speed
        target_tire_angle: Target steering tire angle from control commands
        current_tire_angle: Current measured tire angle (if available)

        speed_control_pwm_offset: PWM offset from speed PID controller
        last_speed: Previous measured speed (for velocity estimation)
        in_reverse: Whether vehicle is in reverse gear
        last_update_time: Timestamp of last update

        # Motor state machine variables
        motor_state: Current motor state (STOPPED, FORWARD, etc.)
        last_pwm_value: Previous PWM value sent to motor
        transition_target: Target state during transitions
        brake_threshold: PWM threshold below init_pwm for brake locking
    """

    target_speed: Optional[float]
    current_speed: Optional[float]

    target_tire_angle: Optional[float]
    current_tire_angle: Optional[float]

    # Control state variables
    speed_control_pwm_offset: float
    last_speed: float
    in_reverse: bool
    last_update_time: float

    # Motor state machine variables
    motor_state: MotorState
    last_pwm_value: int
    transition_target: Optional[MotorState]
    brake_threshold: int

    # Filtered velocities for PID control (with defaults)
    filtered_measured_velocity: float = 0.0
    filtered_target_velocity: float = 0.0


@dataclass
class Config:
    """
    Dataclass to hold the configuration parameters.

    Attributes:
        min_pwm: Minimum PWM value for backward movement
        init_pwm: Initial PWM value (stopped)
        max_pwm: Maximum PWM value for forward movement
        brake_pwm: PWM value for active braking
        min_steer: Minimum steering PWM value (left turn)
        init_steer: Initial steering PWM value (straight)
        max_steer: Maximum steering PWM value (right turn)
        tire_angle_to_steer_ratio: Conversion ratio from tire angle to steering PWM
        steering_speed: Maximum steering speed [rad/s]
        max_steering_angle: Maximum steering angle [rad]
    """

    min_pwm: int
    init_pwm: int
    max_pwm: int
    brake_pwm: int

    min_steer: int
    init_steer: int
    max_steer: int

    tire_angle_to_steer_ratio: float

    # Ackermann-specific parameters
    steering_speed: float
    max_steering_angle: float


def main():
    """
    Main function to initialize and run the node.
    """
    rclpy.init(args=sys.argv)
    node = AutoSdvActuator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Gracefully shutdown on keyboard interrupt
        pass
    finally:
        # Destroy the node explicitly
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
