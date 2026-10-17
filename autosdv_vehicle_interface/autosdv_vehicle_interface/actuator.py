#!/usr/bin/env python3
"""
Actuator control node for AutoSDV using Ackermann control.

This module implements a cascaded PID controller for vehicle's motor and steering using Ackermann steering model.
"""
import atexit
import sys
import math
import time
from typing import Optional

import yaml
try:
    from Adafruit_PCA9685 import PCA9685
except ImportError:  # a laptop: only hardware:=sim or dry_run can run
    PCA9685 = None

import rclpy
from rclpy.node import Node
from rclpy import Parameter
from rclpy.executors import ExternalShutdownException
from rclpy.qos import DurabilityPolicy, QoSProfile
from autoware_adapi_v1_msgs.msg import MrmState
from autoware_control_msgs.msg import Control
from autoware_vehicle_msgs.msg import VelocityReport
from std_msgs.msg import MultiArrayDimension, String
from autoware_internal_debug_msgs.msg import Float32MultiArrayStamped
from tier4_vehicle_msgs.msg import VehicleEmergencyStamped

from autosdv_vehicle_interface.actuator_core import (  # noqa: F401  (State, Config re-exported)
    ActuatorCore, build_controller_params, Config, DEFAULTS, State)
from autosdv_vehicle_interface.hal import check_hardware, HARDWARE_SIM, make_pwm_driver
from autosdv_vehicle_interface.safety import (
    classify_stamp_age,
    mrm_requires_stop,
    StampVerdict,
    write_neutral,
)
# AckermannPID lived in this module before the plug-in split; re-exported so
# existing imports keep working.
from autosdv_vehicle_interface.speed_controller import (  # noqa: F401
    AckermannPID,
    DEFAULT_SPEED_CONTROLLER,
    load_speed_controller,
)


class AutoSdvActuator(Node):
    """
    Node that controls the vehicle's motor and steering servo using an Ackermann control model.

    Subscribes to:
    - Control commands from Autoware (~/input/control_cmd)
    - Velocity reports for current speed (~/input/velocity_status)
    - The vehicle_cmd_gate's emergency verdict (~/input/emergency)
    - The MRM state (~/input/mrm_state)

    The motor command comes from LongitudinalShell (longitudinal.py), whose
    safety modes always win over the plug-in speed controller selected by
    ``speed_controller_class`` (speed_controller.py). Both outputs go to neutral
    when control commands stop arriving and when the node exits.
    """

    def __init__(self, **kwargs):
        # kwargs go to rclpy's Node (parameter_overrides, context, ...)
        super().__init__("autosdv_actuator_node", **kwargs)

        # Declare all parameters
        self.declare_all_parameters()

        # Get parameter values
        parameter_values = self.get_all_parameter_values()

        # Under use_sim_time every time this node uses comes from the ROS clock
        # (the simulator's /clock); otherwise wall time, exactly as before.
        self._use_ros_clock = bool(self.get_parameter("use_sim_time").value)

        # Speed controller and the ROS-free core (config, state, watchdogs)
        self.initialize_pid_controllers(parameter_values)

        # Initialize safety settings (emergency inputs)
        self.initialize_safety(parameter_values)

        # Initialize hardware
        self.driver = self.initialize_pwm_driver(parameter_values)
        self._neutral_written = False
        atexit.register(self.write_neutral_outputs)

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

        # PWM backend: real (PCA9685 over I2C) or sim (the rc_sim plant)
        self.declare_parameter("hardware", "real")

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
        # Vehicle geometry comes from vehicle_info.param.yaml, under Autoware's
        # names, so the planner and this node cannot disagree about the car.
        self.declare_parameter("max_steer_angle", Parameter.Type.DOUBLE)

        self.declare_parameter("tire_angle_to_steer_ratio", Parameter.Type.DOUBLE)

        # Ackermann geometry parameters, also from vehicle_info.param.yaml
        self.declare_parameter("wheel_base", Parameter.Type.DOUBLE)  # L in diagram
        self.declare_parameter("wheel_tread", Parameter.Type.DOUBLE)  # Distance between left/right wheels

        # Output filtering parameters
        self.declare_parameter("derivative_filter_alpha", Parameter.Type.DOUBLE)
        self.declare_parameter("pwm_output_alpha", Parameter.Type.DOUBLE)

        # Speed controller plug-in (see speed_controller.py)
        self.declare_parameter("speed_controller_class", DEFAULTS["speed_controller_class"])
        self.declare_parameter("speed_controller_params", DEFAULTS["speed_controller_params"])
        self.declare_parameter("speed_controller_dt_source", DEFAULTS["speed_controller_dt_source"])
        self.declare_parameter("controller_fault_holdoff", DEFAULTS["controller_fault_holdoff"])
        self.declare_parameter("controller_max_faults", DEFAULTS["controller_max_faults"])

        # Safety: watchdogs, emergency inputs, override behaviour
        self.declare_parameter("control_cmd_timeout", DEFAULTS["control_cmd_timeout"])
        self.declare_parameter("control_cmd_max_stamp_age", DEFAULTS["control_cmd_max_stamp_age"])
        self.declare_parameter("velocity_timeout", DEFAULTS["velocity_timeout"])
        self.declare_parameter("override_brake_duration", DEFAULTS["override_brake_duration"])
        self.declare_parameter("enable_emergency_input", DEFAULTS["enable_emergency_input"])
        self.declare_parameter("emergency_stop_on_any_mrm", DEFAULTS["emergency_stop_on_any_mrm"])
        self.declare_parameter("neutral_write_timeout", DEFAULTS["neutral_write_timeout"])

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

        params["hardware"] = check_hardware(self.get_parameter("hardware").value)

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
            self.get_parameter("max_steer_angle").get_parameter_value().double_value
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

        # Vehicle geometry parameters
        params["wheelbase"] = (
            self.get_parameter("wheel_base").get_parameter_value().double_value
        )
        params["track_width"] = (
            self.get_parameter("wheel_tread").get_parameter_value().double_value
        )

        # Output filtering parameters
        params["derivative_filter_alpha"] = (
            self.get_parameter("derivative_filter_alpha").get_parameter_value().double_value
        )
        params["pwm_output_alpha"] = (
            self.get_parameter("pwm_output_alpha").get_parameter_value().double_value
        )

        # Speed controller plug-in, safety (declared with defaults)
        for name in (
            "speed_controller_class",
            "speed_controller_params",
            "speed_controller_dt_source",
            "controller_fault_holdoff",
            "controller_max_faults",
            "control_cmd_timeout",
            "control_cmd_max_stamp_age",
            "velocity_timeout",
            "override_brake_duration",
            "enable_emergency_input",
            "emergency_stop_on_any_mrm",
            "neutral_write_timeout",
        ):
            params[name] = self.get_parameter(name).value

        return params

    def initialize_pid_controllers(self, params):
        """
        Load the speed controller plug-in and build the longitudinal shell.

        Raises when the plug-in cannot be loaded: the node refuses to start
        rather than drive with a controller nobody selected.

        Args:
            params: Dictionary with parameter values
        """
        controller_params = build_controller_params(
            params, self.parse_speed_controller_params(params["speed_controller_params"]))
        controller_params.update(self.nested_speed_controller_params())

        controller_class = params["speed_controller_class"] or DEFAULT_SPEED_CONTROLLER
        try:
            self.speed_controller = load_speed_controller(controller_class, controller_params)
        except Exception as e:
            self.get_logger().fatal(f"Cannot load speed controller '{controller_class}': {e}")
            raise
        self.get_logger().info(f"Speed controller: {controller_class}")

        self.core = ActuatorCore(params, self.speed_controller,
                                 logger=self.get_logger(), start_time=self._now())
        # The node's long-standing attribute names, kept for its callers and tests
        self.config = self.core.config
        self.state = self.core.state
        self.longitudinal = self.core.longitudinal
        self.cmd_watchdog = self.core.cmd_watchdog
        self.velocity_watchdog = self.core.velocity_watchdog

        # Store filtering and deadband parameters as instance variables
        self.velocity_deadband = params["velocity_deadband"]
        self.full_stop_threshold = params["full_stop_threshold"]
        self.brake_threshold = params["brake_threshold"]
        self.vel_meas_alpha = params["velocity_measurement_filter_alpha"]
        self.vel_cmd_alpha = params["velocity_command_filter_alpha"]

        # PWM output filtering for smoother control (from actuator.yaml)
        self.pwm_output_alpha = params["pwm_output_alpha"]

    @property
    def gate_emergency(self) -> bool:
        return self.core.gate_emergency

    @gate_emergency.setter
    def gate_emergency(self, value: bool) -> None:
        self.core.gate_emergency = value

    @property
    def mrm_emergency(self) -> bool:
        return self.core.mrm_emergency

    @mrm_emergency.setter
    def mrm_emergency(self, value: bool) -> None:
        self.core.mrm_emergency = value

    def _now(self) -> float:
        """Seconds for dt and the shell: the ROS clock under use_sim_time, else wall time."""
        if self._use_ros_clock:
            return self.get_clock().now().nanoseconds * 1e-9
        return time.time()

    def _monotonic(self) -> float:
        """Seconds for the watchdogs: the ROS clock under use_sim_time, else monotonic."""
        if self._use_ros_clock:
            return self.get_clock().now().nanoseconds * 1e-9
        return time.monotonic()

    def nested_speed_controller_params(self):
        """
        Collect ``speed_controller_params.<key>`` overrides.

        A launch file that is handed ``{kp: 15.0}`` for the string parameter
        ``speed_controller_params`` can see it parsed as YAML before it reaches
        the node (ros2 launch and play_launch both do), and the node then gets
        a nested mapping, which ROS flattens into ``speed_controller_params.kp``
        overrides. The string parameter stays empty and the keys would be lost
        without this, so both spellings are accepted; these win on a collision.
        """
        prefix = "speed_controller_params."
        found = {}
        for name, param in getattr(self, "_parameter_overrides", {}).items():
            if name.startswith(prefix) and len(name) > len(prefix):
                found[name[len(prefix):]] = param.value
        return found

    @staticmethod
    def parse_speed_controller_params(text):
        """
        Parse ``speed_controller_params``: a YAML (or JSON) mapping in a string.

        ROS parameters cannot hold a dictionary, so the plug-in's own
        parameters travel as text, e.g. ``"{kp: 20.0, ki: 5.0}"``.
        """
        if not text or not text.strip():
            return {}
        value = yaml.safe_load(text)
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError(
                f"speed_controller_params must be a mapping such as '{{kp: 1.0}}', got {text!r}"
            )
        return value

    def initialize_safety(self, params):
        """
        Set up the stamp check and the emergency inputs (the watchdogs live in the core).

        Args:
            params: Dictionary with parameter values
        """
        self.cmd_max_stamp_age = params["control_cmd_max_stamp_age"]
        self.enable_emergency_input = params["enable_emergency_input"]
        self.emergency_stop_on_any_mrm = params["emergency_stop_on_any_mrm"]
        self.neutral_write_timeout = params["neutral_write_timeout"]

        # Once-per-episode logging of rejected commands
        self._last_cmd_rejection = None

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
            driver = make_pwm_driver(params, node=self, pca_class=PCA9685)
            if params["hardware"] == HARDWARE_SIM:
                self.get_logger().info("Simulated PCA9685: PWM goes to the rc_sim plant")
            else:
                self.get_logger().info(
                    f"PCA9685 initialized successfully on I2C bus {params['i2c_busnum']}, "
                    f"address 0x{params['i2c_address']:02x}")
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

        # Emergency inputs: the gate's verdict and the MRM state
        if self.enable_emergency_input:
            self.emergency_subscription = self.create_subscription(
                VehicleEmergencyStamped,
                "~/input/emergency",
                self.emergency_callback,
                1,
            )
            self.mrm_state_subscription = self.create_subscription(
                MrmState,
                "~/input/mrm_state",
                self.mrm_state_callback,
                1,
            )

        # Motor mode, published on change and latched for late subscribers
        self.mode_publisher = self.create_publisher(
            String,
            "~/debug/mode",
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self._published_mode = None

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
        Handle a velocity report message.

        Updates the current speed.

        Args:
            msg: VelocityReport message containing current vehicle speed
        """
        self.core.on_velocity(self._now(), self._monotonic(), msg.longitudinal_velocity)

    def control_callback(self, msg):
        """
        Handle a control command message.

        Updates the target speed and steering angle state.

        Only a usable command refreshes the command watchdog: one with a
        non-finite field, or a header stamp too far from this node's clock, is
        dropped, so a publisher that keeps sending stale commands still trips
        the watchdog.

        Args:
            msg: Control message containing target velocity and steering angle
        """
        rejection = self.check_control_cmd(msg)
        if rejection is not None:
            if rejection != self._last_cmd_rejection:
                self.get_logger().error(f"Ignoring control commands: {rejection}")
            self._last_cmd_rejection = rejection
            return
        if self._last_cmd_rejection is not None:
            self.get_logger().info("Control commands are usable again")
            self._last_cmd_rejection = None

        self.core.on_command(
            self._monotonic(), msg.longitudinal.velocity, msg.lateral.steering_tire_angle)

    def check_control_cmd(self, msg) -> Optional[str]:
        """
        Return why a control command is unusable, or None if it is usable.

        The reason is a stable string so a run of identical rejections logs once.
        """
        if not (math.isfinite(msg.longitudinal.velocity)
                and math.isfinite(msg.lateral.steering_tire_angle)):
            return "non-finite velocity or steering angle"

        stamp_ns = msg.stamp.sec * 10**9 + msg.stamp.nanosec
        age = None
        if stamp_ns != 0:
            age = (self.get_clock().now().nanoseconds - stamp_ns) / 1e9
        verdict = classify_stamp_age(age, self.cmd_max_stamp_age)
        if verdict == StampVerdict.CLOCK_MISMATCH:
            return (
                "header stamp is more than 10 s from this node's clock -- the publisher "
                "and the actuator disagree about use_sim_time (set "
                "control_cmd_max_stamp_age to 0 to skip this check)"
            )
        if verdict == StampVerdict.STALE:
            return f"header stamp older than control_cmd_max_stamp_age ({self.cmd_max_stamp_age} s)"
        if verdict == StampVerdict.FUTURE:
            return f"header stamp ahead of this node's clock by more than {self.cmd_max_stamp_age} s"
        return None

    def emergency_callback(self, msg):
        """Track vehicle_cmd_gate's emergency output (/control/command/emergency_cmd)."""
        emergency = bool(msg.emergency)
        if emergency != self.gate_emergency:
            if emergency:
                self.get_logger().error("Emergency from vehicle_cmd_gate: braking to a stop")
            else:
                self.get_logger().info("vehicle_cmd_gate emergency cleared")
        self.gate_emergency = emergency

    def mrm_state_callback(self, msg):
        """Track the MRM state (/system/fail_safe/mrm_state)."""
        emergency = mrm_requires_stop(msg.state, msg.behavior, self.emergency_stop_on_any_mrm)
        if emergency != self.mrm_emergency:
            if emergency:
                self.get_logger().error(
                    f"MRM active (state {msg.state}, behavior {msg.behavior}): braking to a stop"
                )
            else:
                self.get_logger().info(f"MRM cleared (state {msg.state})")
        self.mrm_emergency = emergency

    def timer_callback(self):
        """
        Run one control tick at the configured rate.

        Implements the Ackermann control loop similar to CARLA.
        """
        pwm_value, steer_value = self.core.tick(self._now(), self._monotonic())
        self.publish_mode()

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

    def reset_controllers(self) -> None:
        """Reset the speed controller to its initial state."""
        self.speed_controller.reset()

    def publish_mode(self) -> None:
        """Publish the longitudinal mode (and override reason) when it changes."""
        mode = self.longitudinal.mode.value
        if self.longitudinal.override_reason is not None:
            mode = f"{mode}:{self.longitudinal.override_reason}"
        if mode != self._published_mode:
            self.mode_publisher.publish(String(data=mode))
            self._published_mode = mode

    def write_neutral_outputs(self) -> bool:
        """
        Stop the control loop and write neutral motor and centred steering PWM.

        Idempotent, and bounded by ``neutral_write_timeout`` even if the I2C
        bus hangs. Called on every exit path: normal shutdown, SIGINT, SIGTERM,
        an exception out of the executor, and atexit as the last resort.

        Returns:
            bool: True if the neutral values reached the PWM driver.
        """
        if self._neutral_written:
            return True
        self._neutral_written = True

        timer = getattr(self, "timer", None)
        if timer is not None:
            timer.cancel()

        outputs = [(0, self.config.init_pwm), (1, self.config.init_steer)]
        if self.dry_run:
            return True
        ok = write_neutral(self.driver, outputs, self.neutral_write_timeout, self.get_logger())
        if ok:
            self.get_logger().info(
                f"Outputs set to neutral (motor {self.config.init_pwm}, "
                f"steering {self.config.init_steer})"
            )
        return ok

    def publish_debug_control_values(self):
        """
        Publish debug information about control values.

        Publishes PWM offset and direction information.
        """
        msg = Float32MultiArrayStamped()

        # Add timestamp
        msg.stamp = self.get_clock().now().to_msg()

        # Create descriptive layout
        msg.layout.dim.append(MultiArrayDimension())
        msg.layout.dim[0].label = "control_values"
        msg.layout.dim[0].size = 2
        msg.layout.dim[0].stride = 2

        # Sanitize values to prevent NaN/Inf
        pwm_offset = self.longitudinal.speed_control_pwm_offset
        if not math.isfinite(pwm_offset):
            pwm_offset = 0.0

        # Add data: [pwm_offset, in_reverse]
        msg.data = [
            float(pwm_offset),
            1.0 if self.longitudinal.in_reverse else 0.0
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
        def sanitize(value):
            try:
                return float(value) if math.isfinite(value) else 0.0
            except (TypeError, ValueError):
                return 0.0

        # P, I, D terms, if the speed controller exposes them
        terms = (0.0, 0.0, 0.0)
        debug_terms = getattr(self.speed_controller, "debug_terms", None)
        if callable(debug_terms):
            try:
                terms = tuple(debug_terms())[:3]
            except Exception:
                pass

        msg.data = [
            sanitize(target_speed),
            sanitize(current_speed),
            sanitize(target_tire_angle),
            sanitize(current_tire_angle),
            sanitize(terms[0]),
            sanitize(terms[1]),
            sanitize(terms[2]),
            sanitize(self.longitudinal.speed_control_pwm_offset),
        ]

        # Publish message
        self.debug_pid_publisher.publish(msg)


def main():
    """
    Initialize and run the node.

    Every way out leaves the motor in neutral and the steering centred:
    SIGINT raises KeyboardInterrupt, SIGTERM makes rclpy shut down (spin raises
    ExternalShutdownException), and any other exception still runs the
    ``finally``. The node also registers the same write with atexit as soon as
    the PWM driver exists. SIGKILL and a power cut are beyond software: the
    PCA9685 then holds its last duty cycle (see the hardware kill switch).
    """
    rclpy.init(args=sys.argv)
    node = None
    try:
        node = AutoSdvActuator()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # Gracefully shutdown on SIGINT / SIGTERM
        pass
    finally:
        if node is not None:
            node.write_neutral_outputs()
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
