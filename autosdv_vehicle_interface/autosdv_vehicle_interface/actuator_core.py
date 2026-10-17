#!/usr/bin/env python3
"""
The actuator's control logic, free of ROS.

actuator.py wraps this class: it turns ROS messages into the calls below and
writes the returned PWM to the PCA9685. The rc_sim lockstep grader drives the
same class directly, so a graded run is the car's code, tick for tick.

Times are seconds on clocks the caller chooses: ``now`` for the PID's dt and
the shell (the node: wall time, or the ROS clock under use_sim_time), and
``monotonic_now`` for the watchdogs.
"""
from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Tuple

from autosdv_vehicle_interface.longitudinal import (
    LongitudinalShell,
    OVERRIDE_COMMAND_TIMEOUT,
    OVERRIDE_EMERGENCY,
    OVERRIDE_VELOCITY_TIMEOUT,
    ShellConfig,
)
from autosdv_vehicle_interface.safety import InputState, InputWatchdog
from autosdv_vehicle_interface.speed_controller import DEFAULT_SPEED_CONTROLLER

# Parameters the node declares with a default; the others come from
# actuator.yaml and vehicle_info.param.yaml and have none.
DEFAULTS: Dict[str, Any] = {
    'speed_controller_class': DEFAULT_SPEED_CONTROLLER,
    'speed_controller_params': '',
    'speed_controller_dt_source': 'velocity_report',
    'controller_fault_holdoff': 1.0,
    'controller_max_faults': 3,
    'control_cmd_timeout': 0.3,
    'control_cmd_max_stamp_age': 0.5,
    'velocity_timeout': 0.5,
    'override_brake_duration': 1.5,
    'enable_emergency_input': True,
    'emergency_stop_on_any_mrm': False,
    'neutral_write_timeout': 0.5,
}

# vehicle_info.param.yaml's names -> the names this code uses
FILE_KEY_RENAMES = {
    'max_steer_angle': 'max_steering_angle',
    'wheel_base': 'wheelbase',
    'wheel_tread': 'track_width',
}

# Actuator parameters every speed controller receives in its params dict
SPEED_CONTROLLER_BASE_PARAMS = (
    'init_pwm', 'min_pwm', 'max_pwm', 'brake_pwm', 'rate',
    'kp_speed', 'ki_speed', 'kd_speed', 'integral_limit',
    'enable_conditional_integration', 'derivative_filter_alpha',
)


def normalize_params(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Turn merged parameter files into the dict the node builds from its parameters."""
    params = dict(DEFAULTS)
    for key, value in raw.items():
        params[FILE_KEY_RENAMES.get(key, key)] = value
    return params


def build_controller_params(params: Dict[str, Any],
                            overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build the speed controller's params: the actuator's base set, overrides winning."""
    controller_params = {name: params[name] for name in SPEED_CONTROLLER_BASE_PARAMS}
    controller_params.update(overrides or {})
    return controller_params


@dataclass
class State:
    """
    Dataclass to hold the current state of the controller.

    Attributes:
        target_speed: Target longitudinal velocity from control commands
        current_speed: Current measured vehicle speed
        target_tire_angle: Target steering tire angle from control commands
        current_tire_angle: Current measured tire angle (if available)

        last_speed: Previous measured speed (for velocity estimation)
        last_update_time: Wall time of the last velocity report

    The motor-side state (PWM held in deadband, reverse intent, PWM offset)
    lives in LongitudinalShell.
    """

    target_speed: Optional[float]
    current_speed: Optional[float]

    target_tire_angle: Optional[float]
    current_tire_angle: Optional[float]

    # Control state variables
    last_speed: float
    last_update_time: float

    # Filtered values for PID control (with defaults - must be at end)
    filtered_measured_velocity: float = 0.0
    filtered_target_velocity: float = 0.0
    filtered_pwm_output: float = 0.0  # Smoothed PWM output


@dataclass
class Config:
    """
    Dataclass to hold the configuration parameters.

    All values are loaded from actuator.yaml (single source of truth).

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
        wheelbase: Distance between front and rear axles [m]
        track_width: Distance between left and right wheels [m]
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

    # Ackermann geometry (from actuator.yaml)
    wheelbase: float
    track_width: float


def make_config(params: Dict[str, Any]) -> Config:
    return Config(
        min_pwm=params['min_pwm'],
        init_pwm=params['init_pwm'],
        max_pwm=params['max_pwm'],
        brake_pwm=params['brake_pwm'],
        min_steer=params['min_steer'],
        init_steer=params['init_steer'],
        max_steer=params['max_steer'],
        tire_angle_to_steer_ratio=params['tire_angle_to_steer_ratio'],
        steering_speed=params['steering_speed'],
        max_steering_angle=params['max_steering_angle'],
        wheelbase=params['wheelbase'],
        track_width=params['track_width'],
    )


def make_shell_config(params: Dict[str, Any]) -> ShellConfig:
    return ShellConfig(
        init_pwm=params['init_pwm'],
        min_pwm=params['min_pwm'],
        max_pwm=params['max_pwm'],
        brake_pwm=params['brake_pwm'],
        full_stop_threshold=params['full_stop_threshold'],
        brake_threshold=params['brake_threshold'],
        velocity_deadband=params['velocity_deadband'],
        override_brake_duration=params['override_brake_duration'],
        controller_fault_holdoff=params['controller_fault_holdoff'],
        controller_max_faults=params['controller_max_faults'],
        dt_source=params['speed_controller_dt_source'],
    )


class ActuatorCore:
    """One actuator tick: watchdogs, overrides, the longitudinal shell, steering."""

    def __init__(self, params: Dict[str, Any], controller, logger=None, start_time: float = 0.0):
        self.config = make_config(params)
        self.speed_controller = controller
        self.longitudinal = LongitudinalShell(make_shell_config(params), controller, logger=logger)
        self.cmd_watchdog = InputWatchdog('control command', params['control_cmd_timeout'], logger)
        self.velocity_watchdog = InputWatchdog(
            'velocity report', params['velocity_timeout'], logger)
        self.vel_meas_alpha = params['velocity_measurement_filter_alpha']
        self.vel_cmd_alpha = params['velocity_command_filter_alpha']
        self.gate_emergency = False
        self.mrm_emergency = False
        self.state = State(
            target_speed=None,
            current_speed=None,
            target_tire_angle=None,
            current_tire_angle=None,
            last_speed=0.0,
            last_update_time=start_time,
        )

    def on_velocity(self, now: float, monotonic_now: float, speed: float) -> None:
        """Take a velocity report."""
        self.state.last_speed = (
            self.state.current_speed if self.state.current_speed is not None else 0.0
        )
        self.state.current_speed = speed
        self.state.last_update_time = now
        self.velocity_watchdog.feed(monotonic_now)

    def on_command(self, monotonic_now: float, velocity: float, tire_angle: float) -> None:
        """Take a usable control command (the node validates stamps and finiteness)."""
        self.cmd_watchdog.feed(monotonic_now)
        self.state.target_speed = velocity
        max_angle = self.config.max_steering_angle
        self.state.target_tire_angle = max(-max_angle, min(tire_angle, max_angle))

    def select_override(self, cmd_state: InputState, velocity_state: InputState) -> Optional[str]:
        """Return the reason the motor must stop regardless of the command, or None."""
        if self.gate_emergency or self.mrm_emergency:
            return OVERRIDE_EMERGENCY
        if cmd_state == InputState.TIMED_OUT:
            return OVERRIDE_COMMAND_TIMEOUT
        if velocity_state == InputState.TIMED_OUT:
            return OVERRIDE_VELOCITY_TIMEOUT
        return None

    def tick(self, now: float, monotonic_now: float) -> Tuple[int, int]:
        """Run one control tick; return (motor PWM, steering PWM)."""
        delta_time = now - self.state.last_update_time

        # Apply velocity filtering (exponential moving average)
        if self.state.current_speed is not None:
            raw = self.state.current_speed
            if self.state.filtered_measured_velocity == 0.0:
                self.state.filtered_measured_velocity = raw
            else:
                self.state.filtered_measured_velocity = (
                    self.vel_meas_alpha * raw
                    + (1.0 - self.vel_meas_alpha) * self.state.filtered_measured_velocity
                )
        if self.state.target_speed is not None:
            raw = self.state.target_speed
            if self.state.filtered_target_velocity == 0.0:
                self.state.filtered_target_velocity = raw
            else:
                self.state.filtered_target_velocity = (
                    self.vel_cmd_alpha * raw
                    + (1.0 - self.vel_cmd_alpha) * self.state.filtered_target_velocity
                )

        # Safety checks: command watchdog, stale velocity, emergency
        cmd_state = self.cmd_watchdog.check(monotonic_now)
        velocity_state = self.velocity_watchdog.check(monotonic_now)
        override = self.select_override(cmd_state, velocity_state)

        # Lateral control: centre the steering when the commands have stopped
        if cmd_state == InputState.TIMED_OUT:
            steer_value = self.config.init_steer
        else:
            steer_value = self.run_steering_control(delta_time)

        # Longitudinal control (see longitudinal.py)
        current_speed = (
            None if velocity_state == InputState.TIMED_OUT else self.state.current_speed
        )
        pwm_value = self.longitudinal.step(
            now, self.state.target_speed, current_speed, self.state.last_update_time, override)
        return pwm_value, steer_value

    def run_steering_control(self, delta_time: float) -> int:
        """
        Implement the steering control logic using the Ackermann model.

        The Ackermann model calculates inner and outer wheel angles:
        - Inner angle: atan(L / (R - track/2))
        - Outer angle: atan(L / (R + track/2))

        Where L = wheelbase, R = turning radius

        For simplicity with single steering servo, we use the average angle.

        Args:
            delta_time: Time since last update in seconds

        Returns:
            int: PWM value for the steering servo
        """
        # If no target tire angle available, return initial steering value (straight)
        if self.state.target_tire_angle is None:
            return self.config.init_steer

        # Clamp the commanded angle to limits
        commanded_angle = max(
            -self.config.max_steering_angle,
            min(self.state.target_tire_angle, self.config.max_steering_angle)
        )

        # Low-pass filter the commanded angle (from diagram)
        if not hasattr(self.state, 'filtered_commanded_angle'):
            self.state.filtered_commanded_angle = commanded_angle
        else:
            # EMA filter with alpha = 0.3 (from diagram)
            alpha = 0.3
            self.state.filtered_commanded_angle = (
                alpha * commanded_angle +
                (1.0 - alpha) * self.state.filtered_commanded_angle
            )

        clamped_angle = self.state.filtered_commanded_angle

        # Apply Ackermann geometry model
        if abs(clamped_angle) < 0.001:  # Near zero - straight ahead
            inner_angle = 0.0
            outer_angle = 0.0
        else:
            # Calculate turning radius from steering angle
            # R = L / tan(delta) where delta is the steering angle
            turning_radius = self.config.wheelbase / math.tan(abs(clamped_angle))

            # Calculate inner and outer wheel angles using Ackermann formula
            # Inner wheel (sharper turn)
            inner_radius = turning_radius - (self.config.track_width / 2.0)
            inner_angle = math.atan(self.config.wheelbase / inner_radius)

            # Outer wheel (gentler turn)
            outer_radius = turning_radius + (self.config.track_width / 2.0)
            outer_angle = math.atan(self.config.wheelbase / outer_radius)

            # Preserve sign from commanded angle
            if clamped_angle < 0:
                inner_angle = -inner_angle
                outer_angle = -outer_angle

        # For a single servo, use the average of inner and outer angles
        # (In a real Ackermann system, you'd have separate servos)
        average_angle = (inner_angle + outer_angle) / 2.0

        # Store for debugging
        self.state.current_tire_angle = average_angle

        # Convert the steering angle to PWM value
        steer = average_angle * self.config.tire_angle_to_steer_ratio
        steer_pwm = self.config.init_steer + int(steer)

        # Clamp PWM to limits
        steer_pwm = max(self.config.min_steer, min(self.config.max_steer, steer_pwm))

        return steer_pwm
