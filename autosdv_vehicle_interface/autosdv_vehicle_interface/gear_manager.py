#!/usr/bin/env python3
"""
Synthetic Gear Manager for AutoSDV.
Maps motor PWM states to Autoware-compatible gear positions and handles gear commands.
"""
import sys
import time
from enum import Enum
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy import Parameter

from autoware_vehicle_msgs.msg import GearReport, GearCommand, VelocityReport
from autoware_internal_debug_msgs.msg import Float32MultiArrayStamped


class MotorState(Enum):
    """Mirror of motor states from actuator for consistency."""
    STOPPED = "stopped"
    FORWARD = "forward"
    BACKWARD = "backward"
    BRAKE_LOCKED = "brake_locked"
    TRANSITIONING = "transitioning"


class GearManager(Node):
    """
    Synthetic gear management node for RC car platform.
    
    Maps continuous PWM control to discrete gear states for Autoware compatibility.
    Subscribes to gear commands and translates them to appropriate motor behaviors.
    """

    def __init__(self):
        super().__init__('gear_manager_node')

        # Declare gear_manager-specific parameters
        self.declare_parameter('publish_rate', 30.0)
        self.declare_parameter('velocity_threshold', 0.1)  # m/s to consider stopped
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('enable_gear_commands', True)

        # Motor PWM parameters from actuator.yaml (single source of truth)
        # No defaults - these MUST be loaded from actuator.yaml via launch file
        self.declare_parameter('init_pwm')   # PWM value when motor is stopped
        self.declare_parameter('min_pwm')    # PWM value for max reverse
        self.declare_parameter('max_pwm')    # PWM value for max forward
        self.declare_parameter('brake_pwm')  # PWM value for braking

        # Get parameters
        self.publish_rate = self.get_parameter('publish_rate').value
        self.velocity_threshold = self.get_parameter('velocity_threshold').value
        self.frame_id = self.get_parameter('frame_id').value
        self.enable_commands = self.get_parameter('enable_gear_commands').value

        # Motor parameters from actuator.yaml
        self.init_pwm = self.get_parameter('init_pwm').value
        self.min_pwm = self.get_parameter('min_pwm').value
        self.max_pwm = self.get_parameter('max_pwm').value
        self.brake_pwm = self.get_parameter('brake_pwm').value

        self.get_logger().info(
            f'Motor params from actuator.yaml: init_pwm={self.init_pwm}, '
            f'min_pwm={self.min_pwm}, max_pwm={self.max_pwm}, brake_pwm={self.brake_pwm}'
        )

        # State tracking
        self.current_gear = GearReport.PARK
        self.requested_gear = GearReport.PARK
        self.current_velocity = 0.0
        self.motor_state = MotorState.STOPPED
        self.last_pwm_value = self.init_pwm  # Use configured init_pwm
        
        # Publishers
        self.gear_status_pub = self.create_publisher(
            GearReport,
            '/vehicle/status/gear_status',
            1
        )
        
        # Subscribers
        self.velocity_sub = self.create_subscription(
            VelocityReport,
            '/vehicle/status/velocity_status',
            self.velocity_callback,
            1
        )
        
        # Subscribe to PWM debug info from actuator to get motor state
        self.pwm_debug_sub = self.create_subscription(
            Float32MultiArrayStamped,
            '/autosdv/actuator_node/debug/pwm_values',
            self.pwm_debug_callback,
            1
        )
        
        if self.enable_commands:
            self.gear_cmd_sub = self.create_subscription(
                GearCommand,
                '/control/command/gear_cmd',
                self.gear_command_callback,
                1
            )
        
        # Timer for regular status publishing
        self.timer = self.create_timer(
            1.0 / self.publish_rate,
            self.publish_gear_status
        )
        
        self.get_logger().info(
            f'Gear Manager initialized - Publishing at {self.publish_rate}Hz'
        )
    
    def velocity_callback(self, msg: VelocityReport):
        """Update current velocity from velocity report."""
        self.current_velocity = msg.longitudinal_velocity
    
    def pwm_debug_callback(self, msg: Float32MultiArrayStamped):
        """
        Extract motor PWM value from actuator debug info.
        Expected format: [motor_pwm, steer_pwm]
        """
        if len(msg.data) >= 1:
            self.last_pwm_value = int(msg.data[0])
            self.update_motor_state()
    
    def update_motor_state(self):
        """
        Determine motor state from PWM and velocity.
        This mirrors the logic in actuator.py for consistency.
        Uses PWM thresholds from actuator.yaml.
        """
        # PWM deadzone around init_pwm (neutral zone)
        pwm_deadzone = 5

        # Determine state based on PWM and velocity
        if abs(self.current_velocity) < self.velocity_threshold:
            # Vehicle is stopped or nearly stopped
            if abs(self.last_pwm_value - self.init_pwm) < pwm_deadzone:
                self.motor_state = MotorState.STOPPED
            elif self.last_pwm_value <= self.brake_pwm:
                # At or below brake PWM = braking
                self.motor_state = MotorState.BRAKE_LOCKED
            else:
                self.motor_state = MotorState.STOPPED
        else:
            # Vehicle is moving
            if self.last_pwm_value > self.init_pwm + pwm_deadzone:
                self.motor_state = MotorState.FORWARD
            elif self.last_pwm_value < self.init_pwm - pwm_deadzone:
                # Could be braking or reverse
                if self.current_velocity > self.velocity_threshold:
                    # Moving forward but PWM is reverse = braking
                    self.motor_state = MotorState.BRAKE_LOCKED
                else:
                    # Moving backward
                    self.motor_state = MotorState.BACKWARD
            else:
                self.motor_state = MotorState.TRANSITIONING
    
    def map_motor_state_to_gear(self) -> int:
        """
        Map current motor state and velocity to appropriate gear.
        
        Returns:
            Gear state constant from GearReport
        """
        # Update gear based on motor state
        if self.motor_state == MotorState.STOPPED:
            # Vehicle stopped - check if we should be in PARK or NEUTRAL
            if abs(self.current_velocity) < self.velocity_threshold:
                # Fully stopped - use PARK for safety
                return GearReport.PARK
            else:
                # Still coasting - use NEUTRAL
                return GearReport.NEUTRAL
                
        elif self.motor_state == MotorState.FORWARD:
            # Moving forward or ready to move forward
            return GearReport.DRIVE
            
        elif self.motor_state == MotorState.BACKWARD:
            # Moving backward or ready to move backward
            return GearReport.REVERSE
            
        elif self.motor_state == MotorState.BRAKE_LOCKED:
            # Braking - maintain current gear but could switch to PARK if stopped
            if abs(self.current_velocity) < self.velocity_threshold:
                return GearReport.PARK
            else:
                # Keep current gear during braking
                return self.current_gear
                
        elif self.motor_state == MotorState.TRANSITIONING:
            # In transition - use NEUTRAL
            return GearReport.NEUTRAL
            
        else:
            # Default to PARK for safety
            return GearReport.PARK
    
    def gear_command_callback(self, msg: GearCommand):
        """
        Handle gear command requests from Autoware.
        
        Note: The actual gear change is handled by the actuator based on
        velocity commands. This just tracks the requested gear for monitoring.
        """
        self.requested_gear = msg.command
        
        # Log gear command requests (only at debug level to reduce noise)
        gear_name = self.get_gear_name(msg.command)
        self.get_logger().debug(f'Gear command received: {gear_name}')
        
        # Validate gear command feasibility
        if not self.is_gear_change_safe(msg.command):
            self.get_logger().warn(
                f'Gear change to {gear_name} may not be safe at current speed'
            )
    
    def is_gear_change_safe(self, requested_gear: int) -> bool:
        """
        Check if requested gear change is safe given current vehicle state.
        
        Args:
            requested_gear: Requested gear from GearCommand
            
        Returns:
            True if gear change is safe, False otherwise
        """
        # Can always shift to PARK (emergency stop)
        if requested_gear == GearReport.PARK:
            return True
            
        # Check velocity constraints for gear changes
        if abs(self.current_velocity) > 0.5:  # Moving faster than 0.5 m/s
            # Only allow shifts between forward gears or to PARK
            if self.current_gear == GearReport.DRIVE:
                return requested_gear in [GearReport.DRIVE, GearReport.PARK]
            elif self.current_gear == GearReport.REVERSE:
                return requested_gear in [GearReport.REVERSE, GearReport.PARK]
                
        # At low speed or stopped, all gear changes are safe
        return True
    
    def get_gear_name(self, gear: int) -> str:
        """Convert gear constant to readable name."""
        gear_names = {
            GearReport.NONE: "NONE",
            GearReport.NEUTRAL: "NEUTRAL",
            GearReport.DRIVE: "DRIVE",
            GearReport.REVERSE: "REVERSE",
            GearReport.PARK: "PARK",
            GearReport.LOW: "LOW"
        }
        return gear_names.get(gear, f"UNKNOWN({gear})")
    
    def publish_gear_status(self):
        """
        Publish current gear status at regular intervals.
        """
        # If vehicle is stopped and gear command is received, honor the command
        # Otherwise, infer gear from motor state
        if abs(self.current_velocity) < self.velocity_threshold and self.motor_state == MotorState.STOPPED:
            # Vehicle is stopped - honor gear command if safe
            if self.requested_gear != GearReport.NONE and self.is_gear_change_safe(self.requested_gear):
                new_gear = self.requested_gear
            else:
                # Fall back to inferring from motor state
                new_gear = self.map_motor_state_to_gear()
        else:
            # Vehicle is moving - infer gear from motor state
            new_gear = self.map_motor_state_to_gear()
        
        # Log gear changes
        if new_gear != self.current_gear:
            old_name = self.get_gear_name(self.current_gear)
            new_name = self.get_gear_name(new_gear)
            self.get_logger().info(f'Gear change: {old_name} -> {new_name}')
            
        self.current_gear = new_gear
        
        # Create and publish gear report
        msg = GearReport()
        msg.stamp = self.get_clock().now().to_msg()
        msg.report = self.current_gear
        
        self.gear_status_pub.publish(msg)


def main(args=None):
    """Main function to run the gear manager node."""
    rclpy.init(args=args)
    
    try:
        node = GearManager()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error in gear manager: {e}')
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()