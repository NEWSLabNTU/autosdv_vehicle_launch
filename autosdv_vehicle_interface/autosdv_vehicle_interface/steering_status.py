#!/usr/bin/env python3
"""
Steering Status Publisher for AutoSDV.
Estimates steering angle from PWM commands and publishes steering status for Autoware.
"""
import sys
import math

import rclpy
from rclpy.node import Node
from rclpy import Parameter

from autoware_vehicle_msgs.msg import SteeringReport
from autoware_control_msgs.msg import Control
from autoware_internal_debug_msgs.msg import Float32MultiArrayStamped


class SteeringStatus(Node):
    """
    Steering status node for RC car platform.
    
    Since the vehicle lacks a steering angle sensor, this node estimates
    the steering angle from PWM commands sent to the steering servo.
    """

    def __init__(self):
        super().__init__('steering_status_node')
        
        # Declare parameters (matching actuator.yaml values)
        self.declare_parameter('publish_rate', 30.0)
        self.declare_parameter('init_steer', 307)  # PWM value for straight
        self.declare_parameter('min_steer', 204)   # PWM value for max left
        self.declare_parameter('max_steer', 409)   # PWM value for max right
        self.declare_parameter('tire_angle_to_steer_ratio', 130.0)
        self.declare_parameter('max_steering_angle', 0.349)  # rad (20 degrees)
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('steering_response_time', 0.2)  # seconds
        
        # Get parameters
        self.publish_rate = self.get_parameter('publish_rate').value
        self.init_steer = self.get_parameter('init_steer').value
        self.min_steer = self.get_parameter('min_steer').value
        self.max_steer = self.get_parameter('max_steer').value
        self.tire_angle_to_steer_ratio = self.get_parameter('tire_angle_to_steer_ratio').value
        self.max_steering_angle = self.get_parameter('max_steering_angle').value
        self.frame_id = self.get_parameter('frame_id').value
        self.response_time = self.get_parameter('steering_response_time').value
        
        # State tracking
        self.current_steer_pwm = self.init_steer
        self.commanded_steer_angle = 0.0
        self.estimated_steer_angle = 0.0
        self.last_update_time = self.get_clock().now()
        
        # Smoothing filter for angle estimation
        self.angle_filter_alpha = 0.3  # Low-pass filter coefficient
        
        # Publishers
        self.steering_status_pub = self.create_publisher(
            SteeringReport,
            '/vehicle/status/steering_status',
            1
        )
        
        # Subscribers
        # Subscribe to control commands to get commanded steering angle
        self.control_cmd_sub = self.create_subscription(
            Control,
            '/control/command/control_cmd',
            self.control_command_callback,
            1
        )
        
        # Subscribe to PWM debug info from actuator
        self.pwm_debug_sub = self.create_subscription(
            Float32MultiArrayStamped,
            '/autosdv/actuator_node/debug/pwm_values',
            self.pwm_debug_callback,
            1
        )
        
        # Timer for regular status publishing
        self.timer = self.create_timer(
            1.0 / self.publish_rate,
            self.publish_steering_status
        )
        
        self.get_logger().info(
            f'Steering Status Publisher initialized - Publishing at {self.publish_rate}Hz'
        )
    
    def control_command_callback(self, msg: Control):
        """
        Track commanded steering angle from control commands.
        """
        self.commanded_steer_angle = msg.lateral.steering_tire_angle
    
    def pwm_debug_callback(self, msg: Float32MultiArrayStamped):
        """
        Extract steering PWM value from actuator debug info.
        Expected format: [motor_pwm, steer_pwm]
        """
        if len(msg.data) >= 2:
            self.current_steer_pwm = int(msg.data[1])
            # Update estimated angle from PWM
            self.update_estimated_angle()
    
    def pwm_to_angle(self, pwm: int) -> float:
        """
        Convert PWM value to steering angle.
        
        Args:
            pwm: PWM value for steering servo
            
        Returns:
            Estimated steering angle in radians
        """
        # Calculate angle from PWM using the tire_angle_to_steer_ratio
        angle_raw = (pwm - self.init_steer) / self.tire_angle_to_steer_ratio
        
        # Clamp to maximum steering angle
        angle_clamped = max(-self.max_steering_angle, 
                           min(self.max_steering_angle, angle_raw))
        
        return angle_clamped
    
    def update_estimated_angle(self):
        """
        Update the estimated steering angle with filtering.
        """
        # Convert current PWM to angle
        raw_angle = self.pwm_to_angle(self.current_steer_pwm)
        
        # Apply low-pass filter for smoother estimation
        self.estimated_steer_angle = (
            self.angle_filter_alpha * raw_angle +
            (1 - self.angle_filter_alpha) * self.estimated_steer_angle
        )
    
    def publish_steering_status(self):
        """
        Publish steering status report.
        """
        # Create steering report message
        msg = SteeringReport()
        msg.stamp = self.get_clock().now().to_msg()
        
        # Report the estimated actual angle
        msg.steering_tire_angle = self.estimated_steer_angle
        
        # Publish the message
        self.steering_status_pub.publish(msg)
        
        # Periodic debug logging (every 2 seconds)
        current_time = self.get_clock().now()
        if (current_time.nanoseconds - self.last_update_time.nanoseconds) > 2e9:
            self.get_logger().debug(
                f'Steering - PWM: {self.current_steer_pwm}, '
                f'Angle: {math.degrees(self.estimated_steer_angle):.1f}°, '
                f'Commanded: {math.degrees(self.commanded_steer_angle):.1f}°'
            )
            self.last_update_time = current_time


def main(args=None):
    """Main function to run the steering status node."""
    rclpy.init(args=args)
    
    try:
        node = SteeringStatus()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error in steering status: {e}')
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()