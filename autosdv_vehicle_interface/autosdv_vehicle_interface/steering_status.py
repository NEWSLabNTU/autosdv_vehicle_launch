#!/usr/bin/env python3
"""
Steering Status Publisher for AutoSDV.
Reports the commanded steering angle as the current steering status.

Since the vehicle lacks a steering angle sensor, this node reports
the commanded steering angle directly. The actuator applies the commanded
angle to the servo, so reporting the command is the best available estimate.
"""
import sys
import math

import rclpy
from rclpy.node import Node
from rclpy import Parameter

from autoware_vehicle_msgs.msg import SteeringReport
from autoware_control_msgs.msg import Control


class SteeringStatus(Node):
    """
    Steering status node for RC car platform.

    Since the vehicle lacks a steering angle sensor, this node reports
    the commanded steering angle directly as the current steering status.
    This is simpler and more reliable than trying to estimate from PWM values.
    """

    def __init__(self):
        super().__init__('steering_status_node')

        # Declare parameters
        self.declare_parameter('publish_rate', 30.0)
        self.declare_parameter('steering_response_time', 0.1)  # seconds for smoothing

        # From actuator.yaml (single source of truth) - no default
        self.declare_parameter('max_steering_angle')  # rad

        # Get parameters
        self.publish_rate = self.get_parameter('publish_rate').value
        self.max_steering_angle = self.get_parameter('max_steering_angle').value
        self.response_time = self.get_parameter('steering_response_time').value

        # State tracking
        self.commanded_steer_angle = 0.0  # Latest commanded angle
        self.reported_steer_angle = 0.0   # Smoothed angle for reporting
        self.last_update_time = self.get_clock().now()

        # Smoothing filter coefficient (based on response time and publish rate)
        # alpha = 1 means no smoothing, alpha close to 0 means heavy smoothing
        self.angle_filter_alpha = min(1.0, 1.0 / (self.response_time * self.publish_rate))

        # Publishers
        self.steering_status_pub = self.create_publisher(
            SteeringReport,
            '/vehicle/status/steering_status',
            1
        )

        # Subscribers - only need control commands now
        self.control_cmd_sub = self.create_subscription(
            Control,
            '/control/command/control_cmd',
            self.control_command_callback,
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
        self.get_logger().info(
            f'max_steering_angle={self.max_steering_angle:.3f} rad '
            f'({math.degrees(self.max_steering_angle):.1f}°) from actuator.yaml'
        )
        self.get_logger().info(
            f'Reporting commanded steering angle directly (no sensor feedback)'
        )

    def control_command_callback(self, msg: Control):
        """
        Track commanded steering angle from control commands.
        Clamps to maximum steering angle for safety.
        """
        raw_angle = msg.lateral.steering_tire_angle
        # Clamp to maximum steering angle
        self.commanded_steer_angle = max(-self.max_steering_angle,
                                         min(self.max_steering_angle, raw_angle))

    def publish_steering_status(self):
        """
        Publish steering status report.

        Reports the commanded steering angle with optional smoothing to simulate
        servo response time. This provides Autoware with the expected steering state.
        """
        # Apply low-pass filter to simulate steering response time
        self.reported_steer_angle = (
            self.angle_filter_alpha * self.commanded_steer_angle +
            (1.0 - self.angle_filter_alpha) * self.reported_steer_angle
        )

        # Create and publish steering report message
        msg = SteeringReport()
        msg.stamp = self.get_clock().now().to_msg()
        msg.steering_tire_angle = self.reported_steer_angle
        self.steering_status_pub.publish(msg)

        # Periodic debug logging (every 2 seconds)
        current_time = self.get_clock().now()
        if (current_time.nanoseconds - self.last_update_time.nanoseconds) > 2e9:
            self.get_logger().debug(
                f'Steering - Commanded: {math.degrees(self.commanded_steer_angle):.1f}°, '
                f'Reported: {math.degrees(self.reported_steer_angle):.1f}°'
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