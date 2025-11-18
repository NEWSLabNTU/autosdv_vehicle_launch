#!/usr/bin/env python3
"""
Velocity report node for AutoSDV.
This module measures and reports the vehicle's velocity using wheel rotation sensors.
"""
import math
import sys

import Jetson.GPIO as GPIO

import rclpy
from rclpy import Parameter
from rclpy.node import Node
from autoware_vehicle_msgs.msg import VelocityReport


class AutoSdvVelocityReportNode(Node):
    """
    Node that measures and reports vehicle velocity.

    Uses GPIO input from wheel rotation sensors to calculate speed.
    Publishes this information as VelocityReport messages which are used
    by the control system for feedback.
    """
    def __init__(self) -> None:
        super().__init__("autosdv_velocity_report_node")

        # Configure the node parameters
        self.declare_parameter("pin", Parameter.Type.INTEGER)
        self.declare_parameter("rate", Parameter.Type.DOUBLE)
        self.declare_parameter("wheel_diameter", Parameter.Type.DOUBLE)
        self.declare_parameter("markers_per_rotation", Parameter.Type.INTEGER)
        self.declare_parameter("frame_id", Parameter.Type.STRING)

        # Create the publisher for velocity reports
        publisher = self.create_publisher(VelocityReport, "~/output/velocity_status", 1)

        # Setup the GPIO pin for wheel rotation detection
        pin = self.get_parameter("pin").get_parameter_value().integer_value
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(pin, GPIO.IN)
        GPIO.add_event_detect(pin, GPIO.RISING, callback=self.on_gpio_rising)

        # Start a periodic call for publishing velocity
        rate = self.get_parameter("rate").get_parameter_value().double_value
        timer = self.create_timer(1.0 / rate, self.publish_callback)

        # Compute wheel parameters for speed calculation
        wheel_diameter_cm = (
            self.get_parameter("wheel_diameter").get_parameter_value().double_value
        )
        # Convert wheel diameter from cm to meters and calculate circumference
        wheel_circumference_meters = wheel_diameter_cm / 100.0 * math.pi

        # Get reference to ROS clock
        clock = self.get_clock()

        # Store all required variables as instance attributes
        self.wheel_circumference_meters = wheel_circumference_meters
        self.markers_per_rotation = (
            self.get_parameter("markers_per_rotation")
            .get_parameter_value()
            .integer_value
        )
        self.frame_id = (
            self.get_parameter("frame_id").get_parameter_value().string_value
        )
        self.pin = pin
        self.publisher = publisher
        self.timer = timer

        # Velocity measurement using edge-to-edge timing (smoother than counting)
        self.distance_per_marker = wheel_circumference_meters / self.markers_per_rotation
        self.last_edge_time = None  # Time of last marker edge
        self.current_velocity = 0.0  # Current velocity estimate (m/s)
        self.velocity_alpha = 0.3  # EMA filter alpha for velocity smoothing
        self.timeout_threshold = 0.5  # If no markers for 0.5s, assume stopped

        self.clock = clock

    def publish_callback(self) -> None:
        """
        Timer callback that publishes the current vehicle velocity estimate.

        Publishes the velocity calculated from edge-to-edge timing.
        Resets velocity to zero if no markers detected recently (timeout).
        """
        curr_time = self.clock.now()

        # Check for timeout (vehicle stopped if no markers detected recently)
        if self.last_edge_time is not None:
            time_since_last_edge = (curr_time - self.last_edge_time).nanoseconds / (10**9)
            if time_since_last_edge > self.timeout_threshold:
                # No markers for timeout period - vehicle is stopped
                self.current_velocity = 0.0

        # Create and publish the velocity report message
        msg = VelocityReport()
        msg.header.stamp = curr_time.to_msg()
        msg.header.frame_id = self.frame_id
        msg.longitudinal_velocity = self.current_velocity  # Forward/backward velocity
        msg.lateral_velocity = 0.0  # Side-to-side velocity (always 0 for this vehicle)
        msg.heading_rate = 0.0  # Rate of heading change (not measured here)
        self.publisher.publish(msg)

    def on_gpio_rising(self, channel) -> None:
        """
        GPIO event callback that is triggered when a wheel marker is detected.

        Uses edge-to-edge timing to calculate instantaneous velocity.
        This provides much smoother velocity estimates than counting markers.

        Args:
            channel: GPIO channel that triggered the event
        """
        current_time = self.clock.now()

        if self.last_edge_time is not None:
            # Calculate time since last marker
            dt = (current_time - self.last_edge_time).nanoseconds / (10**9)

            # Prevent division by zero for very fast edges
            if dt > 0.001:  # Minimum 1ms between edges (max ~27 m/s for our wheel)
                # Calculate instantaneous velocity from this edge
                instantaneous_velocity = self.distance_per_marker / dt

                # Apply exponential moving average filter for smoothing
                if self.current_velocity == 0.0:
                    # Initialize on first measurement
                    self.current_velocity = instantaneous_velocity
                else:
                    # EMA: new = alpha * measured + (1-alpha) * previous
                    self.current_velocity = (
                        self.velocity_alpha * instantaneous_velocity +
                        (1.0 - self.velocity_alpha) * self.current_velocity
                    )

        # Update last edge time
        self.last_edge_time = current_time


def main():
    """
    Main function to initialize and run the node.
    """
    rclpy.init(args=sys.argv)
    node = AutoSdvVelocityReportNode()
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
