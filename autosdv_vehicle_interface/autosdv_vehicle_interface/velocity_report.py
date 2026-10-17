#!/usr/bin/env python3
"""
Velocity report node for AutoSDV.

Measures the vehicle's speed from a rear wheel's hall-sensor edges. The
arithmetic is VelocityEstimator's (velocity_estimator.py), shared with the
rc_sim lockstep grader; this node supplies the edges and the clock.

``hardware``: ``real`` (Jetson.GPIO, the default) or ``sim`` (SimGPIO: edges
from the autosdv_rc_sim plant, each carrying its true time).
"""
import math
import sys

import rclpy
from rclpy import Parameter
from rclpy.node import Node
from autoware_vehicle_msgs.msg import VelocityReport

from autosdv_vehicle_interface.hal import HARDWARE_REAL, make_gpio
from autosdv_vehicle_interface.velocity_estimator import VelocityEstimator


class AutoSdvVelocityReportNode(Node):
    """Publish VelocityReport messages from hall-sensor edge timing."""

    def __init__(self, **kwargs) -> None:
        super().__init__("autosdv_velocity_report_node", **kwargs)

        self.declare_parameter("pin", Parameter.Type.INTEGER)
        self.declare_parameter("rate", Parameter.Type.DOUBLE)
        self.declare_parameter("wheel_diameter", Parameter.Type.DOUBLE)
        self.declare_parameter("markers_per_rotation", Parameter.Type.INTEGER)
        self.declare_parameter("frame_id", Parameter.Type.STRING)
        self.declare_parameter("hardware", HARDWARE_REAL)

        self.publisher = self.create_publisher(VelocityReport, "~/output/velocity_status", 1)

        wheel_diameter_cm = self.get_parameter("wheel_diameter").get_parameter_value().double_value
        wheel_circumference_meters = wheel_diameter_cm / 100.0 * math.pi
        self.wheel_circumference_meters = wheel_circumference_meters
        self.markers_per_rotation = (
            self.get_parameter("markers_per_rotation").get_parameter_value().integer_value
        )
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        self.distance_per_marker = wheel_circumference_meters / self.markers_per_rotation
        self.estimator = VelocityEstimator(self.distance_per_marker)
        self.clock = self.get_clock()

        # The hall sensor: Jetson.GPIO on the car, the rc_sim plant's edges in simulation
        self.pin = self.get_parameter("pin").get_parameter_value().integer_value
        self.gpio = make_gpio(self.get_parameter("hardware").value, node=self)
        self.gpio.setmode(self.gpio.BOARD)
        self.gpio.setup(self.pin, self.gpio.IN)
        self.gpio.add_event_detect(self.pin, self.gpio.RISING, callback=self.on_gpio_rising)

        rate = self.get_parameter("rate").get_parameter_value().double_value
        self.timer = self.create_timer(1.0 / rate, self.publish_callback)

    def publish_callback(self) -> None:
        """Publish the current estimate (zero once the edges have stopped)."""
        curr_time = self.clock.now()
        msg = VelocityReport()
        msg.header.stamp = curr_time.to_msg()
        msg.header.frame_id = self.frame_id
        msg.longitudinal_velocity = self.estimator.report(curr_time.nanoseconds)
        msg.lateral_velocity = 0.0
        msg.heading_rate = 0.0
        self.publisher.publish(msg)

    def on_gpio_rising(self, channel, stamp_ns=None) -> None:
        """
        Register a marker edge.

        Args:
            channel: GPIO channel that triggered the event
            stamp_ns: The edge's true time [ns], from a simulated backend; the
                real GPIO callback passes none and the edge is timed on arrival.
        """
        if stamp_ns is None:
            stamp_ns = self.clock.now().nanoseconds
        self.estimator.on_edge(stamp_ns)


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
