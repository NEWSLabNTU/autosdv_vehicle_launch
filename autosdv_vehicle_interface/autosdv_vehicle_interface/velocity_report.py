#!/usr/bin/env python3
"""
Velocity report node for AutoSDV.

Measures the vehicle's speed from a rear wheel's hall-sensor edges. The
arithmetic is VelocityEstimator's (velocity_estimator.py), shared with the
rc_sim lockstep grader; this node supplies the edges and the clock.

``hardware``: ``real`` (Jetson.GPIO, the default) or ``sim`` (SimGPIO: edges
from the autosdv_rc_sim plant, each carrying its true time). In sim the node
also acknowledges each report on the plant's clock handshake (hal.py,
LockstepParticipant) after publishing it.
"""
import math
import sys

from autosdv_vehicle_interface.hal import (
    HARDWARE_REAL, make_gpio, make_lockstep_participant, timer_deadline_ns, VELOCITY_REPORT)
from autosdv_vehicle_interface.velocity_estimator import VelocityEstimator
from autoware_vehicle_msgs.msg import VelocityReport
import rclpy
from rclpy import Parameter
from rclpy.node import Node


class AutoSdvVelocityReportNode(Node):
    """Publish VelocityReport messages from hall-sensor edge timing."""

    def __init__(self, **kwargs) -> None:
        super().__init__('autosdv_velocity_report_node', **kwargs)

        self.declare_parameter("pin", Parameter.Type.INTEGER)
        self.declare_parameter("rate", Parameter.Type.DOUBLE)
        self.declare_parameter("wheel_diameter", Parameter.Type.DOUBLE)
        self.declare_parameter("markers_per_rotation", Parameter.Type.INTEGER)
        self.declare_parameter("frame_id", Parameter.Type.STRING)
        self.declare_parameter('hardware', HARDWARE_REAL)
        # 1 = each edge gap alone; 3 rejects a late GPIO callback (velocity_estimator.py)
        self.declare_parameter('gap_median_window', 1)

        self.publisher = self.create_publisher(VelocityReport, '~/output/velocity_status', 1)

        wheel_diameter_cm = self.get_parameter('wheel_diameter').get_parameter_value().double_value
        wheel_circumference_meters = wheel_diameter_cm / 100.0 * math.pi
        self.wheel_circumference_meters = wheel_circumference_meters
        self.markers_per_rotation = (
            self.get_parameter('markers_per_rotation').get_parameter_value().integer_value
        )
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.distance_per_marker = wheel_circumference_meters / self.markers_per_rotation
        self.estimator = VelocityEstimator(
            self.distance_per_marker,
            gap_median=self.get_parameter('gap_median_window').get_parameter_value().integer_value)
        self.clock = self.get_clock()

        # The hall sensor: Jetson.GPIO on the car, the rc_sim plant's edges in simulation
        self.pin = self.get_parameter('pin').get_parameter_value().integer_value
        self.gpio = make_gpio(self.get_parameter('hardware').value, node=self)
        self.gpio.setmode(self.gpio.BOARD)
        self.gpio.setup(self.pin, self.gpio.IN)
        self.gpio.add_event_detect(self.pin, self.gpio.RISING, callback=self.on_gpio_rising)

        rate = self.get_parameter('rate').get_parameter_value().double_value
        self.timer = self.create_timer(1.0 / rate, self.publish_callback)

        # Sim only: the plant's clock handshake. late_edges counts edges that arrived
        # after a report already covering their time (a race the handshake should prevent).
        # A report whose edges the plant has not finished sending is held until they arrive.
        self.lockstep = make_lockstep_participant(
            self.get_parameter('hardware').value, self, VELOCITY_REPORT,
            on_status=self._on_lockstep_status)
        self.last_report_ns = None
        self.late_edges = 0
        self._held_report_ns = None
        if self.lockstep is not None:
            self.gpio.on_until = self._on_edges_until

    def publish_callback(self) -> None:
        """Publish the current estimate (zero once the edges have stopped)."""
        if self.lockstep is None:
            self.publish_report()
            return
        deadline = timer_deadline_ns(self.timer, self.clock.now().nanoseconds)
        if self._held_report_ns is not None:
            # superseded: the plant moved on without the edges (a lockstep timeout)
            self.lockstep.ack(self._held_report_ns, self.timer.timer_period_ns, skipped=1)
            self._held_report_ns = None
        if self.lockstep.active and self.gpio.awaiting_edges(deadline):
            self._held_report_ns = deadline
            return
        self._lockstep_report(deadline)

    def _on_edges_until(self, until_ns: int) -> None:
        held = self._held_report_ns
        if held is not None and until_ns >= held:
            self._held_report_ns = None
            self._lockstep_report(held)

    def _on_lockstep_status(self, active: bool) -> None:
        held = self._held_report_ns
        if not active and held is not None:
            self._held_report_ns = None
            self._lockstep_report(held)

    def _lockstep_report(self, deadline_ns: int) -> None:
        """Publish the report, then acknowledge its deadline."""
        curr_time = self.publish_report()
        self.last_report_ns = curr_time.nanoseconds
        self.lockstep.ack(deadline_ns, self.timer.timer_period_ns, late_edges=self.late_edges)

    def publish_report(self):
        """Publish the estimate at the current time; return that time."""
        curr_time = self.clock.now()
        msg = VelocityReport()
        msg.header.stamp = curr_time.to_msg()
        msg.header.frame_id = self.frame_id
        msg.longitudinal_velocity = self.estimator.report(curr_time.nanoseconds)
        msg.lateral_velocity = 0.0
        msg.heading_rate = 0.0
        self.publisher.publish(msg)
        return curr_time

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
        elif self.last_report_ns is not None and stamp_ns <= self.last_report_ns:
            self.late_edges += 1
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
