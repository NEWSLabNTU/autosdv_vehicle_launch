#!/usr/bin/env python3
"""
Signal Manager for AutoSDV.
Provides mock turn indicators and hazard lights status for Autoware compatibility.
"""
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy import Parameter

from autoware_vehicle_msgs.msg import (
    TurnIndicatorsReport,
    TurnIndicatorsCommand,
    HazardLightsReport,
    HazardLightsCommand
)


class SignalManager(Node):
    """
    Signal management node for RC car platform.

    Since the RC car lacks actual turn signals and hazard lights,
    this node provides mock implementations to satisfy Autoware requirements.
    It acknowledges commands and provides realistic status feedback.
    """

    def __init__(self):
        super().__init__('signal_manager_node')

        # Declare parameters
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('blink_rate', 2.0)  # Hz for blinking simulation
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('simulate_blinking', True)

        # Get parameters
        self.publish_rate = self.get_parameter('publish_rate').value
        self.blink_rate = self.get_parameter('blink_rate').value
        self.frame_id = self.get_parameter('frame_id').value
        self.simulate_blinking = self.get_parameter('simulate_blinking').value

        # State tracking
        self.turn_indicator_state = TurnIndicatorsReport.DISABLE
        self.hazard_lights_state = HazardLightsReport.DISABLE

        # Command tracking
        self.turn_indicator_command = TurnIndicatorsCommand.DISABLE
        self.hazard_lights_command = HazardLightsCommand.DISABLE

        # Blinking simulation
        self.blink_state = False
        self.last_blink_time = time.time()
        self.blink_period = 1.0 / self.blink_rate if self.blink_rate > 0 else 1.0

        # Publishers
        self.turn_indicators_pub = self.create_publisher(
            TurnIndicatorsReport,
            '/vehicle/status/turn_indicators_status',
            1
        )

        self.hazard_lights_pub = self.create_publisher(
            HazardLightsReport,
            '/vehicle/status/hazard_lights_status',
            1
        )

        # Subscribers
        self.turn_indicators_cmd_sub = self.create_subscription(
            TurnIndicatorsCommand,
            '/control/command/turn_indicators_cmd',
            self.turn_indicators_command_callback,
            1
        )

        self.hazard_lights_cmd_sub = self.create_subscription(
            HazardLightsCommand,
            '/control/command/hazard_lights_cmd',
            self.hazard_lights_command_callback,
            1
        )

        # Timer for regular status publishing
        self.timer = self.create_timer(
            1.0 / self.publish_rate,
            self.publish_signal_status
        )

        self.get_logger().info(
            f'Signal Manager initialized - Publishing at {self.publish_rate}Hz'
        )

    def turn_indicators_command_callback(self, msg: TurnIndicatorsCommand):
        """
        Handle turn indicator commands from Autoware.
        """
        self.turn_indicator_command = msg.command

        # Update state based on command
        if msg.command == 0:
            # Command value 0 is uninitialized/no command - treat as disabled
            self.turn_indicator_state = TurnIndicatorsReport.DISABLE
            # Don't log this as it happens frequently

        elif msg.command == TurnIndicatorsCommand.DISABLE:
            self.turn_indicator_state = TurnIndicatorsReport.DISABLE
            self.get_logger().debug('Turn indicators disabled')

        elif msg.command == TurnIndicatorsCommand.ENABLE_LEFT:
            self.turn_indicator_state = TurnIndicatorsReport.ENABLE_LEFT
            self.get_logger().info('Left turn indicator enabled')

        elif msg.command == TurnIndicatorsCommand.ENABLE_RIGHT:
            self.turn_indicator_state = TurnIndicatorsReport.ENABLE_RIGHT
            self.get_logger().info('Right turn indicator enabled')

        else:
            self.get_logger().warn(f'Unknown turn indicator command: {msg.command}')

    def hazard_lights_command_callback(self, msg: HazardLightsCommand):
        """
        Handle hazard lights commands from Autoware.
        """
        self.hazard_lights_command = msg.command

        # Update state based on command
        if msg.command == 0:
            # Command value 0 is uninitialized/no command - treat as disabled
            self.hazard_lights_state = HazardLightsReport.DISABLE
            # Don't log this as it happens frequently

        elif msg.command == HazardLightsCommand.DISABLE:
            self.hazard_lights_state = HazardLightsReport.DISABLE
            self.get_logger().debug('Hazard lights disabled')

        elif msg.command == HazardLightsCommand.ENABLE:
            self.hazard_lights_state = HazardLightsReport.ENABLE
            self.get_logger().info('Hazard lights enabled')

        else:
            self.get_logger().warn(f'Unknown hazard lights command: {msg.command}')

    def update_blink_state(self):
        """
        Update the simulated blinking state for active signals.
        """
        if not self.simulate_blinking:
            return

        current_time = time.time()
        if current_time - self.last_blink_time >= self.blink_period / 2:
            self.blink_state = not self.blink_state
            self.last_blink_time = current_time

    def get_effective_indicator_state(self) -> int:
        """
        Get the effective turn indicator state considering hazard lights.

        Returns:
            Turn indicator state constant
        """
        # Hazard lights override turn indicators
        if self.hazard_lights_state == HazardLightsReport.ENABLE:
            # When hazard lights are on, both turn indicators should be active
            # But we report the underlying turn indicator state
            # Autoware will handle the visual representation
            return self.turn_indicator_state
        else:
            return self.turn_indicator_state

    def publish_signal_status(self):
        """
        Publish turn indicators and hazard lights status.
        """
        # Update blinking simulation
        self.update_blink_state()

        # Publish turn indicators status
        turn_msg = TurnIndicatorsReport()
        turn_msg.stamp = self.get_clock().now().to_msg()
        turn_msg.report = self.get_effective_indicator_state()
        self.turn_indicators_pub.publish(turn_msg)

        # Publish hazard lights status
        hazard_msg = HazardLightsReport()
        hazard_msg.stamp = self.get_clock().now().to_msg()
        hazard_msg.report = self.hazard_lights_state
        self.hazard_lights_pub.publish(hazard_msg)

        # Log state changes for debugging (with blinking simulation)
        if self.simulate_blinking:
            if self.turn_indicator_state != TurnIndicatorsReport.DISABLE:
                indicator_active = self.blink_state
                self.log_blink_state('Turn', indicator_active)

            if self.hazard_lights_state == HazardLightsReport.ENABLE:
                hazard_active = self.blink_state
                self.log_blink_state('Hazard', hazard_active)

    def log_blink_state(self, signal_type: str, active: bool):
        """
        Log the blinking state for debugging (only occasionally).

        Args:
            signal_type: Type of signal (Turn/Hazard)
            active: Whether the signal is currently lit
        """
        # Only log state changes to avoid spam
        # This would be better with a proper state tracker
        pass  # Disabled to avoid log spam


def main(args=None):
    """Main function to run the signal manager node."""
    rclpy.init(args=args)

    try:
        node = SignalManager()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error in signal manager: {e}')
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
