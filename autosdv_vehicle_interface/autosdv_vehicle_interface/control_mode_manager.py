#!/usr/bin/env python3
"""
Control Mode Manager for AutoSDV.
Manages ADAS mode states and handles mode change requests for Autoware compatibility.
"""
import sys
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy import Parameter

from autoware_vehicle_msgs.msg import ControlModeReport
from autoware_vehicle_msgs.srv import ControlModeCommand
from autoware_control_msgs.msg import Control
from std_msgs.msg import Bool
from tier4_system_msgs.msg import OperationModeAvailability


class ControlModeManager(Node):
    """
    Control mode management node for RC car platform.
    
    Infers control mode from system activity and provides mode switching services.
    Handles the gap between physical mode switch and software control requirements.
    """

    def __init__(self):
        super().__init__('control_mode_manager_node')
        
        # Declare parameters
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('command_timeout', 2.0)  # seconds
        self.declare_parameter('startup_delay', 3.0)  # seconds
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('default_to_autonomous', False)
        
        # Get parameters
        self.publish_rate = self.get_parameter('publish_rate').value
        self.command_timeout = self.get_parameter('command_timeout').value
        self.startup_delay = self.get_parameter('startup_delay').value
        self.frame_id = self.get_parameter('frame_id').value
        self.default_autonomous = self.get_parameter('default_to_autonomous').value
        
        # State tracking
        self.current_mode = ControlModeReport.NOT_READY
        self.last_command_time = 0.0
        self.startup_time = time.time()
        self.startup_complete = False
        self.manual_override_active = False
        self.emergency_stop_active = False
        self.control_commands_received = False
        
        # Mode transition tracking
        self.mode_transition_requested = False
        self.pending_mode = None
        self.transition_start_time = None
        
        # Publishers
        self.mode_status_pub = self.create_publisher(
            ControlModeReport,
            '/vehicle/status/control_mode',
            1
        )
        
        self.engage_status_pub = self.create_publisher(
            Bool,
            '/vehicle/engage',
            1
        )

        # Operation mode availability publisher - required by Autoware's operation_mode_transition_manager
        self.operation_mode_availability_pub = self.create_publisher(
            OperationModeAvailability,
            '/system/operation_mode/availability',
            1
        )
        
        # Subscribers
        self.control_cmd_sub = self.create_subscription(
            Control,
            '/control/command/control_cmd',
            self.control_command_callback,
            1
        )
        
        self.emergency_stop_sub = self.create_subscription(
            Bool,
            '/system/emergency/stop',
            self.emergency_stop_callback,
            1
        )
        
        # Services
        self.mode_service = self.create_service(
            ControlModeCommand,
            '/control/control_mode_request',
            self.handle_mode_request
        )
        
        # Timer for regular status publishing and state updates
        self.timer = self.create_timer(
            1.0 / self.publish_rate,
            self.update_and_publish_status
        )
        
        # Startup check timer
        self.startup_timer = self.create_timer(
            0.5,
            self.check_startup_complete
        )
        
        self.get_logger().info(
            f'Control Mode Manager initialized - Publishing at {self.publish_rate}Hz'
        )
    
    def check_startup_complete(self):
        """Check if startup delay has passed and system is ready."""
        if not self.startup_complete:
            elapsed = time.time() - self.startup_time
            if elapsed >= self.startup_delay:
                self.startup_complete = True
                self.startup_timer.cancel()
                self.get_logger().info('Startup complete - System ready for operation')
    
    def control_command_callback(self, msg: Control):
        """
        Track control command reception for activity-based mode detection.
        """
        self.last_command_time = time.time()
        self.control_commands_received = True
    
    def emergency_stop_callback(self, msg: Bool):
        """Handle emergency stop signals."""
        if msg.data:
            self.emergency_stop_active = True
            self.get_logger().warn('Emergency stop activated - Switching to DISENGAGED')
        else:
            self.emergency_stop_active = False
            self.get_logger().info('Emergency stop released')
    
    def determine_current_mode(self) -> int:
        """
        Determine the current control mode based on system state.
        
        Returns:
            ControlModeReport constant representing current mode
        """
        # Emergency stop overrides everything
        if self.emergency_stop_active:
            return ControlModeReport.DISENGAGED
        
        # System not ready during startup
        if not self.startup_complete:
            return ControlModeReport.NOT_READY
        
        # Manual override takes precedence
        if self.manual_override_active:
            return ControlModeReport.MANUAL
        
        # Check for recent control commands
        current_time = time.time()
        time_since_command = current_time - self.last_command_time
        has_recent_commands = time_since_command < self.command_timeout
        
        # Determine mode based on activity
        if has_recent_commands and self.control_commands_received:
            # Receiving commands and system ready
            return ControlModeReport.AUTONOMOUS
        elif self.default_autonomous and self.startup_complete:
            # Default to autonomous if configured
            return ControlModeReport.AUTONOMOUS
        elif not self.control_commands_received:
            # Never received commands, not ready for autonomous
            return ControlModeReport.NOT_READY
        else:
            # No recent commands, system disengaged
            return ControlModeReport.DISENGAGED
    
    def handle_mode_request(
        self,
        request: ControlModeCommand.Request,
        response: ControlModeCommand.Response
    ) -> ControlModeCommand.Response:
        """
        Handle control mode change requests via service.
        
        Args:
            request: Service request with desired mode
            response: Service response to populate
            
        Returns:
            Populated service response
        """
        requested_mode = request.mode
        mode_name = self.get_mode_name(requested_mode)
        
        self.get_logger().info(f'Mode change requested: {mode_name}')
        
        # Validate the request
        if not self.startup_complete:
            response.success = False
            response.message = "System not ready - Still in startup phase"
            return response
        
        if self.emergency_stop_active:
            response.success = False
            response.message = "Cannot change mode - Emergency stop active"
            return response
        
        # Handle specific mode requests
        if requested_mode == ControlModeReport.AUTONOMOUS:
            # Check if safe to engage autonomous mode
            if self.can_engage_autonomous():
                self.manual_override_active = False
                self.mode_transition_requested = True
                self.pending_mode = ControlModeReport.AUTONOMOUS
                response.success = True
                response.message = "Autonomous mode engaged"
                self.get_logger().info('Autonomous mode engaged successfully')
            else:
                response.success = False
                response.message = "Cannot engage autonomous - Safety checks failed"
                
        elif requested_mode == ControlModeReport.MANUAL:
            # Always allow switch to manual
            self.manual_override_active = True
            self.mode_transition_requested = True
            self.pending_mode = ControlModeReport.MANUAL
            response.success = True
            response.message = "Manual mode engaged"
            self.get_logger().info('Manual mode engaged')
            
        elif requested_mode == ControlModeReport.DISENGAGED:
            # Allow disengagement
            self.manual_override_active = False
            self.mode_transition_requested = True
            self.pending_mode = ControlModeReport.DISENGAGED
            response.success = True
            response.message = "System disengaged"
            self.get_logger().info('System disengaged')
            
        else:
            response.success = False
            response.message = f"Unsupported mode: {mode_name}"
        
        return response
    
    def can_engage_autonomous(self) -> bool:
        """
        Check if system is safe to engage autonomous mode.
        
        Returns:
            True if autonomous mode can be safely engaged
        """
        # Check various safety conditions
        if not self.startup_complete:
            return False
            
        if self.emergency_stop_active:
            return False
        
        # Could add more checks here:
        # - Sensor health status
        # - Actuator ready status
        # - Communication link quality
        # - GPS lock status (if applicable)
        
        # For RC car, basic check is sufficient
        return True
    
    def get_mode_name(self, mode: int) -> str:
        """Convert mode constant to readable name."""
        mode_names = {
            ControlModeReport.NO_COMMAND: "NO_COMMAND",
            ControlModeReport.AUTONOMOUS: "AUTONOMOUS",
            ControlModeReport.AUTONOMOUS_STEER_ONLY: "AUTONOMOUS_STEER_ONLY",
            ControlModeReport.AUTONOMOUS_VELOCITY_ONLY: "AUTONOMOUS_VELOCITY_ONLY",
            ControlModeReport.MANUAL: "MANUAL",
            ControlModeReport.DISENGAGED: "DISENGAGED",
            ControlModeReport.NOT_READY: "NOT_READY"
        }
        return mode_names.get(mode, f"UNKNOWN({mode})")
    
    def update_and_publish_status(self):
        """
        Update control mode based on system state and publish status.
        """
        # Determine current mode
        new_mode = self.determine_current_mode()
        
        # Check for mode transitions
        if self.mode_transition_requested and self.pending_mode is not None:
            # Apply pending mode change
            new_mode = self.pending_mode
            self.mode_transition_requested = False
            self.pending_mode = None
        
        # Log mode changes
        if new_mode != self.current_mode:
            old_name = self.get_mode_name(self.current_mode)
            new_name = self.get_mode_name(new_mode)
            self.get_logger().info(f'Mode change: {old_name} -> {new_name}')
            
        self.current_mode = new_mode
        
        # Publish control mode status
        mode_msg = ControlModeReport()
        mode_msg.stamp = self.get_clock().now().to_msg()
        mode_msg.mode = self.current_mode
        self.mode_status_pub.publish(mode_msg)
        
        # Publish engagement status
        engage_msg = Bool()
        engage_msg.data = (self.current_mode == ControlModeReport.AUTONOMOUS)
        self.engage_status_pub.publish(engage_msg)

        # Publish operation mode availability - tells Autoware which modes are available
        availability_msg = OperationModeAvailability()
        availability_msg.stamp = self.get_clock().now().to_msg()
        # When startup is complete, all modes are available
        if self.startup_complete and not self.emergency_stop_active:
            availability_msg.stop = True
            availability_msg.autonomous = True
            availability_msg.local = True
            availability_msg.remote = True
            availability_msg.emergency_stop = True
            availability_msg.comfortable_stop = True
            availability_msg.pull_over = True
        else:
            # During startup or emergency, only emergency stop is available
            availability_msg.stop = True
            availability_msg.autonomous = False
            availability_msg.local = False
            availability_msg.remote = False
            availability_msg.emergency_stop = True
            availability_msg.comfortable_stop = False
            availability_msg.pull_over = False
        self.operation_mode_availability_pub.publish(availability_msg)


def main(args=None):
    """Main function to run the control mode manager node."""
    rclpy.init(args=args)
    
    try:
        node = ControlModeManager()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error in control mode manager: {e}')
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()