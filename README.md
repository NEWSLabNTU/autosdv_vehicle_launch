# AutoSDV Vehicle Interface

This package provides the vehicle interface components for the AutoSDV autonomous vehicle platform. It enables communication between Autoware's control commands and the vehicle's hardware actuators, as well as reporting the vehicle's velocity back to the Autoware stack.

## Overview

The AutoSDV vehicle interface consists of:

1. **Vehicle Description**: URDF and configuration files describing the vehicle's physical properties
2. **Vehicle Interface**: ROS2 nodes that interface with the vehicle's hardware
3. **Launch Files**: Configuration for launching the vehicle interface components

## Components

### 1. AutoSDV Vehicle Description

Located in `autosdv_vehicle_description/`, this package contains:

- URDF models for the vehicle (`urdf/vehicle.xacro`)
- Vehicle information parameters (`config/vehicle_info.param.yaml`)
- 3D meshes of the vehicle (`mesh/`)

### 2. AutoSDV Vehicle Interface

Located in `autosdv_vehicle_interface/`, this package contains:

- **Actuator Node**: Controls the vehicle's motor and steering servo using PWM signals via a PCA9685 PWM driver
- **Velocity Report Node**: Reads hall effect sensor data to calculate and report the vehicle's velocity

#### Actuator Node (`actuator.py`)

The actuator node subscribes to Autoware control commands and converts them into PWM signals for the vehicle's:
- Throttle/brake control (DC motor)
- Steering control (servo motor)

Features:
- Multi-mode longitudinal controller (emergency brake, full stop, deadband hold, active PID control)
- Dual-mode steering controller (open-loop fallback, yaw rate feedback with Ackermann model)
- Configurable parameters for different vehicles and motor setups
- Interfaces with the PCA9685 PWM driver connected via I2C

For detailed control architecture, workflow diagrams, and configuration parameters, see [autosdv_vehicle_interface/README.md](autosdv_vehicle_interface/README.md).

#### Velocity Report Node (`velocity_report.py`)

This node reads hall effect sensor data and publishes vehicle velocity to the Autoware stack.

### 3. AutoSDV Vehicle Launch

Located in `autosdv_vehicle_launch/`, this package contains:
- Launch files for starting the vehicle interface (`launch/vehicle_interface.launch.xml`)
- Configuration for remapping topics between the vehicle interface and Autoware

## Usage

### Launching the Vehicle Interface

The vehicle interface can be launched using:

```bash
ros2 launch autosdv_vehicle_launch vehicle_interface.launch.xml
```

This launch file starts both the actuator and velocity report nodes with the appropriate parameters and topic remappings.

### Integration with Autoware

The vehicle interface integrates with Autoware through the following topics:

- Input: `/control/command/control_cmd` - Control commands from Autoware
- Output: `/vehicle/status/velocity_status` - Vehicle velocity information

## Configuration

The vehicle interface can be customized through parameter files in `autosdv_vehicle_interface/params/`:
- `actuator.yaml`: Motor and steering control parameters
- `velocity_report.yaml`: Velocity reporting parameters

See [autosdv_vehicle_interface/README.md](autosdv_vehicle_interface/README.md) for detailed parameter descriptions.

## Hardware Requirements

- PCA9685 PWM driver (I2C) for motor and servo control
- Hall effect sensor (KY-003, GPIO) for velocity measurement
- DC motor ESC and steering servo
- IMU sensor for yaw rate feedback (steering control)
