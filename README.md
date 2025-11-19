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
- Uses PID controllers for accurate speed and steering control
- Configurable parameters for different vehicles and motor setups
- Interfaces with the PCA9685 PWM driver connected via I2C

##### Longitudinal Control Workflow

The following diagram illustrates how the actuator node processes velocity commands to generate PWM signals for motor control:

```mermaid
flowchart TD
    %% Input Sources
    ControlMsg([Control Command]) -->|target_velocity| FilterCmd[Low-Pass Filter<br/>EMA α]
    VelReport([Velocity Report]) -->|measured_velocity| FilterMeas[Low-Pass Filter<br/>EMA α]

    %% Filtering outputs
    FilterCmd -->|filtered_target| SafetyCheck{Safety Checks}
    FilterMeas -->|filtered_measured| SafetyCheck

    %% Safety and special cases
    SafetyCheck -->|target ≈ 0 AND<br/>measured > threshold| EmergencyBrake[Output BRAKE_PWM]
    EmergencyBrake -->|brake_pwm| OutputMux{Select Output}

    SafetyCheck -->|target ≈ 0 AND<br/>measured ≈ 0| FullStop[Output INIT_PWM]
    FullStop -->|init_pwm| OutputMux

    SafetyCheck -->|error < deadband| Hold[Hold Last PWM]
    Hold -->|last_pwm| OutputMux

    SafetyCheck -->|else: active control| ErrorCalc[Calculate Error<br/>e = target - measured]

    %% Control path
    ErrorCalc -->|velocity_error| ReverseLogic[Determine Direction<br/>Forward/Reverse Logic]

    ReverseLogic -->|in_reverse flag| DirectionMap

    FilterCmd -->|setpoint| PIDController[PID Controller<br/>────────────<br/>P = Kp × e<br/>I = ∫ Ki × e dt<br/>D = -Kd × dv/dt<br/>────────────<br/>Anti-windup:<br/>Integral Limit<br/>Conditional Integration]
    FilterMeas -->|feedback| PIDController

    %% PID output
    PIDController -->|pwm_offset| DirectionMap{Direction?}

    DirectionMap -->|forward| PWMCalcFwd[Add to INIT_PWM<br/>pwm = INIT_PWM + offset]
    DirectionMap -->|reverse| PWMCalcRev[Subtract from INIT_PWM<br/>pwm = INIT_PWM - offset]

    PWMCalcFwd -->|raw_pwm| PWMFilter[Low-Pass Filter<br/>EMA α = 0.25]
    PWMCalcRev -->|raw_pwm| PWMFilter

    PWMFilter -->|filtered_pwm| Saturate[Clamp<br/>MIN_PWM ≤ pwm ≤ MAX_PWM]

    Saturate -->|clamped_pwm| OutputMux

    %% Final output
    OutputMux -->|pwm_value| HardwareDriver[PCA9685 PWM Driver<br/>Channel 0]

    HardwareDriver -->|I2C signal| Motor([Motor ESC])

    %% Styling with darker text for bright backgrounds
    classDef sourceClass fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#01579b
    classDef filterClass fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#4a148c
    classDef decisionClass fill:#fff9c4,stroke:#f57f17,stroke-width:2px,color:#f57f17
    classDef controlClass fill:#ffccbc,stroke:#d84315,stroke-width:3px,color:#bf360c
    classDef calcClass fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20
    classDef outputClass fill:#b39ddb,stroke:#512da8,stroke-width:2px,color:#311b92
    classDef specialClass fill:#ffcdd2,stroke:#c62828,stroke-width:2px,color:#b71c1c
    classDef sinkClass fill:#e0f2f1,stroke:#00695c,stroke-width:2px,color:#004d40

    class ControlMsg,VelReport sourceClass
    class FilterCmd,FilterMeas,PWMFilter filterClass
    class SafetyCheck,DirectionMap,OutputMux decisionClass
    class PIDController controlClass
    class ErrorCalc,ReverseLogic,PWMCalcFwd,PWMCalcRev,Saturate calcClass
    class HardwareDriver outputClass
    class EmergencyBrake,FullStop,Hold specialClass
    class Motor sinkClass
```

Key parameters (in `params/actuator.yaml`):
- PWM frequency and I2C settings
- PID controller parameters for speed and steering
- PWM range settings for motors and servos

#### Velocity Report Node (`velocity_report.py`)

This node:
- Reads from a hall effect sensor connected to a GPIO pin
- Calculates vehicle velocity based on wheel rotation
- Publishes velocity reports to the Autoware stack

Key parameters (in `params/velocity_report.yaml`):
- GPIO pin configuration
- Wheel diameter and markers per rotation
- Publication rate settings

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

The behavior of the vehicle interface can be customized by modifying the parameter files:

- `params/actuator.yaml`: Configure motor control settings, PID parameters, and PWM ranges
- `params/velocity_report.yaml`: Configure velocity reporting settings, wheel dimensions, and sensor settings

## Hardware Requirements

The vehicle interface is designed to work with:

- PCA9685 PWM driver for motor and servo control (connected via I2C)
- Hall effect sensor (KY-003) for velocity measurement (connected to GPIO)
- DC motor and servo motor for vehicle movement and steering

## Dependencies

- ROS 2 Humble
- Adafruit_PCA9685 Python library
- simple_pid Python library
- Jetson.GPIO Python library
