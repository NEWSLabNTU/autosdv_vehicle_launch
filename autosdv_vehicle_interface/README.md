# AutoSDV Vehicle Interface

This package implements the actuator control and velocity reporting for the AutoSDV autonomous vehicle platform.

## Overview

The vehicle interface consists of two main nodes:

- **Actuator Node** (`actuator.py`): Converts Autoware control commands to PWM signals for motor and steering
- **Velocity Report Node** (`velocity_report.py`): Reads hall effect sensor data and publishes vehicle velocity

## Longitudinal Control (Speed Control)

The actuator node implements a multi-mode longitudinal controller that manages motor PWM based on vehicle speed commands.

### Control Modes Overview

```mermaid
flowchart TD
    %% Data Sources - Grouped in invisible subgraph
    subgraph DataSources [" "]
        ControlCmd([Control Command])
        VelReport([Velocity Report])
    end

    %% Control mode selection
    ControlCmd -->|target_velocity| SafetyCheck{Safety &<br/>Mode Check}
    VelReport -->|measured_velocity| SafetyCheck

    %% Emergency Brake Mode
    SafetyCheck -->|target ≈ 0 AND<br/>measured > threshold| EmergencyBrake[Emergency Brake Mode<br/>────────────<br/>Output BRAKE_PWM]

    %% Full Stop Mode
    SafetyCheck -->|target ≈ 0 AND<br/>measured ≈ 0| FullStop[Full Stop Mode<br/>────────────<br/>Output INIT_PWM]

    %% Deadband Hold Mode
    SafetyCheck -->|error < deadband| DeadbandHold[Deadband Hold Mode<br/>────────────<br/>Hold last PWM]

    %% Active Control Mode
    SafetyCheck -->|else: active control| ActiveControl[Active Control Mode<br/>────────────<br/>PID Speed Control<br/>Direction Mapping]

    %% Output Selection
    EmergencyBrake -->|pwm| OutputMux{Mode Select}
    FullStop -->|pwm| OutputMux
    DeadbandHold -->|pwm| OutputMux
    ActiveControl -->|pwm| OutputMux

    %% Final Output
    OutputMux -->|final_pwm| Motor([Motor ESC])

    %% Styling with darker text for bright backgrounds
    classDef sourceClass fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#01579b
    classDef decisionClass fill:#fff9c4,stroke:#f57f17,stroke-width:2px,color:#f57f17
    classDef modeClass fill:#ffccbc,stroke:#d84315,stroke-width:3px,color:#bf360c
    classDef specialClass fill:#ffcdd2,stroke:#c62828,stroke-width:2px,color:#b71c1c
    classDef sinkClass fill:#e0f2f1,stroke:#00695c,stroke-width:2px,color:#004d40
    classDef invisibleGroup fill:none,stroke:none

    class ControlCmd,VelReport sourceClass
    class SafetyCheck,OutputMux decisionClass
    class ActiveControl modeClass
    class EmergencyBrake,FullStop,DeadbandHold specialClass
    class Motor sinkClass
    class DataSources invisibleGroup
```

The controller operates in four modes:
- **Emergency Brake**: Apply brake PWM when target is near zero but vehicle is still moving
- **Full Stop**: Output neutral PWM when both target and measured velocities are near zero
- **Deadband Hold**: Maintain current PWM when velocity error is within deadband (prevents jitter)
- **Active Control**: PID-based speed control with forward/reverse direction mapping

### Active Control Mode Details

```mermaid
flowchart TD
    %% Active Control Mode - Detailed PID Speed Control

    %% Input Sources - Grouped in invisible subgraph
    subgraph DataSources [" "]
        ControlCmd([Control Command])
        VelReport([Velocity Report])
    end

    %% Flow from sources
    ControlCmd -->|target_velocity| FilterCmd[Low-Pass Filter<br/>EMA α]
    VelReport -->|measured_velocity| FilterMeas[Low-Pass Filter<br/>EMA α]

    %% Filtered velocities
    FilterCmd -->|filtered_target| ErrorCalc[Calculate Error<br/>e = target - measured]
    FilterMeas -->|filtered_measured| ErrorCalc

    %% Reverse logic
    ErrorCalc -->|velocity_error| ReverseLogic[Determine Direction<br/>Forward/Reverse Logic]

    ReverseLogic -->|in_reverse flag| DirectionMap

    %% PID Controller
    FilterCmd -->|setpoint| PIDController[PID Controller<br/>────────────<br/>P = Kp × e<br/>I = ∫ Ki × e dt<br/>D = -Kd × dv/dt<br/>────────────<br/>Anti-windup:<br/>Integral Limit<br/>Conditional Integration]
    FilterMeas -->|feedback| PIDController

    %% PID output
    PIDController -->|pwm_offset| DirectionMap{Direction?}

    %% Direction-based PWM calculation
    DirectionMap -->|forward| PWMCalcFwd[Add to INIT_PWM<br/>pwm = INIT_PWM + offset]
    DirectionMap -->|reverse| PWMCalcRev[Subtract from INIT_PWM<br/>pwm = INIT_PWM - offset]

    %% Output filtering
    PWMCalcFwd -->|raw_pwm| PWMFilter[Low-Pass Filter<br/>EMA α = 0.25]
    PWMCalcRev -->|raw_pwm| PWMFilter

    PWMFilter -->|filtered_pwm| Saturate[Clamp<br/>MIN_PWM ≤ pwm ≤ MAX_PWM]

    Saturate -->|final_pwm| HardwareDriver[PCA9685 PWM Driver<br/>Channel 0]

    HardwareDriver -->|I2C signal| Motor([Motor ESC])

    %% Styling with darker text for bright backgrounds
    classDef sourceClass fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#01579b
    classDef filterClass fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#4a148c
    classDef decisionClass fill:#fff9c4,stroke:#f57f17,stroke-width:2px,color:#f57f17
    classDef controlClass fill:#ffccbc,stroke:#d84315,stroke-width:3px,color:#bf360c
    classDef calcClass fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20
    classDef outputClass fill:#b39ddb,stroke:#512da8,stroke-width:2px,color:#311b92
    classDef sinkClass fill:#e0f2f1,stroke:#00695c,stroke-width:2px,color:#004d40
    classDef invisibleGroup fill:none,stroke:none

    class ControlCmd,VelReport sourceClass
    class FilterCmd,FilterMeas,PWMFilter filterClass
    class DirectionMap decisionClass
    class PIDController controlClass
    class ErrorCalc,ReverseLogic,PWMCalcFwd,PWMCalcRev,Saturate calcClass
    class HardwareDriver outputClass
    class Motor sinkClass
    class DataSources invisibleGroup
```

Active control mode features:
- Low-pass filtering on both target and measured velocities
- PID controller with anti-windup protection
- Forward/reverse direction logic
- Direction-specific PWM mapping (INIT_PWM ± offset)
- Output PWM filtering for smooth transitions
- Saturation to hardware limits

## Lateral Control (Steering Control)

The actuator node implements a dual-mode steering controller that switches between open-loop and closed-loop control based on vehicle speed.

### Control Modes Overview

```mermaid
flowchart TD
    %% Data Source
    VelReport([Velocity Report])

    %% Speed-based mode selection
    VelReport -->|vehicle_speed| SpeedCheck{Speed Check<br/>v < 0.3 m/s?}

    %% Fallback Mode
    SpeedCheck -->|Yes: Low speed| FallbackMode[Fallback Mode<br/>────────────<br/>Open-Loop Control<br/>pwm = f steering_angle]

    %% Normal Mode
    SpeedCheck -->|No: Normal speed| NormalMode[Normal Mode<br/>────────────<br/>Yaw Rate Feedback<br/>Ackermann + PID]

    %% Output
    FallbackMode -->|pwm| OutputMux{Mode Select}
    NormalMode -->|pwm| OutputMux

    OutputMux -->|final_pwm| Servo([Steering Servo])

    %% Styling with darker text for bright backgrounds
    classDef sourceClass fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#01579b
    classDef decisionClass fill:#fff9c4,stroke:#f57f17,stroke-width:2px,color:#f57f17
    classDef modeClass fill:#ffccbc,stroke:#d84315,stroke-width:3px,color:#bf360c
    classDef sinkClass fill:#e0f2f1,stroke:#00695c,stroke-width:2px,color:#004d40

    class VelReport sourceClass
    class SpeedCheck,OutputMux decisionClass
    class FallbackMode,NormalMode modeClass
    class Servo sinkClass
```

The controller operates in two modes based on vehicle speed:
- **Fallback Mode** (v < 0.3 m/s): Open-loop feedforward control for standstill/low speeds
- **Normal Mode** (v ≥ 0.3 m/s): Yaw rate feedback control using Ackermann model and PID

### Fallback Mode Details (Low Speed)

```mermaid
flowchart TD
    %% Fallback Mode: Low Speed (v < 0.3 m/s) - Open-Loop Control

    %% Input Sources
    ControlCmd([Control Command])

    %% Flow
    ControlCmd -->|steering_tire_angle| ClampAngle[Clamp to Limits<br/>±MAX_STEER_ANGLE]

    ClampAngle -->|clamped_angle| Feedforward[Feedforward Term<br/>pwm = INIT_STEER + angle × ratio]

    Feedforward -->|pwm| Saturate[Clamp PWM<br/>MIN_STEER ≤ pwm ≤ MAX_STEER]

    Saturate -->|final_pwm| ServoDriver[PCA9685 PWM Driver<br/>Channel 1]

    ServoDriver -->|I2C signal| Servo([Steering Servo])

    %% Styling with darker text for bright backgrounds
    classDef sourceClass fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#01579b
    classDef calcClass fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20
    classDef outputClass fill:#b39ddb,stroke:#512da8,stroke-width:2px,color:#311b92
    classDef sinkClass fill:#e0f2f1,stroke:#00695c,stroke-width:2px,color:#004d40

    class ControlCmd sourceClass
    class ClampAngle,Feedforward,Saturate calcClass
    class ServoDriver outputClass
    class Servo sinkClass
```

Fallback mode features:
- Simple feedforward control: `pwm = INIT_STEER + angle × ratio`
- No feedback required (yaw rate is unreliable at low speeds)
- Direct angle-to-PWM mapping with saturation

### Normal Mode Details (Normal Speed)

```mermaid
flowchart TD
    %% Normal Mode: Normal Speed (v ≥ 0.3 m/s) - Yaw Rate Feedback Control

    %% Input Sources - Grouped in invisible subgraph
    subgraph DataSources [" "]
        ControlCmd([Control Command])
        VelReport([Velocity Report])
        IMU([IMU Sensor])
    end

    %% Flow from sources
    ControlCmd -->|steering_tire_angle| ClampAngle[Clamp to Limits<br/>±MAX_STEER_ANGLE]
    VelReport -->|vehicle_speed| AckermannModel
    IMU -->|yaw_rate_measured| FilterMeas[Low-Pass Filter<br/>EMA α = 0.2]

    %% Ackermann model
    ClampAngle -->|clamped_angle| AckermannModel[Ackermann Model<br/>────────────<br/>ω = v/L × tan δ<br/>────────────<br/>L = wheelbase<br/>δ = steering angle]

    AckermannModel -->|target_yaw_rate| FilterTarget[Low-Pass Filter<br/>EMA α = 0.3]

    %% Control loop
    FilterTarget -->|filtered_target| ErrorCalc[Calculate Error<br/>e = target - measured]
    FilterMeas -->|filtered_measured| ErrorCalc

    ErrorCalc -->|yaw_rate_error| PIDController[PID Controller<br/>────────────<br/>P = Kp × e<br/>I = ∫ Ki × e dt<br/>D = -Kd × dω/dt<br/>────────────<br/>Anti-windup:<br/>Integral Limit]

    PIDController -->|pwm_correction| Combiner[Combine<br/>pwm = pwm_ff + correction]

    %% Feedforward path
    ClampAngle -->|clamped_angle| Feedforward[Feedforward Term<br/>pwm_ff = INIT_STEER + angle × ratio]
    Feedforward -->|pwm_ff| Combiner

    %% Output processing
    Combiner -->|combined_pwm| Saturate[Clamp PWM<br/>MIN_STEER ≤ pwm ≤ MAX_STEER]

    Saturate -->|final_pwm| ServoDriver[PCA9685 PWM Driver<br/>Channel 1]

    ServoDriver -->|I2C signal| Servo([Steering Servo])

    %% Styling with darker text for bright backgrounds
    classDef sourceClass fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#01579b
    classDef filterClass fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#4a148c
    classDef modelClass fill:#ffccbc,stroke:#d84315,stroke-width:3px,color:#bf360c
    classDef controlClass fill:#ffccbc,stroke:#d84315,stroke-width:3px,color:#bf360c
    classDef calcClass fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20
    classDef outputClass fill:#b39ddb,stroke:#512da8,stroke-width:2px,color:#311b92
    classDef sinkClass fill:#e0f2f1,stroke:#00695c,stroke-width:2px,color:#004d40
    classDef invisibleGroup fill:none,stroke:none

    class ControlCmd,VelReport,IMU sourceClass
    class FilterTarget,FilterMeas filterClass
    class AckermannModel modelClass
    class PIDController controlClass
    class ErrorCalc,ClampAngle,Feedforward,Combiner,Saturate calcClass
    class ServoDriver outputClass
    class Servo sinkClass
    class DataSources invisibleGroup
```

Normal mode features:
- Ackermann kinematic model converts steering angle to target yaw rate
- IMU provides measured yaw rate feedback
- PID controller minimizes yaw rate error
- Feedforward + feedback architecture for fast response
- Low-pass filtering on both target and measured signals
- Disturbance rejection (compensates for tire slip, wind, etc.)

## Velocity Reporting

The velocity report node reads hall effect sensor data and publishes vehicle velocity.

### Features
- Reads from hall effect sensor (KY-003) connected to GPIO pin
- Calculates vehicle velocity based on wheel rotation
- Publishes velocity reports to Autoware stack

### Configuration Parameters

Velocity reporting is configured in `params/velocity_report.yaml`:
- GPIO pin configuration
- Wheel diameter and markers per rotation
- Publication rate settings

## Configuration

Control parameters are configured in `params/actuator.yaml`:

### Longitudinal Control Parameters
- `kp_speed`, `ki_speed`, `kd_speed`: PID gains for speed control
- `integral_limit`: Anti-windup limit for integral term
- `enable_conditional_integration`: Stop integration when output saturated
- `velocity_deadband`: Error threshold for hold mode (prevents jitter)
- `full_stop_threshold`: Threshold for full stop detection
- `brake_threshold`: Threshold for emergency brake activation
- `velocity_measurement_filter_alpha`: Low-pass filter coefficient for measured velocity
- `velocity_command_filter_alpha`: Low-pass filter coefficient for target velocity
- PWM values: `min_pwm` (280), `init_pwm` (370), `max_pwm` (460), `brake_pwm` (340)

### Lateral Control Parameters
- `kp_steer`, `ki_steer`, `kd_steer`: PID gains for yaw rate control
- `max_steering_angle`: Vehicle maximum steering angle (0.349 rad ≈ 20°, from `vehicle_info.param.yaml`)
- `tire_angle_to_steer_ratio`: Conversion from radians to PWM units
- `steering_speed`: Maximum steering rate (rad/s)
- PWM values: `min_steer` (350), `init_steer` (400), `max_steer` (450)

## Hardware Interface

- **PCA9685 PWM Driver** (I2C)
  - Channel 0: Motor ESC
  - Channel 1: Steering servo
  - Default: I2C address 0x40, bus 1, 60 Hz PWM frequency

- **IMU Sensor** (for steering yaw rate feedback)
  - Provides angular velocity measurement
  - Required for normal mode steering control

## Topics

### Subscriptions
- `~/input/control_cmd` (`autoware_control_msgs/msg/Control`): Control commands from Autoware
- `~/input/velocity_status` (`autoware_vehicle_msgs/msg/VelocityReport`): Current vehicle velocity
- `~/input/imu` (`sensor_msgs/msg/Imu`): IMU data for yaw rate feedback

### Publications
- `~/debug/control_values` (`autoware_internal_debug_msgs/msg/Float32MultiArrayStamped`): Debug control values
- `~/debug/pwm_values` (`autoware_internal_debug_msgs/msg/Float32MultiArrayStamped`): Debug PWM outputs
- `~/debug/pid_values` (`autoware_internal_debug_msgs/msg/Float32MultiArrayStamped`): Debug PID state
