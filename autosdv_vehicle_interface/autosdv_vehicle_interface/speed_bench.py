#!/usr/bin/env python3
r"""
Speed-controller bench: run a SpeedPID against the motor plant model. No ROS.

The loop is the vehicle's: the same :class:`LongitudinalShell` the actuator
runs at 100 Hz (its measured-speed filter, its dt, its stop modes and its
output clamp), fed by a simulated ``velocity_report`` at 20 Hz. Only the plant
is a model, and its parameters are provisional (see ``motor_plant.py``).

Command line::

    ros2 run autosdv_vehicle_interface speed_bench               # default PID
    ros2 run autosdv_vehicle_interface speed_bench \\
        --controller my_pkg.speed_pid:SpeedPID --params '{kp: 20.0}' \\
        --target 1.0 --csv tmp/step.csv

Reports rise time (10-90 %), overshoot, steady-state error and 5 % settling
time of the true (plant) speed for a step in target speed.
"""
import argparse
import csv
from dataclasses import asdict, dataclass, field
import json
import math
import sys
from typing import Any, Dict, List, Optional

from autosdv_vehicle_interface.longitudinal import LongitudinalShell, ShellConfig
from autosdv_vehicle_interface.motor_plant import HallSpeedSensor, MotorPlant, MotorPlantParams
from autosdv_vehicle_interface.speed_controller import load_speed_controller

# params/actuator.yaml, as far as the speed loop is concerned.
# test_speed_bench.py fails if these drift from the file.
DEFAULT_ACTUATOR_PARAMS: Dict[str, Any] = {
    "rate": 100.0,
    "init_pwm": 370,
    "min_pwm": 360,
    "max_pwm": 470,
    "brake_pwm": 340,
    "brake_threshold": 0.2,
    "kp_speed": 12.0,
    "ki_speed": 10.0,
    "kd_speed": 0.001,
    "integral_limit": 50.0,
    "enable_conditional_integration": True,
    "velocity_deadband": 0.0,
    "full_stop_threshold": 0.1,
    "derivative_filter_alpha": 0.15,
    "override_brake_duration": 1.5,
    "controller_fault_holdoff": 1.0,
    "controller_max_faults": 3,
    "speed_controller_dt_source": "velocity_report",
}

CONTROLLER_PARAM_KEYS = (
    "init_pwm", "min_pwm", "max_pwm", "brake_pwm", "rate", "kp_speed", "ki_speed",
    "kd_speed", "integral_limit", "enable_conditional_integration", "derivative_filter_alpha",
)


@dataclass
class StepMetrics:
    """Step-response figures of the true vehicle speed after the step."""

    target_speed: float
    rise_time: Optional[float]       # 10 % -> 90 % of target [s]; None if never reached
    overshoot_pct: float             # peak above target, % of target
    steady_state_error: float        # target - mean speed over the last 20 % [m/s]
    settling_time: Optional[float]   # last entry into +-5 % of target [s]; None if never
    final_pwm: int


@dataclass
class BenchResult:
    metrics: StepMetrics
    trace: List[Dict[str, Any]] = field(default_factory=list)


def make_shell(controller, actuator_params: Dict[str, Any], logger=None) -> LongitudinalShell:
    """Build the actuator's longitudinal shell from actuator parameters."""
    p = actuator_params
    return LongitudinalShell(
        ShellConfig(
            init_pwm=p["init_pwm"],
            min_pwm=p["min_pwm"],
            max_pwm=p["max_pwm"],
            brake_pwm=p["brake_pwm"],
            full_stop_threshold=p["full_stop_threshold"],
            brake_threshold=p["brake_threshold"],
            velocity_deadband=p["velocity_deadband"],
            override_brake_duration=p["override_brake_duration"],
            controller_fault_holdoff=p["controller_fault_holdoff"],
            controller_max_faults=p["controller_max_faults"],
            dt_source=p["speed_controller_dt_source"],
        ),
        controller,
        logger=logger,
    )


def controller_params(actuator_params: Dict[str, Any], overrides: Optional[Dict] = None) -> Dict:
    """Return the params dict the actuator would pass to a speed controller."""
    params = {k: actuator_params[k] for k in CONTROLLER_PARAM_KEYS}
    params.update(overrides or {})
    return params


def run_step(
    controller,
    target_speed: float = 1.0,
    duration: float = 8.0,
    step_time: float = 0.5,
    actuator_params: Optional[Dict[str, Any]] = None,
    plant_params: Optional[MotorPlantParams] = None,
    report_rate: float = 20.0,
    substep: float = 0.001,
    logger=None,
) -> BenchResult:
    """
    Simulate a step in target speed from standstill.

    Args:
        controller: A SpeedPID instance.
        target_speed: Step size [m/s].
        duration: Simulated time [s].
        step_time: When the target steps up [s]; zero before it.
        actuator_params: Actuator parameters; defaults to actuator.yaml's.
        plant_params: Plant parameters; defaults to the provisional set.
        report_rate: velocity_report publication rate [Hz].
        substep: Plant integration step [s].
    """
    p = dict(DEFAULT_ACTUATOR_PARAMS)
    p.update(actuator_params or {})
    shell = make_shell(controller, p, logger)
    plant = MotorPlant(plant_params)
    sensor = HallSpeedSensor()

    tick = 1.0 / p["rate"]
    report_period = 1.0 / report_rate
    n_ticks = int(round(duration / tick))
    n_sub = max(1, int(round(tick / substep)))

    pwm = p["init_pwm"]
    measured: Optional[float] = None
    last_velocity_time = 0.0
    next_report = 0.0
    trace: List[Dict[str, Any]] = []

    for k in range(n_ticks):
        t = k * tick
        # Plant and sensor up to now, with the last PWM held
        if k > 0:
            h = tick / n_sub
            for i in range(n_sub):
                plant.step(pwm, h)
                sensor.update(t - tick + (i + 1) * h, plant.distance)

        if t + 1e-9 >= next_report:
            measured = sensor.report(t)
            last_velocity_time = t
            next_report += report_period

        target = target_speed if t >= step_time else 0.0
        pwm = shell.step(t, target, measured, last_velocity_time)
        trace.append({
            "t": round(t, 6),
            "target": target,
            "speed": plant.speed,
            "measured": measured if measured is not None else float("nan"),
            "pwm": pwm,
            "mode": shell.mode.value,
        })

    return BenchResult(step_metrics(trace, target_speed, step_time), trace)


def step_metrics(trace: List[Dict[str, Any]], target: float, step_time: float) -> StepMetrics:
    """Compute rise time, overshoot, steady-state error and settling time."""
    after = [r for r in trace if r["t"] >= step_time]
    if not after or target == 0.0:
        raise ValueError("need a non-zero step inside the trace")
    sign = 1.0 if target > 0 else -1.0
    speeds = [sign * r["speed"] for r in after]
    goal = abs(target)

    def first_time(level):
        for r, v in zip(after, speeds):
            if v >= level:
                return r["t"]
        return None

    t10, t90 = first_time(0.1 * goal), first_time(0.9 * goal)
    rise = (t90 - t10) if t10 is not None and t90 is not None else None

    overshoot = max(0.0, (max(speeds) - goal) / goal * 100.0)

    tail = speeds[int(len(speeds) * 0.8):]
    sse = goal - sum(tail) / len(tail)

    settling = None
    band = 0.05 * goal
    outside = [r["t"] for r, v in zip(after, speeds) if abs(v - goal) > band]
    if not outside:
        settling = 0.0
    elif outside[-1] < after[-1]["t"]:
        settling = outside[-1] + (after[1]["t"] - after[0]["t"]) - step_time

    return StepMetrics(
        target_speed=target,
        rise_time=rise,
        overshoot_pct=overshoot,
        steady_state_error=sign * sse,
        settling_time=settling,
        final_pwm=int(trace[-1]["pwm"]),
    )


def _mapping(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        import yaml
        value = yaml.safe_load(text)
    except ImportError:
        value = json.loads(text)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError(f"expected a mapping such as '{{kp: 1.0}}', got {text!r}")
    return value


def _fmt(value, unit="", digits=3):
    if value is None:
        return "never"
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return f"{value:.{digits}f}{unit}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a speed controller against the (provisional) motor plant model."
    )
    parser.add_argument("--controller", default="",
                        help="import path pkg.module:Class (default: the actuator's PID)")
    parser.add_argument("--params", type=_mapping, default={},
                        help="speed_controller_params, a YAML mapping")
    parser.add_argument("--actuator", type=_mapping, default={},
                        help="actuator parameter overrides, e.g. '{max_pwm: 430}'")
    parser.add_argument("--plant", type=_mapping, default={},
                        help="MotorPlantParams overrides, e.g. '{gain: 0.025}'")
    parser.add_argument("--target", type=float, default=1.0, help="step size [m/s]")
    parser.add_argument("--duration", type=float, default=8.0, help="simulated time [s]")
    parser.add_argument("--dt-source", choices=("velocity_report", "loop"), default=None,
                        help="override speed_controller_dt_source")
    parser.add_argument("--csv", help="write the trace to this CSV file")
    parser.add_argument("--json", action="store_true", help="print metrics as JSON")
    args = parser.parse_args(argv)

    actuator = dict(DEFAULT_ACTUATOR_PARAMS)
    actuator.update(args.actuator)
    if args.dt_source:
        actuator["speed_controller_dt_source"] = args.dt_source

    controller = load_speed_controller(args.controller, controller_params(actuator, args.params))
    result = run_step(
        controller,
        target_speed=args.target,
        duration=args.duration,
        actuator_params=actuator,
        plant_params=MotorPlantParams(**args.plant),
    )

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(result.trace[0].keys()))
            writer.writeheader()
            writer.writerows(result.trace)

    m = result.metrics
    if args.json:
        print(json.dumps(asdict(m)))
    else:
        print(f"controller          {args.controller or 'default (AckermannSpeedPID)'}")
        print("plant               PROVISIONAL parameters, not fitted from the vehicle")
        print(f"step                0 -> {m.target_speed:.2f} m/s")
        print(f"rise time 10-90%    {_fmt(m.rise_time, ' s')}")
        print(f"overshoot           {_fmt(m.overshoot_pct, ' %', 1)}")
        print(f"steady-state error  {_fmt(m.steady_state_error, ' m/s')}")
        print(f"settling time 5%    {_fmt(m.settling_time, ' s')}")
        print(f"final PWM           {m.final_pwm}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
