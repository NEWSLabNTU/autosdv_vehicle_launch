# Copyright 2026 NEWSLab, National Taiwan University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Motor plant model and the speed bench harness."""
import json
import os
import xml.etree.ElementTree as ET

from autosdv_vehicle_interface.motor_plant import HallSpeedSensor, MotorPlant, MotorPlantParams
from autosdv_vehicle_interface.speed_bench import (
    DEFAULT_ACTUATOR_PARAMS,
    main,
    run_step,
    step_metrics,
)
from autosdv_vehicle_interface.speed_controller import load_speed_controller
import pytest
import yaml

ACTUATOR_YAML = os.path.join(os.path.dirname(__file__), "..", "params", "actuator.yaml")
VEHICLE_INTERFACE_LAUNCH = os.path.join(
    os.path.dirname(__file__), "..", "..", "autosdv_vehicle_launch", "launch",
    "vehicle_interface.launch.xml")


def settle(plant, pwm, seconds=10.0, dt=0.001):
    for _ in range(int(seconds / dt)):
        plant.step(pwm, dt)
    return plant.speed


def test_plant_deadband_does_not_move():
    p = MotorPlantParams()
    for pwm in range(p.init_pwm - p.deadband, p.init_pwm + p.deadband + 1):
        assert settle(MotorPlant(p), pwm, 2.0) == 0.0


def test_plant_steady_state_gain():
    p = MotorPlantParams(gain=0.03, deadband=5)
    plant = MotorPlant(p)
    assert settle(plant, 400) == pytest.approx(0.03 * 25, rel=1e-3)
    assert plant.steady_state_speed(400) == pytest.approx(0.75)


def test_plant_first_order_time_constant():
    p = MotorPlantParams(time_constant=0.4)
    plant = MotorPlant(p)
    v_ss = plant.steady_state_speed(420)
    v_tau = settle(plant, 420, seconds=0.4)
    assert v_tau == pytest.approx(v_ss * (1 - 1 / 2.718281828), rel=1e-2)


def test_plant_brakes_to_zero_and_not_beyond():
    p = MotorPlantParams(brake_decel=3.0)
    plant = MotorPlant(p, speed=1.5)
    plant.step(340, 0.25)
    assert plant.speed == pytest.approx(0.75)
    settle(plant, 340, 2.0)
    assert plant.speed == 0.0  # allow_reverse is off


def test_plant_reverse_when_allowed():
    plant = MotorPlant(MotorPlantParams(allow_reverse=True))
    assert settle(plant, 340, 5.0) < 0.0


def test_plant_coasts_in_deadband():
    plant = MotorPlant(MotorPlantParams(coast_time_constant=1.0), speed=1.0)
    settle(plant, 370, 1.0)
    assert plant.speed == pytest.approx(0.3679, rel=1e-2)


def test_hall_sensor_reads_constant_speed():
    sensor = HallSpeedSensor()
    v, t, d = 1.0, 0.0, 0.0
    while t < 1.0:
        t += 0.001
        d += v * 0.001
        sensor.update(t, d)
    assert sensor.report(t) == pytest.approx(1.0, rel=0.05)
    # 0.5 s without an edge reads zero, as velocity_report.py does
    assert sensor.report(t + 0.6) == 0.0


def test_actuator_defaults_match_params_file():
    with open(ACTUATOR_YAML) as f:
        params = yaml.safe_load(f)["/**"]["ros__parameters"]
    # The speed_controller_* defaults live in the launch file, not actuator.yaml.
    launch = ET.parse(VEHICLE_INTERFACE_LAUNCH).getroot()
    params |= {a.get("name"): a.get("default") for a in launch.iter("arg")
               if a.get("name", "").startswith("speed_controller_")}
    for key, value in DEFAULT_ACTUATOR_PARAMS.items():
        assert params[key] == value, key


def test_bench_default_controller_reaches_target():
    ctl = load_speed_controller("", DEFAULT_ACTUATOR_PARAMS | {})
    result = run_step(ctl, target_speed=1.0, duration=10.0)
    m = result.metrics
    assert m.rise_time is not None and 0.5 < m.rise_time < 8.0
    assert abs(m.steady_state_error) < 0.1
    assert m.overshoot_pct < 20.0
    modes = {r["mode"] for r in result.trace}
    assert {"full_stop", "active"} <= modes


def test_step_metrics_on_a_known_trace():
    # First-order 1 - exp(-t/0.5) sampled at 10 ms, step at t = 0
    import math
    trace = [{"t": k * 0.01, "speed": 1 - math.exp(-k * 0.01 / 0.5), "pwm": 400}
             for k in range(1000)]
    m = step_metrics(trace, 1.0, 0.0)
    assert m.rise_time == pytest.approx(0.5 * math.log(9), abs=0.02)
    assert m.overshoot_pct == 0.0
    assert m.steady_state_error == pytest.approx(0.0, abs=1e-3)
    assert m.settling_time == pytest.approx(0.5 * math.log(20), abs=0.02)


def test_bench_cli(tmp_path, capsys):
    out = tmp_path / "trace.csv"
    assert main(["--target", "0.8", "--duration", "4", "--csv", str(out), "--json"]) == 0
    metrics = json.loads(capsys.readouterr().out)
    assert metrics["target_speed"] == 0.8
    assert out.read_text().startswith("t,target,speed,measured,pwm,mode")


def test_bench_cli_overrides(capsys):
    assert main(["--duration", "3", "--plant", "{gain: 0.02}", "--params", "{kp_speed: 30.0}",
                 "--dt-source", "loop", "--json"]) == 0
    assert "rise_time" in capsys.readouterr().out
