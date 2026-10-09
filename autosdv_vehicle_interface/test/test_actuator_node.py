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
"""
The actuator node, with the PCA9685 replaced by a fake.

Needs rclpy and the Autoware message packages (skipped without them); never
touches I2C or GPIO.
"""
import os
import signal
import subprocess
import sys
import textwrap
import time

import pytest
import yaml

os.environ.setdefault("ROS_DOMAIN_ID", "75")
rclpy = pytest.importorskip("rclpy")
actuator = pytest.importorskip("autosdv_vehicle_interface.actuator")

from autoware_adapi_v1_msgs.msg import MrmState  # noqa: E402
from autoware_control_msgs.msg import Control  # noqa: E402
from autoware_vehicle_msgs.msg import VelocityReport  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from tier4_vehicle_msgs.msg import VehicleEmergencyStamped  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
ACTUATOR_YAML = os.path.join(PKG, "params", "actuator.yaml")
VEHICLE_INFO_YAML = os.path.join(
    PKG, "..", "autosdv_vehicle_description", "config", "vehicle_info.param.yaml"
)
INIT_PWM, INIT_STEER, BRAKE_PWM = 370, 489, 340


def load_params(**overrides):
    values = {}
    for path in (ACTUATOR_YAML, VEHICLE_INFO_YAML):
        with open(path) as f:
            values.update(yaml.safe_load(f)["/**"]["ros__parameters"])
    values.update(overrides)
    return [Parameter(k, value=v) for k, v in values.items()]


@pytest.fixture(scope="module")
def ros():
    rclpy.init()
    yield
    rclpy.try_shutdown()


@pytest.fixture
def make_node(ros, fake_pca, monkeypatch):
    monkeypatch.setattr(actuator, "PCA9685", fake_pca)
    nodes = []

    def _make(**overrides):
        overrides.setdefault("enable_debug_publishing", False)
        node = actuator.AutoSdvActuator(parameter_overrides=load_params(**overrides))
        nodes.append(node)
        return node

    yield _make
    for node in nodes:
        node._neutral_written = True  # already checked where it matters
        node.destroy_node()


def control(node, velocity, steer=0.0, age=0.0, stamped=True):
    msg = Control()
    msg.longitudinal.velocity = float(velocity)
    msg.lateral.steering_tire_angle = float(steer)
    if stamped:
        now_ns = node.get_clock().now().nanoseconds - int(age * 1e9)
        msg.stamp.sec, msg.stamp.nanosec = divmod(now_ns, 10**9)
    node.control_callback(msg)


def velocity(node, speed):
    msg = VelocityReport()
    msg.longitudinal_velocity = float(speed)
    node.velocity_callback(msg)


def tick(node):
    node.timer_callback()
    return node.driver.last(0), node.driver.last(1)


def expire_commands(node, seconds=1.0):
    node.cmd_watchdog.last_feed_time -= seconds


def test_bad_controller_refuses_to_start_before_touching_hardware(make_node, fake_pca):
    with pytest.raises(ImportError):
        make_node(speed_controller_class="no_such_pkg.speed_pid:SpeedPID")
    assert fake_pca.instances == []


def test_waits_in_neutral_until_commanded(make_node):
    node = make_node()
    assert tick(node) == (INIT_PWM, INIT_STEER)
    assert node.longitudinal.mode.value == "waiting"


def test_watchdog_brakes_then_neutral_and_recovers(make_node):
    node = make_node()
    velocity(node, 1.0)
    control(node, 1.0, steer=0.2)
    motor, steer = tick(node)
    assert motor == node.longitudinal.last_pwm_value and node.longitudinal.mode.value == "active"
    assert steer != INIT_STEER

    # Commands stop: brake (still moving), steering centred
    expire_commands(node)
    velocity(node, 1.0)
    assert tick(node) == (BRAKE_PWM, INIT_STEER)
    assert node.longitudinal.override_reason == "command_timeout"
    # Stopped: neutral
    velocity(node, 0.0)
    assert tick(node) == (INIT_PWM, INIT_STEER)
    assert node.cmd_watchdog.episodes == 1

    # Fresh command: control resumes
    velocity(node, 0.15)
    control(node, 1.0)
    motor, _ = tick(node)
    assert node.longitudinal.override_reason is None
    assert motor > INIT_PWM


def test_stale_and_bad_commands_do_not_feed_the_watchdog(make_node):
    node = make_node()
    velocity(node, 0.0)
    control(node, 1.0, age=2.0)               # stale stamp
    control(node, 1.0, age=-2.0)              # from the future
    control(node, 1.0, age=5000.0)            # clock mismatch
    control(node, float("nan"))
    assert node.cmd_watchdog.last_feed_time is None
    assert node.state.target_speed is None
    assert tick(node)[0] == INIT_PWM

    control(node, 1.0, stamped=False)         # unstamped is accepted
    assert node.state.target_speed == 1.0


def test_stamp_check_can_be_disabled(make_node):
    node = make_node(control_cmd_max_stamp_age=0.0)
    control(node, 1.0, age=5000.0)
    assert node.state.target_speed == 1.0


def test_gate_emergency_brakes(make_node):
    node = make_node()
    velocity(node, 1.0)
    control(node, 1.0)
    tick(node)
    node.emergency_callback(VehicleEmergencyStamped(emergency=True))
    control(node, 1.0)
    assert tick(node)[0] == BRAKE_PWM
    assert node.longitudinal.override_reason == "emergency"
    node.emergency_callback(VehicleEmergencyStamped(emergency=False))
    velocity(node, 0.15)
    control(node, 1.0)
    assert tick(node)[0] > INIT_PWM


def test_mrm_comfortable_stop_brakes_by_default(make_node):
    node = make_node()
    velocity(node, 1.0)
    control(node, 1.0)
    node.mrm_state_callback(MrmState(state=MrmState.MRM_OPERATING,
                                     behavior=MrmState.COMFORTABLE_STOP))
    assert tick(node)[0] == BRAKE_PWM


def test_mrm_comfortable_stop_passes_when_configured(make_node):
    node = make_node(emergency_stop_on_any_mrm=False)
    velocity(node, 0.15)
    control(node, 1.0)
    node.mrm_state_callback(MrmState(state=MrmState.MRM_OPERATING,
                                     behavior=MrmState.COMFORTABLE_STOP))
    assert tick(node)[0] > INIT_PWM
    node.mrm_state_callback(MrmState(state=MrmState.MRM_OPERATING,
                                     behavior=MrmState.EMERGENCY_STOP))
    velocity(node, 1.0)
    assert tick(node)[0] == BRAKE_PWM


@pytest.mark.parametrize("last_speed,expected", [(1.0, BRAKE_PWM), (0.15, INIT_PWM)])
def test_stale_velocity_stops_the_motor(make_node, last_speed, expected):
    node = make_node()
    velocity(node, last_speed)
    control(node, 1.0)
    tick(node)
    node.velocity_watchdog.last_feed_time -= 1.0
    control(node, 1.0)
    motor, _ = tick(node)
    assert node.longitudinal.override_reason == "velocity_timeout"
    assert motor == expected  # judged by the last reading before the lapse


def test_student_controller_selected_by_parameter(make_node):
    node = make_node(
        speed_controller_class="autosdv_vehicle_interface.speed_controller:SpeedPID",
    )
    velocity(node, 0.15)
    control(node, 1.0)
    # The interface base class raises NotImplementedError: neutral, not a crash
    assert tick(node)[0] == INIT_PWM
    assert node.longitudinal.fault_count == 1


def test_speed_controller_params_are_merged(make_node):
    node = make_node(speed_controller_params="{kp_speed: 30.0}")
    assert node.speed_controller.pid.kp == 30.0
    assert node.speed_controller.params["init_pwm"] == INIT_PWM


def test_speed_controller_params_must_be_a_mapping(make_node):
    with pytest.raises(ValueError):
        make_node(speed_controller_params="[1, 2]")


def test_write_neutral_outputs(make_node):
    node = make_node()
    velocity(node, 1.0)
    control(node, 1.0, steer=0.3)
    tick(node)
    assert node.write_neutral_outputs()
    assert node.driver.writes[-2:] == [(0, INIT_PWM), (1, INIT_STEER)]
    assert node.timer.is_canceled()
    n = len(node.driver.writes)
    assert node.write_neutral_outputs()        # idempotent
    assert len(node.driver.writes) == n


def test_write_neutral_outputs_does_not_hang(make_node, hang_event):
    node = make_node(neutral_write_timeout=0.2)
    node.driver.hang = hang_event
    start = time.monotonic()
    assert not node.write_neutral_outputs()
    assert time.monotonic() - start < 1.0


# ------------------------------------------------------- whole process

RUNNER = textwrap.dedent("""
    import os, sys, types
    log = open(os.environ["FAKE_PCA_LOG"], "a", buffering=1)

    class PCA9685:
        def __init__(self, address=0x40, busnum=None, **kw):
            log.write("init\\n")
        def set_pwm_freq(self, freq):
            pass
        def set_pwm(self, channel, on, off):
            log.write(f"{channel} {off}\\n")

    sys.modules["Adafruit_PCA9685"] = types.SimpleNamespace(PCA9685=PCA9685)

    from autosdv_vehicle_interface import actuator
    if os.environ.get("FAIL_IN_TIMER"):
        def boom(self):
            raise RuntimeError("injected failure")
        actuator.AutoSdvActuator.timer_callback = boom
    sys.argv = ["actuator", "--ros-args", "--params-file", sys.argv[1],
                "--params-file", sys.argv[2], "-p", "enable_debug_publishing:=false"]
    actuator.main()
""")


@pytest.mark.parametrize("how", ["sigterm", "sigint", "exception"])
def test_process_exit_leaves_outputs_neutral(tmp_path, how):
    log_path = tmp_path / "pca.log"
    script = tmp_path / "run_actuator.py"
    script.write_text(RUNNER)
    env = dict(os.environ, FAKE_PCA_LOG=str(log_path), ROS_DOMAIN_ID="75",
               PYTHONPATH=PKG + os.pathsep + os.environ.get("PYTHONPATH", ""))
    if how == "exception":
        env["FAIL_IN_TIMER"] = "1"

    proc = subprocess.Popen(
        [sys.executable, str(script), ACTUATOR_YAML, VEHICLE_INFO_YAML],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
    )
    try:
        # Wait for the control loop to write at least once (or, for the
        # injected failure, for the process to exit by itself)
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and proc.poll() is None:
            if log_path.exists() and log_path.read_text().count("\n") >= 3:
                break
            time.sleep(0.05)
        if how == "sigterm":
            proc.send_signal(signal.SIGTERM)
        elif how == "sigint":
            proc.send_signal(signal.SIGINT)
        output = proc.communicate(timeout=10.0)[0].decode(errors="replace")
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)

    writes = [tuple(map(int, line.split()))
              for line in log_path.read_text().splitlines() if line != "init"]
    assert writes[-2:] == [(0, INIT_PWM), (1, INIT_STEER)], output
    # ...and they came from the exit path, not merely from the last tick
    assert "Outputs set to neutral" in output, output
    if how == "exception":
        assert proc.returncode != 0 and "injected failure" in output
