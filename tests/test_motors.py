"""학습용 MJLab 모터 구현을 직접 실행하는 필수 회귀 검사. 별도 모터 모델은 없다."""

from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np
import torch
from mjlab.actuator import BuiltinPositionActuator
from mjlab.actuator.actuator import TransmissionType
from mjlab.utils.buffers import DelayBuffer

from goalkeeper_mjlab import PACKAGE_ROOT
from goalkeeper_mjlab.validation.motor_checks import check_motors, configuration_report, torque_probe
from goalkeeper_mjlab.validation.motor_runtime import MotorSimulation
from goalkeeper_mjlab.robots.k1.constants import K1_JOINT_ORDER
from goalkeeper_mjlab.robots.k1.constants import get_motors, hold_controllers
from goalkeeper_mjlab.scripts.inspect_scene import main
from goalkeeper_mjlab.tasks.goalkeeper.config.k1.env_cfgs import make_goalkeeper_env_cfg


class MotorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = MotorSimulation()

    def setUp(self):
        self.runtime.reset()

    def make_runtime(self, num_envs=2, **kwargs):
        return MotorSimulation(make_goalkeeper_env_cfg(num_envs=num_envs, **kwargs))

    def assert_ctrl(self, runtime, expected):
        torch.testing.assert_close(runtime.sim.data.ctrl[:, runtime.ctrl_ids], expected, atol=1e-6, rtol=0)

    def test_actuator_configuration_requires_joint_transmission(self):
        actuator = self.runtime.cfg.scene.entities["robot"].articulation.actuators[0]
        # 설정 생성 단계에서 거부해 엔진에 전달되기 전에 잘못된 입력을 발견한다.
        with self.assertRaisesRegex(ValueError, "require joint transmission"):
            replace(actuator, transmission_type=TransmissionType.TENDON)

    def test_actuator_configuration_requires_effort_limit(self):
        actuator = self.runtime.cfg.scene.entities["robot"].articulation.actuators[0]
        with self.assertRaisesRegex(ValueError, "require effort_limit"):
            replace(actuator, effort_limit=None)

    def test_default_home_pose_is_inside_actual_soft_limits(self):
        r = self.runtime
        # 기본 장면의 실제 초기 상태를 검사한다. 직립 0 rad로 돌아가 관절 끝값에 서는 회귀를 막는다.
        expected = {"Left_Shoulder_Roll": -1.4, "Right_Shoulder_Roll": 1.4,
                    "Left_Elbow_Yaw": -0.4, "Right_Elbow_Yaw": 0.4,
                    "Left_Hip_Pitch": -0.4, "Right_Hip_Pitch": -0.4,
                    "Left_Knee_Pitch": 0.8, "Right_Knee_Pitch": 0.8,
                    "Left_Ankle_Pitch": -0.4, "Right_Ankle_Pitch": -0.4}
        for name, angle in expected.items():
            self.assertAlmostEqual(r.robot.data.joint_pos[0, K1_JOINT_ORDER.index(name)].item(), angle, places=6)
        q, limits = r.robot.data.joint_pos, r.robot.data.soft_joint_pos_limits
        self.assertTrue(((q > limits[..., 0]) & (q < limits[..., 1])).all())

    def test_recorded_physical_parameters_and_pd_are_applied(self):
        report = configuration_report(self.runtime)
        self.assertTrue(report["passed"], report["errors"])
        reference = json.loads((PACKAGE_ROOT / "robots/k1/motor_reference.json").read_text())["official_motors"]
        for group, motor in get_motors().items():
            self.assertAlmostEqual(motor.peak_effort, reference[group]["effort_limit"])
            self.assertAlmostEqual(motor.armature, reference[group]["armature"])
        self.assertEqual(get_motors()["hip_roll"].peak_effort, 43.)
        self.assertEqual(get_motors("legacy")["hip_roll"].peak_effort, 76.)

    def test_builtin_actuators_share_the_default_delay_buffer(self):
        actuators = self.runtime.robot.actuators
        self.assertTrue(all(isinstance(a, BuiltinPositionActuator) for a in actuators))
        self.assertTrue(all(isinstance(a._delay_buffer, DelayBuffer) for a in actuators))
        self.assertEqual(len({id(a._delay_buffer) for a in actuators}), 1)
        self.assertEqual(self.runtime.robot.actuators[0].cfg.delay_hold_prob, .3)

    def test_real_commands_match_delay_history_with_partial_resets(self):
        r = self.make_runtime(3)
        initial = r.robot.data.default_joint_pos.clone()
        histories = [[], [], []]
        observed_lags = set()
        for step in range(35):
            if step == 12:
                keep_before = r.robot.actuators[0]._delay_buffer.peek()[1].clone()
                r.reset(torch.tensor([0, 2]))
                torch.testing.assert_close(r.robot.actuators[0]._delay_buffer.peek()[1], keep_before)
                histories[0], histories[2] = [], []
            target = initial.clone()
            target[:, 0] += .001 * (step + 1) * torch.tensor([1., -1., .5])
            r.set_target(target)
            r.step_physics()
            lag = r.robot.actuators[0]._delay_buffer.current_lags.clone()
            observed_lags.update(lag.tolist())
            expected = []
            for world in range(3):
                histories[world].append(target[world].clone())
                expected.append(histories[world][max(0, len(histories[world]) - 1 - int(lag[world]))])
            self.assert_ctrl(r, torch.stack(expected))
        # 짧은 난수 실행에서 유지 횟수의 비율을 단정하지 않는다. CPU/CUDA의 난수열도 다를 수 있다.
        self.assertTrue(all(value == 0 or 2 <= value <= 8 for value in observed_lags))

    def test_fixed_and_zero_delays_use_actual_engine_controls(self):
        for lag in (0, 2, 8):
            r = self.make_runtime(1, delay_min_lag=lag, delay_max_lag=lag, delay_hold_prob=0.)
            initial = r.robot.data.default_joint_pos.clone()
            # 실제 준비 명령을 한 스텝 공급해 과거를 만든다. 버퍼에 직접 값을 쓰지 않는다.
            r.step_physics()
            target = initial.clone()
            target[:, 0] += .2
            r.set_target(target)
            for step in range(lag + 1):
                r.step_physics()
                self.assert_ctrl(r, initial if step < lag else target)

    def test_default_reset_has_no_custom_ready_pose_backfill(self):
        r = self.runtime
        command = r.initial_target.copy()
        command[0] = .1
        r.set_target(command)
        r.step()
        r.reset()
        # 기본 버퍼는 reset 후 첫 입력으로 과거를 채운다. 이전 에피소드 명령이 남지 않는다.
        target = r.robot.data.default_joint_pos.clone()
        target[:, 0] += .2
        r.set_target(target)
        r.step_physics()
        self.assert_ctrl(r, target)

    def test_partial_reset_restores_only_selected_entities(self):
        r = self.make_runtime(3)
        for _ in range(3):
            r.step()
        q_before = r.sim.data.qpos[1].clone()
        r.reset(torch.tensor([0, 2]))
        torch.testing.assert_close(r.sim.data.qpos[1], q_before)
        torch.testing.assert_close(r.robot.data.joint_pos[[0, 2]], r.robot.data.default_joint_pos[[0, 2]])
        torch.testing.assert_close(r.scene["goal"].data.root_link_pos_w,
            r.scene.env_origins + torch.tensor(r.cfg.scene.entities["goal"].init_state.pos))
        r.reset(slice(1, 3))
        r.reset(torch.tensor([], dtype=torch.long))

    def test_speed_limits_are_environment_specific(self):
        r = self.make_runtime(2, delay_min_lag=0, delay_max_lag=0)
        q = r.robot.data.default_joint_pos.clone()
        v = torch.zeros_like(q)
        i = K1_JOINT_ORDER.index("Left_Hip_Roll")
        v[1, i] = (8.38 + 12.15) / 2
        r.robot.write_joint_state_to_sim(q, v)
        r.sim.forward()
        r.scene.update(r.physics_dt)
        r.step_physics()
        aid = r.model.actuator("robot/Left_Hip_Roll").id
        torch.testing.assert_close(r.sim.model.actuator_forcerange[:, aid, 1], torch.tensor([43., 21.5]), atol=1e-4, rtol=1e-5)
        self.assertLessEqual(abs(r.sim.data.actuator_force[1, aid].item()), 21.5 + 1e-4)

    def test_independent_220_case_engine_force_probe(self):
        result = torque_probe(make_goalkeeper_env_cfg())
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["joint_cases"], 220)
        self.assertGreater(result["saturated_cases"], 100)

    def test_software_cap_is_applied_after_physical_derating(self):
        cfg = make_goalkeeper_env_cfg(delay_min_lag=0, delay_max_lag=0,
            controllers={k: replace(v, output_limit=5.) for k, v in hold_controllers().items()})
        r = MotorSimulation(cfg)
        q = r.robot.data.default_joint_pos.clone()
        v = torch.zeros_like(q)
        i = K1_JOINT_ORDER.index("Left_Hip_Roll")
        v[:, i] = (8.38 + 12.15) / 2
        r.robot.write_joint_state_to_sim(q, v)
        r.sim.forward()
        r.scene.update(r.physics_dt)
        r.step_physics()
        aid = r.model.actuator("robot/Left_Hip_Roll").id
        self.assertAlmostEqual(r.sim.model.actuator_forcerange[0, aid, 1].item(), 5.)
        self.assertAlmostEqual(abs(r.sim.data.actuator_force[0, aid].item()), 5., places=4)

    def test_missing_startup_binding_fails(self):
        r = self.runtime
        saved = r.robot._motor_model
        r.robot._motor_model = None
        try:
            with self.assertRaisesRegex(RuntimeError, "initialize_motor_limits"):
                r.step_physics()
        finally:
            r.robot._motor_model = saved

    def test_decimation_and_viewer_copy_use_the_actual_simulation(self):
        r = self.runtime
        r.step()
        self.assertEqual((r.physics_steps, r.policy_steps), (4, 1))
        self.assertAlmostEqual(r.data.time, .02, places=6)
        np.testing.assert_allclose(r.data.qpos, r.sim.data.qpos[0].cpu().numpy())
        r.reset()
        self.assertEqual((r.physics_steps, r.policy_steps, r.data.time), (0, 0, 0))

    def test_full_completion_criteria_use_shared_mjlab_code(self):
        report = check_motors(hold_seconds=1.)
        self.assertTrue(report["passed"], report["errors"])

    def test_export_and_cli_use_the_shared_model(self):
        r = self.runtime
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controlled.zip"
            r.scene.write(path.with_suffix(""), zip=True)
            with path.open("rb") as f:
                model = mujoco.MjSpec.from_zip(f).compile()
            self.assertEqual(model.nu, 22)
            for name, angle in zip(K1_JOINT_ORDER, r.initial_target):
                self.assertAlmostEqual(model.key("init_state").ctrl[model.actuator("robot/" + name).id], angle, places=6)
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["--headless"]), 0)
        report = json.loads(output.getvalue())
        self.assertEqual(report["command_delay"]["hold_prob"], .3)
        self.assertTrue(report["checks"]["builtin_actuators"])


    def test_actual_rl_environment_uses_the_same_configuration(self):
        from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
        cfg = make_goalkeeper_env_cfg(num_envs=2)
        self.assertIs(type(cfg), ManagerBasedRlEnvCfg)
        cfg.episode_length_s = 10.0  # 이 검사의 길이이며, 실제 학습 에피소드 설계는 후속 구현이다.
        env = ManagerBasedRlEnv(cfg, device="cpu")
        try:
            env.reset()
            robot = env.scene["robot"]
            robot.set_joint_position_target(robot.data.default_joint_pos)
            env.step(torch.empty((2, 0)))  # 아직 action/관측/보상은 비어 있다. PPO 검사가 아니다.
            self.assertIsNotNone(robot._motor_limits)
            self.assertTrue(torch.isfinite(env.sim.data.qpos).all())
            self.assertEqual(len({id(a._delay_buffer) for a in robot.actuators}), 1)
            limit = env.sim.model.actuator_forcerange[..., 1]
            self.assertTrue((env.sim.data.actuator_force.abs() <= limit + 1e-4).all())
        finally:
            env.close()

    def test_invalid_inputs_are_rejected(self):
        for kwargs in ({"delay_min_lag": -1}, {"delay_min_lag": 9}, {"delay_hold_prob": 1.1}, {"controllers": {}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_goalkeeper_env_cfg(**kwargs)
        for target in (np.zeros(21), np.full(22, np.nan), np.full(22, 100.)):
            with self.assertRaises(ValueError):
                self.runtime.set_target(target)


if __name__ == "__main__":
    unittest.main()
