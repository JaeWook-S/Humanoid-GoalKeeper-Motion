"""2단계 검증: 실제 MuJoCo의 출력·동역학을 확인하며, 서 있기 검사에서는 몸통을 고정하지 않는다."""

from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from goalkeeper_mjlab import PACKAGE_ROOT
from goalkeeper_mjlab.motor_checks import check_motors, configuration_report, torque_probe
from goalkeeper_mjlab.motor_runtime import MotorSimulation
from goalkeeper_mjlab.robots.k1.command_delay import CommandDelay, CommandDelayCfg, JOINT_DELAY_GROUP_IDS
from goalkeeper_mjlab.robots.k1.constants import K1_JOINT_ORDER
from goalkeeper_mjlab.robots.k1.controller_profiles import ControllerParameters, READY_JOINT_POS, official_reference_controllers
from goalkeeper_mjlab.robots.k1.motor_parameters import OFFICIAL_COMMIT, MotorParameters, get_motors, motor_group
from goalkeeper_mjlab.robots.k1.torque_speed import JointParameters
from goalkeeper_mjlab.scripts.inspect_scene import main
from goalkeeper_mjlab.tasks.goalkeeper.config.k1.motor_scene_cfg import Stage2SceneCfg
from goalkeeper_mjlab.tasks.goalkeeper.config.k1.simulation_cfg import PhysicsControlCfg


class Stage2Tests(unittest.TestCase):
    def test_motor_values_match_independent_official_snapshot(self):
        reference = json.loads((PACKAGE_ROOT / "robots/k1/motor_reference.json").read_text())
        self.assertEqual(reference["official_commit"], OFFICIAL_COMMIT)
        for group, motor in get_motors().items():
            expected = reference["official_motors"][group]
            for attr, key in (("armature", "armature"), ("peak_effort", "effort_limit"),
                              ("velocity_limit", "velocity_limit"), ("knee_point_velocity", "knee_point_velocity")):
                self.assertAlmostEqual(getattr(motor, attr), expected[key], places=12)

    def test_hip_roll_and_ankle_differences_are_explicit(self):
        current, legacy = get_motors(), get_motors("legacy")
        self.assertEqual((current["hip_roll"].peak_effort, current["hip_roll"].velocity_limit, current["hip_roll"].knee_point_velocity), (43., 12.15, 8.38))
        self.assertEqual((legacy["hip_roll"].peak_effort, legacy["hip_roll"].velocity_limit, legacy["hip_roll"].knee_point_velocity), (76., 12.57, 2.62))
        self.assertAlmostEqual(current["ankle_pitch"].armature, .0282528 * 1.4)
        self.assertAlmostEqual(current["ankle_roll"].armature, .0282528 * .4)
        self.assertAlmostEqual(legacy["ankle_roll"].armature, .0282528 * 2)

    def test_head_source_knee_is_preserved_and_effective_knee_is_bounded(self):
        head = get_motors()["head"]
        self.assertEqual(head.knee_point_velocity, 10.47)
        self.assertEqual(head.effective_knee_point_velocity, 7.85)

    def test_real_model_contains_all_22_motors_with_physical_parameters(self):
        sim = MotorSimulation()
        self.assertEqual((sim.model.nq, sim.model.nv, sim.model.nu, sim.model.nmocap), (36, 34, 22, 1))
        ref = json.loads((PACKAGE_ROOT / "robots/k1/motor_reference.json").read_text())["official_motors"]
        expected = [ref[motor_group(n)]["armature"] for n in K1_JOINT_ORDER]
        np.testing.assert_allclose(sim.model.dof_armature[sim.dof_ids], expected, rtol=0, atol=1e-12)
        self.assertTrue(configuration_report(sim)["passed"])
        # ctrl은 목표각이다. N*m 한계는 ctrlrange가 아니라 실제 actuator force에 적용한다.
        self.assertFalse(sim.model.actuator_ctrllimited[sim.ctrl_ids].any())
        self.assertTrue(sim.model.actuator_forcelimited[sim.ctrl_ids].all())
        np.testing.assert_allclose(sim.model.key("init_state").ctrl[sim.ctrl_ids], READY_JOINT_POS)

    def test_curve_flat_ramp_zero_and_velocity_symmetry(self):
        p = Stage2SceneCfg().parameters
        np.testing.assert_allclose(p.physical_limit(np.zeros(22)), p.peak)
        ramp = p.knee_speed < p.max_speed
        np.testing.assert_allclose(p.physical_limit((p.knee_speed + p.max_speed) / 2)[ramp], p.peak[ramp] / 2)
        np.testing.assert_allclose(p.physical_limit(p.max_speed), 0)
        np.testing.assert_allclose(p.physical_limit(1.2 * p.max_speed), 0)
        np.testing.assert_allclose(p.physical_limit(-.9 * p.max_speed), p.physical_limit(.9 * p.max_speed))

    def test_controller_cap_is_intersected_after_physical_curve(self):
        cfg = Stage2SceneCfg()
        controllers = {g: replace(c, output_limit=5.) for g, c in cfg.controllers.items()}
        p = JointParameters.build(get_motors(), controllers)
        index = K1_JOINT_ORDER.index("Left_Hip_Roll")
        velocity = (p.knee_speed + p.max_speed) / 2
        self.assertAlmostEqual(p.physical_limit(velocity)[index], 21.5)
        self.assertAlmostEqual(p.effective_limit(velocity)[index], 5.)  # 5*0.5=2.5로 줄이면 오류다.

    def test_engine_really_clips_all_motor_speed_cases(self):
        result = torque_probe(Stage2SceneCfg())
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["joint_cases"], 220)
        self.assertGreater(result["saturated_cases"], 100)

    def test_small_software_cap_is_applied_to_engine_forces(self):
        cfg = Stage2SceneCfg(delay=CommandDelayCfg(0, 0))
        cfg = replace(cfg, controllers={g: replace(c, output_limit=.5) for g, c in cfg.controllers.items()})
        sim = MotorSimulation(cfg)
        sim.set_target(sim.joint_limits[:, 1].copy())
        sim.step_physics()
        self.assertLessEqual(np.abs(sim.last_torque).max(), .5 + 1e-10)
        self.assertGreater(np.abs(sim.last_torque).max(), .49)

    def test_speed_ceiling_zeroes_motor_force_without_overwriting_velocity(self):
        sim = MotorSimulation(Stage2SceneCfg(delay=CommandDelayCfg(0, 0)))
        sim.model.opt.gravity[:] = 0
        sim.data.qpos[2] += 2
        i = K1_JOINT_ORDER.index("Left_Hip_Yaw")
        sim.data.qvel[sim.dof_ids[i]] = 20.
        sim.step_physics()
        self.assertAlmostEqual(sim.last_torque[i], 0.)
        self.assertGreater(sim.data.qvel[sim.dof_ids[i]], 17.59)

    def test_exact_command_delays_and_reset_history(self):
        initial = np.linspace(.01, .22, 22)
        command = initial + .1
        for lag in (0, 2, 8):
            with self.subTest(lag=lag):
                delay = CommandDelay(CommandDelayCfg(lag, lag), initial)
                for _ in range(lag):
                    np.testing.assert_allclose(delay.push(command), initial)
                np.testing.assert_allclose(delay.push(command), command)
                delay.reset(initial)
                np.testing.assert_allclose(delay.push(initial), initial)

    def test_delay_groups_share_lags_and_seed_reproduces(self):
        a = CommandDelay(CommandDelayCfg(), np.zeros(22), seed=17)
        b = CommandDelay(CommandDelayCfg(), np.zeros(22), seed=17)
        np.testing.assert_array_equal(a.joint_lags, b.joint_lags)
        for group in range(4):
            self.assertEqual(len(set(a.joint_lags[JOINT_DELAY_GROUP_IDS == group])), 1)
        self.assertTrue(((a.joint_lags >= 2) & (a.joint_lags <= 8)).all())

    def test_decimation_uses_one_held_target_for_exact_physics_steps(self):
        sim = MotorSimulation()
        official = json.loads((PACKAGE_ROOT / "robots/k1/motor_reference.json").read_text(encoding="utf-8"))["official_execution"]
        self.assertEqual(sim.cfg.execution.physics_dt, official["physics_dt"])
        self.assertEqual(sim.cfg.execution.decimation, official["decimation"])
        sim.step()
        self.assertEqual((sim.physics_steps, sim.policy_steps), (4, 1))
        self.assertAlmostEqual(sim.data.time, .02)
        sim.step()
        self.assertEqual((sim.physics_steps, sim.policy_steps), (8, 2))
        self.assertAlmostEqual(sim.data.time, .04)

    def test_reset_restores_pose_target_clock_and_pending_commands(self):
        sim = MotorSimulation()
        command = sim.target.copy()
        command[0] = .2
        sim.step(command)
        sim.reset()
        self.assertEqual((sim.physics_steps, sim.policy_steps, sim.data.time), (0, 0, 0))
        np.testing.assert_allclose(sim.data.qpos[sim.qpos_ids], READY_JOINT_POS)
        np.testing.assert_allclose(sim.data.ctrl[sim.ctrl_ids], READY_JOINT_POS)
        sim.step_physics()
        np.testing.assert_allclose(sim.last_delayed_target, READY_JOINT_POS)

    def test_all_22_joint_targets_track_in_an_unloaded_free_robot(self):
        # 관절 자체의 명령 추종을 중력/바닥 영향과 구분한다. 몸통 free joint는 그대로 둔다.
        sim = MotorSimulation(Stage2SceneCfg(delay=CommandDelayCfg(2, 2)))
        sim.model.opt.gravity[:] = 0
        sim.data.qpos[2] += 2
        target = np.clip(sim.target + .04, sim.joint_limits[:, 0] + .001, sim.joint_limits[:, 1] - .001)
        sim.set_target(target)
        metrics = sim.run(3.)
        self.assertTrue(metrics["finite"])
        np.testing.assert_allclose(sim.data.qpos[sim.qpos_ids], target, rtol=0, atol=.005)
        self.assertLess(metrics["maximum_torque_violation_nm"], 1e-7)

    def test_full_completion_criteria_in_a_floating_robot(self):
        result = check_motors()
        self.assertTrue(result["passed"], result["errors"])
        self.assertGreater(result["hold"]["minimum_root_height_m"], .5)
        self.assertLess(result["hold"]["maximum_tilt_deg"], 5.)
        self.assertTrue(all(row["passed"] for row in result["tracking"]))

    def test_official_reference_gains_are_separate_from_hold_gains(self):
        cfg = Stage2SceneCfg()
        reference = official_reference_controllers(get_motors())
        self.assertAlmostEqual(reference["hip_roll"].kp, .0339552 * (2 * 3.1415926535 * 4)**2)
        self.assertNotEqual(reference["hip_roll"].kp, cfg.controllers["hip_roll"].kp)
        self.assertEqual(cfg.controllers["hip_roll"].kp, 80.)

    def test_export_contains_initial_pd_target_and_22_actuators(self):
        sim = MotorSimulation()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controlled.zip"
            with path.open("wb") as file:
                sim.spec.to_zip(file)
            with path.open("rb") as file:
                model = mujoco.MjSpec.from_zip(file).compile()
            self.assertEqual(model.nu, 22)
            np.testing.assert_allclose(model.key("init_state").ctrl, READY_JOINT_POS, atol=1e-6)

    def test_invalid_physics_motor_controller_and_target_inputs(self):
        for action in (
            lambda: PhysicsControlCfg(physics_dt=0), lambda: PhysicsControlCfg(decimation=0),
            lambda: PhysicsControlCfg(decimation=True), lambda: PhysicsControlCfg(solver="invalid"),
            lambda: CommandDelayCfg(-1, 8), lambda: CommandDelayCfg(8, 2),
            lambda: MotorParameters(model_name="bad", armature=-1, peak_effort=1, velocity_limit=1, knee_point_velocity=0),
            lambda: ControllerParameters(kp=0, kd=1, output_limit=1), lambda: Stage2SceneCfg(motor_source="unknown"),
            lambda: Stage2SceneCfg(controllers={}),
        ):
            with self.subTest(action=action), self.assertRaises(ValueError):
                action()
        sim = MotorSimulation()
        for bad in (np.zeros(21), np.full(22, np.nan), np.full(22, 100.)):
            with self.subTest(target=bad), self.assertRaises(ValueError):
                sim.set_target(bad)

    def test_main_defaults_to_controlled_stage_and_can_reproduce_stage1(self):
        for args, stage, count in ((["--headless"], 2, 22), (["--stage", "1", "--headless"], 1, 0)):
            with self.subTest(stage=stage), redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(args), 0)
                result = json.loads(output.getvalue())
                self.assertEqual(result["model"]["nu"], count)
                if stage == 2:
                    self.assertEqual(result["stage"], 2)
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["--headless", "--origin", "3", "-2", ".4"]), 0)
            result = json.loads(output.getvalue())
            np.testing.assert_allclose(result["goal"]["position"], [2.25, -2., .4])
            self.assertEqual(result["origin"], [3., -2., .4])


if __name__ == "__main__":
    unittest.main()
