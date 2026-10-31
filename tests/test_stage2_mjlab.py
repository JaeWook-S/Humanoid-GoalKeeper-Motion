"""실제 MJLab/Warp 배치 실행 검사. MJLab 미설치 환경에서는 이 파일의 검사만 skip한다."""

from dataclasses import replace
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("mjlab"), "MJLab training dependencies are not installed")
class Stage2MJLabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from goalkeeper_mjlab.motor_runtime import BatchedMotorSimulation
        from goalkeeper_mjlab.robots.k1.command_delay import CommandDelayCfg
        from goalkeeper_mjlab.tasks.goalkeeper.config.k1.motor_scene_cfg import Stage2SceneCfg
        cls.torch = torch
        cfg = Stage2SceneCfg(delay=CommandDelayCfg(2, 2))
        cls.runtime = BatchedMotorSimulation(replace(cfg, scene=replace(cfg.scene, num_envs=2)), device="cpu")

    def setUp(self):
        self.runtime.reset()

    def test_real_batched_forces_have_independent_speed_limits(self):
        r, torch = self.runtime, self.torch
        q = r.robot.data.default_joint_pos.clone()
        velocity = torch.zeros_like(q)
        index = r.robot.joint_names.index("Left_Hip_Roll")
        velocity[1, index] = (8.38 + 12.15) / 2
        r.robot.write_joint_state_to_sim(q, velocity)
        r.sim.forward()
        r.scene.update(r.cfg.execution.physics_dt)
        r.scene.write_data_to_sim()
        aid = r.sim.mj_model.actuator("robot/Left_Hip_Roll").id
        limits = r.sim.model.actuator_forcerange[:, aid, 1]
        torch.testing.assert_close(limits, torch.tensor([43., 21.5]), atol=1e-4, rtol=1e-5)
        r.sim.step()
        dof = int(r.sim.mj_model.joint("robot/Left_Hip_Roll").dofadr[0])
        self.assertLessEqual(abs(r.sim.data.qfrc_actuator[1, dof].item()), 21.5 + 1e-4)
        self.assertGreater(abs(r.sim.data.qfrc_actuator[1, dof].item()), 20.)

    def test_delayed_targets_and_partial_reset_do_not_leak(self):
        r, torch = self.runtime, self.torch
        target = r.robot.data.default_joint_pos.clone()
        head = r.robot.joint_names.index("Head_Yaw")
        target[:, head] = torch.tensor([.2, -.2])
        r.robot.set_joint_position_target(target)
        aid = r.sim.mj_model.actuator("robot/Head_Yaw").id
        for expected in ([0., 0.], [0., 0.], [.2, -.2]):
            r.step_physics()
            torch.testing.assert_close(r.sim.data.ctrl[:, aid], torch.tensor(expected), atol=1e-6, rtol=0)
        r.reset(torch.tensor([0]))
        r.step_physics()
        torch.testing.assert_close(r.sim.data.ctrl[:, aid], torch.tensor([0., -.2]), atol=1e-6, rtol=0)

    def test_free_stance_and_actual_torque_bounds_over_one_second(self):
        r, torch = self.runtime, self.torch
        for _ in range(50):
            r.step()
            for actuator in r.robot.actuators:
                actual = r.sim.data.actuator_force[:, actuator.global_ctrl_ids]
                self.assertTrue(torch.all(actual.abs() <= actuator.last_limit + 1e-4).item())
        self.assertTrue(torch.isfinite(r.sim.data.qpos).all().item())
        self.assertTrue(torch.all(r.robot.data.root_link_pos_w[:, 2] > .5).item())

    def test_missing_startup_binding_fails_instead_of_skipping_limits(self):
        r = self.runtime
        actuator = r.robot.actuators[0]
        bridge = actuator._motor_model
        actuator._motor_model = None
        try:
            with self.assertRaisesRegex(RuntimeError, "bind_motor_limits"):
                r.scene.write_data_to_sim()
        finally:
            actuator._motor_model = bridge

    def test_simulation_configuration_and_origins_are_applied(self):
        r, torch = self.runtime, self.torch
        self.assertEqual(tuple(r.sim.model.actuator_forcerange.shape), (2, 22, 2))
        torch.testing.assert_close(r.robot.data.root_link_pos_w[:, :2], r.scene.env_origins[:, :2])
        self.assertEqual(r.sim.mj_model.opt.timestep, .005)
        self.assertEqual(r.cfg.execution.decimation, 4)


if __name__ == "__main__":
    unittest.main()
