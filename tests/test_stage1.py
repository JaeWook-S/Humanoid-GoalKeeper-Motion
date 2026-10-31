"""1단계의 실제 MuJoCo 모델·접촉·자산 패키징 회귀 검사. unittest만 사용한다."""

from dataclasses import replace
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import mujoco
import numpy as np

from goalkeeper_mjlab import PACKAGE_ROOT
from goalkeeper_mjlab.assets import validate_model_assets, verify_source_assets
from goalkeeper_mjlab.environment.ball.ball_constants import BallCfg, get_spec as ball_spec
from goalkeeper_mjlab.environment.goal.goal_constants import GoalCfg, get_opening, get_spec as goal_spec
from goalkeeper_mjlab.inspection import inspect_scene
from goalkeeper_mjlab.robots.k1.constants import K1_XML, RobotPoseCfg, get_spec as robot_spec
from goalkeeper_mjlab.scripts.inspect_scene import main
from goalkeeper_mjlab.tasks.goalkeeper.config.k1.scene_cfg import Stage1SceneCfg, compile_scene, make_mjlab_scene_cfg


class Stage1Tests(unittest.TestCase):
    def test_native_scene_contract_and_contacts(self):
        report = inspect_scene()
        self.assertTrue(report["passed"], report["errors"])
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(report["model"], {"nq": 36, "nv": 34, "nu": 0, "nmocap": 1})

    def test_world_translation_preserves_relative_placement(self):
        # 지면 높이까지 바꿔 원점 처리가 XY에만 적용되는 실수를 잡는다.
        first = inspect_scene()
        origin = np.array([3.0, -2.0, 0.4])
        second = inspect_scene(origin=tuple(origin))
        self.assertTrue(second["passed"], second["errors"])
        for name, field in (("robot", "root_position"), ("ball", "position"), ("goal", "position")):
            np.testing.assert_allclose(np.array(second[name][field]) - origin, first[name][field], atol=1e-8)
        np.testing.assert_allclose(first["robot"]["sole_bottoms_above_floor_m"], second["robot"]["sole_bottoms_above_floor_m"], atol=1e-8)
        _, model, data = compile_scene(origin=tuple(origin))
        np.testing.assert_allclose(data.site_xpos[model.site("goal/goal_line_center").id],
                                   origin + (-0.75, 0., get_opening().inner_height / 2))

    def test_ball_mass_radius_and_shell_inertia_across_sizes(self):
        for size in range(1, 6):
            with self.subTest(size=size):
                cfg = BallCfg(size)
                model = ball_spec(cfg).compile()
                bid = model.body("ball").id
                self.assertAlmostEqual(model.body_mass[bid], cfg.mass)
                np.testing.assert_allclose(model.body_inertia[bid], 2 / 3 * cfg.mass * cfg.radius**2)
                self.assertAlmostEqual(model.geom_size[model.geom("ball_collision").id, 0], cfg.radius)
        custom = BallCfg(4, radius_override=0.09, mass_override=0.31)
        model = ball_spec(custom).compile()
        self.assertAlmostEqual(model.body_mass[model.body("ball").id], 0.31)

    def test_goal_visual_and_collision_scaling(self):
        base = get_opening()
        for division, scale in (("small", 1.), ("mid", 240 / 180), ("large", 300 / 180)):
            cfg = GoalCfg(division)
            spec = goal_spec(cfg)
            opening = get_opening(cfg)
            self.assertAlmostEqual(opening.inner_width, base.inner_width * scale)
            self.assertAlmostEqual(opening.inner_height, base.inner_height * scale)
            self.assertTrue(spec.worldbody.bodies[0].mocap)
            self.assertTrue(inspect_scene(replace(Stage1SceneCfg(), goal=cfg))["passed"])

    def test_model_specs_are_fresh_and_do_not_share_edits(self):
        first, second = robot_spec(), robot_spec()
        first.joint("Head_Yaw").range = (-0.1, 0.1)
        self.assertNotEqual(tuple(first.joint("Head_Yaw").range), tuple(second.joint("Head_Yaw").range))
        first_ball, second_ball = ball_spec(), ball_spec()
        first_ball.geom("ball_collision").size = (0.5, 0, 0)
        self.assertNotEqual(tuple(first_ball.geom("ball_collision").size), tuple(second_ball.geom("ball_collision").size))

    def test_pose_and_scene_inputs_reject_invalid_values(self):
        for action in (
            lambda: RobotPoseCfg(joint_pos=(0.,)),
            lambda: RobotPoseCfg().with_joint_overrides({"unknown_joint": 0.}),
            lambda: compile_scene(replace(Stage1SceneCfg(), robot_pose=RobotPoseCfg().with_joint_overrides({"Head_Yaw": 100.}))),
            lambda: BallCfg(4.0), lambda: BallCfg(True), lambda: BallCfg(0),
            lambda: BallCfg(radius_override=-0.1), lambda: BallCfg(mass_override=np.nan),
            lambda: Stage1SceneCfg(ball_position=(0., 0., None)),
            lambda: Stage1SceneCfg(ball_position=(2., 0., 0.01)),
            lambda: Stage1SceneCfg(goal_distance=np.inf), lambda: Stage1SceneCfg(num_envs=0),
        ):
            with self.subTest(action=action):
                with self.assertRaises(ValueError):
                    action()

    def test_source_hashes_and_missing_asset_detection(self):
        self.assertFalse(verify_source_assets()["mismatches"])
        self.assertEqual(len(validate_model_assets(K1_XML)), 26)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            xml = root / "missing.xml"
            xml.write_text('<mujoco><asset><mesh file="not-present.stl"/></asset></mujoco>')
            with self.assertRaises(FileNotFoundError):
                validate_model_assets(xml)
            (root / "asset_manifest.json").write_text(json.dumps({
                "source_commit": "test", "files": {"missing.stl": {"sha256": "not-a-real-hash"}},
            }))
            with patch("goalkeeper_mjlab.assets.PACKAGE_ROOT", root):
                self.assertEqual(verify_source_assets()["mismatches"], ["missing.stl"])

    def test_export_contains_assets_and_round_trips(self):
        spec, model, data = compile_scene()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.zip"
            with path.open("wb") as file:
                spec.to_zip(file)
            with path.open("rb") as file:
                loaded = mujoco.MjSpec.from_zip(file).compile()
            self.assertEqual((loaded.nq, loaded.nv, loaded.nmesh), (model.nq, model.nv, model.nmesh))
            # MuJoCo XML 저장은 수치 문자열을 반올림한다. 원본 런타임 상태와 1 μm 이내로 비교한다.
            np.testing.assert_allclose(loaded.key("init_state").qpos, data.qpos, atol=1e-6)

    def test_passive_ball_fall_reaches_floor_without_nonfinite_state(self):
        # 슈팅 태스크가 아니라 공/바닥의 기본 접촉 검사다.
        _, model, data = compile_scene(replace(Stage1SceneCfg(), ball_position=(2., 0., 0.7)))
        for _ in range(300):
            mujoco.mj_step(model, data)
        self.assertTrue(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
        ball_z = data.xpos[model.body("ball/ball").id, 2]
        self.assertLess(ball_z, 0.7)
        self.assertGreater(ball_z, 0.04)

    def test_cli_runs_from_an_unrelated_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            env = {**os.environ, "PYTHONPATH": str(PACKAGE_ROOT.parent), "PYTHONDONTWRITEBYTECODE": "1"}
            completed = subprocess.run([sys.executable, "-m", "goalkeeper_mjlab.scripts.inspect_scene", "--headless", "--report", str(report)],
                                       cwd=directory, env=env, text=True, capture_output=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(json.loads(report.read_text())["passed"])

    def test_default_execution_opens_viewer_and_headless_skips_it(self):
        # 모델은 실제로 컴파일하고, 테스트 중 GUI만 열지 않도록 경계 함수를 대체한다.
        for args, opens_viewer in (([], True), (["--viewer"], True), (["--headless"], False)):
            with self.subTest(args=args), \
                 patch("goalkeeper_mjlab.scripts.inspect_scene._ensure_viewer_runtime") as runtime, \
                 patch("goalkeeper_mjlab.scripts.inspect_scene.view_scene") as viewer, \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
                self.assertEqual(runtime.call_count, int(opens_viewer))
                self.assertEqual(viewer.call_count, int(opens_viewer))

    def test_main_file_runs_without_installation_or_pythonpath(self):
        entrypoint = PACKAGE_ROOT.parent.parent / "main.py"
        with tempfile.TemporaryDirectory() as directory:
            env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            completed = subprocess.run([sys.executable, str(entrypoint), "--headless"],
                                       cwd=directory, env=env, text=True, capture_output=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(json.loads(completed.stdout)["passed"])

    @unittest.skipUnless(importlib.util.find_spec("mjlab"), "MJLab is not installed in this interpreter")
    def test_mjlab_scene_compiles_with_matching_initial_state(self):
        from mjlab.scene import Scene
        cfg = Stage1SceneCfg()
        scene = Scene(make_mjlab_scene_cfg(cfg), device="cpu")
        model = scene.compile()
        _, native, data = compile_scene(cfg)
        self.assertEqual((model.nq, model.nv, model.nu, model.nmocap), (native.nq, native.nv, native.nu, native.nmocap))
        np.testing.assert_allclose(model.key("init_state").qpos, data.qpos)


if __name__ == "__main__":
    unittest.main()
