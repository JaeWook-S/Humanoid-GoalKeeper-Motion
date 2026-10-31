"""기본 실행은 MuJoCo viewer, --headless는 화면 없이 모델 검사만 수행한다."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct
import sys
import time
import zlib

import mujoco
import numpy as np

from goalkeeper_mjlab import PACKAGE_ROOT
from goalkeeper_mjlab.environment.ball.ball_constants import BallCfg
from goalkeeper_mjlab.environment.goal.goal_constants import GoalCfg
from goalkeeper_mjlab.inspection import inspect_scene
from goalkeeper_mjlab.robots.k1.constants import RobotPoseCfg
from goalkeeper_mjlab.tasks.goalkeeper.config.k1.scene_cfg import (
    Stage1SceneCfg, compile_scene, make_mjlab_scene_cfg,
)


def _ensure_viewer_runtime(argv: list[str]) -> None:
    """macOS에서 일반 python으로 시작해도 Cocoa GUI용 mjpython으로 자동 전환한다."""
    if sys.platform != "darwin":
        return
    import mujoco.viewer as viewer
    # MuJoCo의 macOS 런처가 초기화하는 객체다. 이미 전환했다면 재실행하지 않는다.
    if getattr(viewer, "_MJPYTHON", None) is not None:
        return
    launcher = Path(sys.executable).with_name("mjpython")
    if not launcher.is_file():
        raise RuntimeError("The current Python environment must contain MuJoCo's mjpython launcher")
    env = os.environ.copy()
    # main.py가 임시로 추가한 src 경로도 새 프로세스에 전달한다.
    paths = [str(PACKAGE_ROOT.parent)]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    # 런처를 현재 Python으로 읽어 이전 venv 경로가 남은 shebang도 우회한다.
    os.execve(sys.executable, [sys.executable, str(launcher), "-m",
                             "goalkeeper_mjlab.scripts.inspect_scene", *argv], env)


def _write_png(path: Path, pixels: np.ndarray) -> None:
    """검사용 RGB PNG 저장. 기존 환경에 Pillow 같은 추가 패키지를 요구하지 않는다."""
    height, width, channels = pixels.shape
    if channels != 3 or pixels.dtype != np.uint8:
        raise ValueError("Expected an RGB uint8 image")
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    raw = b"".join(b"\x00" + row.tobytes() for row in pixels)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def render_snapshot(model: mujoco.MjModel, data: mujoco.MjData, path: Path,
                    origin=(0.0, 0.0, 0.0), collisions: bool = False) -> None:
    camera = mujoco.MjvCamera()
    camera.lookat = np.asarray(origin) + (0.4, 0.0, 0.5)
    camera.distance, camera.azimuth, camera.elevation = 4.8, 135.0, -18.0
    option = mujoco.MjvOption()
    if collisions:
        option.geomgroup[:] = 0
        option.geomgroup[0] = option.geomgroup[3] = 1
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=option)
        _write_png(path, renderer.render())


def view_scene(model: mujoco.MjModel, data: mujoco.MjData, collisions: bool = False, simulation=None) -> None:
    import mujoco.viewer
    state = {"simulate": simulation is not None, "reset": False, "head": False, "head_index": 0}
    def key_callback(key: int) -> None:
        if key in (ord("P"), ord("p")):
            state["simulate"] = not state["simulate"]
        elif key in (ord("R"), ord("r")):
            state["reset"] = True
        elif simulation is not None and key in (ord("H"), ord("h")):
            state["head"] = True
    # 2단계는 PD로 서 있는 상태를 보여준다. 1단계의 수동 모델은 정지해서 보여준다.
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        if collisions:
            with viewer.lock():
                viewer.opt.geomgroup[:] = 0
                viewer.opt.geomgroup[0] = viewer.opt.geomgroup[3] = 1
        viewer.cam.lookat = data.xpos[model.body("robot/Trunk").id] + (0.4, 0.0, 0.0)
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 4.8, 135., -18.
        while viewer.is_running():
            started = time.monotonic()
            if state["reset"]:
                if simulation is not None:
                    simulation.reset()
                else:
                    mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
                state.update(reset=False, simulate=simulation is not None, head_index=0)
            if state["head"]:
                from goalkeeper_mjlab.robots.k1.constants import K1_JOINT_ORDER
                target = simulation.target.copy()
                state["head_index"] = (state["head_index"] + 1) % 3
                target[K1_JOINT_ORDER.index("Head_Yaw")] = (0.0, 0.2, -0.2)[state["head_index"]]
                simulation.set_target(target)
                state["head"] = False
            if state["simulate"]:
                if simulation is not None:
                    simulation.step()
                else:
                    mujoco.mj_step(model, data)
            else:
                mujoco.mj_forward(model, data)
            viewer.sync()
            interval = simulation.cfg.execution.policy_dt if simulation is not None else model.opt.timestep
            time.sleep(max(0, interval - (time.monotonic() - started)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="K1 goalkeeper scene: stage 2 controlled motors / stage 1 passive model")
    parser.add_argument("--stage", type=int, choices=(1, 2), default=2, help="Default: 2 with motors; 1 reproduces the passive geometry stage")
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--viewer", dest="viewer", action="store_true", help="Open viewer (default); P: pause, R: reset, H: head target in stage 2")
    display.add_argument("--headless", dest="viewer", action="store_false", help="Check/export the scene without opening a viewer")
    parser.set_defaults(viewer=True)
    parser.add_argument("--render", type=Path, help="Save a 640x480 snapshot")
    parser.add_argument("--collisions", action="store_true", help="Render collision shapes instead of visual meshes")
    parser.add_argument("--export", type=Path, help="Export a self-contained MuJoCo .zip scene")
    parser.add_argument("--report", type=Path, help="Optionally save the same JSON report printed to stdout")
    parser.add_argument("--mjlab-check", action="store_true", help="Also compile the MJLab adapter in the current training environment")
    # main.py로 실행할 때의 공 기본값이다. 직접 지정한 반지름/질량이 호수별 기본값보다 우선한다.
    # 반지름은 m(0.10 = 지름 20 cm), 질량은 kg(0.36 = 360 g)으로 입력한다.
    parser.add_argument("--ball-size", type=int, choices=range(1, 6), default=4)
    parser.add_argument("--ball-radius", type=float)
    parser.add_argument("--ball-mass", type=float)
    parser.add_argument("--ball-x", type=float, default=2.)
    parser.add_argument("--ball-y", type=float, default=0.)
    parser.add_argument("--ball-z", type=float)
    # 골대 크기는 전체 확대 비율로 고른다. small 약 1.8x1.2 m, mid 2.4x1.6 m, large 3.0x2.0 m(외곽 기준).
    parser.add_argument("--goal-division", choices=("small", "mid", "large"), default="mid")
    parser.add_argument("--goal-distance", type=float, default=0.75)
    parser.add_argument("--origin", type=float, nargs=3, default=(0., 0., 0.))
    parser.add_argument("--joint", action="append", default=[], metavar="NAME=RAD")
    parser.add_argument("--target", action="append", default=[], metavar="NAME=RAD", help="Stage-2 motor target, applied through PD and command delay")
    parser.add_argument("--simulate-seconds", type=float, default=0., help="Run physical simulation before rendering or viewing")
    parser.add_argument("--check-motors", action="store_true", help="Stage-2 free hold, target tracking, actual torque and delay tests")
    parser.add_argument("--hold-seconds", type=float, default=10., help="Hold test duration; minimum 1 second")
    parser.add_argument("--motor-source", choices=("official", "legacy"), default="official")
    parser.add_argument("--physics-dt", type=float, default=0.005)
    parser.add_argument("--decimation", type=int, default=4)
    parser.add_argument("--delay-min", type=int, default=2, help="Command delay in physics steps")
    parser.add_argument("--delay-max", type=int, default=8)
    cli_args = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(cli_args)
    try:
        overrides = {}
        for value in args.joint:
            name, angle = value.split("=", 1)
            overrides[name] = float(angle)
        if not np.isfinite(args.simulate_seconds) or args.simulate_seconds < 0:
            raise ValueError("simulate-seconds must be finite and nonnegative")
        if args.export and args.export.suffix != ".zip":
            raise ValueError("export path must end in .zip so assets remain self-contained")
        if args.stage == 1 and (args.target or args.check_motors):
            raise ValueError("target/check-motors require --stage 2")
        from goalkeeper_mjlab.robots.k1.controller_profiles import READY_POSE
        pose = READY_POSE if args.stage == 2 else RobotPoseCfg()
        cfg = Stage1SceneCfg(
            robot_pose=pose.with_joint_overrides(overrides),
            ball=BallCfg(args.ball_size, args.ball_radius, args.ball_mass),
            goal=GoalCfg(args.goal_division),
            ball_position=(args.ball_x, args.ball_y, args.ball_z), goal_distance=args.goal_distance,
        )
        # 옵션이 없으면 화면을 연다. 서버 검사만 할 때는 --headless로 GUI 초기화를 생략한다.
        if args.viewer:
            _ensure_viewer_runtime(cli_args)
        simulation = None
        if args.stage == 2:
            from goalkeeper_mjlab.motor_runtime import MotorSimulation
            from goalkeeper_mjlab.motor_checks import check_motors, configuration_report
            from goalkeeper_mjlab.robots.k1.command_delay import CommandDelayCfg
            from goalkeeper_mjlab.robots.k1.constants import K1_JOINT_ORDER
            from goalkeeper_mjlab.tasks.goalkeeper.config.k1.motor_scene_cfg import Stage2SceneCfg
            from goalkeeper_mjlab.tasks.goalkeeper.config.k1.simulation_cfg import PhysicsControlCfg
            motor_cfg = Stage2SceneCfg(scene=cfg, motor_source=args.motor_source,
                execution=PhysicsControlCfg(physics_dt=args.physics_dt, decimation=args.decimation),
                delay=CommandDelayCfg(args.delay_min, args.delay_max))
            simulation = MotorSimulation(motor_cfg, tuple(args.origin))
            spec, model, data = simulation.spec, simulation.model, simulation.data
            report = check_motors(motor_cfg, args.hold_seconds, tuple(args.origin)) if args.check_motors else configuration_report(simulation)
            target = simulation.initial_target.copy()
            for value in args.target:
                name, angle = value.split("=", 1)
                if name not in K1_JOINT_ORDER:
                    raise ValueError(f"Unknown target joint: {name}")
                target[K1_JOINT_ORDER.index(name)] = float(angle)
            simulation.set_target(target)
        else:
            report = inspect_scene(cfg, tuple(args.origin))
            spec, model, data = compile_scene(cfg, tuple(args.origin))
        if args.mjlab_check:
            from mjlab.scene import Scene
            if args.stage == 2:
                from goalkeeper_mjlab.tasks.goalkeeper.config.k1.motor_scene_cfg import make_controlled_mjlab_scene_cfg
                scene_cfg = make_controlled_mjlab_scene_cfg(motor_cfg)
            else:
                scene_cfg = make_mjlab_scene_cfg(cfg)
            mjlab_model = Scene(scene_cfg, device="cpu").compile()
            actual = (mjlab_model.nq, mjlab_model.nv, mjlab_model.nu, mjlab_model.nmocap)
            report["mjlab_compile"] = {"passed": actual == (36, 34, 22 if args.stage == 2 else 0, 1), "dimensions": list(actual)}
            if not report["mjlab_compile"]["passed"]:
                report["errors"].append("MJLab scene dimensions differ from the native scene")
                report["passed"] = False
        if args.export:
            args.export.parent.mkdir(parents=True, exist_ok=True)
            with args.export.open("wb") as file:
                spec.to_zip(file)
        if simulation is not None and args.simulate_seconds:
            metrics = simulation.run(args.simulate_seconds)
            report["controlled_simulation"] = metrics
            if not metrics["finite"] or metrics["maximum_torque_violation_nm"] > 1e-7:
                report["passed"] = False
                report["errors"].append("Controlled simulation produced invalid state or exceeded a torque limit")
        elif simulation is None:
            for _ in range(int(np.ceil(args.simulate_seconds / model.opt.timestep))):
                mujoco.mj_step(model, data)
        if args.simulate_seconds and simulation is None:
            finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
            report["passive_simulation"] = {"time_s": float(data.time), "finite": finite}
            if not finite:
                report["passed"] = False
                report["errors"].append("Passive simulation produced nonfinite state")
        if args.render:
            render_snapshot(model, data, args.render, args.origin, args.collisions)
        output = json.dumps(report, ensure_ascii=False, indent=2)
        print(output)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(output + "\n")
        if args.viewer:
            view_scene(model, data, args.collisions, simulation=simulation)
        return 0 if report["passed"] else 1
    except (ValueError, FileNotFoundError, RuntimeError, ImportError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
