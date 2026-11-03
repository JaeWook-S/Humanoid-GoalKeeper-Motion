"""화면 실행·검증 전용 CLI. 기본 실행은 MuJoCo viewer, --headless는 화면 없는 검사다.

PPO 학습을 시작하는 스크립트가 아니다.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
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
from goalkeeper_mjlab.validation.geometry import inspect_geometry, compile_geometry_scene, make_geometry_env_cfg, make_geometry_scene_cfg
from goalkeeper_mjlab.robots.k1.constants import K1_JOINT_ORDER, HOME_POSE, get_k1_robot_cfg
from goalkeeper_mjlab.tasks.goalkeeper.config.k1.env_cfgs import goalkeeper_k1_env_cfg


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
    # shots 모드도 같은 viewer를 쓴다. 에피소드 종료/새 공 발사는 실제 MJLab 환경이 처리한다.
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
                # 병렬 슛 모드는 [N,22], 기존 단일 모터 검사는 [22] 목표각을 사용한다.
                target[..., K1_JOINT_ORDER.index("Head_Yaw")] = (0.0, 0.2, -0.2)[state["head_index"]]
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
            interval = simulation.policy_dt if simulation is not None else model.opt.timestep
            time.sleep(max(0, interval - (time.monotonic() - started)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="K1 goalkeeper validation: motors, geometry or random shots")
    parser.add_argument("--mode", choices=("motors", "geometry", "shots"), default="motors", help="motors (default), static geometry, or repeated random shots")
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--viewer", dest="viewer", action="store_true", help="Open viewer (default); P: pause, R: reset, H: head target in motor mode")
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
    # main.py 실행은 이 값이 환경 함수의 기본값(0.75 m)보다 우선한다.
    # 로봇 기준점 x=0에서 뒤쪽 골라인 x=-거리까지의 수평 거리(m)다.
    parser.add_argument("--goal-distance", type=float, default=0.5)
    parser.add_argument("--origin", type=float, nargs=3, default=(0., 0., 0.))
    parser.add_argument("--joint", action="append", default=[], metavar="NAME=RAD")
    parser.add_argument("--target", action="append", default=[], metavar="NAME=RAD", help="Motor target, applied through PD and command delay")
    parser.add_argument("--simulate-seconds", type=float, default=0., help="Run physical simulation before rendering or viewing")
    parser.add_argument("--check-motors", action="store_true", help="Free hold, target tracking, actual torque and delay tests")
    parser.add_argument("--hold-seconds", type=float, default=10., help="Hold test duration; minimum 1 second")
    parser.add_argument("--motor-source", choices=("official", "legacy"), default="official")
    parser.add_argument("--physics-dt", type=float, help="Default: 0.005 s in motors mode, 0.002 s in geometry mode")
    parser.add_argument("--decimation", type=int, default=4)
    parser.add_argument("--delay-min", type=int, default=2, help="Command delay in physics steps")
    parser.add_argument("--delay-max", type=int, default=8)
    parser.add_argument("--delay-hold-prob", type=float, default=0.3, help="Probability of keeping the current MJLab command lag")
    # 아래 범위는 초기 위치가 아니라 매 reset 때 새로 뽑는 슛의 분포다. 거리 m, 속도 m/s.
    parser.add_argument("--shots", type=int, help="Minimum completed shots across all environments; headless default 10, viewer default continuous")
    parser.add_argument("--num-envs", type=int, default=1, help="Parallel shot environments; viewer displays environment 0")
    parser.add_argument("--device", default="cpu", help="Shot simulation device: cpu or cuda:0 on a supported server")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ground-prob", type=float, default=0.5)
    parser.add_argument("--shot-speed", type=float, nargs=2, default=(3., 8.), metavar=("MIN", "MAX"))
    parser.add_argument("--shot-distance", type=float, nargs=2, default=(1.5, 3.), metavar=("MIN", "MAX"))
    parser.add_argument("--shot-y", type=float, nargs=2, default=(-1., 1.), metavar=("MIN", "MAX"))
    parser.add_argument("--air-launch-height", type=float, nargs=2, default=(.2, .6), metavar=("MIN", "MAX"))
    parser.add_argument("--target-y-fraction", type=float, nargs=2, default=(-.85, .85), metavar=("MIN", "MAX"))
    parser.add_argument("--air-target-height-fraction", type=float, nargs=2, default=(.05, .85), metavar=("MIN", "MAX"))
    parser.add_argument("--episode-seconds", type=float, default=3.)
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
        if args.mode == "geometry" and args.target:
            raise ValueError("target requires a controlled mode")
        if args.mode != "motors" and args.check_motors:
            raise ValueError("check-motors requires --mode motors")
        if args.mode != "shots" and (args.shots is not None or args.num_envs != 1 or args.device != "cpu"):
            raise ValueError("shots/num-envs/device options require --mode shots")
        if args.shots is not None and args.shots < 1:
            raise ValueError("shots must be positive")
        if args.physics_dt is not None and (not np.isfinite(args.physics_dt) or args.physics_dt <= 0):
            raise ValueError("physics-dt must be finite and positive")
        if args.decimation < 1 or args.seed < 0:
            raise ValueError("decimation must be positive and seed must be nonnegative")
        pose = HOME_POSE
        factory = make_geometry_env_cfg if args.mode == "geometry" else goalkeeper_k1_env_cfg
        shot_options = {}
        if args.mode == "shots":
            from goalkeeper_mjlab.tasks.goalkeeper.mdp.events import ShotCfg
            shot_options = dict(num_envs=args.num_envs, shots=ShotCfg(
                ground_probability=args.ground_prob, speed=tuple(args.shot_speed),
                launch_distance=tuple(args.shot_distance), launch_y=tuple(args.shot_y),
                air_launch_height=tuple(args.air_launch_height),
                target_y_fraction=tuple(args.target_y_fraction),
                air_target_height_fraction=tuple(args.air_target_height_fraction),
                episode_timeout_s=args.episode_seconds,
            ))
        cfg = factory(
            ball=BallCfg(args.ball_size, args.ball_radius, args.ball_mass),
            goal=GoalCfg(args.goal_division),
            ball_position=(args.ball_x, args.ball_y, args.ball_z), goal_distance=args.goal_distance,
            **shot_options,
        )
        # 학습 환경 함수에는 검사 전용 옵션을 노출하지 않는다.
        # wirlwind처럼 반환된 기본 MJLab 설정을 수정한다. CLI에서 비교 검사할 때만 바꾸는 값이다.
        cfg.scene.entities["robot"] = get_k1_robot_cfg(
            pose=pose.with_joint_overrides(overrides), motor_source=args.motor_source,
            delay_min_lag=args.delay_min, delay_max_lag=args.delay_max,
            delay_hold_prob=args.delay_hold_prob,
        )
        if args.physics_dt is not None:
            cfg.sim.mujoco.timestep = args.physics_dt
        cfg.decimation = args.decimation
        cfg.seed = args.seed
        # 옵션이 없으면 화면을 연다. 서버 검사만 할 때는 --headless로 GUI 초기화를 생략한다.
        if args.viewer:
            _ensure_viewer_runtime(cli_args)
        simulation = None
        if args.mode == "motors":
            from goalkeeper_mjlab.validation.motor_runtime import MotorSimulation
            from goalkeeper_mjlab.validation.motor_checks import check_motors, configuration_report
            # Warp 초기화 메시지는 stderr에 두어 stdout의 JSON 보고서를 유지한다.
            with redirect_stdout(sys.stderr):
                simulation = MotorSimulation(cfg, tuple(args.origin))
            model, data = simulation.model, simulation.data
            with redirect_stdout(sys.stderr):
                report = check_motors(cfg, args.hold_seconds, tuple(args.origin)) if args.check_motors else configuration_report(simulation)
            target = simulation.initial_target.copy()
            for value in args.target:
                name, angle = value.split("=", 1)
                if name not in K1_JOINT_ORDER:
                    raise ValueError(f"Unknown target joint: {name}")
                target[K1_JOINT_ORDER.index(name)] = float(angle)
            simulation.set_target(target)
        elif args.mode == "shots":
            from goalkeeper_mjlab.validation.shots import ShotSimulation
            with redirect_stdout(sys.stderr):
                simulation = ShotSimulation(cfg, tuple(args.origin), device=args.device)
            model, data = simulation.model, simulation.data
            target = simulation.initial_target.copy()
            for value in args.target:
                name, angle = value.split("=", 1)
                if name not in K1_JOINT_ORDER:
                    raise ValueError(f"Unknown target joint: {name}")
                target[..., K1_JOINT_ORDER.index(name)] = float(angle)
            simulation.set_target(target)
            with redirect_stdout(sys.stderr):
                # 화면에서는 바로 시작한다. 서버에서는 완료된 에피소드를 반복 수집한다.
                report = simulation.run_episodes(args.shots or 10) if (args.shots is not None or not args.viewer) else simulation.run(0.)
        else:
            report = inspect_geometry(cfg, tuple(args.origin))
            geometry_scene, model, data = compile_geometry_scene(cfg, tuple(args.origin))
        if args.mjlab_check:
            from mjlab.scene import Scene
            scene_cfg = make_geometry_scene_cfg(cfg) if args.mode == "geometry" else cfg.scene
            mjlab_model = Scene(scene_cfg, device="cpu").compile()
            actual = (mjlab_model.nq, mjlab_model.nv, mjlab_model.nu, mjlab_model.nmocap)
            report["mjlab_compile"] = {"passed": actual == (36, 34, 0 if args.mode == "geometry" else 22, 1), "dimensions": list(actual)}
            if not report["mjlab_compile"]["passed"]:
                report["errors"].append("MJLab scene dimensions differ from the native scene")
                report["passed"] = False
        if args.export:
            args.export.parent.mkdir(parents=True, exist_ok=True)
            if simulation is not None:
                # 기본 exporter가 MJLab 합성 XML의 default 클래스와 메시 경로를 정리한다.
                simulation.scene.write(args.export.with_suffix(""), zip=True)
            else:
                geometry_scene.write(args.export.with_suffix(""), zip=True)
        if simulation is not None and args.simulate_seconds:
            with redirect_stdout(sys.stderr):
                metrics = simulation.run(args.simulate_seconds)
            report["shot_simulation" if args.mode == "shots" else "controlled_simulation"] = metrics
            if not metrics["finite"] or (args.mode == "motors" and metrics["maximum_torque_violation_nm"] > 1e-4):
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
        if args.mode == "shots":
            simulation.close()
        return 0 if report["passed"] else 1
    except (ValueError, FileNotFoundError, RuntimeError, ImportError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
