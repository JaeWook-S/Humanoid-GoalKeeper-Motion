"""모델 구조·배치·접촉 검증. 보상이나 학습 성공 판정과는 독립적이다."""

from __future__ import annotations

from dataclasses import asdict, fields, replace
from copy import deepcopy

from mjlab.entity import EntityCfg
from mjlab.scene import Scene
from goalkeeper_mjlab.environment.goal.goal_constants import GoalOpening

import mujoco
import numpy as np

from goalkeeper_mjlab.assets import verify_source_assets
from goalkeeper_mjlab.robots.k1.constants import (
    FOOT_COLLISION_GEOMS, HAND_BODIES, HAND_COLLISION_GEOMS, K1_JOINT_ORDER, geom_world_bounds,
)
from goalkeeper_mjlab.tasks.goalkeeper.config.k1.env_cfgs import goalkeeper_k1_env_cfg
from goalkeeper_mjlab.robots.k1.constants import get_k1_robot_cfg


def make_geometry_env_cfg(*, robot_pose=None, **kwargs):
    """실제 K1 환경에서 형상 검사에 필요한 값만 바꾼다. 학습 설정의 인자를 늘리지 않는다."""
    cfg = goalkeeper_k1_env_cfg(**kwargs)
    cfg.sim.mujoco.timestep = 0.002
    if robot_pose is not None:
        cfg.scene.entities["robot"] = get_k1_robot_cfg(pose=robot_pose)
    return cfg


def make_geometry_scene_cfg(cfg=None):
    """학습 장면을 복사해 모터만 뺀다. 이 함수는 검증 패키지 밖에서 사용하지 않는다."""
    cfg = cfg if cfg is not None else make_geometry_env_cfg()
    scene = deepcopy(cfg.scene)
    scene.num_envs = 1
    # 학습 환경에는 항상 공 접촉 센서가 있다. 정적 형상 검사에서는 추가 센서를 사용하지 않는다.
    scene.sensors = ()
    robot = scene.entities["robot"]
    scene.entities["robot"] = EntityCfg(**{
        f.name: None if f.name == "articulation" else getattr(robot, f.name) for f in fields(EntityCfg)
    })
    return scene


def compile_geometry_scene(cfg=None, origin=(0.0, 0.0, 0.0)):
    """기본 MJLab 장면 조립을 사용한 정적 MuJoCo 검사. Scene·모델·초기 상태를 반환한다."""
    if len(origin) != 3 or not np.isfinite(origin).all():
        raise ValueError("origin must contain three finite numbers")
    cfg = cfg if cfg is not None else make_geometry_env_cfg()
    scene_cfg = make_geometry_scene_cfg(cfg)
    for entity in scene_cfg.entities.values():
        entity.init_state = replace(entity.init_state, pos=tuple(np.asarray(entity.init_state.pos) + origin))
    scene = Scene(scene_cfg, device="cpu")
    scene.spec.modelname = "goalkeeper_geometry_check"
    scene.spec.geom("terrain").pos = origin
    model = scene.compile()
    cfg.sim.mujoco.apply(model)
    # 정적 ZIP 출력에도 같은 물리 옵션을 기록한다. 수치는 학습 cfg.sim에서만 정한다.
    for name in ("timestep", "integrator", "solver", "iterations", "ls_iterations", "tolerance", "cone"):
        setattr(scene.spec.option, name, getattr(model.opt, name))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
    mujoco.mj_forward(model, data)
    return scene, model, data


def contact_records(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict]:
    return [{
        "geom1": model.geom(int(c.geom1)).name,
        "geom2": model.geom(int(c.geom2)).name,
        "distance_m": float(c.dist),
    } for c in data.contact]


def ball_contact_probe(model: mujoco.MjModel, data: mujoco.MjData,
                       position: np.ndarray, target_prefix: str) -> bool:
    """독립 MjData에서만 공을 옮겨 충돌 검사를 한다. 실제 장면 상태는 변경하지 않는다."""
    probe = mujoco.MjData(model)
    probe.qpos[:] = data.qpos
    probe.mocap_pos[:] = data.mocap_pos
    probe.mocap_quat[:] = data.mocap_quat
    address = int(model.joint("ball/ball_joint").qposadr[0])
    probe.qpos[address:address + 3] = position
    mujoco.mj_forward(model, probe)
    for c in probe.contact:
        names = (model.geom(int(c.geom1)).name, model.geom(int(c.geom2)).name)
        if any(n.startswith("ball/") for n in names) and any(n.startswith(target_prefix) for n in names):
            return True
    return False


def inspect_geometry(cfg=None,
                  origin: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> dict:
    cfg = cfg if cfg is not None else make_geometry_env_cfg()
    _, model, data = compile_geometry_scene(cfg, origin)
    # 기대 물성은 자산 설정이 만든 원본 spec에서 읽고, 합성 모델에서 질량이 중복되지 않았는지 검사한다.
    source_ball = cfg.scene.entities["ball"].spec_fn()
    radius = float(source_ball.geom("ball_collision").size[0])
    mass = float(source_ball.geom("ball_collision").mass)
    source_goal = cfg.scene.entities["goal"].spec_fn()
    left, right, bar = (source_goal.geom(n) for n in ("post_negative_x", "post_positive_x", "crossbar"))
    opening = GoalOpening(float(right.pos[0] + right.size[0] - left.pos[0] + left.size[0]),
        float(bar.pos[2] + bar.size[2]), float(right.pos[0] - right.size[0] - left.pos[0] - left.size[0]),
        float(bar.pos[2] - bar.size[2]))
    errors: list[str] = []
    checks: dict[str, bool] = {}

    def check(name: str, passed: bool, message: str) -> None:
        checks[name] = bool(passed)
        if not passed:
            errors.append(message)

    source = verify_source_assets()
    check("source_assets", not source["mismatches"], f"Original assets changed: {source['mismatches']}")
    names = tuple(model.joint(i).name.removeprefix("robot/") for i in range(model.njnt)
                  if model.joint(i).name.startswith("robot/") and model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE)
    check("joint_order", names == K1_JOINT_ORDER, "K1 joint order differs from the 22-joint contract")
    joint_ids = [model.joint("robot/" + n).id for n in K1_JOINT_ORDER]
    check("joint_axes", np.allclose(np.linalg.norm(model.jnt_axis[joint_ids], axis=1), 1.),
          "A joint axis is not normalized")
    limits = model.jnt_range[joint_ids]
    check("joint_limits", np.isfinite(limits).all() and np.all(limits[:, 0] < limits[:, 1]),
          "Joint range contains invalid bounds")
    check("degrees_of_freedom", (model.nq, model.nv, model.nu, model.nmocap) == (36, 34, 0, 1),
          "Expected a passive 22-joint robot, free ball and one fixed mocap goal")
    check("finite_initial_state", np.isfinite(data.qpos).all(), "Initial state is not finite")
    robot_ids = [i for i in range(model.nbody) if model.body(i).name.startswith("robot/")]
    check("positive_robot_inertia", all(model.body_mass[i] > 0 and np.all(model.body_inertia[i] > 0) for i in robot_ids),
          "Robot link mass or principal inertia is not positive")
    robot_colliders = [i for i in range(model.ngeom) if model.geom(i).name.startswith("robot/") and "_collision" in model.geom(i).name]
    check("whole_body_collision", len(robot_colliders) == 22 and all(model.geom_contype[i] == 1 and model.geom_conaffinity[i] == 1 for i in robot_colliders),
          "A robot collision shape is disabled")
    sole_heights = [float(geom_world_bounds(model, data, model.geom("robot/" + n).id)[0][2] - origin[2])
                    for n in FOOT_COLLISION_GEOMS]
    check("sole_clearance", min(sole_heights) >= -1e-7,
          "A foot sole starts below the configured floor")
    contacts = contact_records(model, data)
    check("initial_penetration", not any(c["distance_m"] < -1e-6 for c in contacts),
          "Initial pose contains intersecting collision shapes; inspect the contacts report")
    ball_id = model.body("ball/ball").id
    check("ball_mass", np.isclose(model.body_mass[ball_id], mass), "Ball visual meshes inflated the physical mass")
    expected_inertia = 2.0 / 3.0 * mass * radius**2
    check("ball_shell_inertia", np.allclose(model.body_inertia[ball_id], expected_inertia), "Ball shell inertia is incorrect")
    offset = np.asarray(origin)
    check("ball_placement", np.allclose(data.xpos[ball_id], offset + np.asarray(cfg.scene.entities["ball"].init_state.pos)), "Ball placement differs from the scene configuration")
    goal_id = model.body("goal/frame").id
    check("goal_placement", np.allclose(data.xpos[goal_id], offset + np.asarray(cfg.scene.entities["goal"].init_state.pos)), "Goal placement differs from the scene configuration")
    # 골대의 로컬 -Y가 장면의 -X를 향해야 골대 뒤쪽이 로봇 쪽으로 돌출되지 않는다.
    goal_rotation = data.xmat[goal_id].reshape(3, 3)
    check("goal_orientation", np.allclose(goal_rotation @ np.array([0., -1., 0.]), [-1., 0., 0.]), "Goal depth does not point behind the goal line")
    mouth_center = offset + np.asarray(cfg.scene.entities["goal"].init_state.pos) + (0., 0., opening.inner_height / 2)
    check("open_goal_mouth", not ball_contact_probe(model, data, mouth_center, "goal/"), "An invisible collider blocks the goal mouth")
    post_position = data.geom_xpos[model.geom("goal/post_positive_x").id]
    check("goal_post_collision", ball_contact_probe(model, data, post_position, "goal/"), "Ball cannot contact the goal post")
    # 손 위치의 공이 다리와도 겹칠 수 있으므로, 양쪽 손 끝 충돌체 이름까지 확인한다.
    hand_contacts = [ball_contact_probe(model, data, data.xpos[model.body("robot/" + body).id], "robot/" + geom)
                     for body, geom in zip(HAND_BODIES, HAND_COLLISION_GEOMS, strict=True)]
    check("hand_ball_collision", all(hand_contacts), "Ball cannot contact both robot hand-end colliders")
    return {
        "mode": "geometry", "passed": not errors, "checks": checks, "errors": errors,
        "scope": "Geometry and contact checks; no PD, policy, hardware or goalkeeping performance validation",
        "mujoco_version": mujoco.__version__, "source_assets": source,
        "origin": list(origin), "joint_names": list(names),
        "model": {"nq": model.nq, "nv": model.nv, "nu": model.nu, "nmocap": model.nmocap},
        "robot": {
            "compiled_mass_kg": float(model.body_mass[robot_ids].sum()),
            "root_position": data.xpos[model.body("robot/Trunk").id].tolist(),
            "sole_bottoms_above_floor_m": sole_heights,
        },
        "ball": {"radius_m": radius, "mass_kg": float(model.body_mass[ball_id]), "position": data.xpos[ball_id].tolist()},
        "goal": {**asdict(opening), "position": data.xpos[goal_id].tolist()},
        "initial_contacts": contacts,
    }
