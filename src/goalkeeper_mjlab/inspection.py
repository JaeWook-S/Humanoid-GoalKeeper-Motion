"""모델 구조·배치·접촉 검증. 보상이나 학습 성공 판정과는 독립적이다."""

from __future__ import annotations

from dataclasses import asdict

import mujoco
import numpy as np

from goalkeeper_mjlab.assets import verify_source_assets
from goalkeeper_mjlab.environment.goal.goal_constants import get_opening
from goalkeeper_mjlab.robots.k1.constants import (
    FOOT_COLLISION_GEOMS, HAND_BODIES, HAND_COLLISION_GEOMS, K1_JOINT_ORDER, geom_world_bounds,
)
from goalkeeper_mjlab.tasks.goalkeeper.config.k1.scene_cfg import Stage1SceneCfg, compile_scene


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


def inspect_scene(cfg: Stage1SceneCfg = Stage1SceneCfg(),
                  origin: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> dict:
    _, model, data = compile_scene(cfg, origin)
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
    check("sole_clearance", np.isclose(min(sole_heights), cfg.robot_pose.sole_clearance, atol=1e-7),
          "Foot sole height does not match the configured clearance")
    contacts = contact_records(model, data)
    check("initial_penetration", not any(c["distance_m"] < -1e-6 for c in contacts),
          "Initial pose contains intersecting collision shapes; inspect the contacts report")
    ball_id = model.body("ball/ball").id
    check("ball_mass", np.isclose(model.body_mass[ball_id], cfg.ball.mass), "Ball visual meshes inflated the physical mass")
    expected_inertia = 2.0 / 3.0 * cfg.ball.mass * cfg.ball.radius**2
    check("ball_shell_inertia", np.allclose(model.body_inertia[ball_id], expected_inertia), "Ball shell inertia is incorrect")
    offset = np.asarray(origin)
    check("ball_placement", np.allclose(data.xpos[ball_id], offset + cfg.resolved_ball_position), "Ball placement differs from the scene configuration")
    goal_id = model.body("goal/frame").id
    check("goal_placement", np.allclose(data.xpos[goal_id], offset + cfg.goal_position), "Goal placement differs from the scene configuration")
    # 골대의 로컬 -Y가 장면의 -X를 향해야 골대 뒤쪽이 로봇 쪽으로 돌출되지 않는다.
    goal_rotation = data.xmat[goal_id].reshape(3, 3)
    check("goal_orientation", np.allclose(goal_rotation @ np.array([0., -1., 0.]), [-1., 0., 0.]), "Goal depth does not point behind the goal line")
    opening = get_opening(cfg.goal)
    mouth_center = offset + np.array([-cfg.goal_distance, 0., opening.inner_height / 2])
    check("open_goal_mouth", not ball_contact_probe(model, data, mouth_center, "goal/"), "An invisible collider blocks the goal mouth")
    post_position = data.geom_xpos[model.geom("goal/post_positive_x").id]
    check("goal_post_collision", ball_contact_probe(model, data, post_position, "goal/"), "Ball cannot contact the goal post")
    # 손 위치의 공이 다리와도 겹칠 수 있으므로, 양쪽 손 끝 충돌체 이름까지 확인한다.
    hand_contacts = [ball_contact_probe(model, data, data.xpos[model.body("robot/" + body).id], "robot/" + geom)
                     for body, geom in zip(HAND_BODIES, HAND_COLLISION_GEOMS, strict=True)]
    check("hand_ball_collision", all(hand_contacts), "Ball cannot contact both robot hand-end colliders")
    return {
        "passed": not errors, "checks": checks, "errors": errors,
        "scope": "Stage 1 geometry and contact checks; no PD, policy, hardware or goalkeeping performance validation",
        "mujoco_version": mujoco.__version__, "source_assets": source,
        "origin": list(origin), "joint_names": list(names),
        "model": {"nq": model.nq, "nv": model.nv, "nu": model.nu, "nmocap": model.nmocap},
        "robot": {
            "compiled_mass_kg": float(model.body_mass[robot_ids].sum()),
            "root_position": data.xpos[model.body("robot/Trunk").id].tolist(),
            "sole_bottoms_above_floor_m": sole_heights,
        },
        "ball": {"radius_m": cfg.ball.radius, "mass_kg": float(model.body_mass[ball_id]), "position": data.xpos[ball_id].tolist()},
        "goal": {**asdict(opening), "position": data.xpos[goal_id].tolist()},
        "initial_contacts": contacts,
    }
