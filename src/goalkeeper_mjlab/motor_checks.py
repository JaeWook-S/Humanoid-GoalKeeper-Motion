"""2단계 완료 기준을 실제 MuJoCo 동역학으로 검사한다. 기본 자세에서 몸통을 붙잡지 않는다."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import mujoco
import numpy as np

from goalkeeper_mjlab.motor_runtime import MotorSimulation
from goalkeeper_mjlab import PACKAGE_ROOT
from goalkeeper_mjlab.assets import verify_source_assets
from goalkeeper_mjlab.environment.goal.goal_constants import get_opening
from goalkeeper_mjlab.robots.k1.command_delay import CommandDelayCfg
from goalkeeper_mjlab.robots.k1.constants import K1_JOINT_ORDER
from goalkeeper_mjlab.robots.k1.motor_parameters import get_motors, OFFICIAL_COMMIT, OFFICIAL_K1_SOURCE, OFFICIAL_MOTOR_SOURCE, LOCAL_REFERENCE_COMMIT
from goalkeeper_mjlab.tasks.goalkeeper.config.k1.motor_scene_cfg import Stage2SceneCfg


def configuration_report(sim: MotorSimulation) -> dict:
    m, p, cfg = sim.model, sim.parameters, sim.cfg
    # 기준 숫자도 함께 배포한 JSON에서 읽는다. 기록된 GitHub URL에 접속하거나 참조 저장소 폴더를 열지 않는다.
    reference = json.loads((PACKAGE_ROOT / "robots/k1/motor_reference.json").read_text(encoding="utf-8"))
    expected = reference["official_motors" if cfg.motor_source == "official" else "local_reference_motors"]
    assets = verify_source_assets()
    reference_matches = all(all(np.isclose(getattr(motor, attr), expected[group][source_attr]) for attr, source_attr in
        (("armature", "armature"), ("peak_effort", "effort_limit"), ("velocity_limit", "velocity_limit"), ("knee_point_velocity", "knee_point_velocity")))
        for group, motor in get_motors(cfg.motor_source).items())
    checks = {
        "recorded_motor_reference": reference_matches,
        "original_assets": not assets["mismatches"],
        "actuator_count": m.nu == 22,
        "floating_robot": int(m.joint("robot/floating_base_joint").type[0]) == mujoco.mjtJoint.mjJNT_FREE,
        "physical_armature_applied": bool(np.allclose(m.dof_armature[sim.dof_ids], p.armature)),
        "pd_gains_applied": bool(np.allclose(m.actuator_gainprm[sim.ctrl_ids, 0], p.kp) and
                                 np.allclose(m.actuator_biasprm[sim.ctrl_ids, 1], -p.kp) and
                                 np.allclose(m.actuator_biasprm[sim.ctrl_ids, 2], -p.kd)),
        "force_limit_enabled": bool(m.actuator_forcelimited[sim.ctrl_ids].all()),
        "physics_timestep": bool(np.isclose(m.opt.timestep, cfg.execution.physics_dt)),
        "integrator": int(m.opt.integrator) == {"implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST, "euler": mujoco.mjtIntegrator.mjINT_EULER}[cfg.execution.integrator],
        "solver_iterations": m.opt.iterations == cfg.execution.iterations and m.opt.ls_iterations == cfg.execution.ls_iterations,
        "elliptic_contacts": int(m.opt.cone) == mujoco.mjtCone.mjCONE_ELLIPTIC,
    }
    return {"stage": 2, "passed": all(checks.values()), "checks": checks,
        "scope": "Stage-2 configuration checks; --check-motors additionally runs hold/tracking/force/delay dynamics",
        "mujoco_version": mujoco.__version__, "source_commit": OFFICIAL_COMMIT if cfg.motor_source == "official" else LOCAL_REFERENCE_COMMIT,
        "official_comparison_commit": OFFICIAL_COMMIT,
        "source_urls": [OFFICIAL_MOTOR_SOURCE, OFFICIAL_K1_SOURCE], "motor_source": cfg.motor_source,
        "model": {"nq": m.nq, "nv": m.nv, "nu": m.nu, "nmocap": m.nmocap},
        "execution": {**asdict(cfg.execution), "policy_dt": cfg.execution.policy_dt,
                      "policy_hz": 1 / cfg.execution.policy_dt, "physics_hz": 1 / cfg.execution.physics_dt},
        "source_assets": assets, "joint_names": list(K1_JOINT_ORDER), "origin": list(sim.origin),
        "motors": {name: {**asdict(value), "effective_knee_point_velocity": value.effective_knee_point_velocity} for name, value in get_motors(cfg.motor_source).items()},
        "controllers": {name: asdict(value) for name, value in cfg.controllers.items()},
        "command_delay": {**asdict(cfg.delay), "sampled_group_lags": sim.delay.group_lags.tolist(),
                          "group_order": ["legs", "feet", "arm", "head"],
                          "minimum_ms": cfg.delay.min_steps * cfg.execution.physics_dt * 1000,
                          "maximum_ms": cfg.delay.max_steps * cfg.execution.physics_dt * 1000},
        "robot": {"root_position": sim.data.xpos[m.body("robot/Trunk").id].tolist(),
                  "compiled_mass_kg": float(sum(m.body_mass[i] for i in range(m.nbody) if m.body(i).name.startswith("robot/")))},
        "ball": {"radius_m": cfg.scene.ball.radius, "mass_kg": cfg.scene.ball.mass,
                 "position": sim.data.xpos[m.body("ball/ball").id].tolist()},
        "goal": {"division": cfg.scene.goal.division, **asdict(get_opening(cfg.scene.goal)),
                 "position": sim.data.xpos[m.body("goal/frame").id].tolist()},
        "errors": [name for name, ok in checks.items() if not ok]}


def torque_probe(cfg: Stage2SceneCfg) -> dict:
    # 고속 경계 검사는 공중/무중력의 별도 복제 장면에서만 한다. 서 있기 검사와 섞지 않는다.
    sim = MotorSimulation(replace(cfg, delay=CommandDelayCfg(0, 0)))
    sim.model.opt.gravity[:] = 0.0
    p = sim.parameters
    maximum_error, violation, saturation = 0.0, 0.0, 0
    for sign in (-1.0, 1.0):
        for speed in (np.zeros(22), p.knee_speed, (p.knee_speed + p.max_speed) / 2, p.max_speed, p.max_speed * 1.1):
            sim.reset()
            root = int(sim.model.joint("robot/floating_base_joint").qposadr[0])
            sim.data.qpos[root + 2] += 2.0
            sim.data.qvel[sim.dof_ids] = sign * speed
            target = sim.joint_limits[:, 1 if sign > 0 else 0].copy()
            sim.set_target(target)
            # 코어의 clip 구현과 독립적으로, Booster 공식 코드의 세 구간을 직접 계산해 기대값을 만든다.
            fraction = np.ones(22)
            ramp = (speed > p.knee_speed) & (speed < p.max_speed)
            fraction[ramp] = (p.max_speed[ramp] - speed[ramp]) / (p.max_speed[ramp] - p.knee_speed[ramp])
            fraction[speed >= p.max_speed] = 0.0
            expected_limit = np.minimum(p.controller_cap, p.peak * fraction)
            raw = p.kp * (target - sim.data.qpos[sim.qpos_ids]) - p.kd * sim.data.qvel[sim.dof_ids]
            expected = np.clip(raw, -expected_limit, expected_limit)
            sim.step_physics()
            maximum_error = max(maximum_error, float(np.max(np.abs(sim.last_torque - expected))))
            violation = max(violation, float(np.max(np.abs(sim.last_torque) - expected_limit)))
            saturation += int(np.sum(np.abs(raw) > expected_limit + 1e-5))
    return {"maximum_engine_torque_error_nm": maximum_error, "maximum_violation_nm": max(0.0, violation),
            "saturated_cases": saturation, "joint_cases": 220,
            "passed": maximum_error < 1e-7 and violation < 1e-7 and saturation > 0}


def check_motors(cfg: Stage2SceneCfg | None = None, hold_seconds: float = 10.0, origin=(0.0, 0.0, 0.0)) -> dict:
    cfg = cfg or Stage2SceneCfg()
    if not np.isfinite(hold_seconds) or hold_seconds < 1.0:
        raise ValueError("Hold verification must run for at least 1 second")
    sim = MotorSimulation(cfg, origin)
    report = configuration_report(sim)
    initial_height = float(sim.data.xpos[sim.model.body("robot/Trunk").id, 2] - origin[2])
    hold = sim.run(hold_seconds)
    hold["passed"] = bool(hold["finite"] and hold["minimum_root_height_m"] - origin[2] > initial_height * 0.9 and
                          hold["maximum_tilt_deg"] < 5.0 and hold["maximum_joint_error_rad"] < 0.1 and
                          hold["maximum_torque_violation_nm"] < 1e-7)
    head = K1_JOINT_ORDER.index("Head_Yaw")
    tracking = []
    for target_angle in (0.2, -0.2):
        target = sim.initial_target.copy()
        target[head] = target_angle
        sim.set_target(target)
        metrics = sim.run(2.0)
        error = float(abs(sim.data.qpos[sim.qpos_ids[head]] - target_angle))
        tracking.append({"target_rad": target_angle, "actual_rad": float(sim.data.qpos[sim.qpos_ids[head]]),
                         "error_rad": error, "passed": error < 0.015 and metrics["maximum_tilt_deg"] < 5.0 and metrics["maximum_torque_violation_nm"] < 1e-7})
    probe = torque_probe(cfg)
    delay_sim = MotorSimulation(replace(cfg, delay=CommandDelayCfg(2, 2)), origin)
    command = delay_sim.initial_target.copy()
    command[head] = 0.2
    delay_sim.set_target(command)
    received = []
    for _ in range(3):
        delay_sim.step_physics()
        received.append(float(delay_sim.data.ctrl[delay_sim.ctrl_ids[head]]))
    delay_sim.reset()
    delay_sim.step_physics()
    delay_pass = bool(np.allclose(received, [0.0, 0.0, 0.2]) and
                      np.allclose(delay_sim.data.ctrl[delay_sim.ctrl_ids], delay_sim.initial_target))
    report.update(scope="Stage-2 native MuJoCo hold, joint target, actual torque and command-delay validation",
                  hold=hold, tracking=tracking, torque_probe=probe,
                  delay_probe={"received_targets_rad": received, "passed": delay_pass})
    for name, passed in (("free_base_hold", hold["passed"]), ("joint_target_tracking", all(p["passed"] for p in tracking)),
                         ("engine_torque_limits", probe["passed"]), ("command_delay_and_reset", delay_pass)):
        report["checks"][name] = passed
        if not passed:
            report["errors"].append(name)
    report["passed"] = all(report["checks"].values())
    return report
