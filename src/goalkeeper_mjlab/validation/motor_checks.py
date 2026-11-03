"""공통 MJLab 실행 결과를 검사한다. 검증용 모터 모델이나 지연 버퍼를 만들지 않는다."""

from dataclasses import asdict
from copy import deepcopy
import json
import math
import numpy as np
import torch

from goalkeeper_mjlab import PACKAGE_ROOT
from goalkeeper_mjlab.assets import verify_source_assets
from goalkeeper_mjlab.validation.motor_runtime import MotorSimulation
from goalkeeper_mjlab.robots.k1.constants import K1_JOINT_ORDER
from goalkeeper_mjlab.robots.k1.constants import get_motors, motor_group
from goalkeeper_mjlab.tasks.goalkeeper.config.k1.env_cfgs import goalkeeper_k1_env_cfg


def _motor_settings(cfg):
    # 별도 검사용 설정표 없이, 실제 actuator에 전달한 물리값과 PD 설정을 읽는다.
    actuators = cfg.scene.entities["robot"].articulation.actuators
    motors = {motor_group(a.target_names_expr[0]): a.motor for a in actuators}
    controls = {motor_group(a.target_names_expr[0]): a.controller for a in actuators}
    source = next((name for name in ("official", "legacy") if motors == get_motors(name)), "custom")
    return motors, controls, source


def configuration_report(runtime):
    m, cfg = runtime.model, runtime.cfg
    motors, controllers, source = _motor_settings(cfg)
    physical = [motors[motor_group(n)] for n in K1_JOINT_ORDER]
    controls = [controllers[motor_group(n)] for n in K1_JOINT_ORDER]
    assets = verify_source_assets()
    reference = json.loads((PACKAGE_ROOT / "robots/k1/motor_reference.json").read_text())
    expected = reference["official_motors" if source == "official" else "local_reference_motors"]
    checks = {
        "recorded_motor_reference": all(np.isclose(getattr(v, a), expected[g][b]) for g, v in motors.items()
            for a, b in (("armature", "armature"), ("peak_effort", "effort_limit"), ("velocity_limit", "velocity_limit"), ("knee_point_velocity", "knee_point_velocity"))),
        "original_assets": not assets["mismatches"],
        "actuator_count": m.nu == 22,
        "physical_armature_applied": np.allclose(m.dof_armature[runtime.dof_ids], [v.armature for v in physical]),
        "pd_gains_applied": np.allclose(m.actuator_gainprm[runtime.ctrl_ids, 0], [v.stiffness for v in controls]) and
                            np.allclose(m.actuator_biasprm[runtime.ctrl_ids, 2], [-v.damping for v in controls]),
        "force_limit_enabled": m.actuator_forcelimited[runtime.ctrl_ids].all(),
        "physics_timestep": np.isclose(m.opt.timestep, cfg.sim.mujoco.timestep),
        "builtin_actuators": all(type(a).__name__ == "BuiltinPositionActuator" for a in runtime.robot.actuators),
    }
    checks = {k: bool(v) for k, v in checks.items()}
    return {"mode": "motors", "passed": all(checks.values()), "checks": checks,
            "errors": [k for k, v in checks.items() if not v], "scope": "Shared MJLab motor implementation",
            "model": {"nq": m.nq, "nv": m.nv, "nu": m.nu, "nmocap": m.nmocap}, "source_assets": assets,
            "motor_source": source, "motors": {k: asdict(v) for k, v in motors.items()},
            "controllers": {k: asdict(v) for k, v in controllers.items()},
            "execution": {"physics_dt": runtime.physics_dt, "decimation": cfg.decimation,
                          "integrator": cfg.sim.mujoco.integrator, "solver": cfg.sim.mujoco.solver,
                          "iterations": cfg.sim.mujoco.iterations, "ls_iterations": cfg.sim.mujoco.ls_iterations,
                          "tolerance": cfg.sim.mujoco.tolerance, "physics_hz": 1 / runtime.physics_dt,
                          "policy_hz": 1 / runtime.policy_dt, "policy_dt": runtime.policy_dt},
            "command_delay": {"min_lag": runtime.robot.actuators[0].cfg.delay_min_lag,
                              "max_lag": runtime.robot.actuators[0].cfg.delay_max_lag,
                              "hold_prob": runtime.robot.actuators[0].cfg.delay_hold_prob}, "origin": list(runtime.origin)}


def torque_probe(cfg):
    probe_cfg = deepcopy(cfg)
    probe_cfg.scene.num_envs = 1
    for actuator in probe_cfg.scene.entities["robot"].articulation.actuators:
        actuator.delay_min_lag = actuator.delay_max_lag = 0
    runtime = MotorSimulation(probe_cfg)
    by_group, controllers, _ = _motor_settings(probe_cfg)
    motors = [by_group[motor_group(n)] for n in K1_JOINT_ORDER]
    peak = np.array([m.peak_effort for m in motors])
    knee = np.array([m.effective_knee_point_velocity for m in motors])
    maximum = np.array([m.velocity_limit for m in motors])
    cap = np.array([controllers[motor_group(n)].output_limit for n in K1_JOINT_ORDER])
    max_error = violation = 0.
    saturated = 0
    for sign in (-1., 1.):
        for speed in (np.zeros(22), knee, (knee + maximum) / 2, maximum, maximum * 1.1):
            runtime.reset()
            state = runtime.robot.data.default_root_state.clone()
            state[:, 2] += 2.
            runtime.robot.write_root_state_to_sim(state)
            q = runtime.robot.data.default_joint_pos.clone()
            velocity = torch.tensor(sign * speed, dtype=torch.float32).unsqueeze(0)
            runtime.robot.write_joint_state_to_sim(q, velocity)
            runtime.sim.forward()
            runtime.scene.update(cfg.sim.mujoco.timestep)
            target = runtime.robot.data.joint_pos_limits[0, :, 1 if sign > 0 else 0].cpu().numpy().copy()
            runtime.set_target(target)
            # 기대값은 독립적인 세 구간 조건으로 계산한다. 학습용 max_effort() 자체를 정답으로 사용하지 않는다.
            fraction = np.ones(22)
            ramp = (speed > knee) & (speed < maximum)
            fraction[ramp] = (maximum[ramp] - speed[ramp]) / (maximum[ramp] - knee[ramp])
            fraction[speed >= maximum] = 0.
            limit = np.minimum(cap, peak * fraction)
            stiffness = np.array([controllers[motor_group(n)].stiffness for n in K1_JOINT_ORDER])
            damping = np.array([controllers[motor_group(n)].damping for n in K1_JOINT_ORDER])
            raw = stiffness * (target - q[0].cpu().numpy()) - damping * sign * speed
            expected = np.clip(raw, -limit, limit)
            runtime.step_physics()
            actual = runtime.sim.data.actuator_force[0, runtime.ctrl_ids].cpu().numpy()
            max_error = max(max_error, float(np.max(np.abs(actual - expected))))
            violation = max(violation, float(np.max(np.abs(actual) - limit)))
            saturated += int(np.sum(np.abs(raw) > limit + 1e-5))
    return {"passed": max_error < 3e-4 and violation < 1e-4, "maximum_engine_torque_error_nm": max_error,
            "maximum_violation_nm": max(0., violation), "joint_cases": 220, "saturated_cases": saturated}


def check_motors(cfg=None, hold_seconds=10., origin=(0., 0., 0.)):
    cfg = cfg if cfg is not None else goalkeeper_k1_env_cfg()
    if not math.isfinite(hold_seconds) or hold_seconds < 1:
        raise ValueError("Hold verification must run for at least 1 second")
    runtime = MotorSimulation(cfg, origin)
    report = configuration_report(runtime)
    initial_height = runtime.data.xpos[runtime.model.body("robot/Trunk").id, 2] - origin[2]
    hold = runtime.run(hold_seconds)
    # 높이 조건이 실패하면 NumPy bool이 반환될 수 있다. 실패 보고서도 JSON으로 저장되도록 기본 bool로 바꾼다.
    hold["passed"] = bool(hold["finite"] and hold["minimum_root_height_m"] - origin[2] > initial_height * .9 and hold["maximum_tilt_deg"] < 5 and hold["maximum_torque_violation_nm"] < 1e-4)
    tracking = []
    for angle in (.2, -.2):
        target = runtime.initial_target.copy()
        target[0] = angle
        runtime.set_target(target)
        result = runtime.run(2.)
        error = abs(runtime.data.qpos[runtime.qpos_ids[0]] - angle)
        tracking.append({"target_rad": angle, "error_rad": float(error), "passed": bool(error < .015 and result["maximum_tilt_deg"] < 5)})
    probe = torque_probe(cfg)
    report.update(hold=hold, tracking=tracking, torque_probe=probe)
    for name, ok in (("free_base_hold", hold["passed"]), ("joint_target_tracking", all(v["passed"] for v in tracking)), ("engine_torque_limits", probe["passed"])):
        report["checks"][name] = bool(ok)
        if not ok:
            report["errors"].append(name)
    report["passed"] = all(report["checks"].values())
    return report
