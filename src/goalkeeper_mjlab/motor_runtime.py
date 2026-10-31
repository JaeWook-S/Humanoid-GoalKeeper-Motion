"""실제 MuJoCo 운동 계산과 PD·속도별 힘 제한·명령 지연을 묶는 실행기."""

from __future__ import annotations

import mujoco
import numpy as np

from goalkeeper_mjlab.robots.k1.command_delay import CommandDelay
from goalkeeper_mjlab.robots.k1.constants import K1_JOINT_ORDER
from goalkeeper_mjlab.tasks.goalkeeper.config.k1.motor_scene_cfg import Stage2SceneCfg, compile_controlled_scene


class MotorSimulation:
    def __init__(self, cfg: Stage2SceneCfg | None = None, origin=(0.0, 0.0, 0.0)):
        self.cfg = cfg or Stage2SceneCfg()
        self.spec, self.model, self.data = compile_controlled_scene(self.cfg, origin)
        self.origin = tuple(float(value) for value in origin)
        self.parameters = self.cfg.parameters
        # 로봇과 공이 같은 qpos에 들어 있으므로 순서로 추측하지 않고 이름으로 주소를 찾는다.
        self.qpos_ids = np.array([int(self.model.joint("robot/" + n).qposadr[0]) for n in K1_JOINT_ORDER])
        self.dof_ids = np.array([int(self.model.joint("robot/" + n).dofadr[0]) for n in K1_JOINT_ORDER])
        self.ctrl_ids = np.array([self.model.actuator("robot/" + n).id for n in K1_JOINT_ORDER])
        self.joint_limits = np.array([self.model.joint("robot/" + n).range for n in K1_JOINT_ORDER])
        self.initial_target = self.data.qpos[self.qpos_ids].copy()
        self.delay = CommandDelay(self.cfg.delay, self.initial_target, self.cfg.seed)
        self.target = self.initial_target.copy()
        self.physics_steps = 0
        self.policy_steps = 0
        self.last_delayed_target = self.target.copy()
        self.last_raw_torque = np.zeros(22)
        self.last_limit = np.minimum(self.parameters.peak, self.parameters.controller_cap)
        self.last_velocity = np.zeros(22)
        self.last_torque = np.zeros(22)

    def reset(self) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("init_state").id)
        self.target = self.initial_target.copy()
        self.delay.reset(self.initial_target)
        self.physics_steps = self.policy_steps = 0
        self.last_delayed_target = self.target.copy()
        self.last_raw_torque[:] = self.last_velocity[:] = self.last_torque[:] = 0.0
        self.last_limit = np.minimum(self.parameters.peak, self.parameters.controller_cap)
        mujoco.mj_forward(self.model, self.data)

    def set_target(self, target: np.ndarray) -> None:
        target = np.asarray(target, dtype=float)
        if target.shape != (22,) or not np.isfinite(target).all():
            raise ValueError("Target must contain 22 finite joint angles in radians")
        if np.any(target < self.joint_limits[:, 0]) or np.any(target > self.joint_limits[:, 1]):
            raise ValueError("Joint target is outside the model joint limits")
        self.target = target.copy()

    def step_physics(self) -> None:
        # 정책은 느리게 목표각을 바꾸지만, 모터 제어와 속도별 한계는 매 물리 스텝에 갱신한다.
        self.last_delayed_target = self.delay.push(self.target)
        self.last_velocity = self.data.qvel[self.dof_ids].copy()
        raw, _, limit = self.parameters.command(
            self.last_delayed_target, self.data.qpos[self.qpos_ids], self.last_velocity,
        )
        self.last_raw_torque, self.last_limit = raw, limit
        # 현재 속도에 맞춰 엔진의 실제 힘 상한을 먼저 바꾼 뒤, 지연된 목표각을 전달한다.
        self.model.actuator_forcerange[self.ctrl_ids] = np.stack((-limit, limit), axis=-1)
        self.data.ctrl[self.ctrl_ids] = self.last_delayed_target
        mujoco.mj_step(self.model, self.data)
        # 실제 엔진이 낸 힘을 읽는다. 요청한 ctrl 값만 기록해서 적용됐다고 판단하지 않는다.
        self.last_torque = self.data.qfrc_actuator[self.dof_ids].copy()
        self.physics_steps += 1

    def step(self, target: np.ndarray | None = None) -> None:
        if target is not None:
            self.set_target(target)
        for _ in range(self.cfg.execution.decimation):
            self.step_physics()
        self.policy_steps += 1

    def run(self, seconds: float) -> dict:
        if not np.isfinite(seconds) or seconds < 0:
            raise ValueError("Duration must be finite and nonnegative")
        root = self.model.body("robot/Trunk").id
        minimum_height, maximum_tilt, violation = float(self.data.xpos[root, 2]), 0.0, 0.0
        max_joint_error, max_torque = 0.0, np.zeros(22)
        for _ in range(int(np.ceil(seconds / self.cfg.execution.policy_dt))):
            # 검사도 실제 policy/physics 시간 관계를 그대로 사용한다.
            for _ in range(self.cfg.execution.decimation):
                self.step_physics()
                minimum_height = min(minimum_height, float(self.data.xpos[root, 2]))
                tilt = np.arccos(np.clip(self.data.xmat[root].reshape(3, 3)[2, 2], -1.0, 1.0))
                maximum_tilt = max(maximum_tilt, float(tilt))
                violation = max(violation, float(np.max(np.abs(self.last_torque) - self.last_limit)))
                max_joint_error = max(max_joint_error, float(np.max(np.abs(self.data.qpos[self.qpos_ids] - self.target))))
                max_torque = np.maximum(max_torque, np.abs(self.last_torque))
            self.policy_steps += 1
        return {"time_s": float(self.data.time), "minimum_root_height_m": minimum_height,
                "maximum_tilt_deg": float(np.degrees(maximum_tilt)), "maximum_joint_error_rad": max_joint_error,
                "maximum_torque_violation_nm": max(0.0, violation), "peak_applied_torque_nm": max_torque.tolist(),
                "finite": bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())}


class BatchedMotorSimulation:
    """MJLab의 여러 환경에서도 동일한 물리/정책 주기로 실행한다. torch는 이 경로에서만 필요하다."""

    def __init__(self, cfg: Stage2SceneCfg | None = None, device: str = "cpu"):
        from mjlab.scene import Scene
        from mjlab.sim import Simulation
        from goalkeeper_mjlab.robots.k1.mjlab_actuators import bind_motor_limits
        from goalkeeper_mjlab.tasks.goalkeeper.config.k1.motor_scene_cfg import make_controlled_mjlab_scene_cfg
        self.cfg = cfg or Stage2SceneCfg()
        self.scene = Scene(make_controlled_mjlab_scene_cfg(self.cfg), device=device)
        self.sim = Simulation(self.cfg.scene.num_envs, self.cfg.execution.to_mjlab(),
                              model=self.scene.compile(), device=device)
        # MJLab의 tensor bridge를 전달한다. raw Warp 배열을 넘기면 환경별 tensor 인덱싱이 동작하지 않는다.
        self.scene.initialize(self.sim.mj_model, self.sim.model, self.sim.data)
        bind_motor_limits(self.scene, self.sim)
        self.robot = self.scene["robot"]
        self.reset()

    def reset(self, env_ids=None) -> None:
        from goalkeeper_mjlab.tasks.goalkeeper.config.k1.scene_cfg import place_entities_at_origins
        self.sim.reset(env_ids)
        self.scene.reset(env_ids)
        place_entities_at_origins(self.scene, env_ids)
        self.sim.forward()
        self.scene.update(self.cfg.execution.physics_dt)

    def step_physics(self) -> None:
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.cfg.execution.physics_dt)

    def step(self, target=None) -> None:
        if target is not None:
            import torch
            target = torch.as_tensor(target, dtype=torch.float32, device=self.scene.device)
            if target.shape != (self.cfg.scene.num_envs, 22) or not torch.isfinite(target).all():
                raise ValueError("Batched target must have shape (num_envs, 22) and finite angles")
            limits = self.robot.data.joint_pos_limits
            if ((target < limits[..., 0]) | (target > limits[..., 1])).any():
                raise ValueError("Batched target is outside the joint limits")
            self.robot.set_joint_position_target(target)
        for _ in range(self.cfg.execution.decimation):
            self.step_physics()
