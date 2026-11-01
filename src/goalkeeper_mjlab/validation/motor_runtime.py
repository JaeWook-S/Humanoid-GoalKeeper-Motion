"""화면과 검사가 학습용 MJLab 모터 구현을 그대로 실행하는 공통 실행기.

PD·지연·토크 제한을 여기서 재구현하지 않는다. PPO 루프는 후속 ManagerBasedRlEnv가 담당한다.
"""

import math
import mujoco
import numpy as np
import torch
from mjlab.scene import Scene
from mjlab.sim import Simulation
from mjlab.envs.mdp.events import reset_scene_to_default

from goalkeeper_mjlab.robots.k1.constants import K1_JOINT_ORDER
from goalkeeper_mjlab.robots.k1.torque_speed import bind_motor_limits
from goalkeeper_mjlab.tasks.goalkeeper.config.k1.env_cfgs import make_goalkeeper_env_cfg
from copy import deepcopy


class MotorSimulation:
    def __init__(self, cfg=None, origin=(0., 0., 0.), device="cpu"):
        self.cfg = deepcopy(cfg if cfg is not None else make_goalkeeper_env_cfg())
        self.physics_dt = self.cfg.sim.mujoco.timestep
        self.policy_dt = self.physics_dt * self.cfg.decimation
        if len(origin) != 3 or not np.isfinite(origin).all():
            raise ValueError("origin must contain three finite numbers")
        self.origin = tuple(float(v) for v in origin)
        self.device = device
        self.num_envs = self.cfg.scene.num_envs
        # 실제 학습은 전체 학습 seed를 설정한다. 이 실행기도 기본 MJLab 지연의 난수를 같은 seed로 시작한다.
        torch.manual_seed(self.cfg.seed)
        self.scene = Scene(self.cfg.scene, device=device)
        self.scene.env_origins.add_(torch.tensor(origin, device=device))
        self.spec = self.scene.spec
        self.spec.modelname = "goalkeeper_motor_check"
        self.spec.geom("terrain").pos = origin
        model = self.scene.compile()
        # 내보낸 모델도 첫 목표각은 준비 자세다. 런타임 지연/속도별 제한은 이 ZIP에 포함되지 않는다.
        controls = np.zeros(model.nu)
        for name in K1_JOINT_ORDER:
            controls[model.actuator("robot/" + name).id] = model.key("init_state").qpos[int(model.joint("robot/" + name).qposadr[0])]
        self.spec.key("init_state").ctrl = controls
        self.sim = Simulation(self.num_envs, self.cfg.sim, model=self.scene.compile(), device=device)
        # 내보내는 XML도 실제 Simulation에 적용된 물리 옵션을 반영한다. 별도 물리 설정표는 없다.
        for name in ("timestep", "integrator", "solver", "iterations", "ls_iterations", "tolerance", "cone"):
            setattr(self.spec.option, name, getattr(self.sim.mj_model.opt, name))
        self.scene.initialize(self.sim.mj_model, self.sim.model, self.sim.data)
        bind_motor_limits(self.scene, self.sim)
        self.robot = self.scene["robot"]
        if tuple(self.robot.joint_names) != K1_JOINT_ORDER:
            raise RuntimeError("K1 joint order differs from the command contract")
        self.model, self.data = self.sim.mj_model, self.sim.mj_data
        self.qpos_ids = [int(self.model.joint("robot/" + n).qposadr[0]) for n in K1_JOINT_ORDER]
        self.dof_ids = [int(self.model.joint("robot/" + n).dofadr[0]) for n in K1_JOINT_ORDER]
        self.ctrl_ids = [self.model.actuator("robot/" + n).id for n in K1_JOINT_ORDER]
        self.physics_steps = self.policy_steps = 0
        self.reset()
        self.spec.key("init_state").qpos = self.data.qpos.copy()

    @property
    def initial_target(self):
        value = self.robot.data.default_joint_pos.detach().cpu().numpy().copy()
        return value[0] if self.num_envs == 1 else value

    @property
    def target(self):
        value = self.robot.data.joint_pos_target.detach().cpu().numpy().copy()
        return value[0] if self.num_envs == 1 else value

    def reset(self, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device)
        if env_ids is not None:
            ids = ids[env_ids] if isinstance(env_ids, slice) else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if ids.numel() == 0:
            return
        self.sim.reset(ids)
        self.scene.reset(ids)
        # 기본 reset 이벤트로 로봇·공·mocap 골대를 설정한 초기 배치로 되돌린다.
        reset_scene_to_default(self, ids)
        # 정책 대신 준비 자세를 유지하는 검사 입력이다. 지연 버퍼를 직접 채우거나 시간을 진행하지 않는다.
        self.robot.set_joint_position_target(self.robot.data.default_joint_pos[ids], env_ids=ids)
        self.sim.forward()
        self.scene.update(self.physics_dt)
        if ids.numel() == self.num_envs:
            self.physics_steps = self.policy_steps = 0
        self.sync_viewer_data()

    def set_target(self, target):
        target = torch.as_tensor(target, dtype=torch.float32, device=self.device)
        if self.num_envs == 1 and target.shape == (22,):
            target = target.unsqueeze(0)
        if target.shape != (self.num_envs, 22) or not torch.isfinite(target).all():
            raise ValueError("Target must contain 22 finite joint angles per environment")
        limits = self.robot.data.joint_pos_limits
        if ((target < limits[..., 0]) | (target > limits[..., 1])).any():
            raise ValueError("Joint target is outside the model joint limits")
        self.robot.set_joint_position_target(target)

    def step_physics(self):
        # ManagerBasedRlEnv와 같은 순서다. 제어 처리는 실제 학습용 Entity/Actuator가 수행한다.
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.physics_dt)
        self.physics_steps += 1

    def step(self, target=None):
        if target is not None:
            self.set_target(target)
        for _ in range(self.cfg.decimation):
            self.step_physics()
        self.policy_steps += 1
        self.sync_viewer_data()

    def sync_viewer_data(self, env_id=0):
        # 화면용 CPU MjData에는 결과만 복사한다. 여기서 mj_step을 호출하면 별도의 물리 계산이 된다.
        for name in ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat"):
            getattr(self.data, name)[:] = getattr(self.sim.data, name)[env_id].detach().cpu().numpy()
        self.data.time = float(self.sim.data.time[env_id].item())
        self.model.actuator_forcerange[:] = self.sim.model.actuator_forcerange[env_id].detach().cpu().numpy()
        mujoco.mj_forward(self.model, self.data)

    def run(self, seconds):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Duration must be finite and nonnegative")
        minimum_height = float(self.robot.data.root_link_pos_w[:, 2].min().item())
        max_tilt = max_error = violation = 0.0
        peak = torch.zeros(22, device=self.device)
        for _ in range(math.ceil(seconds / self.policy_dt)):
            for _ in range(self.cfg.decimation):
                self.step_physics()
                minimum_height = min(minimum_height, float(self.robot.data.root_link_pos_w[:, 2].min().item()))
                trunk = self.model.body("robot/Trunk").id
                up = self.sim.data.xmat[:, trunk].reshape(self.num_envs, 3, 3)[:, 2, 2]
                max_tilt = max(max_tilt, float(torch.rad2deg(torch.acos(up.clamp(-1, 1))).max().item()))
                max_error = max(max_error, float((self.robot.data.joint_pos - self.robot.data.joint_pos_target).abs().max().item()))
                force = self.sim.data.actuator_force[:, self.ctrl_ids].abs()
                limit = self.sim.model.actuator_forcerange[:, self.ctrl_ids, 1]
                violation = max(violation, float((force - limit).max().item()))
                peak = torch.maximum(peak, force.max(dim=0).values)
            self.policy_steps += 1
        self.sync_viewer_data()
        return {"time_s": self.data.time, "minimum_root_height_m": minimum_height,
                "maximum_tilt_deg": max_tilt, "maximum_joint_error_rad": max_error,
                "maximum_torque_violation_nm": max(0., violation), "peak_applied_torque_nm": peak.cpu().tolist(),
                "finite": bool(torch.isfinite(self.sim.data.qpos).all() and torch.isfinite(self.sim.data.qvel).all())}
