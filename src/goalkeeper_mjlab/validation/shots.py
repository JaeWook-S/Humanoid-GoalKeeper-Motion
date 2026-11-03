"""실제 학습 환경의 슛 기능을 실행하는 검증 도구. PPO 학습 파일이 아니다.

발사·접촉·득점·부분 reset은 모두 tasks/goalkeeper/mdp와 MJLab이 담당한다.
이 파일은 정책 대신 HOME 목표각을 공급하고 화면/JSON을 위해 결과만 읽는다.
"""

from __future__ import annotations

from copy import deepcopy
import math

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionActionCfg

from goalkeeper_mjlab.robots.k1.constants import K1_JOINT_ORDER
from goalkeeper_mjlab.tasks.goalkeeper.config.k1.env_cfgs import goalkeeper_k1_env_cfg
from goalkeeper_mjlab.tasks.goalkeeper.mdp.metrics import get_shot_tracker


class ShotSimulation:
    """기본 ManagerBasedRlEnv를 감싼 검사기. 별도 물리 루프/가짜 공 이동은 없다."""

    def __init__(self, cfg=None, origin=(0., 0., 0.), device="cpu"):
        self.cfg = deepcopy(cfg if cfg is not None else goalkeeper_k1_env_cfg())
        if "shot_state" not in self.cfg.metrics:
            raise ValueError("ShotSimulation requires a goalkeeper environment with shot tracking")
        # 연속 발사는 MJLab 기본 자동 reset을 사용한다. 별도 재발사 관리자를 만들지 않는다.
        if not self.cfg.auto_reset:
            raise ValueError("ShotSimulation requires auto_reset=True for repeated shot validation")
        if len(origin) != 3 or not np.isfinite(origin).all():
            raise ValueError("origin must contain three finite numbers")
        self.origin = tuple(float(x) for x in origin)
        # 관측/action 설계는 4단계다. 여기서는 MJLab 기본 위치 action에 0을 넣어 HOME을
        # 명령한다. 이는 검증기의 입력일 뿐, 학습 정책이나 새 제어기를 구현한 것이 아니다.
        if self.cfg.actions:
            raise ValueError("ShotSimulation supplies validation actions; use the training environment for policy actions")
        self.cfg.actions = {"validation_home": JointPositionActionCfg(
            entity_name="robot", actuator_names=K1_JOINT_ORDER, preserve_order=True,
            scale=1.0, use_default_offset=True,
        )}
        if any(self.origin):
            original_fn = self.cfg.scene.spec_fn

            def translated_floor(spec):
                # 원점을 높이 옮기는 검사에서도 공 아래의 바닥을 함께 옮긴다.
                if original_fn is not None:
                    original_fn(spec)
                spec.geom("terrain").pos = self.origin

            self.cfg.scene.spec_fn = translated_floor
        self.env = ManagerBasedRlEnv(self.cfg, device=device)
        self.scene, self.sim = self.env.scene, self.env.sim
        self.scene.env_origins.add_(torch.tensor(self.origin, device=device))
        self.model, self.data = self.sim.mj_model, self.sim.mj_data
        self.robot = self.scene["robot"]
        self.tracker = get_shot_tracker(self.env)
        self.physics_dt, self.policy_dt = self.env.physics_dt, self.env.step_dt
        self.num_envs, self.device = self.env.num_envs, self.env.device
        self._target = self.robot.data.default_joint_pos.clone()
        self.completed_episodes = 0  # 반복 실행 종료에 필요한 개수만 센다. 슛별 로그는 쌓지 않는다.
        self.reset()

    @property
    def initial_target(self):
        value = self.robot.data.default_joint_pos.detach().cpu().numpy().copy()
        return value[0] if self.num_envs == 1 else value

    @property
    def target(self):
        value = self._target.detach().cpu().numpy().copy()
        return value[0] if self.num_envs == 1 else value

    def set_target(self, target):
        value = torch.as_tensor(target, dtype=torch.float32, device=self.device)
        if self.num_envs == 1 and value.shape == (len(K1_JOINT_ORDER),):
            value = value.unsqueeze(0)
        limits = self.robot.data.joint_pos_limits
        if (value.shape != self._target.shape or not torch.isfinite(value).all()
                or ((value < limits[..., 0]) | (value > limits[..., 1])).any()):
            raise ValueError("Target must contain valid finite joint angles per environment")
        self._target.copy_(value)

    def reset(self, env_ids=None):
        """R 키와 부분 초기화 검사 모두 MJLab의 공개 reset API로 연결한다."""
        ids = torch.arange(self.num_envs, device=self.device)
        if env_ids is not None:
            ids = ids[env_ids] if isinstance(env_ids, slice) else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if ids.numel() == 0:
            return
        self.env.reset(env_ids=ids)
        self._target[ids] = self.robot.data.default_joint_pos[ids]
        self.sync_viewer_data()

    def step(self, target=None):
        if target is not None:
            self.set_target(target)
        # default offset=True이므로 action=목표각-HOME으로 주면 실제 목표각이 정확히 전달된다.
        action = self._target - self.robot.data.default_joint_pos
        result = self.env.step(action)
        done = result[2] | result[3]
        # 종료 여부는 env.step의 기본 반환값으로 읽는다. 직전 결과를 따로 복사하지 않는다.
        self.completed_episodes += int(done.sum().item())
        self._target[done] = self.robot.data.default_joint_pos[done]
        self.sync_viewer_data()
        return result

    def sync_viewer_data(self, env_id=0):
        """화면용 CPU 데이터에 복사만 한다. 실제 물리 계산은 위 env.step 한 곳에서만 한다."""
        for name in ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat"):
            getattr(self.data, name)[:] = getattr(self.sim.data, name)[env_id].detach().cpu().numpy()
        self.data.time = float(self.sim.data.time[env_id].item())
        self.model.actuator_forcerange[:] = self.sim.model.actuator_forcerange[env_id].detach().cpu().numpy()
        mujoco.mj_forward(self.model, self.data)

    def _report(self, completed):
        finite = bool(torch.isfinite(self.sim.data.qpos).all() and torch.isfinite(self.sim.data.qvel).all())
        return {"passed": finite, "errors": [] if finite else ["Nonfinite simulation state"],
                "finite": finite, "num_envs": self.num_envs,
                "physics_hz": 1 / self.physics_dt, "policy_hz": 1 / self.policy_dt,
                "completed_episodes": completed}

    def run_episodes(self, count=10):
        """전체 완료 슛이 count개 이상이면 끝낸다. 동시 종료로 최대 N-1개 더 끝날 수 있다."""
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("Shot count must be a positive integer")
        first = self.completed_episodes
        # 모든 슛은 timeout을 가지므로 이 횟수 안에는 완료되어야 한다. 무한 검사 루프를 피한다.
        limit = count * (math.ceil(float(self.tracker.timeout_limit.max()) / self.policy_dt) + 2)
        for _ in range(limit):
            self.step()
            if self.completed_episodes - first >= count:
                break
        else:
            raise RuntimeError("Shot episodes did not finish within the configured timeout")
        return self._report(self.completed_episodes - first)

    def run(self, seconds):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Duration must be finite and nonnegative")
        first = self.completed_episodes
        for _ in range(math.ceil(seconds / self.policy_dt)):
            self.step()
        return self._report(self.completed_episodes - first)

    def close(self):
        self.env.close()
