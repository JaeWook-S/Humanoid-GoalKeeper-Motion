"""MJLab에서도 매 물리 스텝에 목표각 지연과 실제 힘 상한을 적용하는 모터 어댑터.

이 파일만 MJLab/torch를 import한다. 네이티브 MuJoCo 검사는 이 의존성 없이 실행할 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch

from mjlab.actuator.actuator import Actuator, ActuatorCfg, TransmissionType
from mjlab.entity import Entity, EntityCfg
from mjlab.managers.event_manager import requires_model_fields

from goalkeeper_mjlab.robots.k1.actuators import add_position_actuator
from goalkeeper_mjlab.robots.k1.command_delay import CommandDelayCfg
from goalkeeper_mjlab.robots.k1.constants import K1_JOINT_ORDER
from goalkeeper_mjlab.robots.k1.controller_profiles import ControllerParameters
from goalkeeper_mjlab.robots.k1.motor_parameters import MotorParameters, delay_group, get_motors, motor_group


@dataclass(kw_only=True)
class K1PositionActuatorCfg(ActuatorCfg):
    motor: MotorParameters
    controller: ControllerParameters
    command_delay: CommandDelayCfg
    command_group: str

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.transmission_type != TransmissionType.JOINT or self.delay_max_lag != 0:
            raise ValueError("K1 actuator uses joint transmission and its own command delay")

    def build(self, entity, target_ids, target_names):
        if not isinstance(entity, MotorEntity):
            raise TypeError("K1PositionActuatorCfg requires MotorEntityCfg")
        return K1PositionActuator(self, entity, target_ids, target_names)


class K1PositionActuator(Actuator):
    def edit_spec(self, spec, target_names) -> None:
        for name in target_names:
            self._mjs_actuators.append(add_position_actuator(
                spec, name, self.cfg.motor.armature,
                min(self.cfg.controller.output_limit, self.cfg.motor.peak_effort),
                self.cfg.controller.kp, self.cfg.controller.kd,
            ))

    def initialize(self, mj_model, model, data, device) -> None:
        super().initialize(mj_model, model, data, device)
        self._motor_model = None
        # 엔티티의 데이터 생성보다 actuator 초기화가 먼저다. 합쳐진 keyframe에서 목표각을 읽는다.
        ids = self.global_ctrl_ids.cpu().numpy()
        joint_ids = mj_model.actuator_trnid[ids, 0]
        positions = mj_model.key("init_state").qpos[mj_model.jnt_qposadr[joint_ids]].copy()
        self.initial = torch.tensor(positions, dtype=torch.float32, device=device).repeat(data.nworld, 1)
        self.history = self.initial.unsqueeze(0).repeat(self.cfg.command_delay.max_steps + 1, 1, 1)
        self.cursor = 0
        self.rows = torch.arange(data.nworld, device=device)
        self.last_limit = torch.full_like(self.initial, min(self.cfg.motor.peak_effort, self.cfg.controller.output_limit))
        self.last_velocity = torch.zeros_like(self.initial)
        self.reset()

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        if not hasattr(self, "initial"):
            return
        idx = slice(None) if env_ids is None else env_ids
        self.history[:, idx] = self.initial[idx].unsqueeze(0)
        if env_ids is None:
            self.cursor = 0
        # 다리/발목/팔/머리 그룹마다 reset 때 한 번 선택한다. 동일 그룹 관절은 같은 지연을 쓴다.
        group = self.cfg.command_group
        cache = self.entity._motor_delay_groups
        if group not in cache:
            cache[group] = {"epoch": -1, "lags": torch.zeros(len(self.rows), dtype=torch.long, device=self.initial.device)}
        entry = cache[group]
        if entry["epoch"] != self.entity._motor_reset_epoch:
            count = self.initial[idx].shape[0]
            entry["lags"][idx] = torch.randint(self.cfg.command_delay.min_steps,
                self.cfg.command_delay.max_steps + 1, (count,), device=self.initial.device,
                generator=self.entity._motor_rng)
            entry["epoch"] = self.entity._motor_reset_epoch
        self.lags = entry["lags"]

    def compute(self, cmd):
        if self._motor_model is None:
            raise RuntimeError("Call bind_motor_limits(scene, sim), or register initialize_motor_limits startup event")
        # 목표각만 늦게 전달한다. 모터가 읽는 현재 각도/속도는 최신 값이다.
        self.history[self.cursor] = cmd.position_target
        delayed = self.history[(self.cursor - self.lags) % len(self.history), self.rows]
        self.cursor = (self.cursor + 1) % len(self.history)
        motor = self.cfg.motor
        width = max(motor.velocity_limit - motor.effective_knee_point_velocity, 1e-6)
        fraction = ((motor.velocity_limit - cmd.vel.abs()) / width).clamp(0.0, 1.0)
        self.last_velocity = cmd.vel.clone()
        self.last_limit = (motor.peak_effort * fraction).clamp(max=self.cfg.controller.output_limit)
        # 세계별 배열을 매번 bridge에서 읽는다. 확장 후 예전 배열을 계속 참조하는 문제를 피한다.
        self._motor_model.actuator_forcerange[:, self.global_ctrl_ids, :] = torch.stack(
            (-self.last_limit, self.last_limit), dim=-1,
        )
        return delayed


@dataclass
class MotorEntityCfg(EntityCfg):
    motor_seed: int = 0

    def build(self):
        return MotorEntity(self)


class MotorEntity(Entity):
    def initialize(self, mj_model, model, data, device) -> None:
        self._motor_reset_epoch = 0
        self._motor_delay_groups = {}
        self._motor_rng = torch.Generator(device=device).manual_seed(self.cfg.motor_seed)
        super().initialize(mj_model, model, data, device)
        self.reset()

    def reset(self, env_ids=None) -> None:
        self._motor_reset_epoch += 1
        super().reset(env_ids)
        # reset 후 0 rad가 아니라 설정한 준비 자세를 목표로 유지한다.
        idx = slice(None) if env_ids is None else env_ids
        self.set_joint_position_target(self.data.default_joint_pos[idx], env_ids=env_ids)


def make_motor_actuators(cfg) -> tuple:
    motors = get_motors(cfg.motor_source)
    result = []
    for group, motor in motors.items():
        names = tuple(name for name in K1_JOINT_ORDER if motor_group(name) == group)
        result.append(K1PositionActuatorCfg(target_names_expr=names, motor=motor,
            controller=cfg.controllers[group], command_delay=cfg.delay, command_group=delay_group(names[0])))
    return tuple(result)


def bind_motor_limits(scene, sim) -> None:
    """Scene/Simulation 초기화 후 한 번 연결한다. 여러 환경의 힘 상한이 서로 섞이지 않게 한다."""
    if "actuator_forcerange" not in sim.expanded_fields:
        sim.expand_model_fields(("actuator_forcerange",))
    for entity in scene.entities.values():
        for actuator in entity.actuators:
            if isinstance(actuator, K1PositionActuator):
                actuator._motor_model = sim.model


@requires_model_fields("actuator_forcerange")
def initialize_motor_limits(env, env_ids) -> None:
    """후속 ManagerBasedRlEnv에서는 startup 이벤트에 등록한다. 초기화를 빼면 조용히 통과하지 않는다."""
    del env_ids
    bind_motor_limits(env.scene, env.sim)
