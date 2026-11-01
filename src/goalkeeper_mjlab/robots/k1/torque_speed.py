"""학습·검증 공통: wirlwind 방식의 배치 토크-속도 제한을 실제 엔진에 적용한다."""

from dataclasses import dataclass

import torch
from mjlab.entity import Entity, EntityCfg
from mjlab.managers.event_manager import requires_model_fields

from goalkeeper_mjlab.robots.k1.actuators import MotorPositionActuatorCfg


class TorqueSpeedLimits:
    def __init__(self, actuators):
        self.target_ids = torch.cat([a.target_ids for a in actuators])
        self.ctrl_ids = torch.cat([a.global_ctrl_ids for a in actuators])
        values = {name: [] for name in ("peak", "speed", "width", "cap")}
        for actuator in actuators:
            motor, control = actuator.cfg.motor, actuator.cfg.controller
            count = actuator.target_ids.numel()
            for name, value in (("peak", motor.peak_effort), ("speed", motor.velocity_limit),
                                ("width", max(motor.velocity_limit - motor.effective_knee_point_velocity, 1e-6)),
                                ("cap", control.output_limit)):
                values[name].extend([value] * count)
        for name, value in values.items():
            setattr(self, name, torch.tensor(value, dtype=torch.float32, device=self.target_ids.device))
        self.last_limit = None
        self.last_velocity = None

    def max_effort(self, velocity):
        # 천천히 돌 때 최대 힘, 감소 시작 속도부터 직선 감소, 최고 속도 이상에서 0이다.
        # 현재 속도의 부호와 무관하게 같은 상한을 쓴다. 센서 피드백에는 명령 지연을 적용하지 않는다.
        fraction = ((self.speed - velocity.abs()) / self.width).clamp(0.0, 1.0)
        # [wirlwind와 구분] 물리 최대 토크와 프로그램 상한을 분리한다.
        # cap에 먼저 속도 비율을 곱하면 작은 프로그램 상한에서 힘을 과도하게 줄이므로 min을 마지막에 취한다.
        return torch.minimum(self.peak * fraction, self.cap)

    def update(self, entity, model):
        self.last_velocity = entity.data.joint_vel[:, self.target_ids].clone()
        self.last_limit = self.max_effort(self.last_velocity)
        # 숫자만 계산하지 않고 실제 엔진의 환경별 힘 제한을 갱신한다.
        # 모델 배열 확장으로 주소가 바뀔 수 있어, bridge에서 매번 최신 배열을 읽는다.
        model.actuator_forcerange[:, self.ctrl_ids, :] = torch.stack((-self.last_limit, self.last_limit), dim=-1)


@dataclass
class MotorEntityCfg(EntityCfg):
    def build(self):
        return MotorEntity(self)


class MotorEntity(Entity):
    _motor_limits = None
    _motor_model = None

    def write_data_to_sim(self):
        if self._motor_limits is None or self._motor_model is None:
            raise RuntimeError("MotorEntity requires initialize_motor_limits startup event or bind_motor_limits(scene, sim)")
        # wirlwind와 같은 실행 위치다. 기본 actuator가 지연된 명령을 쓰고, 물리를 진행하기 전에 힘 상한을 갱신한다.
        super().write_data_to_sim()
        self._motor_limits.update(self, self._motor_model)


def bind_motor_limits(scene, sim):
    if "actuator_forcerange" not in sim.expanded_fields:
        sim.expand_model_fields(("actuator_forcerange",))
    for entity in scene.entities.values():
        actuators = [a for a in entity.actuators if isinstance(a.cfg, MotorPositionActuatorCfg)]
        if actuators:
            if not isinstance(entity, MotorEntity):
                raise TypeError("MotorPositionActuatorCfg requires MotorEntityCfg")
            entity._motor_limits = TorqueSpeedLimits(actuators)
            entity._motor_model = sim.model


@requires_model_fields("actuator_forcerange")
def initialize_motor_limits(env, env_ids):
    """실제 ManagerBasedRlEnv의 startup 이벤트에 등록한다. 검사 실행기도 같은 연결 함수를 사용한다."""
    del env_ids
    bind_motor_limits(env.scene, env.sim)
