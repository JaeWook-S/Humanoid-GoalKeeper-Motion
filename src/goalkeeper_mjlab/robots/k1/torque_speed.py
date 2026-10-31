"""현재 회전 속도에서 허용할 힘을 계산한다. 수치는 모터/제어기 설정에서 받는다."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from goalkeeper_mjlab.robots.k1.constants import K1_JOINT_ORDER
from goalkeeper_mjlab.robots.k1.controller_profiles import ControllerParameters
from goalkeeper_mjlab.robots.k1.motor_parameters import MotorParameters, motor_group


@dataclass(frozen=True)
class JointParameters:
    armature: np.ndarray
    peak: np.ndarray
    max_speed: np.ndarray
    knee_speed: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    controller_cap: np.ndarray

    @classmethod
    def build(cls, motors: dict[str, MotorParameters], controllers: dict[str, ControllerParameters]):
        motor = [motors[motor_group(n)] for n in K1_JOINT_ORDER]
        control = [controllers[motor_group(n)] for n in K1_JOINT_ORDER]
        return cls(*(np.asarray(values, dtype=float) for values in (
            [m.armature for m in motor], [m.peak_effort for m in motor],
            [m.velocity_limit for m in motor], [m.effective_knee_point_velocity for m in motor],
            [c.kp for c in control], [c.kd for c in control], [c.output_limit for c in control],
        )))

    def physical_limit(self, velocity: np.ndarray) -> np.ndarray:
        # 느릴 때는 최대 힘을 쓰고, 감소 시작 속도부터 최고 속도까지는 직선으로 0까지 줄인다.
        # 공식과 같은 단순화: 최고 속도 이상에서는 가속뿐 아니라 제동용 모터 힘도 0으로 제한한다.
        # 1e-6은 공식과 같은 0 나눗셈 방지다. Head처럼 시작 속도=최고 속도가 되는 경우도 처리한다.
        fraction = np.clip((self.max_speed - np.abs(velocity)) / np.maximum(self.max_speed - self.knee_speed, 1e-6), 0.0, 1.0)
        return self.peak * fraction

    def effective_limit(self, velocity: np.ndarray) -> np.ndarray:
        # 먼저 모터 물리 한계를 계산한 뒤 프로그램의 상한과 비교한다.
        # 프로그램 상한 자체에 속도 비율을 곱하면 필요 이상으로 힘을 줄이는 오류가 생긴다.
        return np.minimum(self.controller_cap, self.physical_limit(velocity))

    def command(self, target: np.ndarray, position: np.ndarray, velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """늦게 도착한 목표각과 최신 각도/속도로 PD 요청값·제한 후 출력·상한을 계산한다."""
        raw = self.kp * (target - position) - self.kd * velocity
        limit = self.effective_limit(velocity)
        return raw, np.clip(raw, -limit, limit), limit
