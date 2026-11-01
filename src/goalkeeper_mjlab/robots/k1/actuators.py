"""wirlwind와 같은 역할: 모터·PD 자료형과 기본 위치 actuator의 설정 클래스.

설정 수치는 constants.py에서 관리한다. PD·지연·부분 reset은 MJLab 기본 구현을 사용한다.
"""

from dataclasses import dataclass
import math

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.actuator.actuator import TransmissionType


@dataclass(frozen=True, kw_only=True)
class MotorParameters:
    model_name: str
    # armature는 마찰이 아니다. 모터 내부 회전 부품 때문에 관절 속도를 바꾸기 어려운 정도다.
    # 단위 kg*m². 값이 크면 같은 회전력으로도 속도가 더 천천히 변한다.
    armature: float
    peak_effort: float  # 최대 회전력(N*m). 일상적인 '미는 힘'을 관절의 회전에 적용한 값이다.
    # rad/s는 1초 동안 변하는 각도다. 이 속도부터는 단순 모델의 모터 출력이 0이 된다.
    # 외력으로 더 빠르게 움직일 수는 있다. 실제 관절 속도를 이 숫자로 강제 수정하지 않는다.
    velocity_limit: float
    # rad/s: 여기부터 속도가 빨라질수록 낼 수 있는 회전력이 줄어든다.
    # 이름의 knee는 무릎 관절이 아니라, 힘-속도 그래프가 꺾이는 지점을 뜻한다.
    knee_point_velocity: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.armature, self.peak_effort, self.velocity_limit, self.knee_point_velocity)):
            raise ValueError("Motor values must be finite")
        if self.armature < 0 or self.peak_effort <= 0 or self.velocity_limit <= 0 or self.knee_point_velocity < 0:
            raise ValueError("Invalid motor inertia, torque or speed")

    @property
    def effective_knee_point_velocity(self) -> float:
        # 공식 Head 값은 감소 시작 속도가 최고 속도보다 크다. 공식 실행 코드처럼 범위 안으로 제한한다.
        return min(self.knee_point_velocity, self.velocity_limit)


@dataclass(frozen=True, kw_only=True)
class ControllerParameters:
    # PD의 기본 생각: 목표와 차이가 크면 밀어 주고(P), 너무 빨리 움직이면 제동을 건다(D).
    # 예: stiffness=80, 목표와 차이가 0.1 rad, 속도가 0이면 우선 8 N*m를 요청한다.
    # 요청이 커도 모터 한계와 output_limit을 넘는 실제 힘은 나오지 않도록 뒤에서 제한한다.
    # stiffness가 크면 목표 각도에서 벗어났을 때 더 세게 되돌린다. 단위 N*m/rad.
    stiffness: float
    # damping은 빠르게 움직이는 관절에 브레이크를 거는 정도다. 단위 N*m*s/rad.
    # 관절의 수동 마찰/모터 armature와 별개이며, 현재 속도를 보고 제어기가 내는 회전력이다.
    damping: float
    output_limit: float  # N*m: 소프트웨어 상한. 실제 출력은 이 값과 모터 물리 한계 중 작은 값이다.

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.stiffness, self.damping, self.output_limit)) or self.stiffness <= 0 or self.damping < 0 or self.output_limit <= 0:
            raise ValueError("Invalid PD gains or controller output limit")


@dataclass(kw_only=True)
class MotorPositionActuatorCfg(BuiltinPositionActuatorCfg):
    # 실제 PD·명령 지연·부분 reset은 상속한 MJLab 기본 actuator가 처리한다.
    # 아래 두 필드는 속도별 토크 갱신에 필요한 원래의 물리값과 프로그램 상한이다.
    motor: MotorParameters
    controller: ControllerParameters

    def build(self, entity, target_ids, target_names):
        from goalkeeper_mjlab.robots.k1.torque_speed import MotorEntity
        if not isinstance(entity, MotorEntity):
            raise TypeError("MotorPositionActuatorCfg requires MotorEntityCfg")
        return super().build(entity, target_ids, target_names)

    def __post_init__(self) -> None:
        super().__post_init__()
        # wirlwind tracking과 같은 두 검사만 추가한다. 설정 객체를 만드는 시점에 실행된다.
        # 이 토크 제한은 관절 제어용이므로 다른 제어 대상을 받으면 오류를 낸다.
        if self.transmission_type != TransmissionType.JOINT:
            raise ValueError("Motor torque-speed limits require joint transmission")
        # 제한값을 비워 두면 엔진에서 힘 제한이 꺼질 수 있으므로 생성을 거부한다.
        if self.effort_limit is None:
            raise ValueError("Motor torque-speed limits require effort_limit")
