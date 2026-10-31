"""목표각까지 얼마나 세게 밀고, 움직임을 얼마나 억제할지 정하는 제어기 설정.

모터 자체의 한계는 motor_parameters.py, 프로그램이 요청하는 힘의 상한은 이 파일에 둔다.
아래 hold gain은 wirlwind 팀 코드의 PD를 사용한다. 공식 최신 PD와 다름을 의도적으로 표시한다.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from goalkeeper_mjlab.robots.k1.constants import K1_JOINT_ORDER, RobotPoseCfg
from goalkeeper_mjlab.robots.k1.motor_parameters import MotorParameters


@dataclass(frozen=True, kw_only=True)
class ControllerParameters:
    # PD의 기본 생각: 목표와 차이가 크면 밀어 주고(P), 너무 빨리 움직이면 제동을 건다(D).
    # 예: kp=80, 목표와 차이가 0.1 rad, 속도가 0이면 우선 8 N*m를 요청한다.
    # 요청이 커도 모터 한계와 output_limit을 넘는 실제 힘은 나오지 않도록 뒤에서 제한한다.
    # kp가 크면 목표 각도에서 벗어났을 때 더 세게 되돌린다. 단위 N*m/rad.
    kp: float
    # kd는 빠르게 움직이는 관절에 브레이크를 거는 정도다. 단위 N*m*s/rad.
    # 관절의 수동 마찰/모터 armature와 별개이며, 현재 속도를 보고 제어기가 내는 회전력이다.
    kd: float
    output_limit: float  # N*m: 소프트웨어 상한. 실제 출력은 이 값과 모터 물리 한계 중 작은 값이다.

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.kp, self.kd, self.output_limit)) or self.kp <= 0 or self.kd < 0 or self.output_limit <= 0:
            raise ValueError("Invalid PD gains or controller output limit")


def hold_controllers() -> dict[str, ControllerParameters]:
    # [공식과 다름] gain은 wirlwind 팀 코드의 k1_constants.py에 있는 head 4/.25, arm 10/1, leg 80/4, ankle 50/2.
    # 출력 상한은 현재 공식 모터의 저속 한계로 명시했다. Hip Roll은 wirlwind 팀 코드의 76 대신 Booster 공식 값 43이다.
    # 실물용 gain을 보증하는 설정이 아니라 현재 MJCF에서 준비 자세 유지를 검증할 설정이다.
    return {
        "head": ControllerParameters(kp=4.0, kd=0.25, output_limit=6.0),
        "arm": ControllerParameters(kp=10.0, kd=1.0, output_limit=14.0),
        "hip_pitch": ControllerParameters(kp=80.0, kd=4.0, output_limit=68.0),
        "hip_roll": ControllerParameters(kp=80.0, kd=4.0, output_limit=43.0),
        "hip_yaw": ControllerParameters(kp=80.0, kd=4.0, output_limit=38.3),
        "knee": ControllerParameters(kp=80.0, kd=4.0, output_limit=112.0),
        "ankle_pitch": ControllerParameters(kp=50.0, kd=2.0, output_limit=38.3),
        "ankle_roll": ControllerParameters(kp=50.0, kd=2.0, output_limit=38.3),
    }


def official_reference_controllers(motors: dict[str, MotorParameters]) -> dict[str, ControllerParameters]:
    """공식 K1 gain 계산을 비교 자료로 제공한다. 현재 모델의 기본 hold 설정과 구분한다."""
    result = {}
    for group, motor in motors.items():
        # 이 주파수는 공식이 gain을 계산할 때 쓰는 반응 특성이다. 정책 호출 주기 50 Hz와 다른 값이다.
        frequency = 10.0 if group in ("head", "arm") else 4.0
        ratio = 2.0 if group in ("head", "arm") else 1.0 if group == "knee" else 1.5
        # 공식 소스가 사용한 pi 자리수까지 유지해 출처와 수치 대조가 가능하도록 한다.
        omega = 2.0 * 3.1415926535 * frequency
        result[group] = ControllerParameters(kp=motor.armature * omega**2,
                                             kd=2.0 * ratio * motor.armature * omega,
                                             output_limit=motor.peak_effort)
    return result


# 공식 BOOSTER_K1_CFG.init_state의 관절각: 어깨 Roll만 -1.3/+1.3, 나머지 0 rad.
# 몸통 높이는 소스의 0.57을 복사하지 않고, 현재 MJCF의 발바닥 위치에서 다시 계산한다.
# 1단계의 다리를 굽힌 검사 자세는 이 gain으로 서 있지 못해, 검증용 준비 자세를 따로 둔다.
READY_JOINT_POS = tuple(-1.3 if name == "Left_Shoulder_Roll" else
                        1.3 if name == "Right_Shoulder_Roll" else 0.0 for name in K1_JOINT_ORDER)
READY_POSE = RobotPoseCfg(joint_pos=READY_JOINT_POS)
