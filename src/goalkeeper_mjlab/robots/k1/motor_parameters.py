"""모터가 실제로 낼 수 있다고 모델링한 힘·속도·회전 관성. PD 설정은 controller_profiles.py에 둔다.

공식 참조는 BoosterRobotics/booster_train의 고정 커밋이다. 실물 모터를 측정한 결과는 아니다.
로컬 booster_mjlab은 공식 저장소와 다른 프로젝트이므로 두 출처를 분리한다.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

OFFICIAL_COMMIT = "83037fde138c49cd8d6459a4b592538e6e979ffb"
OFFICIAL_BASE = f"https://github.com/BoosterRobotics/booster_train/blob/{OFFICIAL_COMMIT}/source/booster_train/booster_train/assets/robots"
OFFICIAL_MOTOR_SOURCE = OFFICIAL_BASE + "/actuator.py"
OFFICIAL_K1_SOURCE = OFFICIAL_BASE + "/booster_k1.py"
LOCAL_REFERENCE_COMMIT = "d92e11395d961096d639446c0a8a176f12bb51da"


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


# 공식 actuator.py의 동일 모터 모델. Hip Roll의 43/12.15/8.38은 로컬의 76/12.57/2.62와 다르다.
HEAD_MOTOR = MotorParameters(model_name="HT4438", armature=0.001, peak_effort=6.0, velocity_limit=7.85, knee_point_velocity=10.47)
ARM_MOTOR = MotorParameters(model_name="R14", armature=0.001, peak_effort=14.0, velocity_limit=33.51, knee_point_velocity=5.24)
HIP_PITCH_MOTOR = MotorParameters(model_name="E6408", armature=0.0478125, peak_effort=68.0, velocity_limit=14.66, knee_point_velocity=1.88)
HIP_ROLL_MOTOR = MotorParameters(model_name="E4315", armature=0.0339552, peak_effort=43.0, velocity_limit=12.15, knee_point_velocity=8.38)
HIP_YAW_MOTOR = MotorParameters(model_name="E4310", armature=0.0282528, peak_effort=38.3, velocity_limit=17.59, knee_point_velocity=7.85)
KNEE_MOTOR = MotorParameters(model_name="E6416", armature=0.095625, peak_effort=112.0, velocity_limit=12.57, knee_point_velocity=2.09)
# 발목은 두 모터가 연결 기구를 통해 움직인다. 현재 serial 모델에는 공식의 환산된 값을 적용한다.
# 공식 booster_k1.py가 armature_ratio=(1.4, 0.4)를 명시한다. wrapper 기본값 (2, 2)보다 이 override가 우선이다.
ANKLE_PITCH_MOTOR = replace(HIP_YAW_MOTOR, model_name="K1 ankle pitch / E4310", armature=HIP_YAW_MOTOR.armature * 1.4)
ANKLE_ROLL_MOTOR = replace(HIP_YAW_MOTOR, model_name="K1 ankle roll / E4310", armature=HIP_YAW_MOTOR.armature * 0.4)

OFFICIAL_MOTORS = {
    "head": HEAD_MOTOR, "arm": ARM_MOTOR,
    "hip_pitch": HIP_PITCH_MOTOR, "hip_roll": HIP_ROLL_MOTOR, "hip_yaw": HIP_YAW_MOTOR,
    "knee": KNEE_MOTOR, "ankle_pitch": ANKLE_PITCH_MOTOR, "ankle_roll": ANKLE_ROLL_MOTOR,
}
# 로컬 모터 테이블을 대조하는 옵션이다. gain/지연/정책까지 과거 학습 전체를 재현하는 설정은 아니다.
LEGACY_MOTORS = {
    **OFFICIAL_MOTORS,
    "hip_roll": replace(HIP_ROLL_MOTOR, peak_effort=76.0, velocity_limit=12.57, knee_point_velocity=2.62),
    "ankle_pitch": replace(ANKLE_PITCH_MOTOR, armature=HIP_YAW_MOTOR.armature * 2.0),
    "ankle_roll": replace(ANKLE_ROLL_MOTOR, armature=HIP_YAW_MOTOR.armature * 2.0),
}


def motor_group(joint_name: str) -> str:
    if joint_name.startswith("Head_"):
        return "head"
    for part, group in (("Hip_Pitch", "hip_pitch"), ("Hip_Roll", "hip_roll"), ("Hip_Yaw", "hip_yaw"),
                        ("Knee_Pitch", "knee"), ("Ankle_Pitch", "ankle_pitch"), ("Ankle_Roll", "ankle_roll")):
        if joint_name.endswith(part):
            return group
    if "_Shoulder_" in joint_name or "_Elbow_" in joint_name:
        return "arm"
    raise ValueError(f"Unknown K1 joint: {joint_name}")


def delay_group(joint_name: str) -> str:
    group = motor_group(joint_name)
    return "feet" if group.startswith("ankle") else group if group in ("head", "arm") else "legs"


def get_motors(source: str = "official") -> dict[str, MotorParameters]:
    if source not in ("official", "legacy"):
        raise ValueError("motor source must be official or legacy")
    return dict(OFFICIAL_MOTORS if source == "official" else LEGACY_MOTORS)
