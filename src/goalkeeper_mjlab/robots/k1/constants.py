"""
    K1 공통 설정: 모델·준비 자세·충돌·모터 물리값·PD·지연·로봇 생성.

    구조 참고: wirlwind 팀 코드의 k1_constants.py (d92e11395d961096d639446c0a8a176f12bb51da).
    물리 한계와 PD는 같은 파일에서도 서로 다른 자료형으로 구분한다. 검증 코드는 이 설정을 불러 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping
import math

import mujoco
import numpy as np
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from goalkeeper_mjlab.assets import load_model_spec
from goalkeeper_mjlab.robots.k1.actuators import ControllerParameters, MotorParameters, MotorPositionActuatorCfg

K1_XML = Path(__file__).parent / "xmls" / "k1.xml"
K1_JOINT_ORDER = (
    "Head_Yaw", "Head_Pitch",
    "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch",
    "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch",
    "Right_Ankle_Pitch", "Right_Ankle_Roll",
)
TRUNK_BODY = "Trunk"
HAND_BODIES = ("left_hand_end_ball", "right_hand_end_ball")
HAND_COLLISION_GEOMS = ("left_hand_end_collision", "right_hand_end_collision")
FOOT_BODIES = ("left_foot_link", "right_foot_link")
FOOT_COLLISION_GEOMS = ("left_foot_collision", "right_foot_collision")

# 학습과 정적 검사에서 같은 수치를 사용한다. 아래 값을 공통 MJLab CollisionCfg에 전달한다.
# contype/conaffinity: 충돌 대상 선택, condim: 접촉 때 계산할 힘의 종류, solref/solmix: 접촉 반응 설정.
# wirlwind HOME keyframe의 관절각이다. 학습·모터 화면·형상 검사의 기본 자세로 함께 사용한다.
_SOURCE_HOME = {
    "Left_Shoulder_Roll": -1.4, "Left_Elbow_Yaw": -0.4,
    "Right_Shoulder_Roll": 1.4, "Right_Elbow_Yaw": 0.4,
    "Left_Hip_Pitch": -0.4, "Left_Knee_Pitch": 0.8, "Left_Ankle_Pitch": -0.4,
    "Right_Hip_Pitch": -0.4, "Right_Knee_Pitch": 0.8, "Right_Ankle_Pitch": -0.4,
}
HOME_JOINT_POS = tuple(_SOURCE_HOME.get(name, 0.0) for name in K1_JOINT_ORDER)


@dataclass(frozen=True)
class RobotPoseCfg:
    joint_pos: tuple[float, ...] = HOME_JOINT_POS
    yaw: float = 0.0
    sole_clearance: float = 0.001  # 발바닥이 지면 안에 들어가지 않게 두는 초기 여유(m). 0.001 m = 1 mm.

    def __post_init__(self) -> None:
        if len(self.joint_pos) != len(K1_JOINT_ORDER):
            raise ValueError("Robot pose must contain exactly 22 joint angles")
        if not np.isfinite((*self.joint_pos, self.yaw, self.sole_clearance)).all():
            raise ValueError("Robot pose values must be finite")
        if self.sole_clearance < 0:
            raise ValueError("sole_clearance must be nonnegative")

    def with_joint_overrides(self, values: Mapping[str, float]) -> RobotPoseCfg:
        unknown = set(values) - set(K1_JOINT_ORDER)
        if unknown:
            raise ValueError(f"Unknown robot joints: {sorted(unknown)}")
        return replace(self, joint_pos=tuple(values.get(n, q) for n, q in zip(K1_JOINT_ORDER, self.joint_pos)))


def get_spec() -> mujoco.MjSpec:
    """원본 자산의 독립 spec을 반환한다. 접촉 설정은 공통 MJLab CollisionCfg에서 적용한다."""
    spec = load_model_spec(K1_XML)
    for geom in spec.geoms:
        if "_collision" in geom.name.lower():
            # group은 화면 표시 분류다. wirlwind get_spec()처럼 충돌 형상을 같은 표시 그룹에 둔다.
            geom.group = 3
    return spec


# wirlwind의 FULL_COLLISION과 같은 위치에서 관리한다.
FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision.*",),
    contype=1, conaffinity=1, solref=(0.01, 1.0), condim=3, priority=0, solmix=0.001,
    # 발바닥은 1.0, 나머지 몸체는 0.45의 미끄럼 마찰을 사용한다.
    friction={r"^(left|right)_foot_collision$": (1.0,), ".*": (0.45,)},
    # [wirlwind와 다름] 선택하지 않은 geom의 XML 충돌 마스크를 보존한다. 기본 True로 바꾸면 동작이 달라진다.
    disable_other_geoms=False,
)


def yaw_quaternion(yaw: float) -> tuple[float, float, float, float]:
    """MuJoCo의 quaternion 순서는 w, x, y, z이다."""
    return (float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2)))


def set_robot_pose(model: mujoco.MjModel, data: mujoco.MjData, pose: RobotPoseCfg,
                   position: tuple[float, float, float], prefix: str = "") -> None:
    """관절 이름으로 주소를 찾아 배치한다. 합성 장면의 qpos 인덱스를 추측하지 않는다."""
    root_joint = model.joint(prefix + "floating_base_joint")
    address = int(root_joint.qposadr[0])
    data.qpos[address:address + 3] = position
    data.qpos[address + 3:address + 7] = yaw_quaternion(pose.yaw)
    for name, angle in zip(K1_JOINT_ORDER, pose.joint_pos, strict=True):
        joint = model.joint(prefix + name)
        low, high = joint.range
        if joint.limited[0] and not low <= angle <= high:
            raise ValueError(f"{name}={angle} rad is outside [{low}, {high}]")
        data.qpos[int(joint.qposadr[0])] = angle
    mujoco.mj_forward(model, data)


def geom_world_bounds(model: mujoco.MjModel, data: mujoco.MjData,
                      geom_id: int) -> tuple[np.ndarray, np.ndarray]:
    """메시는 실제 vertex, 그 외 형상은 로컬 AABB로 월드 경계를 계산한다."""
    mesh_id = int(model.geom_dataid[geom_id])
    if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
        start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
        points = model.mesh_vert[start:start + count].astype(float)
    else:
        center, half = model.geom_aabb[geom_id].reshape(2, 3)
        signs = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        points = center + signs * half
    points = points @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
    return points.min(axis=0), points.max(axis=0)


def initial_root_height(pose: RobotPoseCfg) -> float:
    """발바닥 충돌 mesh의 최저점으로 root 높이를 정한다. PD로 서 있게 만드는 기능은 아니다."""
    model = get_spec().compile()
    data = mujoco.MjData(model)
    set_robot_pose(model, data, pose, (0.0, 0.0, 0.0))
    bottom = min(geom_world_bounds(model, data, model.geom(n).id)[0][2] for n in FOOT_COLLISION_GEOMS)
    return float(-bottom + pose.sole_clearance)


OFFICIAL_COMMIT = "83037fde138c49cd8d6459a4b592538e6e979ffb"
OFFICIAL_BASE = f"https://github.com/BoosterRobotics/booster_train/blob/{OFFICIAL_COMMIT}/source/booster_train/booster_train/assets/robots"
OFFICIAL_MOTOR_SOURCE = OFFICIAL_BASE + "/actuator.py"
OFFICIAL_K1_SOURCE = OFFICIAL_BASE + "/booster_k1.py"
LOCAL_REFERENCE_COMMIT = "d92e11395d961096d639446c0a8a176f12bb51da"


# 공식 actuator.py의 동일 모터 모델. Hip Roll의 43/12.15/8.38은 wirlwind 팀 코드의 76/12.57/2.62와 다르다. -> 확인 필요
HEAD_MOTOR = MotorParameters(model_name="HT4438", armature=0.001, peak_effort=6.0, velocity_limit=7.85, knee_point_velocity=10.47)
ARM_MOTOR = MotorParameters(model_name="R14", armature=0.001, peak_effort=14.0, velocity_limit=33.51, knee_point_velocity=5.24)
HIP_PITCH_MOTOR = MotorParameters(model_name="E6408", armature=0.0478125, peak_effort=68.0, velocity_limit=14.66, knee_point_velocity=1.88)
HIP_ROLL_MOTOR = MotorParameters(model_name="E4315", armature=0.0339552, peak_effort=43.0, velocity_limit=12.15, knee_point_velocity=8.38)
HIP_YAW_MOTOR = MotorParameters(model_name="E4310", armature=0.0282528, peak_effort=38.3, velocity_limit=17.59, knee_point_velocity=7.85)
KNEE_MOTOR = MotorParameters(model_name="E6416", armature=0.095625, peak_effort=112.0, velocity_limit=12.57, knee_point_velocity=2.09)
# 발목은 두 모터가 연결 기구를 통해 움직인다. 현재 serial 모델에는 공식의 환산된 값을 적용한다.
# 공식 booster_k1.py가 armature_ratio=(1.4, 0.4)를 명시한다. wrapper 기본값 (2, 2)보다 이 override가 우선이다. -> 확인 필요
ANKLE_PITCH_MOTOR = replace(HIP_YAW_MOTOR, model_name="K1 ankle pitch / E4310", armature=HIP_YAW_MOTOR.armature * 1.4)
ANKLE_ROLL_MOTOR = replace(HIP_YAW_MOTOR, model_name="K1 ankle roll / E4310", armature=HIP_YAW_MOTOR.armature * 0.4)

OFFICIAL_MOTORS = {
    "head": HEAD_MOTOR, "arm": ARM_MOTOR,
    "hip_pitch": HIP_PITCH_MOTOR, "hip_roll": HIP_ROLL_MOTOR, "hip_yaw": HIP_YAW_MOTOR,
    "knee": KNEE_MOTOR, "ankle_pitch": ANKLE_PITCH_MOTOR, "ankle_roll": ANKLE_ROLL_MOTOR,
}
# wirlwind 팀의 모터 테이블을 대조하는 옵션이다. gain/지연/정책까지 과거 학습 전체를 재현하는 설정은 아니다.
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


def get_motors(source: str = "official") -> dict[str, MotorParameters]:
    if source not in ("official", "legacy"):
        raise ValueError("motor source must be official or legacy")
    return dict(OFFICIAL_MOTORS if source == "official" else LEGACY_MOTORS)


def hold_controllers() -> dict[str, ControllerParameters]:
    # [Booster 공식과 다름] gain은 wirlwind 팀 코드의 k1_constants.py에 있는 head 4/.25, arm 10/1, leg 80/4, ankle 50/2.
    # 출력 상한은 현재 공식 모터의 저속 한계로 명시했다. Hip Roll은 wirlwind 팀 코드의 76 대신 Booster 공식 값 43이다.
    # 실물용 gain을 보증하는 설정이 아니라 현재 MJCF에서 준비 자세 유지를 검증할 설정이다.
    return {
        "head": ControllerParameters(stiffness=4.0, damping=0.25, output_limit=6.0),
        "arm": ControllerParameters(stiffness=10.0, damping=1.0, output_limit=14.0),
        "hip_pitch": ControllerParameters(stiffness=80.0, damping=4.0, output_limit=68.0),
        "hip_roll": ControllerParameters(stiffness=80.0, damping=4.0, output_limit=43.0),
        "hip_yaw": ControllerParameters(stiffness=80.0, damping=4.0, output_limit=38.3),
        "knee": ControllerParameters(stiffness=80.0, damping=4.0, output_limit=112.0),
        "ankle_pitch": ControllerParameters(stiffness=50.0, damping=2.0, output_limit=38.3),
        "ankle_roll": ControllerParameters(stiffness=50.0, damping=2.0, output_limit=38.3),
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
        result[group] = ControllerParameters(
            stiffness=motor.armature * omega**2,
            damping=2.0 * ratio * motor.armature * omega,
            output_limit=motor.peak_effort,
        )
    return result


# root 높이는 wirlwind의 고정 0.5125 m를 복사하지 않고, 기존 발바닥 기준 자동 계산을 유지한다.
HOME_POSE = RobotPoseCfg()

# 0.3은 초가 아니라 이전 지연 유지 확률
ACTUATOR_DELAY = dict(delay_min_lag=2, delay_max_lag=8, delay_hold_prob=0.3)


def make_k1_articulation(*, motor_source="official", controllers=None,
                         delay_min_lag=ACTUATOR_DELAY["delay_min_lag"],
                         delay_max_lag=ACTUATOR_DELAY["delay_max_lag"],
                         delay_hold_prob=ACTUATOR_DELAY["delay_hold_prob"]):
    """wirlwind의 K1_ARTICULATION에 해당한다. 매 호출마다 독립적인 설정을 만든다."""
    motors = get_motors(motor_source)
    controllers = hold_controllers() if controllers is None else controllers
    if set(controllers) != set(motors):
        raise ValueError("Controller groups must cover every motor group exactly once")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (delay_min_lag, delay_max_lag)) or not 0 <= delay_min_lag <= delay_max_lag:
        raise ValueError("Command delay requires integers: 0 <= delay_min_lag <= delay_max_lag")
    if not math.isfinite(delay_hold_prob) or not 0 <= delay_hold_prob <= 1:
        raise ValueError("delay_hold_prob must be in [0, 1]")
    actuators = []
    for group, motor in motors.items():
        control = controllers[group]
        actuators.append(MotorPositionActuatorCfg(
            target_names_expr=tuple(n for n in K1_JOINT_ORDER if motor_group(n) == group),
            motor=motor, controller=control, armature=motor.armature,
            stiffness=control.stiffness, damping=control.damping,
            effort_limit=min(motor.peak_effort, control.output_limit),
            delay_min_lag=delay_min_lag, delay_max_lag=delay_max_lag, delay_hold_prob=delay_hold_prob,
        ))

    return EntityArticulationInfoCfg(actuators=tuple(actuators), soft_joint_pos_limit_factor=0.9)


K1_ARTICULATION = make_k1_articulation(**ACTUATOR_DELAY)


def get_k1_robot_cfg(*, pose=HOME_POSE, motor_source="official", controllers=None,
                     delay_min_lag=ACTUATOR_DELAY["delay_min_lag"],
                     delay_max_lag=ACTUATOR_DELAY["delay_max_lag"],
                     delay_hold_prob=ACTUATOR_DELAY["delay_hold_prob"]):
    """학습용 K1 로봇 설정. 검증 실행기도 이 함수를 그대로 호출한다."""
    from goalkeeper_mjlab.robots.k1.torque_speed import MotorEntityCfg
    return MotorEntityCfg(
        spec_fn=get_spec, collisions=(FULL_COLLISION,),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, initial_root_height(pose)), rot=yaw_quaternion(pose.yaw),
            joint_pos=dict(zip(K1_JOINT_ORDER, pose.joint_pos, strict=True)), joint_vel={".*": 0.0},
        ),
        articulation=make_k1_articulation(motor_source=motor_source, controllers=controllers,
            delay_min_lag=delay_min_lag, delay_max_lag=delay_max_lag, delay_hold_prob=delay_hold_prob),
    )
