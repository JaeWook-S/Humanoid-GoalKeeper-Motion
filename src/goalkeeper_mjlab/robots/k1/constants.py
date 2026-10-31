"""K1 serial-ankle 모델의 구조와 검사 전용 초기 자세.

참조 출처: wirlwind 팀 코드(저장소명: booster_mjlab), 커밋 d92e11395d961096d639446c0a8a176f12bb51da.
모터·PD·실행 구성은 2단계 모듈에서 연결하고, 정책 action scale은 4단계에서 설정한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import mujoco
import numpy as np

from goalkeeper_mjlab.assets import load_model_spec

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

# HOME의 관절각
_SOURCE_HOME = {
    "Left_Shoulder_Roll": -1.4, "Left_Elbow_Yaw": -0.4,
    "Right_Shoulder_Roll": 1.4, "Right_Elbow_Yaw": 0.4,
    "Left_Hip_Pitch": -0.4, "Left_Knee_Pitch": 0.8, "Left_Ankle_Pitch": -0.4,
    "Right_Hip_Pitch": -0.4, "Right_Knee_Pitch": 0.8, "Right_Ankle_Pitch": -0.4,
}
INSPECTION_JOINT_POS = tuple(_SOURCE_HOME.get(name, 0.0) for name in K1_JOINT_ORDER)


@dataclass(frozen=True)
class RobotPoseCfg:
    joint_pos: tuple[float, ...] = INSPECTION_JOINT_POS
    yaw: float = 0.0
    sole_clearance: float = 0.001  # 검사 시 지면 관통을 피하기 위한 1 mm 여유. 학습값은 미확정.

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
        from dataclasses import replace
        return replace(self, joint_pos=tuple(values.get(n, q) for n, q in zip(K1_JOINT_ORDER, self.joint_pos)))


def get_spec() -> mujoco.MjSpec:
    """wirlwind 참조 자산을 보존하고 호출마다 독립적인 MjSpec을 반환한다."""
    spec = load_model_spec(K1_XML)
    for geom in spec.geoms:
        if "_collision" in geom.name.lower():
            # wirlwind 팀의 FULL_COLLISION과 같은 접촉 설정. 손/팔/몸통도 공과 접촉한다.
            geom.group = 3
            geom.contype = geom.conaffinity = 1
            geom.solref = (0.01, 1.0)
            geom.condim = 3
            geom.priority = 0
            geom.solmix = 0.001
            geom.friction[0] = 1.0 if geom.name in FOOT_COLLISION_GEOMS else 0.45
    return spec


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
