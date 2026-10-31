"""MuJoCo 내장 위치 제어기에 모터 관성·PD·실제 출력 상한을 연결한다."""

from __future__ import annotations

import mujoco

from goalkeeper_mjlab.robots.k1.constants import K1_JOINT_ORDER
from goalkeeper_mjlab.robots.k1.torque_speed import JointParameters


def add_position_actuator(spec: mujoco.MjSpec, joint_name: str, armature: float, cap: float, kp: float, kd: float):
    # 회전 관성은 joint에 반영해야 실제 운동 계산에 사용된다. 메타데이터에 적기만 해서는 적용되지 않는다.
    spec.joint(joint_name).armature = armature
    actuator = spec.add_actuator(name=joint_name, target=joint_name)
    # 엔진 안에서 PD를 계산해야 implicitfast 적분기가 제동 효과까지 반영해 빠른 관절의 진동을 줄인다.
    # 공식 Isaac의 명시적 PD와 구현 방식이 다르며, wirlwind 팀 코드의 MuJoCo 이식 방식을 따른다.
    actuator.set_to_position(kp=kp, kv=kd)
    actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
    actuator.gear = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # ctrl은 목표각(rad), forcerange는 실제 회전력(N*m)이다. 서로 다른 단위를 혼합하지 않는다.
    actuator.ctrllimited = False
    actuator.forcelimited = True
    actuator.forcerange = (-cap, cap)  # 엔진 안에도 고정 상한을 두어 실제 actuator_force까지 제한한다.
    return actuator


def configure_actuators(spec: mujoco.MjSpec, parameters: JointParameters, prefix: str = "robot/") -> None:
    for i, name in enumerate(K1_JOINT_ORDER):
        cap = float(min(parameters.peak[i], parameters.controller_cap[i]))
        add_position_actuator(spec, prefix + name, float(parameters.armature[i]), cap,
                              float(parameters.kp[i]), float(parameters.kd[i]))
