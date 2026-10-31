"""1단계 장면 조립. 학습용 action/reward/reset/PPO는 아직 연결하지 않는다."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from goalkeeper_mjlab.environment.ball.ball_constants import BallCfg, get_spec as ball_spec
from goalkeeper_mjlab.environment.goal.goal_constants import GoalCfg, get_opening, get_spec as goal_spec
from goalkeeper_mjlab.robots.k1.constants import (
    K1_JOINT_ORDER, RobotPoseCfg, get_spec as robot_spec,
    initial_root_height, set_robot_pose, yaw_quaternion,
)

if TYPE_CHECKING:
    from mjlab.scene import Scene, SceneCfg


@dataclass(frozen=True)
class Stage1SceneCfg:
    robot_pose: RobotPoseCfg = field(default_factory=RobotPoseCfg)
    ball: BallCfg = field(default_factory=BallCfg)
    goal: GoalCfg = field(default_factory=GoalCfg)
    # 공 중심의 (로봇 앞쪽 거리, 좌우 위치, 바닥에서의 높이). 단위는 m, None이면 바닥에 놓는다.
    ball_position: tuple[float, float, float | None] = (2.0, 0.0, None)
    # 로봇 뒤쪽에서 골대 입구까지 거리(m). 기본 0.75는 검사 화면의 배치용 값이다.
    goal_distance: float = 0.75  # 최종 학습 배치 거리는 후속 단계에서 정한다.
    num_envs: int = 1
    env_spacing: float = 4.0  # 독립 물리 월드의 표시용 간격. 충돌 격리용 값이 아니다.
    preview_timestep: float = 0.002  # wirlwind 참조 K1 XML 값. 2단계에서 물리/제어 주기를 확정한다.

    def __post_init__(self) -> None:
        if isinstance(self.num_envs, bool) or not isinstance(self.num_envs, int) or self.num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        values = [self.goal_distance, self.env_spacing, self.preview_timestep]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Distances and preview_timestep must be finite and positive")
        if len(self.ball_position) != 3 or self.ball_position[0] is None or self.ball_position[0] <= 0:
            raise ValueError("ball_position must be (x > 0, y, z or None)")
        if self.ball_position[1] is None or not np.isfinite([v for v in self.ball_position if v is not None]).all():
            raise ValueError("ball_position must be finite")
        if self.ball_position[2] is not None and self.ball_position[2] < self.ball.radius:
            raise ValueError("Ball center must be at least one radius above the floor")

    @property
    def resolved_ball_position(self) -> tuple[float, float, float]:
        """공을 바닥에 놓으려면 중심을 반지름만큼 띄워야 하므로, 높이 생략 시 반지름을 사용한다."""
        x, y, z = self.ball_position
        return (x, y, self.ball.radius if z is None else z)

    @property
    def goal_position(self) -> tuple[float, float, float]:
        """로봇 앞쪽이 +X이므로, 골대는 -X 쪽에 놓는다. 높이 0은 골대 바닥이 바닥에 닿는 위치다."""
        return (-self.goal_distance, 0.0, 0.0)


def apply_scene_settings(spec: mujoco.MjSpec, cfg: Stage1SceneCfg) -> None:
    # 자식 XML의 option은 attach 시 전파되지 않는다. 전역 옵션은 합성 후 여기서 통일한다.
    spec.option.timestep = cfg.preview_timestep
    spec.option.gravity = (0.0, 0.0, -9.81)
    # wirlwind 공 설정의 rolling resistance 변환은 elliptic friction cone을 전제로 한다.
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    opening = get_opening(cfg.goal)
    # site는 위치를 눈으로 확인하는 표시다. 여기서는 골문 가운데에 작은 빨간 점을 찍는다.
    # 이 점은 공을 막지 않는다. 높이를 안쪽 골문 높이의 절반으로 두어 중앙에 표시한다.
    # 골대에 붙여 두므로, 환경별로 골대를 옮기면 점도 함께 따라간다.
    spec.body("goal/frame").add_site(name="goal/goal_line_center", pos=(0, 0, opening.inner_height / 2),
                                     size=(0.008, 0, 0), rgba=(1, 0.2, 0.2, 1))


def build_scene_spec(cfg: Stage1SceneCfg = Stage1SceneCfg(),
                     origin: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> mujoco.MjSpec:
    """MuJoCo만으로 조립/검사가 가능한 단일 환경 spec을 만든다."""
    if len(origin) != 3 or not np.isfinite(origin).all():
        raise ValueError("origin must contain three finite numbers")
    offset = np.asarray(origin, dtype=float)
    spec = mujoco.MjSpec()
    spec.modelname = "goalkeeper_stage1"
    spec.worldbody.add_geom(
        name="terrain", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=(0, 0, 0.1), pos=offset, rgba=(0.25, 0.30, 0.25, 1),
        contype=1, conaffinity=1, condim=3, priority=0,
        friction=(0, 0, 0), solmix=1.0,
    )
    height = initial_root_height(cfg.robot_pose)
    robot = robot_spec()
    robot.worldbody.bodies[0].pos = offset + (0.0, 0.0, height)
    robot.worldbody.bodies[0].quat = yaw_quaternion(cfg.robot_pose.yaw)
    ball = ball_spec(cfg.ball)
    ball.worldbody.bodies[0].pos = offset + cfg.resolved_ball_position
    goal = goal_spec(cfg.goal)
    goal.worldbody.bodies[0].pos = offset + cfg.goal_position
    # wirlwind 자산 파일의 골대 방향을 현재 장면에 맞춰 바닥 위에서 -90도 돌린다.
    # 그러면 골문 입구는 로봇 쪽을 향하고, 뒤쪽 연결대는 로봇 뒤쪽(-X)으로 뻗는다.
    # wirlwind 골대 자산의 가로 방향 X는 장면의 Y 방향이 된다. np.pi/2 rad가 90도다.
    goal.worldbody.bodies[0].quat = yaw_quaternion(-np.pi / 2)
    for name, child in (("robot", robot), ("ball", ball), ("goal", goal)):
        spec.attach(child, prefix=name + "/", frame=spec.worldbody.add_frame())
    apply_scene_settings(spec, cfg)
    model = spec.compile()
    data = mujoco.MjData(model)
    # qpos의 앞 22개가 로봇 관절이라고 가정하지 않고 이름으로 찾아 초기화한다.
    set_robot_pose(model, data, cfg.robot_pose, tuple(offset + (0.0, 0.0, height)), "robot/")
    spec.add_key(name="init_state", qpos=data.qpos.copy(), qvel=data.qvel.copy())
    return spec


def compile_scene(cfg: Stage1SceneCfg = Stage1SceneCfg(),
                  origin: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> tuple[mujoco.MjSpec, mujoco.MjModel, mujoco.MjData]:
    spec = build_scene_spec(cfg, origin)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
    mujoco.mj_forward(model, data)
    return spec, model, data


def make_mjlab_scene_cfg(cfg: Stage1SceneCfg = Stage1SceneCfg()) -> SceneCfg:
    """같은 모델을 기존 학습 환경의 MJLab SceneCfg로 연결한다. 아직 actuator는 없다."""
    try:
        from mjlab.entity import EntityCfg
        from mjlab.scene import SceneCfg
        from mjlab.terrains import TerrainEntityCfg
        from mjlab.utils.spec_config import CollisionCfg
    except ImportError as exc:
        raise RuntimeError("MJLab adapter requires the existing MJLab training environment") from exc
    initial = EntityCfg.InitialStateCfg
    height = initial_root_height(cfg.robot_pose)
    return SceneCfg(
        num_envs=cfg.num_envs, env_spacing=cfg.env_spacing,
        terrain=TerrainEntityCfg(terrain_type="plane", collisions=(CollisionCfg(
            geom_names_expr=(".*",), contype=1, conaffinity=1, condim=3,
            priority=0, friction=(0.0, 0.0, 0.0), solmix=1.0,
        ),)),
        entities={
            "robot": EntityCfg(spec_fn=robot_spec, init_state=initial(
                pos=(0.0, 0.0, height), rot=yaw_quaternion(cfg.robot_pose.yaw),
                joint_pos=dict(zip(K1_JOINT_ORDER, cfg.robot_pose.joint_pos, strict=True)),
            )),
            "ball": EntityCfg(spec_fn=partial(ball_spec, cfg.ball), init_state=initial(pos=cfg.resolved_ball_position)),
            "goal": EntityCfg(spec_fn=partial(goal_spec, cfg.goal), init_state=initial(
                pos=cfg.goal_position, rot=yaw_quaternion(-np.pi / 2),
            )),
        },
        spec_fn=partial(apply_scene_settings, cfg=cfg),
    )


def place_entities_at_origins(scene: Scene, env_ids=None) -> None:
    """MJLab 초기화 후 사용한다. 3단계 reset에서 재사용할 배치 API다."""
    import torch
    ids = torch.arange(scene.num_envs, device=scene.device) if env_ids is None else env_ids
    origins = scene.env_origins[ids]
    for name in ("robot", "ball"):
        entity = scene[name]
        state = entity.data.default_root_state[ids].clone()
        state[:, :3] += origins
        entity.write_root_link_pose_to_sim(state[:, :7], env_ids=ids)
        entity.write_root_link_velocity_to_sim(state[:, 7:], env_ids=ids)
    robot = scene["robot"]
    robot.write_joint_state_to_sim(robot.data.default_joint_pos[ids], robot.data.default_joint_vel[ids], env_ids=ids)
    # 고정 골대까지 같은 원점으로 옮겨야 병렬 환경별 상대 배치가 일치한다.
    goal = scene["goal"]
    pose = goal.data.default_root_state[ids].clone()
    pose[:, :3] += origins
    goal.write_mocap_pose_to_sim(pose, env_ids=ids)
