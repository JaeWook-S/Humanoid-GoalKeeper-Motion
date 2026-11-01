"""골키퍼 환경 설정. wirlwind처럼 MJLab 기본 ManagerBasedRlEnvCfg를 반환한다.

현재는 장면·모터·물리 주기·기본 reset까지 구성한다.
관측/action/보상/종료/PPO는 후속 구현 대상이며, 검사용 Stage 설정은 여기에 두지 않는다.
"""

from functools import partial
import math

import mujoco
import numpy as np
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import EventTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.spec_config import CollisionCfg

from goalkeeper_mjlab.environment.ball.ball_constants import BallCfg, get_spec as ball_spec
from goalkeeper_mjlab.environment.goal.goal_constants import GoalCfg, get_opening, get_spec as goal_spec
from goalkeeper_mjlab.robots.k1.constants import ACTUATOR_DELAY, HOME_POSE, get_k1_robot_cfg, yaw_quaternion
from goalkeeper_mjlab.robots.k1.torque_speed import initialize_motor_limits


def _add_goal_center(spec: mujoco.MjSpec, goal: GoalCfg) -> None:
    # 골문 가운데를 표시하는 점이다. 충돌체가 아니므로 공을 막지 않는다.
    spec.body("goal/frame").add_site(name="goal/goal_line_center",
        pos=(0, 0, get_opening(goal).inner_height / 2), size=(0.008, 0, 0), rgba=(1, 0.2, 0.2, 1))


def make_goalkeeper_env_cfg(*, num_envs=1, env_spacing=4.0, robot_pose=HOME_POSE,
        ball=BallCfg(), goal=GoalCfg(), ball_position=(2.0, 0.0, None), goal_distance=0.75,
        physics_dt=0.005, decimation=4, integrator="implicitfast", solver="newton",
        iterations=10, ls_iterations=20, tolerance=1e-8, motor_source="official", controllers=None,
        delay_min_lag=ACTUATOR_DELAY["delay_min_lag"], delay_max_lag=ACTUATOR_DELAY["delay_max_lag"],
        delay_hold_prob=ACTUATOR_DELAY["delay_hold_prob"], seed=0) -> ManagerBasedRlEnvCfg:
    """학습·화면·검사가 공유하는 환경 설정. 반환값은 MJLab의 기본 설정 객체다."""
    for name, value in (("num_envs", num_envs), ("decimation", decimation),
                        ("iterations", iterations), ("ls_iterations", ls_iterations)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    values = (env_spacing, goal_distance, physics_dt, tolerance)
    if not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError("Distances, physics timestep and tolerance must be finite and positive")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if integrator not in ("implicitfast", "euler") or solver not in ("newton", "cg"):
        raise ValueError("Unsupported integrator or solver")
    if len(ball_position) != 3 or ball_position[0] is None or ball_position[1] is None:
        raise ValueError("ball_position must be (x > 0, y, z or None)")
    x, y, z = ball_position
    z = ball.radius if z is None else z
    if not all(math.isfinite(v) for v in (x, y, z)) or x <= 0 or z < ball.radius:
        raise ValueError("Ball position must be finite, in front, and at least one radius above the floor")
    initial = EntityCfg.InitialStateCfg
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(num_envs=num_envs, env_spacing=env_spacing,
            terrain=TerrainEntityCfg(terrain_type="plane", collisions=(CollisionCfg(
                geom_names_expr=(".*",), contype=1, conaffinity=1, condim=3,
                priority=0, friction=(0.0, 0.0, 0.0), solmix=1.0,
            ),)),
            entities={
                "robot": get_k1_robot_cfg(pose=robot_pose, motor_source=motor_source, controllers=controllers,
                    delay_min_lag=delay_min_lag, delay_max_lag=delay_max_lag, delay_hold_prob=delay_hold_prob),
                "ball": EntityCfg(spec_fn=partial(ball_spec, ball), init_state=initial(pos=(x, y, z))),
                # 로봇 앞은 +X, 골대는 -X다. 자산을 -90도 돌려 골문 입구를 로봇 쪽으로 향하게 한다.
                "goal": EntityCfg(spec_fn=partial(goal_spec, goal),
                    init_state=initial(pos=(-goal_distance, 0.0, 0.0), rot=yaw_quaternion(-np.pi / 2))),
            }, spec_fn=partial(_add_goal_center, goal=goal)),
        # 5 ms마다 물리 계산(200 Hz), 4회마다 정책/목표각 갱신(50 Hz).
        decimation=decimation, seed=seed,
        sim=SimulationCfg(nconmax=80, njmax=1500, mujoco=MujocoCfg(
            timestep=physics_dt, integrator=integrator, solver=solver,
            # 접촉 솔버의 최대 반복/탐색 횟수다. 물리 스텝 수를 늘리는 값이 아니다.
            iterations=iterations, ls_iterations=ls_iterations, tolerance=tolerance, cone="elliptic",
        )),
    )
    # 기본 reset_scene_to_default 이벤트는 그대로 둔다. 환경별 상태와 원점 처리는 MJLab이 맡는다.
    cfg.events["motor_limits"] = EventTermCfg(mode="startup", func=initialize_motor_limits)
    cfg.viewer.body_name = "Trunk"
    return cfg
