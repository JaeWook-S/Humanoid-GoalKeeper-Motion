"""
    골키퍼 공통 환경 설정.
    항목별 설정을 먼저 정의하고 마지막 ManagerBasedRlEnvCfg에서 한 번에 조립한다.
    K1 모델과 모터 연결은 config/k1/env_cfgs.py의 goalkeeper_k1_env_cfg()가 맡는다.
"""

from dataclasses import replace
from functools import partial
import math

import mujoco
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.managers import EventTermCfg, MetricsTermCfg, TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.spec_config import CollisionCfg
from mjlab.viewer.viewer_config import ViewerConfig

from goalkeeper_mjlab.environment.ball.ball_constants import (
    BallCfg,
    get_spec as ball_spec,
)
from goalkeeper_mjlab.environment.goal.goal_constants import (
    GoalCfg,
    get_opening,
    get_spec as goal_spec,
)
from goalkeeper_mjlab.tasks.goalkeeper.mdp.events import ShotCfg, reset_ball_shot
from goalkeeper_mjlab.tasks.goalkeeper.mdp.metrics import ShotTracker, shot_metric
from goalkeeper_mjlab.tasks.goalkeeper.mdp.terminations import (
    shot_goal,
    shot_out,
    shot_timeout,
)


def _add_goal_center(spec: mujoco.MjSpec, goal: GoalCfg) -> None:
    """골문 중앙을 표시한다. 공과 충돌하는 물체는 아니다."""
    spec.body("goal/frame").add_site(
        name="goal/goal_line_center",
        pos=(0, 0, get_opening(goal).inner_height / 2),
        size=(0.008, 0, 0),
        rgba=(1, 0.2, 0.2, 1),
    )


def make_goalkeeper_env_cfg(
    *,
    ball=BallCfg(),
    goal=GoalCfg(),
    shots: ShotCfg = ShotCfg(),
    ball_position=(2.0, 0.0, None),
    goal_distance=0.75,
) -> ManagerBasedRlEnvCfg:
    """
        공·골대·슛의 공통 설정. 실제 실행에는 K1 환경 진입점을 사용한다.
        wirlwind와 같이 설정을 항목별로 읽고, 맨 아래 return에서 연결 관계를 확인한다.
        반환 후 cfg에 슛 설정을 하나씩 덧붙이는 구성은 사용하지 않는다.
    """
    # ShotCfg는 커리큘럼에서 수정할 수 있다. 호출마다 복사해 기본 설정을 공유하지 않는다.
    # 골키퍼 환경에는 항상 슛이 있으므로 None으로 기능을 끄는 분기는 두지 않는다.
    shots = replace(shots)

    # 배치값 확인: 공 중심이 바닥 속에 들어가거나 골대 거리가 잘못되는 경우를 막는다.
    # 모터값은 로봇 설정이, 랜덤 슛의 범위와 도달 가능성은 발사 이벤트가 검사한다.
    if not math.isfinite(goal_distance) or goal_distance <= 0:
        raise ValueError("goal_distance must be finite and positive")
    if len(ball_position) != 3 or ball_position[0] is None or ball_position[1] is None:
        raise ValueError("ball_position must be (x > 0, y, z or None)")
    x, y, z = ball_position
    z = ball.radius if z is None else z
    if not all(math.isfinite(v) for v in (x, y, z)) or x <= 0 or z < ball.radius:
        raise ValueError(
            "Ball position must be finite, in front, and at least one radius above the floor"
        )

    opening = get_opening(goal)
    dimensions = {
        "ball_radius": ball.radius,
        "goal_width": opening.inner_width,
        "goal_height": opening.inner_height,
        "goal_distance": goal_distance,
    }

    ##
    # Observations / Actions / Commands
    ##

    # 아직 정책 입력·출력과 명령은 구현하지 않았다. 빈 dict도 아래 조립부에 명시해
    # 어떤 기능이 연결되어 있고 어떤 기능이 남았는지 한눈에 확인할 수 있게 한다.
    observations = {}
    actions = {}
    commands = {}

    ##
    # Events
    ##

    # events를 직접 넘기면 MJLab 설정 객체의 기본 dict를 대체한다.
    # 따라서 기존과 같은 기본 reset을 먼저 명시하고 그 뒤에 공 발사를 연결해야 한다.
    # reset 자체는 직접 구현하지 않고 MJLab 기본 함수를 그대로 사용한다.
    events = {
        "reset_scene_to_default": EventTermCfg(
            func=reset_scene_to_default,
            mode="reset",
        ),
        "launch_ball": EventTermCfg(
            func=reset_ball_shot,
            mode="reset",
            params={"shot_cfg": shots, **dimensions},
        ),
    }

    ##
    # Rewards / Terminations / Curriculum / Metrics / Sensors
    ##

    # 보상과 자동 난이도 변경은 후속 구현이다. 슛 센서·기록·종료 조건은 항상 연결한다.
    rewards = {}
    curriculum = {}

    # 로봇과 공의 접촉 여부만 기본 ContactSensor로 측정한다. 접촉 힘은 수집하지 않는다.
    # secondary를 공으로 제한하므로 발-바닥 접촉은 공 접촉으로 기록되지 않는다.
    sensors = (
        ContactSensorCfg(
            name="robot_ball_contact",
            primary=ContactMatch(mode="body", pattern=".*", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="ball", entity="ball"),
            fields=("found",),
            reduce="maxforce",
            num_slots=1,
        ),
    )

    # per_substep=True: 정책 사이의 물리 계산도 모두 확인해 짧은 접촉과 빠른 통과를 잡는다.
    # 나머지 metric은 ShotTracker가 기록한 사실을 에피소드 통계로 읽는다.
    metrics = {
        "shot_state": MetricsTermCfg(
            func=ShotTracker,
            params=dimensions,
            per_substep=True,
            reduce="last",
        ),
        "shot_goal": MetricsTermCfg(
            func=shot_metric,
            params={"name": "goal"},
            reduce="last",
        ),
        "shot_touched": MetricsTermCfg(
            func=shot_metric,
            params={"name": "touched"},
            reduce="last",
        ),
        "shot_out": MetricsTermCfg(
            func=shot_metric,
            params={"name": "out"},
            reduce="last",
        ),
        "shot_timeout": MetricsTermCfg(
            func=shot_metric,
            params={"name": "timeout"},
            reduce="last",
        ),
    }

    terminations = {
        "goal": TerminationTermCfg(func=shot_goal),
        "ball_out": TerminationTermCfg(func=shot_out),
        # 시간 초과는 결과를 더 관찰하지 않고 다음 슛으로 넘어가는 truncation이다.
        "shot_timeout": TerminationTermCfg(func=shot_timeout, time_out=True),
    }
    episode_length_s = shots.episode_timeout_s

    ##
    # Environment
    ##

    # 위에서 정의한 항목을 이곳에서 한 번에 조립한다. 환경 전체를 볼 때는 이 return부터 읽는다.
    # K1 설정은 여기에서 반환한 공통 장면에 robot과 모터 startup 이벤트를 추가한다.
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(
                collisions=(
                    CollisionCfg(
                        geom_names_expr=(".*",),
                        contype=1,
                        conaffinity=1,
                        condim=3,
                        priority=0,
                        friction=(0.0, 0.0, 0.0),
                        solmix=1.0,
                    ),
                ),
                terrain_type="plane",
            ),
            entities={
                "ball": EntityCfg(
                    spec_fn=partial(ball_spec, ball),
                    init_state=EntityCfg.InitialStateCfg(pos=(x, y, z)),
                ),
                # 로봇 앞은 +X, 골문 뒤는 -X다. 골대를 -90도 돌려 입구가 로봇 쪽을 향하게 한다.
                "goal": EntityCfg(
                    spec_fn=partial(goal_spec, goal),
                    init_state=EntityCfg.InitialStateCfg(
                        pos=(-goal_distance, 0.0, 0.0),
                        rot=(math.cos(-math.pi / 4), 0.0, 0.0, math.sin(-math.pi / 4)),
                    ),
                ),
            },
            sensors=sensors,
            num_envs=1,
            env_spacing=4.0,
            spec_fn=partial(_add_goal_center, goal=goal),
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum=curriculum,
        metrics=metrics,
        viewer=ViewerConfig(),
        sim=SimulationCfg(
            nconmax=80,
            njmax=1500,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
                # Newton·implicitfast·tolerance=1e-8은 wirlwind처럼 MJLab 기본값을 사용한다.
                # elliptic은 기존 땅볼의 구름 마찰을 위해 유지한다. wirlwind 기본값과 다르다.
                cone="elliptic",
            ),
        ),
        decimation=4,  # 물리 200 Hz, 정책 50 Hz.
        episode_length_s=episode_length_s,
        # seed=0,
    )
