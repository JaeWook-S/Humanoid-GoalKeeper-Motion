"""K1 전용 연결. wirlwind처럼 공통 과제 설정을 만든 뒤 로봇을 붙인다."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import EventTermCfg

from goalkeeper_mjlab.environment.ball.ball_constants import BallCfg
from goalkeeper_mjlab.environment.goal.goal_constants import GoalCfg
from goalkeeper_mjlab.robots.k1.constants import get_k1_robot_cfg
from goalkeeper_mjlab.robots.k1.torque_speed import initialize_motor_limits
from goalkeeper_mjlab.tasks.goalkeeper.goalkeeper_env_cfg import make_goalkeeper_env_cfg
from goalkeeper_mjlab.tasks.goalkeeper.mdp.events import ShotCfg, reset_robot_joint_targets


def goalkeeper_k1_env_cfg(
    *,
    num_envs=1,
    ball=BallCfg(),
    goal=GoalCfg(),
    shots: ShotCfg = ShotCfg(),
    ball_position=(2.0, 0.0, None),
    goal_distance=0.75,  # 직접 호출할 때 로봇 기준점~뒤쪽 골라인 거리(m). CLI는 옵션값을 넘긴다.
) -> ManagerBasedRlEnvCfg:
    """자주 바꾸는 환경 개수·공·골대·슛·배치만 받는 실제 K1 환경 진입점."""
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
        raise ValueError("num_envs must be a positive integer")

    cfg = make_goalkeeper_env_cfg(
        ball=ball,
        goal=goal,
        shots=shots,
        ball_position=ball_position,
        goal_distance=goal_distance,
    )
    cfg.scene.num_envs = num_envs
    # HOME 자세·PD·지연·모터값은 constants.py의 기본 로봇 설정을 사용한다.
    # 로봇을 첫 항목에 둬 기존 관절/자산 배치 순서도 보존한다.
    cfg.scene.entities = {"robot": get_k1_robot_cfg(), **cfg.scene.entities}
    cfg.events["motor_limits"] = EventTermCfg(
        mode="startup", func=initialize_motor_limits
    )
    # [wirlwind와 다름] 기본 장면 reset 다음, 첫 제어 명령을 쓰기 전에 HOME 목표각을 지정한다.
    # 실제 자세만 HOME이고 지연 버퍼의 첫 목표가 0 rad인 시작/에피소드 전환 오류를 막는다.
    cfg.events["robot_joint_targets"] = EventTermCfg(
        mode="reset", func=reset_robot_joint_targets
    )
    cfg.viewer.body_name = "Trunk"
    return cfg
