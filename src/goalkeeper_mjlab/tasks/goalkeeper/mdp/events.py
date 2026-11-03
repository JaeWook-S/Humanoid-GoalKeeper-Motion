"""학습과 화면 검사가 함께 사용하는 랜덤 슛 생성·초기화 이벤트.

이 파일은 공을 어느 곳에서 어떤 속도로 출발시킬지만 정한다. 실제 이동·중력·충돌은
MuJoCo가 계산한다. 화면 검사에서만 따로 공을 움직이는 로직을 만들지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from goalkeeper_mjlab.environment.ball.ball_constants import BALL_ROLLING_RESISTANCE


@dataclass
class ShotCfg:
    """에피소드가 시작될 때 적용할 슛 분포. 거리 m, 시간 s, 속도 m/s 단위다.

    나중에 커리큘럼은 이 객체의 범위·확률을 바꾸면 된다. reset 시점에만 읽으므로
    설정을 변경하더라도 이미 날아가고 있는 공의 속도나 위치를 바꾸지 않는다.
    여기의 기본 범위는 초기 검사용 선택값이며, 실물 슈팅 측정에서 얻은 값은 아니다.
    """

    ground_probability: float = 0.5  # 0: 전부 뜬볼, 1: 전부 땅볼, 0.5: 각각 약 절반.
    launch_distance: tuple[float, float] = (1.0, 3.0)  # 로봇 앞(+X)의 출발 거리 범위.
    launch_y: tuple[float, float] = (-2.0, 2.0)  # 로봇 기준 좌우 출발 위치. +Y는 왼쪽이다.
    air_launch_height: tuple[float, float] = (0.2, 0.6)  # 뜬볼 공 중심의 출발 높이.
    speed: tuple[float, float] = (3.0, 8.0)  # 땅볼·뜬볼 모두 출발 순간의 이동 속력.
    # 골문에 공이 들어갈 수 있는 좌우 영역을 -1~+1로 표현한다.
    # 0은 중앙, ±1은 공 반지름까지 고려한 좌우 끝이다. 골대 크기가 바뀌어도 대응된다.
    target_y_fraction: tuple[float, float] = (-0.85, 0.85)
    # 공 중심이 갈 수 있는 최저 높이(반지름)를 0, 최고 높이(가로대 아래-반지름)를 1로 둔다.
    # 따라서 0.5는 골문 높이의 중간이다. 이것은 m 단위의 절대 높이가 아닌 비율이다.
    air_target_height_fraction: tuple[float, float] = (0.05, 0.85)
    max_sampling_attempts: int = 128  # 불가능한 위치·속도 조합을 무한히 다시 뽑지 않도록 제한.
    episode_timeout_s: float = 3.0  # 슛 하나를 관찰할 최대 시간. 실제 종료 연결은 env_cfgs에서 한다.
    out_margin: float = 2.0  # 발사 영역·골대 주변에 더 허용할 거리. 영역 이탈 판정에 사용한다.


@dataclass(frozen=True)
class ShotBatch:
    """엔진에 넣을 N개 공의 초기 상태만 반환한다. 세 텐서는 모두 [N, 3]이다.

    position은 환경 원점 기준 좌표, 두 velocity는 이동·회전 속도다.
    목표점·슛 종류·예상 도착 시간은 엔진 입력이 아니므로 별도 기록으로 남기지 않는다.
    """

    position: torch.Tensor
    linear_velocity: torch.Tensor
    angular_velocity: torch.Tensor


def _validate_shot_cfg(cfg: ShotCfg, ball_radius: float, goal_width: float,
                       goal_height: float, goal_distance: float, gravity: float) -> None:
    """커리큘럼이 바꾼 값도 reset 때 다시 확인해 잘못된 슛을 조용히 만들지 않는다."""
    values = (ball_radius, goal_width, goal_height, goal_distance, gravity,
              cfg.episode_timeout_s, cfg.out_margin)
    if not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError("Ball/goal dimensions, gravity, episode timeout and out margin must be finite and positive")
    if goal_width <= 2 * ball_radius or goal_height <= 2 * ball_radius:
        raise ValueError("The goal opening must be larger than the ball diameter")
    if not math.isfinite(cfg.ground_probability) or not 0 <= cfg.ground_probability <= 1:
        raise ValueError("ground_probability must be between 0 and 1")
    if (isinstance(cfg.max_sampling_attempts, bool) or not isinstance(cfg.max_sampling_attempts, int)
            or cfg.max_sampling_attempts < 1):
        raise ValueError("max_sampling_attempts must be a positive integer")
    for name in ("launch_distance", "launch_y", "air_launch_height", "speed",
                 "target_y_fraction", "air_target_height_fraction"):
        limits = getattr(cfg, name)
        if len(limits) != 2 or not all(math.isfinite(v) for v in limits) or limits[0] > limits[1]:
            raise ValueError(f"{name} must be a finite (minimum, maximum) range")
    if cfg.launch_distance[0] <= 0 or cfg.speed[0] <= 0:
        raise ValueError("launch_distance and speed must have positive minimums")
    # 땅볼만 발사한다면 뜬볼 출발 높이는 사용하지 않는다. 다른 공 크기를 쓰는 경우도 고려한다.
    if cfg.ground_probability < 1 and cfg.air_launch_height[0] < ball_radius:
        raise ValueError("air_launch_height must keep the whole ball above the floor")
    if cfg.target_y_fraction[0] < -1 or cfg.target_y_fraction[1] > 1:
        raise ValueError("target_y_fraction must stay within [-1, 1]")
    if cfg.air_target_height_fraction[0] < 0 or cfg.air_target_height_fraction[1] > 1:
        raise ValueError("air_target_height_fraction must stay within [0, 1]")


def sample_shots(cfg: ShotCfg, count: int, ball_radius: float, goal_width: float,
                 goal_height: float, goal_distance: float, gravity: float = 9.81,
                 device: str = "cpu", generator: torch.Generator | None = None) -> ShotBatch:
    """
        골문을 향하는 땅볼·뜬볼을 Torch 텐서로 한 번에 뽑는다.
        출발점과 목표점을 먼저 고른 뒤, 그 두 점을 연결하는 초기 속도를 계산한다.
        모든 속도 성분을 독립적으로 뽑으면 골문과 관계없는 방향으로 날아가기 때문이다.
        generator를 생략하면 MJLab이 seed로 초기화한 기본 난수 발생기를 그대로 사용한다.
    """
    _validate_shot_cfg(cfg, ball_radius, goal_width, goal_height, goal_distance, gravity)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a nonnegative integer")

    def uniform(limits, n):
        # 두 끝이 같으면 고정된 조건이다. 테스트나 특정 슛을 관찰할 때도 같은 함수를 쓴다.
        return limits[0] + (limits[1] - limits[0]) * torch.rand(n, device=device, generator=generator)

    position = torch.empty((count, 3), device=device)
    velocity = torch.empty_like(position)
    angular_velocity = torch.zeros_like(position)
    # 종류는 한 번만 고른다. 불가능한 뜬볼을 땅볼로 바꾸면 설정한 땅볼 비율이 달라진다.
    is_ground = torch.rand(count, device=device, generator=generator) < cfg.ground_probability
    pending = torch.arange(count, device=device)
    rolling_deceleration = BALL_ROLLING_RESISTANCE * gravity

    for _ in range(cfg.max_sampling_attempts):
        n = pending.numel()
        if n == 0:
            break
        ground = is_ground[pending]
        start = torch.stack((uniform(cfg.launch_distance, n), uniform(cfg.launch_y, n),
                             uniform(cfg.air_launch_height, n)), dim=-1)
        start[:, 2] = torch.where(ground, ball_radius, start[:, 2])
        aim = torch.empty_like(start)
        aim[:, 0] = -goal_distance
        # 기둥·가로대의 바깥 크기 대신 실제 빈 골문 크기를 쓰고 공 반지름도 빼 준다.
        aim[:, 1] = uniform(cfg.target_y_fraction, n) * (goal_width / 2 - ball_radius)
        aim[:, 2] = ball_radius + uniform(cfg.air_target_height_fraction, n) * (goal_height - 2 * ball_radius)
        aim[:, 2] = torch.where(ground, ball_radius, aim[:, 2])
        launch_speed = uniform(cfg.speed, n)
        displacement = aim - start
        distance = torch.linalg.vector_norm(displacement[:, :2], dim=-1)
        direction_xy = displacement[:, :2] / distance[:, None]

        # 땅볼은 구름 저항 때문에 느려진다. v^2 - 2*a*거리 > 0이면 근사상 골문까지 도달한다.
        # 이 계산은 발사 조건을 거르는 용도이며, 시뮬레이션 중 속도를 덮어쓰는 보정이 아니다.
        residual_speed_sq = launch_speed.square() - 2 * rolling_deceleration * distance
        ground_valid = residual_speed_sq > 0

        # 뜬볼은 중력으로 아래로 휘어진다. 같은 속력으로 목표에 닿는 두 포물선 중
        # 낮게 날아가는 쪽(low arc)을 선택한다. 공기 저항·회전으로 휘는 효과는 아직 없다.
        # discriminant가 음수이면 해당 속력으로는 목표점에 도달할 수 없다는 뜻이다.
        speed_sq = launch_speed.square()
        dz = displacement[:, 2]
        discriminant = speed_sq.square() - gravity * (gravity * distance.square() + 2 * dz * speed_sq)
        air_valid = discriminant >= 0
        root = discriminant.clamp_min(0).sqrt()
        # (v^2 - sqrt(discriminant))/(g*d)를 수치 오차가 적은 동치식으로 계산한다.
        tangent = (gravity * distance.square() + 2 * dz * speed_sq) / (distance * (speed_sq + root))
        horizontal_speed = launch_speed / torch.sqrt(1 + tangent.square())
        horizontal_speed = torch.where(ground, launch_speed, horizontal_speed)
        candidate_velocity = torch.zeros_like(start)
        candidate_velocity[:, :2] = direction_xy * horizontal_speed[:, None]
        candidate_velocity[:, 2] = torch.where(ground, 0.0, horizontal_speed * tangent)
        valid = torch.where(ground, ground_valid, air_valid)
        valid &= torch.isfinite(candidate_velocity).all(dim=-1)
        # 양 끝 높이가 반지름 이상이고 아래로 휜 포물선이므로, 목표 전에 바닥을 뚫지 않는다.
        accepted = pending[valid]
        position[accepted] = start[valid]
        velocity[accepted] = candidate_velocity[valid]
        pending = pending[~valid]
    if pending.numel():
        raise ValueError(
            f"Could not sample {pending.numel()} reachable shots after {cfg.max_sampling_attempts} attempts; "
            "increase speed, shorten launch distance, or lower the aerial target"
        )

    # 바닥을 미끄러지며 출발하는 대신 처음부터 굴러가게 한다. 회전 속도의 단위는 rad/s다.
    # 접촉점의 이동이 0이 되는 omega = 위쪽 단위벡터 × 이동속도 / 반지름을 사용한다.
    angular_velocity[is_ground, 0] = -velocity[is_ground, 1] / ball_radius
    angular_velocity[is_ground, 1] = velocity[is_ground, 0] / ball_radius
    return ShotBatch(position, velocity, angular_velocity)


def reset_ball_shot(env, env_ids, shot_cfg: ShotCfg, ball_radius: float, goal_width: float,
                    goal_height: float, goal_distance: float) -> ShotBatch:
    """기본 reset_scene_to_default 다음에 실행하는 MJLab reset 이벤트.

    지정한 환경만 새 슛으로 바꾼다. 로봇 초기화·모터 지연 버퍼 초기화는 이미 MJLab이 하므로
    여기서는 공의 초기 상태와 과제 기록만 설정한다. 다른 환경의 진행 상태에는 손대지 않는다.
    """
    # MJLab이 만든 동일한 기록 객체를 찾아 새 공의 판정 상태만 초기화한다.
    from goalkeeper_mjlab.tasks.goalkeeper.mdp.metrics import get_shot_tracker

    all_ids = torch.arange(env.num_envs, device=env.device)
    if env_ids is None:
        ids = all_ids
    elif isinstance(env_ids, slice):
        ids = all_ids[env_ids]
    else:
        ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    # 실제 모델에 설정된 중력을 읽는다. 속도 계산과 MuJoCo의 중력이 달라지면 목표도 달라진다.
    gravity_vector = env.sim.mj_model.opt.gravity
    if abs(float(gravity_vector[0])) > 1e-9 or abs(float(gravity_vector[1])) > 1e-9:
        raise ValueError("Shot sampling currently requires gravity along the negative Z axis")

    batch = sample_shots(shot_cfg, ids.numel(), ball_radius, goal_width, goal_height,
                         goal_distance, gravity=-float(gravity_vector[2]), device=env.device)
    if ids.numel() == 0:
        return batch
    # root_state는 [위치 xyz, 회전 wxyz, 이동속도 xyz, 회전속도 xyz]의 13개 값이다.
    state = torch.zeros((ids.numel(), 13), device=env.device)
    # 병렬 환경은 서로 다른 원점을 갖는다. 로컬 좌표에 해당 환경 원점을 정확히 한 번 더한다.
    state[:, :3] = batch.position + env.scene.env_origins[ids]
    state[:, 3] = 1.0  # 공의 초기 방향은 단위 quaternion. 회전 속도는 별도로 지정한다.
    state[:, 7:10] = batch.linear_velocity
    state[:, 10:13] = batch.angular_velocity
    env.scene["ball"].write_root_state_to_sim(state, env_ids=ids)
    # 이벤트가 가진 실제 분포 설정을 함께 넘긴다. MJLab 매니저들은 설정을 각각 복사하므로
    # 기록 객체의 별도 복사본을 읽으면 커리큘럼으로 바꾼 제한 시간/영역이 반영되지 않는다.
    get_shot_tracker(env).begin_shots(ids, batch.position, shot_cfg)
    return batch


def reset_robot_joint_targets(env, env_ids) -> None:
    """
        기본 reset이 복원한 HOME 자세와 모터의 첫 목표각을 일치시킨다.
        MJLab의 Entity.reset()은 모터 목표각을 0으로 지운다. 기본 장면 reset은 실제 관절각을
        HOME으로 복원하지만 목표각까지 복원하지는 않는다. 그 상태에서 env.reset()이 첫
        제어 명령을 쓰면 0 rad가 기본 지연 버퍼의 첫 값이 되어 시작할 때 로봇이 움찔한다.
        이 이벤트에서 목표각만 지정하면 기본 DelayBuffer가 HOME을 첫 값으로 채운다.
        버퍼를 직접 수정하거나 지연을 끄거나 물리를 미리 진행하지 않는다.
    """
    robot = env.scene["robot"]
    ids = slice(None) if env_ids is None else env_ids
    # 종료한 환경만 변경한다. 진행 중인 다른 환경의 목표각은 그대로 유지한다.
    robot.set_joint_position_target(robot.data.default_joint_pos[ids], env_ids=env_ids)
