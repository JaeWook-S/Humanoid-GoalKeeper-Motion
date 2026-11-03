"""슛 하나의 접촉·골라인 통과 기록. 학습과 검증이 같은 구현을 사용한다.

MJLab MetricsManager의 per_substep 기능에 연결한다. 물리 루프나 reset 관리자를
새로 만들지 않고, 환경마다 필요한 작은 Torch 버퍼만 보관한다.
"""

from __future__ import annotations

import torch
from mjlab.managers.manager_base import ManagerTermBase

# 기본 공/접촉 모델의 실제 땅볼 검사에서 바닥 침투 약 0.53 mm를 관찰했다.
# 1 mm는 판정에만 쓰는 바닥 허용 오차다. 공의 반지름·마찰·충돌 물리를 바꾸는 값이 아니다.
# 옆 기둥/가로대에는 적용하지 않는다. tests의 3 m/s 땅볼 회귀 검사로 필요성을 확인한다.
GOAL_FLOOR_TOLERANCE = 0.001


def get_shot_tracker(env) -> ShotTracker:
    """MJLab이 생성한 상태 기록 객체를 이벤트·종료 조건에서 함께 참조한다."""
    # pinned MJLab은 MetricsManager.get_term()을 제공하지 않는다.
    # manager.cfg의 func는 초기화 과정에서 클래스에서 실제 객체로 바뀐다.
    return env.metrics_manager.cfg["shot_state"].func


def shot_metric(env, name: str) -> torch.Tensor:
    """이미 기록된 사실만 통계로 반환한다. 여기서 공의 상태를 다시 진행하지 않는다."""
    return getattr(get_shot_tracker(env), name).float()


class ShotTracker(ManagerTermBase):
    """
        N개 환경에 각각 독립적인 '현재 슛의 기록'을 둔다.
        touched와 goal은 한 번 True가 되면 해당 에피소드에서 계속 True다.
        따라서 잠깐 공에 닿은 뒤 떨어져도 기록이 남고, 접촉 후 실점도 표현할 수 있다.
    """

    def __init__(self, cfg, env):
        super().__init__(env)
        self.radius = cfg.params["ball_radius"]
        self.half_width = cfg.params["goal_width"] / 2
        self.height = cfg.params["goal_height"]
        self.goal_distance = cfg.params["goal_distance"]
        self.sensor = env.scene["robot_ball_contact"]
        # freejoint의 앞 3칸이 공 중심의 월드 좌표다. 관절 이름으로 주소를 찾는다.
        # mj_step 직후 xpos/site_xpos는 한 물리 스텝 전 값일 수 있어 사용하지 않는다.
        self.ball_qpos_adr = int(env.sim.mj_model.joint("ball/ball_joint").qposadr[0])
        n, device = self.num_envs, self.device
        # elapsed는 슛 제한 시간, previous_position은 두 물리 스텝 사이의 골라인 통과에 쓴다.
        self.elapsed = torch.zeros(n, device=device)
        self.previous_position = torch.zeros((n, 3), device=device)
        self.goal = torch.zeros(n, dtype=torch.bool, device=device)  # 공 전체가 골문 안으로 넘어갔는지?
        self.entered_opening = torch.zeros_like(self.goal)  # 공 중심도 빈 골문을 통해 들어왔는지?
        self.touched = torch.zeros_like(self.goal)  # 이번 공이 로봇에 한 번이라도 닿았는지?
        self.out = torch.zeros_like(self.goal)  # 허용 영역 밖으로 넘어갔는지?
        self.timeout = torch.zeros_like(self.goal)  # 제한 시간이 지나갔는지?
        # 커리큘럼 설정은 다음 reset 때만 반영한다. 진행 중인 공은 당시의 종료 범위를 유지한다.
        self.timeout_limit = torch.zeros(n, device=device)
        self.out_x_min = torch.zeros(n, device=device)
        self.out_x_max = torch.zeros(n, device=device)
        self.out_y_max = torch.zeros(n, device=device)
        self.out_z_max = torch.zeros(n, device=device)

    def begin_shots(self, env_ids, position, shot_cfg) -> None:
        """발사 이벤트가 호출한다. 지정한 환경의 현재 슛 판정만 초기화한다."""
        ids = env_ids
        self.elapsed[ids] = 0
        self.previous_position[ids] = position + self._env.scene.env_origins[ids]
        for name in ("goal", "entered_opening", "touched", "out", "timeout"):
            getattr(self, name)[ids] = False
        # 발사 이벤트가 실제 사용한 설정을 환경별로 저장한다. 커리큘럼을 바꿔도 이미
        # 날아오는 공의 제한 시간과 영역은 그대로이고, reset한 환경만 새 값을 사용한다.
        self.timeout_limit[ids] = shot_cfg.episode_timeout_s
        self.out_x_min[ids] = -self.goal_distance - shot_cfg.out_margin
        self.out_x_max[ids] = shot_cfg.launch_distance[1] + shot_cfg.out_margin
        self.out_y_max[ids] = max(self.half_width, *(abs(y) for y in shot_cfg.launch_y)) + shot_cfg.out_margin
        self.out_z_max[ids] = max(self.height, shot_cfg.air_launch_height[1]) + shot_cfg.out_margin
        # MJLab은 이 발사 이벤트 뒤에 MetricsManager.reset을 호출한다. 별도 reset을
        # 구현해 여기의 새 상태를 다시 지우지 않는다. 이전 에피소드 통계는 기본 매니저가 낸다.

    def __call__(self, env, **_) -> torch.Tensor:
        """물리 계산 한 번마다 호출한다. 반환값은 단순 통계이고 실제 기록은 버퍼에 남는다."""
        dt = env.physics_dt
        current = env.sim.data.qpos[:, self.ball_qpos_adr:self.ball_qpos_adr + 3]
        origin = env.scene.env_origins
        previous_local = self.previous_position - origin
        current_local = current - origin

        # 현재 장면은 로봇 앞이 +X, 골문 뒤가 -X다. 골문은 -goal_distance에 있다.
        # 공 중심이 골라인을 반지름만큼 더 지나가야 공 전체가 선을 넘는다.
        full_cross_x = -self.goal_distance - self.radius
        dx = current_local[:, 0] - previous_local[:, 0]
        crossing = (~self.goal & (previous_local[:, 0] > full_cross_x)
                    & (current_local[:, 0] <= full_cross_x) & (dx < 0))
        # 빠른 공은 한 번에 선을 건너뛴다. '선 근처에 있는지' 검사하지 않고 두 위치 사이에서
        # 선을 넘는 순간을 보간한다. alpha=0은 이전 위치, alpha=1은 현재 위치다.
        safe_dx = torch.where(dx.abs() > 1e-12, dx, torch.ones_like(dx))
        alpha = ((full_cross_x - previous_local[:, 0]) / safe_dx).clamp(0, 1)
        at_line = previous_local + alpha[:, None] * (current_local - previous_local)
        # 공 중심이 기둥/가로대와 반지름 이상 떨어져야 공 전체가 빈 골문 안에 있다.
        # 바닥은 접촉 솔버의 미세한 겹침을 허용한다. float32 오차와는 별개의 현상이다.
        eps = 1e-6

        def in_opening(point):
            return ((point[:, 1].abs() <= self.half_width - self.radius + eps)
                    & (point[:, 2] >= self.radius - GOAL_FLOOR_TOLERANCE - eps)
                    & (point[:, 2] <= self.height - self.radius + eps))

        # 실제 골라인(공 중심이 선을 지나는 시점)도 확인한다. 공 전체가 넘어간 시점만 보면
        # 기둥 바깥에서 대각선으로 들어온 경로를 골문 안으로 통과한 것으로 오인할 수 있다.
        plane_x = -self.goal_distance
        center_crossing = ((previous_local[:, 0] > plane_x)
                           & (current_local[:, 0] <= plane_x) & (dx < 0))
        center_alpha = ((plane_x - previous_local[:, 0]) / safe_dx).clamp(0, 1)
        at_center_line = previous_local + center_alpha[:, None] * (current_local - previous_local)
        self.entered_opening = torch.where(center_crossing, in_opening(at_center_line), self.entered_opening)
        # 다시 선 앞쪽으로 튕겨 나오면 다음 진입 때 새로 확인한다. 이미 기록된 득점은 유지한다.
        self.entered_opening &= current_local[:, 0] <= plane_x
        scored = crossing & self.entered_opening & in_opening(at_line)
        self.goal |= scored

        # 네이티브 ContactSensor가 실제 엔진 접촉을 측정한다. 거리로 접촉을 추측하지 않는다.
        # 여러 로봇 부위 중 하나라도 닿으면 True다. 잠깐 닿았다가 떨어져도 reset까지 유지한다.
        self.touched |= (self.sensor.data.found > 0).reshape(self.num_envs, -1).any(dim=1)
        self.elapsed += dt

        # 반사되어 앞/옆으로 멀리 나간 공도 언젠가 종료되어야 반복 발사가 가능하다.
        # 로봇에 닿았다는 이유만으로 끝내지 않아 접촉 후 득점도 정상적으로 기록한다.
        outside = ((current_local[:, 0] < self.out_x_min)
                   | (current_local[:, 0] > self.out_x_max)
                   | (current_local[:, 1].abs() > self.out_y_max)
                   | (current_local[:, 2] > self.out_z_max))
        self.out |= outside
        self.timeout |= self.elapsed >= self.timeout_limit - 1e-6
        self.previous_position.copy_(current)
        return self.goal.float()
