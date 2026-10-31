"""
wirlwind 팀 코드에서 가져온 공 설정
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from goalkeeper_mjlab.assets import load_model_spec

BALL_XML = Path(__file__).parent / "xmls" / "ball.xml"

# 공 크기·질량의 출처: wirlwind 팀 코드의 environment/ball/ball_constants.py.
# wirlwind 팀 코드는 5호 공을 '반지름 11 cm, 질량 430 g'으로 정하고, 다른 호수는 이 값에 배율을 곱함
BALL_SIZE_5_RADIUS: float = 0.11  # 반지름(m)
BALL_SIZE_5_MASS: float = 0.430  # 질량(kg)

# wirlwind의 공 메시는 반지름 0.11 m로 만들어져 있다. 원하는 반지름 / 이 값으로 화면 크기를 맞춤
# 공 크기 변경은 BallCfg.radius_override로 하고, 이 값은 wirlwind 메시 파일 크기의 기준
BALL_MESH_REFERENCE_RADIUS: float = 0.11  # m: wirlwind 공 메시의 반지름

# 같은 크기·무게라도 무게가 공의 중심에 모였는지, 겉에 모였는지에 따라 돌리기 힘든 정도가 다르다.
# 여기서는 축구공을 '속이 비고, 무게가 겉껍질에 고르게 모인 공'으로 가정
# 2/3은 이 모양의 회전 관성(회전 속도를 바꾸기 어려운 정도)을 계산할 때 정해지는 계수
BALL_SHELL_INERTIA_RATIO: float = 2.0 / 3.0

# 아래 마찰값도 wirlwind 팀 코드의 설정이다. 값을 정한 실험 자료는 해당 참조 파일에 적혀 있지 않다.
# 1) 미끄럼: 공과 바닥이 닿은 부분이 서로 쓸려 지나갈 때 생기는 저항이다.
#    같은 접촉 조건에서 값을 키우면 미끄러지는 것을 더 강하게 막는다. 단위는 없다.
BALL_SLIDING_FRICTION: float = 0.5
# 2) 비틀림: 바닥에 놓인 공을 팽이처럼 제자리에서 빙글 돌릴 때 그 회전을 방해하는 저항이다.
#    같은 접촉 조건에서 값을 키우면 이 방향의 회전을 더 강하게 막는다.
BALL_TORSIONAL_FRICTION: float = 0.002  # MuJoCo에서 이 계수의 단위는 m이다.
# 3) 구름: 툭 밀어 굴러가는 공이 바닥의 저항으로 얼마나 빨리 느려지는지를 정한다.
#    같은 조건에서 값을 키우면 더 빨리 멈추는 방향이다.
#    0.060은 평지에서 미끄럼 없이 구를 때, 1초마다 속도가 약 0.59 m/s 줄어드는 정도를 목표로 한다.
#    이 목표값을 아래 get_spec()에서 MuJoCo가 받는 계수로 바꿔서 전달한다.
BALL_ROLLING_RESISTANCE: float = 0.060

# 부딪힘, 미끄러짐, 팽이 같은 회전, 굴러감을 모두 접촉 계산에 포함한다.
# 6은 세 방향의 힘과 세 방향의 회전 저항을 함께 다룬다는 뜻이다.
BALL_COLLISION_CONDIM: int = 6
# 두 물체가 부딪히면 양쪽에 설정된 마찰·접촉 값 중 어떤 쪽을 우선할지 정해야 한다.
# 공은 1, 현재 로봇/바닥은 0이므로 공 쪽 설정을 우선 사용한다.
BALL_COLLISION_PRIORITY: int = 1
# 충돌 계산에서는 물체가 아주 조금 겹칠 수 있다. solref는 이를 어떻게 밀어내 복원할지 정한다.
# 앞의 0.05: 복원 속도를 정하는 시간 기준(s, 시간상수). 커지면 더 부드럽게 복원하는 쪽이다.
# 뒤의 0.15: 충돌 후 튕김·출렁임을 억제하는 정도(감쇠비). 1보다 작아서 튕김이 남을 수 있다.
# 0.15를 '충돌 전 속도의 15%로 튄다'는 뜻으로 읽으면 안 된다.
BALL_COLLISION_SOLREF: tuple[float, float] = (0.05, 0.15)

# 호수별 SIZE_SCALES[호수][0]은 크기, [1]은 질량에 곱한다.
# 예: 4호 반지름 = 0.11 * 0.9384, 질량 = 0.430 * 0.8605.
# 이 표는 wirlwind 팀 코드와 같으며, 각 배율을 어떻게 측정하거나 계산했는지는 참조 코드에 설명이 없다.
SIZE_SCALES = {
    1: (0.6304, 0.4419), 2: (0.8188, 0.6163), 3: (0.8551, 0.7209),
    4: (0.9384, 0.8605), 5: (1.0, 1.0),
}


@dataclass(frozen=True)
class BallCfg:
    """사용할 공의 크기와 질량. 직접 지정한 값이 있으면 호수별 계산보다 우선한다."""

    size: int = 4  # 기본은 4호 공. 1~5 중 선택한다.
    radius_override: float | None = None  
    mass_override: float | None = None

    def __post_init__(self) -> None:
        """없는 호수, 0 이하 크기·질량, NaN 같은 잘못된 입력을 미리 차단한다."""
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size not in SIZE_SCALES:
            raise ValueError("ball size must be an integer from 1 to 5")
        for name in ("radius_override", "mass_override"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")

    @property
    def radius(self) -> float:
        """직접 지정한 반지름을 쓰거나, 5호 반지름에 선택한 호수의 크기 배율을 곱한다."""
        return self.radius_override if self.radius_override is not None else BALL_SIZE_5_RADIUS * SIZE_SCALES[self.size][0]

    @property
    def mass(self) -> float:
        """직접 지정한 질량을 쓰거나, 5호 질량에 선택한 호수의 질량 배율을 곱한다."""
        return self.mass_override if self.mass_override is not None else BALL_SIZE_5_MASS * SIZE_SCALES[self.size][1]


def get_spec(cfg: BallCfg = BallCfg()) -> mujoco.MjSpec:
    """공 설정을 MuJoCo 모델에 적용한다. 겉모양과 실제 충돌 계산용 공의 크기를 함께 맞춘다."""
    spec = load_model_spec(BALL_XML)
    # 예: 사용할 반지름이 0.10 m라면 wirlwind의 반지름 0.11 m 공 메시를 0.10/0.11배로 줄인다.
    scale = cfg.radius / BALL_MESH_REFERENCE_RADIUS
    for mesh in spec.meshes:
        mesh.scale = (scale, scale, scale)
    # 위의 '얼마나 빨리 느려질지' 설정을 MuJoCo의 구름 저항 계수로 바꾼다.
    # wirlwind 팀 코드의 변환식은 공의 이동과 회전을 함께 고려해 (1 + 2/3) * 목표값 * 반지름을 사용한다.
    # 이 식은 현재 설정인 평지의 미끄럼 없는 구름과 elliptic 접촉 모델을 기준으로 한다.
    # 기본 4호 공이면 약 0.0103224 m가 나온다. 자세한 식과 출처는 docs/구현1단계.md에 있다.
    rolling_friction = (1.0 + BALL_SHELL_INERTIA_RATIO) * BALL_ROLLING_RESISTANCE * cfg.radius
    friction = (BALL_SLIDING_FRICTION, BALL_TORSIONAL_FRICTION, rolling_friction)  # 순서: 미끄럼, 팽이 회전, 굴러감
    for geom in spec.worldbody.bodies[0].geoms:
        if geom.name.startswith("ball_visual"):
            # ball_visual은 화면에 공 무늬를 그리는 모양이다. 공 전체의 질량은 아래 충돌용 구에 넣는다.
            # 그림용 모양에도 무게가 생기면 질량이 두 번 더해지므로, 그림 쪽 질량·밀도는 0으로 둔다.
            geom.mass = geom.density = 0.0
        if geom.name == "ball_collision":
            # ball_collision은 공이 바닥/손/골대에 닿았는지 계산하는, 눈에 보이는 무늬 없는 단순한 구다.
            geom.size = (cfg.radius, 0.0, 0.0)  # 구는 첫 칸의 반지름만 사용한다. 뒤의 두 칸은 사용하지 않는다.
            geom.mass = cfg.mass
            # 회전 계산도 위의 2/3 계수와 일치하도록 '속이 빈 공'으로 설정한다.
            geom.typeinertia = mujoco.mjtGeomInertia.mjINERTIA_SHELL
            geom.priority = BALL_COLLISION_PRIORITY
            geom.condim = BALL_COLLISION_CONDIM
            geom.solref = BALL_COLLISION_SOLREF
        geom.friction = friction
    return spec
