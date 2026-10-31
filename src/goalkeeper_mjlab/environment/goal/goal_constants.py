"""
    WirlWind 골대 설정
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco

from goalkeeper_mjlab.assets import load_model_spec

GOAL_XML = Path(__file__).parent / "xmls" / "goal.xml"

# 출처: booster_mjlab/environment/goal/goal_constants.py의 DIVISION_SCALE.
# 원본 small 모델의 바깥 폭은 약 180 cm다. '원하는 폭 / 원본 폭'이 아래 확대 배율이다.
# small: 약 180 x 120 cm, mid: 약 240 x 160 cm, large: 약 300 x 200 cm.
# 폭뿐 아니라 높이·깊이·기둥 두께도 같은 비율로 커진다.
DIVISION_SCALES = {"small": 1.0, "mid": 240 / 180, "large": 300 / 180}

# 원본 XML의 충돌용 박스 11개에, 어떤 기둥인지 알아볼 수 있는 이름을 붙인다.
# negative_x/positive_x는 골대 파일 자체의 X 좌표가 작은 쪽/큰 쪽을 뜻한다.
# 장면에 넣을 때 골대를 회전하므로 로봇 기준의 왼쪽/오른쪽과 이름을 바로 대응시키지 않는다.
_COLLISION_NAMES = (
    "post_negative_x",  # 입구의 한쪽 세로 기둥
    "post_positive_x",  # 입구의 반대쪽 세로 기둥
    "back_post_negative_x",  # 뒤쪽 한쪽 세로 기둥
    "back_post_positive_x",  # 뒤쪽 반대쪽 세로 기둥
    "back_top",  # 뒤쪽 위 가로대
    "back_bottom",  # 뒤쪽 바닥 가로대
    "side_top_negative_x",  # 한쪽 옆면의 위 연결대
    "side_top_positive_x",  # 반대쪽 옆면의 위 연결대
    "side_bottom_negative_x",  # 한쪽 옆면의 바닥 연결대
    "side_bottom_positive_x",  # 반대쪽 옆면의 바닥 연결대
    "crossbar",  # 입구 위쪽을 가로지르는 가로대
)


@dataclass(frozen=True)
class GoalCfg:
    """골대 크기를 선택한다. 설정 객체의 기본값은 small이며, 실행 기본값은 CLI에서 따로 정한다."""

    division: str = "small" 

    def __post_init__(self) -> None:
        if self.division not in DIVISION_SCALES:
            raise ValueError(f"Unknown goal division: {self.division}")

    @property
    def scale(self) -> float:
        """원본 모양을 몇 배로 키울지 반환한다. 예: mid는 240/180, 약 1.333배다."""
        return DIVISION_SCALES[self.division]


@dataclass(frozen=True)
class GoalOpening:
    """골대 바깥 크기와 입구 안쪽의 빈 공간 크기를 보관한다. 모두 m 단위다."""

    outer_width: float  # 기둥의 바깥 면부터 반대쪽 기둥의 바깥 면까지 잰 전체 폭
    outer_height: float  # 바닥부터 입구 위 가로대의 꼭대기까지 잰 전체 높이
    inner_width: float  # 양쪽 입구 기둥의 안쪽 면 사이 거리
    inner_height: float  # 바닥부터 입구 위 가로대의 아래 면까지 거리


def get_spec(cfg: GoalCfg = GoalCfg()) -> mujoco.MjSpec:
    """원본 골대의 겉모양과 충돌용 기둥들을 함께 확대해 MuJoCo에 전달한다."""
    spec = load_model_spec(GOAL_XML)
    root = spec.worldbody.bodies[0]
    # frame은 골대 전체를 묶는 본체 이름이다. 장면에 붙인 뒤에는 goal/frame으로 찾을 수 있다.
    root.name = "frame"
    # mocap은 여기서 '코드로 위치를 정하는 고정 물체' 기능으로 사용한다.
    # 공이 맞아도 골대가 밀려나지 않으며, 병렬 학습 환경마다 필요한 위치에 배치할 수 있다.
    root.mocap = True
  
    for mesh in spec.meshes:
        mesh.scale = (cfg.scale,) * 3
    # 골대 전체 모양을 하나의 충돌 물체로 처리하면, 빈 골문까지 막힌 것으로 계산될 수 있다.
    # 원본은 기둥/연결대마다 단순한 박스 11개를 두어 골문 가운데를 비워 둔다.
    # contype/conaffinity는 충돌 허용 설정이며, 둘 중 하나라도 켜진 도형을 아래에서 선택한다.
    colliders = [g for g in root.geoms if g.contype or g.conaffinity]
    if len(colliders) != len(_COLLISION_NAMES):
        raise ValueError("Goal collision layout differs from the source model")
    for geom, name in zip(colliders, _COLLISION_NAMES, strict=True):
        geom.name = name
        # group 3은 viewer에서 충돌 도형을 모아 표시하기 위한 번호다.
        # 실제로 충돌하는지는 위의 contype/conaffinity 설정이 결정한다.
        geom.group = 3
        # 화면만 커지고 충돌 기둥은 작게 남는 일이 없도록, 박스도 같은 비율로 키운다.
        # pos는 각 박스의 중심 위치다. 위치까지 확대해야 양쪽 기둥 사이 거리도 함께 늘어난다.
        geom.pos = tuple(float(x) * cfg.scale for x in geom.pos)
        # 원본 11개는 모두 박스다. MuJoCo 박스의 size에는 가로·세로·높이의 절반이 들어 있다.
        # 예: size의 폭 성분이 0.05 m이면 그 박스의 전체 두께는 0.10 m다.
        geom.size = tuple(float(x) * cfg.scale for x in geom.size)
    for geom in root.geoms:
        if geom.meshname:
            # visual은 화면용 모양이다. 원본 XML에서 충돌을 끄고, 충돌 계산은 위 박스들이 맡는다.
            geom.name = "visual"
    return spec


def get_opening(cfg: GoalCfg = GoalCfg()) -> GoalOpening:
    """공이 지나갈 입구의 빈 공간을 기둥 안쪽 면에서 계산한다. 높이는 골대 바닥 기준이다."""
    spec = get_spec(cfg)
    left, right, bar = (spec.geom(n) for n in ("post_negative_x", "post_positive_x", "crossbar"))
    # 원본의 약 1.8 x 1.2 m에는 기둥 두께가 포함되어 있다.
    # 각 기둥의 중심(pos)과 반쪽 두께(size)로, 골문 쪽을 향한 안쪽 면의 위치를 구한다.
    # X가 작은 기둥의 안쪽 면 = 중심 + 반쪽 두께, X가 큰 기둥의 안쪽 면 = 중심 - 반쪽 두께.
    inner_width = (right.pos[0] - right.size[0]) - (left.pos[0] + left.size[0])
    # 실제 실점 판정에서는 공 반지름까지 고려해야 한다. 이 함수는 골문 치수만 제공한다.
    return GoalOpening(
        outer_width=float((right.pos[0] + right.size[0]) - (left.pos[0] - left.size[0])),
        outer_height=float(bar.pos[2] + bar.size[2]),
        inner_width=float(inner_width),
        inner_height=float(bar.pos[2] - bar.size[2]),  # 가로대 중심 높이 - 반쪽 두께 = 아래 면 높이
    )
