"""모터에 목표각이 도착하기까지의 기다리는 시간을 물리 스텝 수로 표현한다."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from goalkeeper_mjlab.robots.k1.constants import K1_JOINT_ORDER
from goalkeeper_mjlab.robots.k1.motor_parameters import delay_group

DELAY_GROUPS = ("legs", "feet", "arm", "head")
JOINT_DELAY_GROUP_IDS = np.array([DELAY_GROUPS.index(delay_group(n)) for n in K1_JOINT_ORDER])


@dataclass(frozen=True)
class CommandDelayCfg:
    # 공식 K1 min_delay=2/max_delay=8. dt=0.005이면 10~40 ms다.
    # reset마다 그룹별 지연을 선택하고 다음 reset까지 유지한다. 센서 피드백은 지연시키지 않는다.
    min_steps: int = 2
    max_steps: int = 8

    def __post_init__(self) -> None:
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (self.min_steps, self.max_steps)) or not 0 <= self.min_steps <= self.max_steps:
            raise ValueError("Command delay must use integers: 0 <= min_steps <= max_steps")


class CommandDelay:
    def __init__(self, cfg: CommandDelayCfg, initial: np.ndarray, seed: int = 0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.reset(initial)

    def reset(self, initial: np.ndarray) -> None:
        # 과거 명령이 없는 구간은 준비 자세로 채운다. 0 rad로 채우면 reset 직후 잘못된 명령이 된다.
        self.history = np.repeat(np.asarray(initial, dtype=float)[None, :], self.cfg.max_steps + 1, axis=0)
        self.cursor = 0
        self.group_lags = self.rng.integers(self.cfg.min_steps, self.cfg.max_steps + 1, len(DELAY_GROUPS))
        self.joint_lags = self.group_lags[JOINT_DELAY_GROUP_IDS]

    def push(self, target: np.ndarray) -> np.ndarray:
        self.history[self.cursor] = target
        rows = (self.cursor - self.joint_lags) % len(self.history)
        delayed = self.history[rows, np.arange(len(K1_JOINT_ORDER))].copy()
        self.cursor = (self.cursor + 1) % len(self.history)
        return delayed
