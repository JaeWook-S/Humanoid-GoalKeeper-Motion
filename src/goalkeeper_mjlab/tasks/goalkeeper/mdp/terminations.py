"""3단계 반복 발사에 필요한 최소 종료 조건. 보상/선방 성공 판정은 후속 단계다."""

from goalkeeper_mjlab.tasks.goalkeeper.mdp.metrics import get_shot_tracker


def shot_goal(env):
    """공 전체가 빈 골문을 통과했다. 접촉 여부와 관계없이 득점으로 종료한다."""
    return get_shot_tracker(env).goal


def shot_out(env):
    """공이 검사 영역을 벗어나 더 관찰할 필요가 없다."""
    return get_shot_tracker(env).out


def shot_timeout(env):
    """관찰 시간 소진. cfg에서 time_out=True로 연결해 실패 종료와 구분한다."""
    return get_shot_tracker(env).timeout
