"""물리 계산의 갱신 횟수와 정책이 목표각을 결정하는 횟수를 나누어 관리한다."""

from __future__ import annotations

from dataclasses import dataclass
import math
import mujoco


@dataclass(frozen=True)
class PhysicsControlCfg:
    # 0.005초마다 충돌과 운동을 계산한다. 공식/로컬 Booster의 50 Hz 구성과 같은 시간 관계다.
    physics_dt: float = 0.005
    decimation: int = 4  # 정책 목표각 1번당 물리를 4번 진행한다. 정책 주기는 0.020초=50 Hz다.
    # 적분기는 지금의 위치·속도·힘에서 다음 순간의 상태를 계산하는 방법이다.
    # implicitfast는 MuJoCo 내장 PD의 제동 효과도 계산에 포함한다. 공식 Isaac/PhysX와 엔진이 다르다.
    integrator: str = "implicitfast"
    # 접촉 솔버는 여러 물체가 겹치지 않게 밀어내는 힘을 계산한다. Newton 방식으로 반복해서 푼다.
    solver: str = "newton"
    iterations: int = 10  # 한 스텝에 허용할 반복 횟수. 로컬 booster_mjlab과 같다.
    ls_iterations: int = 20
    tolerance: float = 1e-8

    def __post_init__(self) -> None:
        if not math.isfinite(self.physics_dt) or self.physics_dt <= 0 or not math.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("Physics timestep/tolerance must be finite and positive")
        for name in ("decimation", "iterations", "ls_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.integrator not in ("implicitfast", "euler") or self.solver not in ("newton", "cg"):
            raise ValueError("Unsupported stage-2 integrator or solver")

    @property
    def policy_dt(self) -> float:
        return self.physics_dt * self.decimation

    def apply(self, spec: mujoco.MjSpec) -> None:
        spec.option.timestep = self.physics_dt
        spec.option.integrator = {"implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST, "euler": mujoco.mjtIntegrator.mjINT_EULER}[self.integrator]
        spec.option.solver = {"newton": mujoco.mjtSolver.mjSOL_NEWTON, "cg": mujoco.mjtSolver.mjSOL_CG}[self.solver]
        spec.option.iterations = self.iterations
        spec.option.ls_iterations = self.ls_iterations
        spec.option.tolerance = self.tolerance
        # 1단계 공의 구름 저항 설정이 전제로 하는 elliptic 마찰 계산을 유지한다.
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC

    def to_mjlab(self):
        from mjlab.sim import MujocoCfg, SimulationCfg
        return SimulationCfg(nconmax=80, njmax=1500, mujoco=MujocoCfg(
            timestep=self.physics_dt, integrator=self.integrator, solver=self.solver,
            iterations=self.iterations, ls_iterations=self.ls_iterations,
            tolerance=self.tolerance, cone="elliptic",
        ))
