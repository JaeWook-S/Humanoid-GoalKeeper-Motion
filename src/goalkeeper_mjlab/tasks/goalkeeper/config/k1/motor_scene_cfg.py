"""1단계 장면에 22개 모터와 물리/제어 실행 설정을 연결한다."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
import mujoco

from goalkeeper_mjlab.robots.k1.actuators import configure_actuators
from goalkeeper_mjlab.robots.k1.command_delay import CommandDelayCfg
from goalkeeper_mjlab.robots.k1.constants import K1_JOINT_ORDER
from goalkeeper_mjlab.robots.k1.controller_profiles import ControllerParameters, READY_POSE, hold_controllers
from goalkeeper_mjlab.robots.k1.motor_parameters import get_motors
from goalkeeper_mjlab.robots.k1.torque_speed import JointParameters
from goalkeeper_mjlab.tasks.goalkeeper.config.k1.scene_cfg import Stage1SceneCfg, build_scene_spec, make_mjlab_scene_cfg
from goalkeeper_mjlab.tasks.goalkeeper.config.k1.simulation_cfg import PhysicsControlCfg


@dataclass(frozen=True)
class Stage2SceneCfg:
    scene: Stage1SceneCfg = field(default_factory=lambda: Stage1SceneCfg(robot_pose=READY_POSE))
    execution: PhysicsControlCfg = field(default_factory=PhysicsControlCfg)
    controllers: dict[str, ControllerParameters] = field(default_factory=hold_controllers)
    delay: CommandDelayCfg = field(default_factory=CommandDelayCfg)
    motor_source: str = "official"
    seed: int = 0  # 시뮬레이션 검사에서 지연 선택 결과를 다시 재현하기 위한 값이다.

    def __post_init__(self) -> None:
        motors = get_motors(self.motor_source)
        if set(self.controllers) != set(motors):
            raise ValueError("Controller groups must cover every motor group exactly once")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")

    @property
    def parameters(self) -> JointParameters:
        return JointParameters.build(get_motors(self.motor_source), self.controllers)


def compile_controlled_scene(cfg: Stage2SceneCfg | None = None, origin=(0.0, 0.0, 0.0)):
    cfg = cfg or Stage2SceneCfg()
    spec = build_scene_spec(cfg.scene, origin)
    spec.modelname = "goalkeeper_stage2"
    configure_actuators(spec, cfg.parameters)
    cfg.execution.apply(spec)
    model = spec.compile()
    # ZIP을 다시 열 때도 첫 목표각이 준비 자세가 되도록 keyframe의 ctrl까지 기록한다.
    positions = model.key("init_state").qpos
    spec.key("init_state").ctrl = [positions[int(model.joint("robot/" + n).qposadr[0])] for n in K1_JOINT_ORDER]
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
    mujoco.mj_forward(model, data)
    return spec, model, data


def make_controlled_mjlab_scene_cfg(cfg: Stage2SceneCfg | None = None):
    """같은 모터/PD/지연 설정을 학습용 MJLab actuator에 연결한다."""
    from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
    from goalkeeper_mjlab.robots.k1.mjlab_actuators import MotorEntityCfg, make_motor_actuators
    cfg = cfg or Stage2SceneCfg()
    # SceneCfg는 1단계의 자산/배치 설정을 재사용한다. 실제 시간 설정은 SimulationCfg로 별도 전달한다.
    scene = make_mjlab_scene_cfg(replace(cfg.scene, preview_timestep=cfg.execution.physics_dt))
    scene.entities["robot"].articulation = EntityArticulationInfoCfg(
        actuators=make_motor_actuators(cfg), soft_joint_pos_limit_factor=1.0,
    )
    robot = scene.entities["robot"]
    scene.entities["robot"] = MotorEntityCfg(**{f.name: getattr(robot, f.name) for f in fields(EntityCfg)}, motor_seed=cfg.seed)
    return scene
