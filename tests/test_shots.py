"""3단계 회귀 검사: 학습에서 사용하는 슛 생성·reset·접촉·득점 코드를 직접 검사한다.

여기서 별도의 공 발사기나 득점 판정기를 만들지 않는다. 수학으로 계산할 수 있는
궤적은 독립적인 계산으로 확인하고, 상태 변화는 실제 MJLab 환경에서 확인한다.
"""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import unittest

import torch

from goalkeeper_mjlab.environment.ball.ball_constants import BallCfg
from goalkeeper_mjlab.environment.goal.goal_constants import get_opening
from goalkeeper_mjlab.robots.k1.constants import K1_JOINT_ORDER
from goalkeeper_mjlab.tasks.goalkeeper.config.k1.env_cfgs import goalkeeper_k1_env_cfg
from goalkeeper_mjlab.tasks.goalkeeper.mdp.events import ShotCfg, sample_shots
from goalkeeper_mjlab.tasks.goalkeeper.mdp.metrics import get_shot_tracker
from goalkeeper_mjlab.validation.shots import ShotSimulation


class ShotSamplingTests(unittest.TestCase):
    """시뮬레이션 실행 이전에 슛의 범위·재현성·궤적 계산을 확인한다."""

    def sample(self, cfg=None, count=256, seed=42, **kwargs):
        opening = get_opening()
        parameters = dict(ball_radius=BallCfg().radius, goal_width=opening.inner_width,
                          goal_height=opening.inner_height, goal_distance=0.75)
        parameters.update(kwargs)
        return sample_shots(cfg or ShotCfg(), count,
                            generator=torch.Generator(device="cpu").manual_seed(seed), **parameters)

    def test_random_samples_respect_type_position_speed_and_opening(self):
        cfg = ShotCfg()
        batch = self.sample(cfg, count=2048)
        radius = BallCfg().radius
        # 2,048개의 서로 다른 슛을 생성한다. 범위만 검사하면 난수가 항상 같은 값을
        # 내는 버그를 놓칠 수 있으므로 실제 값의 다양성도 함께 확인한다.
        self.assertGreater(torch.unique(batch.position[:, 0]).numel(), 100)
        self.assertTrue(((batch.position[:, 0] >= cfg.launch_distance[0]) &
                         (batch.position[:, 0] <= cfg.launch_distance[1])).all())
        self.assertTrue(((batch.position[:, 1] >= cfg.launch_y[0]) &
                         (batch.position[:, 1] <= cfg.launch_y[1])).all())
        speed = batch.linear_velocity.norm(dim=-1)
        self.assertTrue(((speed >= cfg.speed[0] - 1e-5) & (speed <= cfg.speed[1] + 1e-5)).all())
        # 정해진 난수열에서도 극단적으로 잘못된 비율은 검출하되, 정확히 50%라는
        # 조건을 요구하지 않는다. 랜덤 선택은 유한 개를 뽑으면 비율이 조금 달라진다.
        proportion = torch.isclose(batch.position[:, 2], torch.tensor(radius)).float().mean().item()
        self.assertGreater(proportion, 0.4)
        self.assertLess(proportion, 0.6)
        self.assertTrue(torch.isfinite(batch.linear_velocity).all())

    def test_air_shots_reach_sampled_target_under_gravity(self):
        # 목표를 고정하고, 실제 반환된 속도로 도착 시간·위치를 독립 계산한다.
        cfg = replace(ShotCfg(), ground_probability=0.0,
                      target_y_fraction=(0.3, 0.3), air_target_height_fraction=(0.6, 0.6))
        batch = self.sample(cfg, count=1024)
        duration = ((-0.75 - batch.position[:, 0]) / batch.linear_velocity[:, 0])[:, None]
        # 초기 위치 + 이동한 거리 + 중력이 떨어뜨린 거리로 도착점을 독립 계산한다.
        # 발사 코드에서 목표 높이와 수직 속도의 관계가 잘못되어도 이 검사가 잡는다.
        endpoint = batch.position + batch.linear_velocity * duration
        endpoint[:, 2] -= 0.5 * 9.81 * duration[:, 0].square()
        opening, radius = get_opening(), BallCfg().radius
        target = torch.tensor([-0.75, 0.3 * (opening.inner_width / 2 - radius),
                               radius + 0.6 * (opening.inner_height - 2 * radius)])
        torch.testing.assert_close(endpoint, target.expand_as(endpoint), atol=3e-5, rtol=2e-5)
        self.assertTrue((duration > 0).all())
        self.assertTrue((batch.linear_velocity[:, 0] < 0).all())
        self.assertTrue((batch.angular_velocity == 0).all())

    def test_ground_shots_start_on_floor_and_roll_without_initial_slip(self):
        batch = self.sample(replace(ShotCfg(), ground_probability=1.0))
        radius = BallCfg().radius
        torch.testing.assert_close(batch.position[:, 2], torch.full((256,), radius))
        self.assertTrue((batch.linear_velocity[:, 2] == 0).all())
        # 바닥에 닿는 공 표면의 속도 = 공 중심의 이동 속도 + 회전에 의한 속도다.
        # 두 값이 상쇄되면 공이 처음부터 바닥 위를 쓸려 나가지 않고 구른다.
        center_to_floor = torch.zeros_like(batch.position)
        center_to_floor[:, 2] = -radius
        contact_velocity = batch.linear_velocity + torch.cross(
            batch.angular_velocity, center_to_floor, dim=-1)
        torch.testing.assert_close(contact_velocity, torch.zeros_like(contact_velocity), atol=1e-6, rtol=0)

    def test_seed_reproduces_shots_and_different_seed_changes_them(self):
        first, repeated, different = self.sample(seed=31), self.sample(seed=31), self.sample(seed=32)
        for field in ("position", "linear_velocity", "angular_velocity"):
            torch.testing.assert_close(getattr(first, field), getattr(repeated, field))
        self.assertFalse(torch.equal(first.position, different.position))

    def test_curriculum_can_change_ranges_for_the_next_sampling(self):
        # 커리큘럼 매니저를 구현하는 단계는 아니지만, 같은 설정 객체의 범위와
        # 비율을 바꾸면 다음 슛부터 적용할 수 있다는 연결 지점을 확인한다.
        cfg = ShotCfg()
        before = self.sample(cfg)
        saved = before.position.clone()
        cfg.ground_probability = 1.0
        cfg.launch_distance = (1.5, 1.5)
        cfg.launch_y = (0.0, 0.0)
        cfg.speed = (4.0, 4.0)
        after = self.sample(cfg)
        torch.testing.assert_close(after.position[:, 2], torch.full((256,), BallCfg().radius))
        torch.testing.assert_close(after.position[:, 0], torch.full((256,), 1.5))
        torch.testing.assert_close(after.position[:, 1], torch.zeros(256))
        torch.testing.assert_close(after.linear_velocity.norm(dim=-1), torch.full((256,), 4.0))
        torch.testing.assert_close(before.position, saved)

    def test_invalid_ranges_and_impossible_trajectories_fail_explicitly(self):
        for changes in ({"ground_probability": -0.1}, {"ground_probability": float("nan")},
                        {"launch_distance": (3.0, 1.0)}, {"launch_y": (0.0, float("inf"))},
                        {"speed": (0.0, 2.0)}, {"target_y_fraction": (-1.1, 0.5)},
                        {"max_sampling_attempts": 0}, {"episode_timeout_s": -1.0}):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    self.sample(replace(ShotCfg(), **changes), count=4)
        # 공이 골문보다 크거나, 먼 골문을 향해 너무 느리게 쏘도록 요구한 경우는
        # 조용히 속도를 바꾸지 말고 잘못된 설정이라고 알려야 한다.
        with self.assertRaises(ValueError):
            self.sample(count=4, ball_radius=1.0)
        for probability in (0.0, 1.0):
            with self.subTest(ground_probability=probability):
                with self.assertRaises(ValueError):
                    self.sample(replace(ShotCfg(), ground_probability=probability,
                                        launch_distance=(20.0, 20.0), speed=(0.5, 0.5),
                                        max_sampling_attempts=3), count=4)


class ShotEnvironmentTests(unittest.TestCase):
    """실제 ManagerBasedRlEnv에서 환경별 reset·접촉·골라인 판정을 확인한다."""

    @classmethod
    def setUpClass(cls):
        # 짧은 제한 시간은 timeout 검사를 빨리 반복하기 위한 검사 설정이다.
        # 실제 슛의 기본 제한 시간을 바꾸거나, 넘어지는 로봇을 고정하지 않는다.
        cfg = goalkeeper_k1_env_cfg(num_envs=3,
                                   shots=replace(ShotCfg(), episode_timeout_s=0.12))
        cfg.seed = 17  # 실행 조건은 반환된 MJLab 기본 설정에서 바꾼다.
        cls.runtime = ShotSimulation(cfg)

    @classmethod
    def tearDownClass(cls):
        cls.runtime.env.close()

    def setUp(self):
        self.runtime.reset()

    def place_ball(self, local_position, velocity=(0.0, 0.0, 0.0)):
        """정해진 경계 조건을 만들기 위해 실제 공의 자유 관절 상태를 설정한다."""
        r = self.runtime
        position = torch.as_tensor(local_position, dtype=torch.float32, device=r.env.device)
        if position.ndim == 1:
            position = position.expand(r.env.num_envs, -1)
        state = torch.zeros((r.env.num_envs, 13), device=r.env.device)
        state[:, :3] = position + r.scene.env_origins
        state[:, 3] = 1.0
        state[:, 7:10] = torch.as_tensor(velocity, device=r.env.device)
        r.scene["ball"].write_root_state_to_sim(state)
        return state[:, :3]

    def test_native_reset_only_changes_selected_environments(self):
        r, tracker = self.runtime, self.runtime.tracker
        self.assertIs(tracker, get_shot_tracker(r.env))
        r.step()
        qpos = r.sim.data.qpos[1].clone()
        qvel = r.sim.data.qvel[1].clone()
        elapsed = tracker.elapsed.clone()
        previous = tracker.previous_position.clone()
        tracker.touched[:] = True
        r.reset(torch.tensor([0, 2], device=r.env.device))
        # 원점이 다른 1번 환경의 물리 상태와 과거 기록은 그대로 남아야 한다.
        torch.testing.assert_close(r.sim.data.qpos[1], qpos, atol=0, rtol=0)
        torch.testing.assert_close(r.sim.data.qvel[1], qvel, atol=0, rtol=0)
        torch.testing.assert_close(tracker.elapsed[1], elapsed[1])
        torch.testing.assert_close(tracker.previous_position[1], previous[1])
        self.assertTrue(tracker.touched[1])
        self.assertTrue((tracker.elapsed[[0, 2]] == 0).all())
        self.assertFalse(tracker.touched[[0, 2]].any())
        # reset된 공의 위치에는 각 환경의 원점이 정확히 한 번 더해져야 한다.
        joint = r.model.joint("ball/ball_joint")
        address = int(joint.qposadr[0])
        world = r.sim.data.qpos[:, address:address + 3]
        torch.testing.assert_close(world[[0, 2]], tracker.previous_position[[0, 2]])

    def test_partial_reset_restores_home_targets_without_changing_other_targets(self):
        r = self.runtime
        target = r.initial_target.copy()
        target[:, K1_JOINT_ORDER.index("Head_Yaw")] = 0.2
        r.set_target(target)
        for _ in range(3):
            r.step()
        kept_target = r.robot.data.joint_pos_target[1].clone()
        r.reset([0, 2])
        # 실제 자세와 모터 목표각을 선택한 환경에서만 HOME으로 복원해야 한다.
        torch.testing.assert_close(r.robot.data.joint_pos_target[[0, 2]],
                                   r.robot.data.default_joint_pos[[0, 2]])
        torch.testing.assert_close(r.robot.data.joint_pos_target[1], kept_target)
        ctrl_ids = [r.model.actuator("robot/" + name).id for name in K1_JOINT_ORDER]
        torch.testing.assert_close(r.sim.data.ctrl[[0, 2]][:, ctrl_ids],
                                   r.robot.data.default_joint_pos[[0, 2]], atol=1e-6, rtol=0)

    def test_goal_requires_the_entire_ball_and_latches_once(self):
        r, tracker = self.runtime, self.runtime.tracker
        radius = BallCfg().radius
        # 공 중심이 선을 넘었지만 공의 뒷부분이 아직 선 앞이면 득점이 아니다.
        tracker.previous_position[:] = r.scene.env_origins + torch.tensor([0.0, 0.0, 0.5])
        self.place_ball((-0.75 - radius + 0.01, 0.0, 0.5))
        tracker(r.env)
        self.assertFalse(tracker.goal.any())
        self.place_ball((-0.75 - radius - 0.01, 0.0, 0.5))
        tracker(r.env)
        self.assertTrue(tracker.goal.all())
        elapsed = tracker.elapsed.clone()
        # 득점 뒤 튕겨 나오더라도 득점 기록은 취소되지 않고, 새 에피소드도 생성하지 않는다.
        self.place_ball((0.1, 0.0, 0.5))
        tracker(r.env)
        self.assertTrue(tracker.goal.all())
        torch.testing.assert_close(tracker.elapsed, elapsed + r.physics_dt)

    def test_fast_crossing_uses_current_root_state_and_interpolated_opening(self):
        r, tracker = self.runtime, self.runtime.tracker
        radius = BallCfg().radius
        # 한번의 물리 스텝 사이에서 선을 크게 건너뛰는 경우다. xpos를 갱신하는
        # forward()를 일부러 호출하지 않아 최신 qpos를 읽는지도 확인한다.
        tracker.previous_position[:] = r.scene.env_origins + torch.tensor([0.5, 0.0, 0.5])
        # 마지막 위치의 y=1m는 골문 밖이지만 선을 통과한 순간의 y는 약0.677m로
        # 공 반지름을 제외한 골문 안이다. 현재 위치만 검사하면 득점을 놓치게 된다.
        self.place_ball((-1.5, 1.0, 0.5), velocity=(-20.0, 0.0, 0.0))
        tracker(r.env)
        self.assertTrue(tracker.goal.all())
        self.assertLess(-1.5, -0.75 - radius)

    def test_wrong_direction_and_outside_opening_are_not_goals(self):
        r, tracker = self.runtime, self.runtime.tracker
        opening = get_opening()
        # 세 환경에 서로 다른 오답 조건을 넣어 환경별 판정이 섞이지 않는지도 본다.
        previous = torch.tensor([[-1.5, 0.0, 0.5],
                                 [0.0, opening.inner_width / 2 + 0.2, 0.5],
                                 [0.0, 0.0, opening.inner_height + 0.2]])
        current = torch.tensor([[0.0, 0.0, 0.5],
                                [-1.5, opening.inner_width / 2 + 0.2, 0.5],
                                [-1.5, 0.0, opening.inner_height + 0.2]])
        tracker.previous_position[:] = previous + r.scene.env_origins
        self.place_ball(current)
        tracker(r.env)
        self.assertFalse(tracker.goal.any())

    def test_opening_boundary_accounts_for_the_ball_radius(self):
        r, tracker = self.runtime, self.runtime.tracker
        radius, opening = BallCfg().radius, get_opening()
        usable_y = opening.inner_width / 2 - radius
        local = torch.tensor([[-1.0, usable_y - 0.002, 0.5],
                              [-1.0, usable_y + 0.002, 0.5],
                              [-1.0, 0.0, opening.inner_height - radius + 0.002]])
        before = local.clone()
        before[:, 0] = 0.0
        tracker.previous_position[:] = before + r.scene.env_origins
        self.place_ball(local)
        tracker(r.env)
        # 공 중심이 기둥/가로대 안쪽에 있다는 것만으로는 충분하지 않다.
        # 공의 크기까지 골문 안에 들어오는 첫 번째 환경만 인정해야 한다.
        torch.testing.assert_close(tracker.goal, torch.tensor([True, False, False]))

    def test_entering_from_the_side_behind_the_goal_line_is_not_a_goal(self):
        r, tracker = self.runtime, self.runtime.tracker
        usable_y = get_opening().inner_width / 2 - BallCfg().radius
        # 공 중심이 골라인을 넘을 때는 골문 옆이었다가, 선 뒤에서 골문 안쪽으로
        # 들어오는 대각선 경로다. 마지막 순간만 보면 득점으로 잘못 셀 수 있다.
        tracker.previous_position[:] = r.scene.env_origins + torch.tensor([-0.5, usable_y + 0.3, 0.5])
        self.place_ball((-1.0, usable_y - 0.2, 0.5))
        tracker(r.env)
        self.assertFalse(tracker.goal.any())

    def test_max_default_speed_ball_bounces_off_the_actual_goal_post(self):
        r, tracker = self.runtime, self.runtime.tracker
        post_id = r.model.geom("goal/post_negative_x").id
        post = torch.as_tensor(r.data.geom_xpos[post_id].copy(), dtype=torch.float32)
        local = post - r.scene.env_origins[0]
        local[0] += 0.35
        local[2] = 0.5
        tracker.previous_position[:] = self.place_ball(local, velocity=(-8.0, 0.0, 0.0))
        velocity_address = int(r.model.joint("ball/ball_joint").dofadr[0])
        bounced = torch.zeros(r.env.num_envs, dtype=torch.bool)
        for _ in range(5):
            r.step()
            bounced |= r.sim.data.qvel[:, velocity_address] > 0
        # 기둥에 정면으로 맞은 공이 계속 골문 뒤로 통과하면 안 된다. 별도 충돌
        # 공식을 대입하지 않고 실제 엔진이 계산한 이동 방향의 반전을 확인한다.
        self.assertTrue(bounced.all())
        self.assertFalse(tracker.goal.any())

    def test_native_robot_contact_is_remembered_even_if_a_goal_follows(self):
        r, tracker = self.runtime, self.runtime.tracker
        radius = BallCfg().radius
        # 몸통의 앞면에서 공을 아주 조금 겹치게 시작해 실제 접촉을 만든다.
        # 이 검사는 contact flag를 직접 True로 쓰지 않는다. 실제 기본 센서가 읽어야 한다.
        root = r.scene["robot"].data.root_link_pos_w.clone()
        local = root - r.scene.env_origins + torch.tensor([0.06 + radius - 0.01, 0.0, 0.1])
        tracker.previous_position[:] = self.place_ball(local, velocity=(-1.0, 0.0, 0.0))
        r.step()
        self.assertTrue(tracker.touched.all())
        # 접촉은 선방 성공과 다르다. 접촉 기록을 남긴 상태에서 이후 득점도 기록해야 한다.
        tracker.previous_position[:] = r.scene.env_origins + torch.tensor([0.0, 0.0, 0.5])
        self.place_ball((-0.75 - radius - 0.02, 0.0, 0.5))
        tracker(r.env)
        self.assertTrue(tracker.touched.all())
        self.assertTrue(tracker.goal.all())

    def test_max_default_speed_contact_is_latched_after_the_ball_separates(self):
        r, tracker = self.runtime, self.runtime.tracker
        radius = BallCfg().radius
        # 기본 상한 8m/s의 공을 몸통 앞에서 발사한다. 시작 시에는 6cm의 틈이 있어
        # 이미 겹친 상태를 감지하는 검사와 구분된다. 짧은 접촉도 기록되어야 한다.
        local = r.scene["robot"].data.root_link_pos_w - r.scene.env_origins
        local = local + torch.tensor([0.06 + radius + 0.06, 0.0, 0.1])
        tracker.previous_position[:] = self.place_ball(local, velocity=(-8.0, 0.0, 0.0))
        r.step()
        self.assertTrue(tracker.touched.all())
        # 공을 다시 멀리 옮겨 이번 스텝에는 접촉이 사라지게 한다. 센서의 현재 접촉은
        # 없어져도 에피소드의 접촉 기록은 남아야 보상/평가에서 사용할 수 있다.
        tracker.previous_position[:] = self.place_ball((1.5, 0.0, 0.5))
        r.step()
        self.assertFalse((tracker.sensor.data.found > 0).any())
        self.assertTrue(tracker.touched.all())

    def test_timeout_uses_native_auto_reset_and_starts_a_new_shot(self):
        r, tracker = self.runtime, self.runtime.tracker
        timed_out = torch.zeros(r.num_envs, dtype=torch.bool)
        for _ in range(8):
            _, _, _, truncated, _ = r.step()
            timed_out |= truncated
            # 자동 reset에도 첫 목표가 0 rad로 바뀌면 안 된다. 실제 엔진 명령을 읽는다.
            ctrl_ids = [r.model.actuator("robot/" + name).id for name in K1_JOINT_ORDER]
            torch.testing.assert_close(r.sim.data.ctrl[:, ctrl_ids],
                                       r.robot.data.default_joint_pos, atol=1e-6, rtol=0)
        # 이 설정의 슛은 제한 시간 0.12초 안에 골문에 도착할 수 없다.
        # 따라서 실제 기본 환경의 timeout → 부분 reset → 새 발사를 확인한다.
        self.assertTrue(timed_out.all())
        self.assertTrue((tracker.elapsed < 0.12 + r.physics_dt).all())

    def test_curriculum_runtime_settings_only_apply_to_reset_environments(self):
        r, tracker = self.runtime, self.runtime.tracker
        term = r.env.event_manager.get_term_cfg("launch_ball")
        original = term.params["shot_cfg"]
        previous_before = tracker.previous_position.clone()
        limits_before = tracker.timeout_limit.clone()
        out_before = tracker.out_x_max.clone()
        qpos_before = r.sim.data.qpos.clone()
        try:
            # 실제 커리큘럼은 이 발사 이벤트의 설정을 바꾸면 된다. 매니저들이 설정을
            # 따로 복사하므로, 원래 env_cfg 객체만 바꾸는 것과는 구분해야 한다.
            term.params["shot_cfg"] = replace(original, ground_probability=1.0,
                launch_distance=(2.0, 2.0), launch_y=(0.2, 0.2), speed=(4.0, 4.0),
                episode_timeout_s=1.0, out_margin=0.5)
            # 설정을 바꾸는 순간 이미 날아가는 공이나 종료 조건이 바뀌면 안 된다.
            torch.testing.assert_close(r.sim.data.qpos[:], qpos_before)
            torch.testing.assert_close(tracker.previous_position, previous_before)
            torch.testing.assert_close(tracker.timeout_limit, limits_before)
            torch.testing.assert_close(tracker.out_x_max, out_before)
            r.reset([0])
            address = tracker.ball_qpos_adr
            local = r.sim.data.qpos[:, address:address + 3] - r.scene.env_origins
            torch.testing.assert_close(local[0], torch.tensor([2.0, 0.2, BallCfg().radius]))
            self.assertAlmostEqual(tracker.timeout_limit[0].item(), 1.0)
            self.assertAlmostEqual(tracker.out_x_max[0].item(), 2.5)
            torch.testing.assert_close(tracker.previous_position[1:], previous_before[1:])
            torch.testing.assert_close(r.sim.data.qpos[1:], qpos_before[1:])
            torch.testing.assert_close(tracker.timeout_limit[1:], limits_before[1:])
            torch.testing.assert_close(tracker.out_x_max[1:], out_before[1:])
        finally:
            term.params["shot_cfg"] = original

    def test_native_outside_termination_resets_only_the_outside_balls(self):
        r, tracker = self.runtime, self.runtime.tracker
        address = tracker.ball_qpos_adr
        local = r.sim.data.qpos[:, address:address + 3] - r.scene.env_origins
        local[0, 0] = tracker.out_x_max[0] + 0.3
        local[1, 1] = tracker.out_y_max[1] + 0.3
        self.place_ball(local)
        _, _, terminated, truncated, info = r.step()
        torch.testing.assert_close(terminated, torch.tensor([True, True, False]))
        self.assertFalse(truncated.any())
        torch.testing.assert_close(r.env.episode_length_buf, torch.tensor([0, 0, 1]))
        self.assertEqual(info["log"]["Episode_Termination/ball_out"], 2)

    def test_shifted_xyz_origin_preserves_launch_and_goal_coordinates(self):
        r = ShotSimulation(goalkeeper_k1_env_cfg(num_envs=2, shots=ShotCfg()), origin=(3.0, -2.0, 0.4))
        try:
            tracker = r.tracker
            address = int(r.model.joint("ball/ball_joint").qposadr[0])
            torch.testing.assert_close(r.sim.data.qpos[:, address:address + 3],
                                       tracker.previous_position)
            tracker.previous_position[:] = r.scene.env_origins + torch.tensor([0.0, 0.0, 0.5])
            state = torch.zeros((2, 13))
            state[:, 3] = 1.0
            state[:, :3] = r.scene.env_origins + torch.tensor([-1.0, 0.0, 0.5])
            r.scene["ball"].write_root_state_to_sim(state)
            tracker(r.env)
            self.assertTrue(tracker.goal.all())
        finally:
            r.env.close()

    def test_real_rolling_shot_scores_with_soft_floor_contact(self):
        opening, radius = get_opening(), BallCfg().radius
        fraction = 0.5 / (opening.inner_width / 2 - radius)
        shots = replace(ShotCfg(), ground_probability=1.0, launch_distance=(1.5, 1.5),
                        launch_y=(0.5, 0.5), speed=(3.0, 3.0),
                        target_y_fraction=(fraction, fraction), episode_timeout_s=2.0)
        r = ShotSimulation(goalkeeper_k1_env_cfg(shots=shots))
        try:
            # 공 발사 자체를 검사하는 장면이다. 로봇만 10m 옮겨 골문을 비워 두고,
            # 공·바닥·골대·모터·접촉 솔버는 실제 학습 설정을 그대로 사용한다.
            robot = r.scene["robot"]
            state = robot.data.default_root_state.clone()
            state[:, :3] += r.scene.env_origins
            state[:, 0] += 10.0
            robot.write_root_state_to_sim(state)
            completed = False
            for _ in range(110):
                _, _, terminated, truncated, info = r.step()
                if (terminated | truncated).any():
                    completed = True
                    break
            self.assertTrue(completed)
            # 부드러운 바닥 접촉에서는 공 중심이 반지름보다 아주 조금 낮을 수 있다.
            # 그 물리적인 접촉 오차 때문에 모든 땅볼을 골문 밖으로 판정하면 안 된다.
            self.assertEqual(info["log"]["Episode_Termination/goal"], 1)
        finally:
            r.env.close()


class ShotValidationEntryTests(unittest.TestCase):
    def test_first_reset_and_delayed_commands_start_at_home(self):
        r = ShotSimulation()
        try:
            ctrl_ids = [r.model.actuator("robot/" + name).id for name in K1_JOINT_ORDER]
            home = r.robot.data.default_joint_pos.clone()
            # 첫 장면을 만든 직후부터 자세와 명령이 같아야 한다. 첫 step 뒤만 검사하면
            # 그전에 지연 버퍼를 오염시킨 0 rad 명령을 놓칠 수 있다.
            torch.testing.assert_close(r.robot.data.joint_pos_target, home)
            torch.testing.assert_close(r.sim.data.ctrl[:, ctrl_ids], home, atol=1e-6, rtol=0)

            history = []
            original_write = r.robot.write_data_to_sim

            def record_actual_control():
                original_write()
                history.append(r.sim.data.ctrl[:, ctrl_ids].clone())

            r.robot.write_data_to_sim = record_actual_control
            try:
                # 최대 지연 8스텝보다 오래 실행한다. 중간의 0↔HOME 명령 전환도 검사한다.
                for _ in range(3):
                    r.step()
            finally:
                r.robot.write_data_to_sim = original_write
            self.assertEqual(len(history), 12)
            for command in history:
                torch.testing.assert_close(command, home, atol=1e-6, rtol=0)
        finally:
            r.close()

    """사용자가 실행하는 진입점도 같은 학습용 슛 기능에 연결되는지 확인한다."""

    def test_validation_rejects_disabled_auto_reset_before_compilation(self):
        cfg = goalkeeper_k1_env_cfg(shots=ShotCfg())
        cfg.auto_reset = False
        # 연속 발사 도구는 MJLab 기본 자동 reset을 전제로 하므로 컴파일 전에 확인한다.
        with self.assertRaisesRegex(ValueError, "auto_reset=True"):
            ShotSimulation(cfg)

    def test_headless_cli_collects_requested_parallel_shots(self):
        repo = Path(__file__).resolve().parents[1]
        # 실제 main.py를 새 Python 프로세스로 실행한다. import 상태나 기존 테스트의
        # 환경 객체에 기대지 않고 서버용 --headless 옵션이 끝까지 동작하는지 본다.
        result = subprocess.run([
            sys.executable, str(repo / "main.py"), "--mode", "shots", "--headless",
            "--shots", "3", "--num-envs", "2", "--seed", "7", "--episode-seconds", "0.12",
        ], cwd=repo, text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report["passed"])
        self.assertEqual(report["num_envs"], 2)
        self.assertGreaterEqual(report["completed_episodes"], 3)
        self.assertLess(report["completed_episodes"], 5)
        self.assertTrue(report["finite"])


if __name__ == "__main__":
    unittest.main()
