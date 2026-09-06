"""
스타크래프트식 네비게이터 및 주행 제어기 (navigator.py)
"""
from enum import Enum
import math
import time
from typing import List, Optional, Tuple
import numpy as np

from pathfinding.astar import AStarPlanner


class NavState(Enum):
    IDLE = "IDLE"
    ROTATING = "ROTATING"
    MOVING = "MOVING"
    ARRIVED = "ARRIVED"
    LOST_ROBOT = "LOST_ROBOT"


class StarcraftNavigator:
    def __init__(
        self,
        kp_angle: float = 1.8,
        kd_angle: float = 0.12,
        kp_dist: float = 0.65,
        kd_dist: float = 0.08,
        rot_threshold_deg: float = 25.0,
        waypoint_dist_px: float = 40.0,
        final_goal_dist_px: float = 30.0,
        max_speed: int = 150,
        min_forward_pwm: int = 75,
        min_rot_pwm: int = 115,
        max_rot_pwm: int = 175,
        diff_weight: float = 38.0,
        robot_radius_px: int = 38,
        grid_size: int = 12,
        step_drive_enabled: bool = True,
        step_move_sec: float = 0.22,
        step_pause_sec: float = 0.09
    ):
        self.kp_angle = kp_angle
        self.kd_angle = kd_angle
        self.kp_dist = kp_dist
        self.kd_dist = kd_dist
        self.rot_threshold_deg = rot_threshold_deg
        self.waypoint_dist_px = waypoint_dist_px
        self.final_goal_dist_px = final_goal_dist_px
        self.max_speed = max_speed
        self.min_forward_pwm = min_forward_pwm
        self.min_rot_pwm = min_rot_pwm
        self.max_rot_pwm = max_rot_pwm
        self.diff_weight = diff_weight
        
        # 스텝 주행 설정
        self.step_drive_enabled = step_drive_enabled
        self.step_move_sec = step_move_sec
        self.step_pause_sec = step_pause_sec
        self.step_start_time = time.time()

        self.planner = AStarPlanner(robot_radius_px=robot_radius_px, grid_size=grid_size)

        # 경로 및 목표 상태
        self.final_goal: Optional[Tuple[int, int]] = None
        self.waypoints: List[Tuple[int, int]] = []
        self.current_wp_idx: int = 0
        self.state: NavState = NavState.IDLE

        # PID 에러 추적
        self.prev_angle_error: float = 0.0
        self.prev_dist_error: float = 0.0
        self.last_update_time: float = time.time()

        # 우클릭 핑 위치
        self.ping_pos: Optional[Tuple[int, int]] = None
        self.ping_start_time: float = 0.0

    @staticmethod
    def normalize_angle(angle_rad: float) -> float:
        # 각도를 -pi ~ pi 범위로 맞춤
        while angle_rad > math.pi:
            angle_rad -= 2 * math.pi
        while angle_rad < -math.pi:
            angle_rad += 2 * math.pi
        return angle_rad

    def set_goal(
        self,
        robot_pos: Optional[Tuple[int, int]],
        goal_pos: Tuple[int, int],
        obstacle_mask: Optional[np.ndarray] = None
    ):
        self.final_goal = goal_pos
        self.ping_pos = goal_pos
        self.ping_start_time = time.time()
        self.prev_angle_error = 0.0
        self.prev_dist_error = 0.0
        self.step_start_time = time.time()

        if robot_pos is None:
            self.waypoints = [goal_pos]
            self.current_wp_idx = 0
            self.state = NavState.LOST_ROBOT
            print(f"[NAV] 목표 설정: {goal_pos} (로봇 미감지)")
            return

        # A* 경로 계산
        path = []
        if obstacle_mask is not None and np.any(obstacle_mask > 0):
            try:
                path = self.planner.plan(robot_pos, goal_pos, obstacle_mask)
            except Exception as e:
                print(f"[A* 오류]: {e}")

        # 경로 없으면 직선 경로로 대체
        if not path:
            path = [robot_pos, goal_pos]

        self.waypoints = path

        # 첫 번째 점 시작 인덱스 결정
        if len(self.waypoints) > 1:
            d0 = math.hypot(self.waypoints[0][0] - robot_pos[0], self.waypoints[0][1] - robot_pos[1])
            self.current_wp_idx = 1 if d0 < 35.0 else 0
        else:
            self.current_wp_idx = 0

        self.state = NavState.MOVING
        print(f"[NAV] 목표 이동: {goal_pos} (웨이포인트 {len(self.waypoints)}개)")

    def return_to_home(
        self,
        current_pos: Tuple[int, int],
        home_pos: Tuple[int, int],
        obstacle_mask: Optional[np.ndarray] = None,
        is_blind: bool = False
    ):
        print(f"[NAV] 홈 복귀: {home_pos}")
        self.set_goal(current_pos, home_pos, obstacle_mask)

    def update_control(
        self,
        robot_pos: Optional[Tuple[int, int]],
        robot_angle_deg: float,
        is_odom_fallback: bool = False
    ) -> Tuple[float, float, NavState]:
        now = time.time()
        dt = max(0.01, min(0.1, now - self.last_update_time))
        self.last_update_time = now

        if self.final_goal is None or self.state == NavState.ARRIVED or self.state == NavState.IDLE:
            return 0.0, 0.0, NavState.IDLE

        if robot_pos is None:
            self.state = NavState.LOST_ROBOT
            return 0.0, 0.0, NavState.LOST_ROBOT

        rx, ry = robot_pos

        # 1. 최종 목적지 도착 검사
        fx, fy = self.final_goal
        dist_to_final = math.hypot(fx - rx, fy - ry)
        if dist_to_final <= self.final_goal_dist_px:
            self.state = NavState.ARRIVED
            self.final_goal = None
            self.waypoints.clear()
            print(f"[NAV] 목적지 도착 완료 (오차: {dist_to_final:.1f}px)")
            return 0.0, 0.0, NavState.ARRIVED

        # 2. 현재 타겟 웨이포인트 선택
        if self.current_wp_idx >= len(self.waypoints):
            target_wp = (fx, fy)
        else:
            target_wp = self.waypoints[self.current_wp_idx]

        tx, ty = target_wp
        dist_to_wp = math.hypot(tx - rx, ty - ry)

        # 웨이포인트 근접 시 다음으로 넘김
        if dist_to_wp <= self.waypoint_dist_px and self.current_wp_idx < len(self.waypoints) - 1:
            self.current_wp_idx += 1
            target_wp = self.waypoints[self.current_wp_idx]
            tx, ty = target_wp
            dist_to_wp = math.hypot(tx - rx, ty - ry)

        # 3. 목표 방향 각도 계산
        dx = tx - rx
        dy = ty - ry
        target_angle_deg = math.degrees(math.atan2(-dy, dx)) % 360.0

        angle_error_rad = math.radians(target_angle_deg - robot_angle_deg)
        angle_error_rad = self.normalize_angle(angle_error_rad)
        angle_error_deg = math.degrees(angle_error_rad)

        # 4. PID 계산
        angle_deriv = (angle_error_rad - self.prev_angle_error) / dt
        w = (self.kp_angle * angle_error_rad) + (self.kd_angle * angle_deriv)
        self.prev_angle_error = angle_error_rad

        dist_deriv = (dist_to_wp - self.prev_dist_error) / dt
        v_raw = (self.kp_dist * dist_to_wp) + (self.kd_dist * dist_deriv)
        self.prev_dist_error = dist_to_wp

        # 5. 각도 오차에 따라 회전/전진 분기 (히스테리시스로 채터링/뒤뚱거림 억제)
        # 회전 중일 때는 55% 각도(약 15도)까지 충분히 정렬된 후 전진으로 전환
        rot_threshold = self.rot_threshold_deg if self.state != NavState.ROTATING else (self.rot_threshold_deg * 0.55)

        if abs(angle_error_deg) > rot_threshold:
            # 제자리 회전
            self.state = NavState.ROTATING
            # 각도 오차가 작아질수록 회전 PWM을 부드럽게 감속 (오버슈트/뒤뚱거림 원천 차단)
            scale = min(1.0, max(0.0, abs(angle_error_deg) / 50.0))
            rot_magnitude = self.min_rot_pwm + scale * (self.max_rot_pwm - self.min_rot_pwm)
            rot_magnitude = float(np.clip(rot_magnitude, self.min_rot_pwm, self.max_rot_pwm))
            
            if angle_error_deg > 0:
                v_left = -rot_magnitude
                v_right = rot_magnitude
            else:
                v_left = rot_magnitude
                v_right = -rot_magnitude
        else:
            # 전진 및 조향
            self.state = NavState.MOVING
            
            # 스텝 주행 펄스 체크
            if self.step_drive_enabled:
                cycle = self.step_move_sec + self.step_pause_sec
                phase = (now - self.step_start_time) % cycle
                if phase > self.step_move_sec:
                    return 0.0, 0.0, NavState.MOVING

            v_forward = float(np.clip(v_raw, self.min_forward_pwm, self.max_speed))
            # 조향 편차 (급격한 쏠림 방지 클리핑)
            steer_bias = float(np.clip(w * self.diff_weight, -30.0, 30.0))
            
            v_left = float(np.clip(v_forward - steer_bias, -self.max_speed, self.max_speed))
            v_right = float(np.clip(v_forward + steer_bias, -self.max_speed, self.max_speed))

        return v_left, v_right, self.state

    def is_current_path_blocked(self, obstacle_mask: Optional[np.ndarray], robot_pos: Optional[Tuple[int, int]]) -> bool:
        """현재 남아있는 웨이포인트 경로상에 장애물이 침범했는지 검사"""
        if obstacle_mask is None or robot_pos is None or not self.waypoints:
            return False
        
        remaining = self.waypoints[self.current_wp_idx:]
        if not remaining:
            return False
        
        pts = [robot_pos] + remaining
        h, w = obstacle_mask.shape[:2]
        
        for i in range(len(pts) - 1):
            p1, p2 = pts[i], pts[i + 1]
            dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            steps = max(2, int(dist / 8.0))
            for s in range(1, steps):
                t = s / steps
                cx = int(round(p1[0] + t * (p2[0] - p1[0])))
                cy = int(round(p1[1] + t * (p2[1] - p1[1])))
                if 0 <= cx < w and 0 <= cy < h and obstacle_mask[cy, cx] > 0:
                    return True
        return False

    def reset(self):
        """네비게이션 상태 초기화"""
        self.final_goal = None
        self.waypoints.clear()
        self.current_wp_idx = 0
        self.state = NavState.IDLE
        self.prev_angle_error = 0.0
        self.prev_dist_error = 0.0
        self.ping_pos = None
