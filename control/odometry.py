"""
모터 데드 레코닝 및 오도메트리 추정기 (odometry.py)
ArUco 마커를 놓쳤을 때 모터 PWM 및 차동 키네마틱스를 적분하여
로봇의 위치(x, y)와 방향(angle)을 실시간 추정합니다.
"""
import math
import time
from typing import Optional, Tuple
import numpy as np


class OdometryEstimator:
    def __init__(
        self,
        kv_scale: float = 1.6,   # PWM 대비 초당 이동 픽셀 비례 계수 (px / PWM*sec)
        kw_scale: float = 2.4,   # PWM 차이 대비 초당 회전 각속도 비례 계수 (deg / PWM*sec)
        track_width_px: float = 70.0 # 좌우 바퀴 간격 (px)
    ):
        self.kv_scale = kv_scale
        self.kw_scale = kw_scale
        self.track_width = track_width_px

        # 추정 상태
        self.est_x: float = 640.0
        self.est_y: float = 360.0
        self.est_angle_deg: float = 0.0

        # 초기 홈 위치 (처음 ArUco 마커가 감지된 위치)
        self.home_pos: Optional[Tuple[int, int]] = None
        self.home_angle_deg: float = 0.0

        self.last_update_time: float = time.time()
        self.is_initialized = False

    def sync_with_vision(self, robot_pos: Tuple[int, int], robot_angle_deg: float):
        """실제 ArUco 마커 비전 측정값으로 오도메트리 위치를 보정 및 동기화"""
        self.est_x = float(robot_pos[0])
        self.est_y = float(robot_pos[1])
        self.est_angle_deg = float(robot_angle_deg)
        self.last_update_time = time.time()

        # 홈 위치가 아직 없으면 첫 감지 위치를 홈으로 등록
        if self.home_pos is None:
            self.home_pos = (int(self.est_x), int(self.est_y))
            self.home_angle_deg = float(robot_angle_deg)
            print(f"[HOME REGISTERED] 기준 홈 위치 등록 완료: {self.home_pos} ({self.home_angle_deg:.1f}°)")

        self.is_initialized = True

    def update_from_motors(self, v_left: float, v_right: float) -> Tuple[Tuple[int, int], float]:
        """
        모터 출력값(v_left, v_right)을 기반으로 위치와 방향을 적분(Dead-Reckoning)
        :return: ((est_x, est_y), est_angle_deg)
        """
        now = time.time()
        dt = max(0.005, min(0.1, now - self.last_update_time))
        self.last_update_time = now

        if not self.is_initialized:
            return (int(self.est_x), int(self.est_y)), self.est_angle_deg

        # 1. 차동 구동 속도 계산
        # 선속도 v (px/s)
        v_avg = (v_left + v_right) / 2.0
        linear_speed = v_avg * self.kv_scale

        # 각속도 w (deg/s) (오른쪽이 크면 반시계 회전)
        diff_speed = (v_right - v_left)
        angular_speed_deg = (diff_speed / 2.0) * self.kw_scale

        # 2. 자세 각도 업데이트 (0 ~ 360 정규화)
        self.est_angle_deg = (self.est_angle_deg + angular_speed_deg * dt) % 360.0

        # 3. 2D 위치 업데이트 (화면 좌표계: y축 아래 방향)
        rad = math.radians(self.est_angle_deg)
        self.est_x += linear_speed * math.cos(rad) * dt
        self.est_y -= linear_speed * math.sin(rad) * dt

        # 맵 영역(1280x720) 클리핑
        self.est_x = float(np.clip(self.est_x, 10.0, 1270.0))
        self.est_y = float(np.clip(self.est_y, 10.0, 710.0))

        return (int(round(self.est_x)), int(round(self.est_y))), self.est_angle_deg

    def get_estimated_pose(self) -> Tuple[Tuple[int, int], float]:
        """현재 추정된 위치와 각도 반환"""
        return (int(round(self.est_x)), int(round(self.est_y))), self.est_angle_deg

    def set_home_manual(self, pos: Tuple[int, int], angle_deg: float = 0.0):
        """수동으로 홈 위치 설정"""
        self.home_pos = pos
        self.home_angle_deg = angle_deg
        print(f"[HOME MANUAL] 홈 위치 수동 재설정: {self.home_pos}")
