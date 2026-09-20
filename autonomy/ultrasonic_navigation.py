"""초음파 센서 기반 점유 격자와 기존 A*를 결합한 자율주행 예제.

좌표계는 로봇 기준으로 x=좌우, y=전방이며, 지도는 5 cm 셀을 사용한다.
실제 주행 전에는 HC-SR04 Echo가 반드시 3.3 V 이하가 되도록 전압 분압을
확인하고, 모터 PWM 핀과 GPIO 25가 중복되지 않도록 배선을 조정해야 한다.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from pathfinding.astar import AStarPlanner

try:
    from gpiozero import DistanceSensor
except ImportError:  # 개발 PC에서 지도/필터를 테스트할 수 있도록 지연 실패
    DistanceSensor = None


CELL_SIZE_M = 0.05
MAP_WIDTH_M = 0.5
MAP_DEPTH_M = 0.5
GRID_WIDTH = int(MAP_WIDTH_M / CELL_SIZE_M)
GRID_HEIGHT = int(MAP_DEPTH_M / CELL_SIZE_M)

LEFT_TRIG, LEFT_ECHO = 4, 17
RIGHT_TRIG, RIGHT_ECHO = 14, 15
ULTRASONIC_OBSTACLE_DISTANCE_M = 0.20


class KalmanFilter1D:
    """상수 상태/상수 측정 모델의 1차원 칼만 필터."""

    def __init__(
        self,
        process_variance: float = 0.002,
        measurement_variance: float = 0.01,
        initial_estimate: float = 0.5,
        max_jump_m: float = 0.20,
    ) -> None:
        if process_variance <= 0 or measurement_variance <= 0:
            raise ValueError("분산은 0보다 커야 합니다.")
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = initial_estimate
        self.error_covariance = measurement_variance
        self.max_jump_m = max_jump_m
        self.initialized = False

    def update(self, measurement: float) -> float:
        if not math.isfinite(measurement) or measurement < 0:
            raise ValueError(f"유효하지 않은 거리 측정값: {measurement!r}")
        if not self.initialized:
            self.estimate = measurement
            self.initialized = True
            return self.estimate

        # 한 샘플에서 물리적으로 불가능한 큰 튐은 필터 입력을 제한한다.
        measurement = self.estimate + np.clip(
            measurement - self.estimate, -self.max_jump_m, self.max_jump_m
        )
        predicted_covariance = self.error_covariance + self.process_variance
        gain = predicted_covariance / (predicted_covariance + self.measurement_variance)
        self.estimate += gain * (measurement - self.estimate)
        self.error_covariance = (1.0 - gain) * predicted_covariance
        return self.estimate


class OccupancyGrid:
    """로그 오즈(log-odds) 기반의 확률 점유 격자."""

    def __init__(
        self,
        width_m: float = MAP_WIDTH_M,
        depth_m: float = MAP_DEPTH_M,
        cell_size_m: float = CELL_SIZE_M,
    ) -> None:
        if min(width_m, depth_m, cell_size_m) <= 0:
            raise ValueError("지도 크기와 셀 크기는 0보다 커야 합니다.")
        self.cell_size_m = cell_size_m
        self.width = int(round(width_m / cell_size_m))
        self.height = int(round(depth_m / cell_size_m))
        self.log_odds = np.zeros((self.height, self.width), dtype=np.float32)

    @property
    def probabilities(self) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.log_odds))

    def _cell(self, x_m: float, y_m: float) -> Optional[Tuple[int, int]]:
        col = int(math.floor((x_m + self.width * self.cell_size_m / 2) / self.cell_size_m))
        row = int(math.floor(y_m / self.cell_size_m))
        if 0 <= row < self.height and 0 <= col < self.width:
            return row, col
        return None

    def _update_cell(self, cell: Optional[Tuple[int, int]], delta: float) -> None:
        if cell is not None:
            row, col = cell
            self.log_odds[row, col] = np.clip(
                self.log_odds[row, col] + delta, -5.0, 5.0
            )

    def mark_obstacle(self, x_m: float, y_m: float, confidence: float = 0.98) -> None:
        """센서 임계값에 걸린 셀을 즉시 장애물로 표시한다."""
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence는 0과 1 사이여야 합니다.")
        cell = self._cell(x_m, y_m)
        if cell is not None:
            self.log_odds[cell] = max(
                self.log_odds[cell], math.log(confidence / (1.0 - confidence))
            )

    def update_beam(
        self,
        distance_m: float,
        angle_deg: float,
        max_range_m: Optional[float] = None,
    ) -> None:
        if not math.isfinite(distance_m) or distance_m <= 0:
            return
        max_range_m = max_range_m or self.height * self.cell_size_m
        measured_range = min(distance_m, max_range_m)
        angle = math.radians(angle_deg)
        end_x = math.sin(angle) * measured_range
        end_y = math.cos(angle) * measured_range

        start = self._cell(0.0, 0.0)
        end = self._cell(end_x, end_y)
        if start is None:
            return
        end_for_ray = end if end is not None else self._cell(
            math.sin(angle) * max_range_m, math.cos(angle) * max_range_m
        )
        if end_for_ray is None:
            return

        cells = list(self._ray_cells(start, end_for_ray))
        hit_obstacle = distance_m < max_range_m and end is not None
        for cell in cells[:-1] if hit_obstacle else cells:
            self._update_cell(cell, -0.45)
        if hit_obstacle:
            self._update_cell(end, 0.95)

    @staticmethod
    def _ray_cells(
        start: Tuple[int, int], end: Tuple[int, int]
    ) -> Iterable[Tuple[int, int]]:
        row0, col0 = start
        row1, col1 = end
        steps = max(abs(row1 - row0), abs(col1 - col0), 1)
        for index in range(steps + 1):
            ratio = index / steps
            yield (
                int(round(row0 + (row1 - row0) * ratio)),
                int(round(col0 + (col1 - col0) * ratio)),
            )

    def obstacle_mask(self, threshold: float = 0.65) -> np.ndarray:
        return (self.probabilities >= threshold).astype(np.uint8) * 255


@dataclass(frozen=True)
class SensorReading:
    left_m: float
    right_m: float


class UltrasonicSensors:
    """지정된 GPIO 핀으로 좌/우 DistanceSensor를 생성하는 입력 계층."""

    def __init__(self, max_distance_m: float = 0.5, queue_len: int = 5) -> None:
        if DistanceSensor is None:
            raise RuntimeError("gpiozero가 필요합니다. 라즈베리 파이에서 실행하세요.")
        self.left = DistanceSensor(
            echo=LEFT_ECHO, trigger=LEFT_TRIG, max_distance=max_distance_m,
            queue_len=queue_len
        )
        self.right = DistanceSensor(
            echo=RIGHT_ECHO, trigger=RIGHT_TRIG, max_distance=max_distance_m,
            queue_len=queue_len
        )

    def read_m(self) -> SensorReading:
        # gpiozero.distance는 0~1 정규화 거리이므로 max_distance로 환산한다.
        return SensorReading(
            left_m=self.left.distance * self.left.max_distance,
            right_m=self.right.distance * self.right.max_distance,
        )

    def close(self) -> None:
        self.left.close()
        self.right.close()


class UltrasonicAStarController:
    """필터링된 센서 -> 점유지도 -> 기존 AStarPlanner -> 차동 구동 명령."""

    def __init__(
        self,
        sensors: UltrasonicSensors,
        grid: Optional[OccupancyGrid] = None,
        obstacle_distance_m: float = ULTRASONIC_OBSTACLE_DISTANCE_M,
    ) -> None:
        self.sensors = sensors
        self.grid = grid or OccupancyGrid()
        self.left_filter = KalmanFilter1D()
        self.right_filter = KalmanFilter1D()
        self.obstacle_distance_m = obstacle_distance_m
        self.planner = AStarPlanner(robot_radius_px=0, grid_size=1, smooth_path=False)
        self.last_path: List[Tuple[int, int]] = []
        self.replan_count = 0
        self.last_log_time = 0.0

    def step(self) -> Tuple[float, float, List[Tuple[int, int]]]:
        raw = self.sensors.read_m()
        left_m = self.left_filter.update(raw.left_m)
        right_m = self.right_filter.update(raw.right_m)
        self.grid.update_beam(left_m, -35.0)
        self.grid.update_beam(right_m, 35.0)

        left_blocked = left_m <= self.obstacle_distance_m
        right_blocked = right_m <= self.obstacle_distance_m
        blocked_sides = []
        if left_blocked:
            self.grid.mark_obstacle(-left_m, max(left_m, 0.05))
            blocked_sides.append("LEFT")
        if right_blocked:
            self.grid.mark_obstacle(right_m, max(right_m, 0.05))
            blocked_sides.append("RIGHT")

        mask = self.grid.obstacle_mask()
        start = (self.grid.width // 2, 0)
        goal = (self.grid.width // 2, self.grid.height - 1)
        path = self.planner.plan(start, goal, mask)
        self.replan_count += 1
        self.last_path = path

        now = time.monotonic()
        if blocked_sides or now - self.last_log_time >= 1.0:
            sides = "+".join(blocked_sides)
            print(
                f"[ULTRASONIC] raw(L/R)={raw.left_m * 100:5.1f}/"
                f"{raw.right_m * 100:5.1f}cm "
                f"filtered(L/R)={left_m * 100:5.1f}/"
                f"{right_m * 100:5.1f}cm "
                f"blocked={sides or 'NONE'} "
                f"map={'UPDATED' if blocked_sides else 'stable'} "
                f"A*=#{self.replan_count} waypoints={len(path)}"
            )
            self.last_log_time = now

        # 양쪽이 동시에 막히면 전진 경로가 없으므로 정지하고 다음 샘플에서 재탐색한다.
        if left_blocked and right_blocked:
            print("[PATH] 좌우 모두 20 cm 이내 -> 정지 (A* 재탐색 대기)")
            return 0.0, 0.0, path

        if len(path) < 2:
            print("[PATH] 유효한 A* 우회 경로 없음 -> 정지")
            return 0.0, 0.0, path

        next_x, _ = path[1]
        steer = float(np.clip((next_x - start[0]) * 20.0, -35.0, 35.0))
        speed = 55.0 if mask[1, self.grid.width // 2] == 0 else 35.0
        if left_blocked and not right_blocked:
            steer = min(steer, -25.0)
            print("[PATH] 왼쪽 장애물 회피 -> 오른쪽으로 조향")
        elif right_blocked and not left_blocked:
            steer = max(steer, 25.0)
            print("[PATH] 오른쪽 장애물 회피 -> 왼쪽으로 조향")
        return speed - steer, speed + steer, path

    def run(self, period_s: float = 0.1) -> None:
        try:
            while True:
                left, right, path = self.step()
                print(f"[AUTO] L={left:+.0f} R={right:+.0f} path={len(path)}")
                time.sleep(period_s)
        except KeyboardInterrupt:
            print("\n[종료] 초음파 자율주행을 종료합니다.")
        finally:
            self.sensors.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="초음파 점유지도 + A* 자율주행")
    parser.add_argument("--period", type=float, default=0.1)
    args = parser.parse_args()
    sensors = UltrasonicSensors()
    controller = UltrasonicAStarController(sensors=sensors)
    controller.run(period_s=max(0.02, args.period))


if __name__ == "__main__":
    main()
