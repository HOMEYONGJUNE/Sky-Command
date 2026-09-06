import heapq
import math
from typing import List, Optional, Tuple
import cv2
import numpy as np


class AStarPlanner:
    def __init__(self, robot_radius_px: int = 38, grid_size: int = 12, smooth_path: bool = True):
        self.robot_radius_px = robot_radius_px
        self.grid_size = grid_size
        self.smooth_path = smooth_path
        
        # 장애물 팽창용 커널
        k_size = int(robot_radius_px * 2) + 1
        self.inflation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

    def create_inflated_obstacle_map(self, obstacle_mask: np.ndarray, radius_px: Optional[int] = None) -> np.ndarray:
        if obstacle_mask is None or obstacle_mask.size == 0 or np.count_nonzero(obstacle_mask) == 0:
            return np.zeros((100, 100), dtype=np.uint8)
        r = radius_px if radius_px is not None else self.robot_radius_px
        if r is None or r <= 0:
            return obstacle_mask.copy()
        k_size = max(3, int(r * 2) + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        return cv2.dilate(obstacle_mask, kernel)

    def plan(
        self,
        start_pt: Tuple[int, int],
        goal_pt: Tuple[int, int],
        obstacle_mask: np.ndarray
    ) -> List[Tuple[int, int]]:
        if start_pt is None or goal_pt is None:
            return []

        h, w = obstacle_mask.shape[:2]

        # 맵 범위 안으로 클리핑
        sx = int(np.clip(start_pt[0], 0, w - 1))
        sy = int(np.clip(start_pt[1], 0, h - 1))
        gx = int(np.clip(goal_pt[0], 0, w - 1))
        gy = int(np.clip(goal_pt[1], 0, h - 1))

        cols = max(1, w // self.grid_size)
        rows = max(1, h // self.grid_size)

        # 화면에 표시되는 파란색 뻥튀기 마스크(obstacle_mask)와 실제 A* 탐색 영역을 1:1로 직접 연동
        # 이미 마스크 자체에 회피 마진이 반영되어 있으므로, 2차 과도 팽창을 제거하고 미세 마진만 적용
        radii_to_try = [4, 0]

        # 스무딩 검사용 안전 맵 (화면상 뻥튀기 영역과 1:1)
        safety_check_map = obstacle_mask.copy()

        for r_px in radii_to_try:
            inflated_map = self.create_inflated_obstacle_map(obstacle_mask, r_px)

            # 그리드 축소 및 장애물 판정
            small_inflated = cv2.resize(
                inflated_map, (cols, rows), interpolation=cv2.INTER_AREA
            )
            grid_blocked = np.zeros((rows, cols), dtype=bool)
            grid_blocked[small_inflated > 15] = True

            start_grid = (sx // self.grid_size, sy // self.grid_size)
            goal_grid = (gx // self.grid_size, gy // self.grid_size)

            start_grid = self._snap_to_free_cell(start_grid, grid_blocked, cols, rows, search_radius=30)
            goal_grid = self._snap_to_free_cell(goal_grid, grid_blocked, cols, rows, search_radius=30)

            if start_grid is None or goal_grid is None:
                continue

            grid_path = self._astar_search(start_grid, goal_grid, grid_blocked, cols, rows)
            if not grid_path:
                continue

            # 그리드 좌표를 픽셀 좌표로 변환
            pixel_path = []
            for (gc, gr) in grid_path:
                px = int(gc * self.grid_size + self.grid_size / 2)
                py = int(gr * self.grid_size + self.grid_size / 2)
                pixel_path.append((px, py))

            pixel_path[0] = (sx, sy)
            pixel_path[-1] = (gx, gy)

            # 안전 맵 기준으로 스무딩 (화면상 파란 뻥튀기 경계면을 매끄럽게 통과)
            if self.smooth_path and len(pixel_path) > 2:
                pixel_path = self._smooth_waypoints(pixel_path, safety_check_map)

            return pixel_path

        # A* 격자 경로 실패 시: 직선 경로 안전성 점검
        if self._is_line_clear((sx, sy), (gx, gy), safety_check_map, step_px=5):
            return [(sx, sy), (gx, gy)]

        # 직선 경로상에 장애물이 가로막고 있는 경우: 기하학적 우회 경로(Detour) 생성
        detour_path = self._generate_detour_path((sx, sy), (gx, gy), obstacle_mask, self.robot_radius_px)
        if detour_path:
            return detour_path

        return [(sx, sy), (gx, gy)]

    def _generate_detour_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        obstacle_mask: np.ndarray,
        car_radius: int
    ) -> Optional[List[Tuple[int, int]]]:
        """직선상 장애물을 만났을 때 장애물 윤곽선을 기반으로 안전 우회점을 계산"""
        sx, sy = start
        gx, gy = goal
        h, w = obstacle_mask.shape[:2]

        # 직선과 장애물이 충돌하는 지점들 탐색
        dist = math.hypot(gx - sx, gy - sy)
        if dist < 1.0:
            return None

        ux = (gx - sx) / dist
        uy = (gy - sy) / dist
        # 법선 벡터 (좌/우 수직 방향)
        nx = -uy
        ny = ux

        steps = max(2, int(dist / 5))
        hit_points = []
        for s in range(1, steps):
            cx = int(round(sx + (s / steps) * (gx - sx)))
            cy = int(round(sy + (s / steps) * (gy - sy)))
            if 0 <= cx < w and 0 <= cy < h:
                if obstacle_mask[cy, cx] > 0:
                    hit_points.append((cx, cy))

        if not hit_points:
            return None

        # 장애물 중심점
        hit_arr = np.array(hit_points, dtype=np.float32)
        mid_x = float(np.mean(hit_arr[:, 0]))
        mid_y = float(np.mean(hit_arr[:, 1]))

        # 좌/우 우회 후보점 생성 (지나치게 크게 돌지 않고 화면상 장애물 테두리 바로 바깥으로 우회)
        detour_margin = max(24.0, min(40.0, float(car_radius * 0.45)))

        cand1 = (int(np.clip(mid_x + nx * detour_margin, 20, w - 21)),
                 int(np.clip(mid_y + ny * detour_margin, 20, h - 21)))
        cand2 = (int(np.clip(mid_x - nx * detour_margin, 20, w - 21)),
                 int(np.clip(mid_y - ny * detour_margin, 20, h - 21)))

        # 두 후보점 중 장애물과 덜 겹치고 시작/목표점과 이동 거리가 짧은 후보 선택
        safety_map = self.create_inflated_obstacle_map(obstacle_mask, int(car_radius * 0.5))
        for cand in [cand1, cand2]:
            if self._is_line_clear(start, cand, safety_map) and self._is_line_clear(cand, goal, safety_map):
                return [start, cand, goal]

        for cand in [cand1, cand2]:
            if 0 <= cand[0] < w and 0 <= cand[1] < h and obstacle_mask[cand[1], cand[0]] == 0:
                return [start, cand, goal]

        return None

    def _snap_to_free_cell(
        self,
        grid_pt: Tuple[int, int],
        grid_blocked: np.ndarray,
        cols: int,
        rows: int,
        search_radius: int = 15
    ) -> Optional[Tuple[int, int]]:
        gc, gr = grid_pt
        if 0 <= gc < cols and 0 <= gr < rows and not grid_blocked[gr, gc]:
            return grid_pt

        best_pt = None
        min_dist = float('inf')

        for dr in range(-search_radius, search_radius + 1):
            for dc in range(-search_radius, search_radius + 1):
                nc, nr = gc + dc, gr + dr
                if 0 <= nc < cols and 0 <= nr < rows:
                    if not grid_blocked[nr, nc]:
                        dist = dc * dc + dr * dr
                        if dist < min_dist:
                            min_dist = dist
                            best_pt = (nc, nr)

        return best_pt

    def _astar_search(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        grid_blocked: np.ndarray,
        cols: int,
        rows: int
    ) -> List[Tuple[int, int]]:
        # 8방향 이동
        moves = [
            (0, -1, 1.0), (0, 1, 1.0), (-1, 0, 1.0), (1, 0, 1.0),
            (-1, -1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (1, 1, 1.414)
        ]

        def heuristic(a, b):
            dx = abs(a[0] - b[0])
            dy = abs(a[1] - b[1])
            return (dx + dy) + (1.414 - 2.0) * min(dx, dy)

        open_set = []
        heapq.heappush(open_set, (0.0, 0, start))

        came_from = {}
        g_score = {start: 0.0}
        tie_breaker = 0

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            cur_c, cur_r = current
            cur_g = g_score[current]

            for dc, dr, move_cost in moves:
                nc, nr = cur_c + dc, cur_r + dr

                if not (0 <= nc < cols and 0 <= nr < rows):
                    continue

                if grid_blocked[nr, nc]:
                    continue

                # 대각선 코너 끼임 방지
                if dc != 0 and dr != 0:
                    if grid_blocked[cur_r, nc] or grid_blocked[nr, cur_c]:
                        continue

                tentative_g = cur_g + move_cost
                neighbor = (nc, nr)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    came_from[neighbor] = current
                    tie_breaker += 1
                    heapq.heappush(open_set, (f_score, tie_breaker, neighbor))

        return []

    def _smooth_waypoints(
        self,
        pixel_path: List[Tuple[int, int]],
        inflated_map: np.ndarray
    ) -> List[Tuple[int, int]]:
        if len(pixel_path) <= 2:
            return pixel_path

        smoothed = [pixel_path[0]]
        current_idx = 0

        while current_idx < len(pixel_path) - 1:
            farthest_idx = current_idx + 1
            for next_idx in range(len(pixel_path) - 1, current_idx, -1):
                if self._is_line_clear(smoothed[-1], pixel_path[next_idx], inflated_map):
                    farthest_idx = next_idx
                    break
            
            smoothed.append(pixel_path[farthest_idx])
            current_idx = farthest_idx

        return smoothed

    def _is_line_clear(
        self,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        inflated_map: np.ndarray,
        step_px: int = 5
    ) -> bool:
        # 직선 구간 장애물 확인
        x1, y1 = p1
        x2, y2 = p2
        dist = math.hypot(x2 - x1, y2 - y1)
        if dist == 0:
            return True

        steps = max(1, int(dist / step_px))
        h, w = inflated_map.shape[:2]

        for s in range(1, steps):
            t = s / steps
            cx = int(round(x1 + t * (x2 - x1)))
            cy = int(round(y1 + t * (y2 - y1)))

            if 0 <= cx < w and 0 <= cy < h:
                if inflated_map[cy, cx] > 0:  # 장애물 충돌
                    return False
            else:
                return False

        return True
