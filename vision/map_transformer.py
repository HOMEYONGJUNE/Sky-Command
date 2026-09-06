from typing import List, Optional, Tuple
import cv2
import numpy as np


class MapTransformer:
    def __init__(self, map_width: int = 1000, map_height: int = 1000):
        self.map_width = map_width
        self.map_height = map_height
        self.matrix: Optional[np.ndarray] = None
        self.inv_matrix: Optional[np.ndarray] = None
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def set_perspective_from_rect(self, x1: int, y1: int, x2: int, y2: int, target_w: int = 1280, target_h: int = 720):
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)
        
        w = max_x - min_x
        h = max_y - min_y
        if w <= 10 or h <= 10:
            return False

        self.map_width = target_w
        self.map_height = target_h

        src_pts = np.float32([
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y]
        ])
        dst_pts = np.float32([
            [0, 0],
            [self.map_width, 0],
            [self.map_width, self.map_height],
            [0, self.map_height]
        ])

        self.matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.inv_matrix = cv2.getPerspectiveTransform(dst_pts, src_pts)
        return True

    def warp(self, frame: np.ndarray) -> np.ndarray:
        if self.matrix is None:
            return frame
        return cv2.warpPerspective(frame, self.matrix, (self.map_width, self.map_height))

    def extract_blue_obstacle_mask(
        self,
        hsv_map: np.ndarray,
        h_min: int = 90,
        h_max: int = 135,
        s_min: int = 70,
        v_min: int = 40,
        dilate_px: Optional[int] = None,
        dilate_x: Optional[int] = None,
        dilate_y: Optional[int] = None
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        # 파란색 범위 마스킹 및 노이즈 제거
        h_min = min(h_min, h_max)
        lower_blue = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper_blue = np.array([h_max, 255, 255], dtype=np.uint8)

        # 좌우(dx) 및 상하(dy) 팽창 반경 결정
        if dilate_x is not None and dilate_y is not None:
            dx = max(0, int(dilate_x))
            dy = max(0, int(dilate_y))
        else:
            base_r = dilate_px if dilate_px is not None else 30
            dx = max(4, int(round(base_r * 0.35)))
            dy = max(10, int(round(base_r * 1.3)))

        raw_mask = cv2.inRange(hsv_map, lower_blue, upper_blue)
        raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, self.morph_kernel, iterations=1)
        raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, self.morph_kernel, iterations=2)

        obstacle_mask = np.zeros_like(raw_mask)
        cone_base_points: List[Tuple[int, int]] = []
        contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 10:  # 미세 노이즈 무시
                continue

            pts = cnt.reshape(-1, 2)
            x, y, w, h = cv2.boundingRect(cnt)
            peri = cv2.arcLength(cnt, True)

            # 최소 회전 사각형 계산 (길이, 두께, 종횡비)
            rect = cv2.minAreaRect(cnt)
            dim1, dim2 = rect[1]
            length = max(dim1, dim2)
            thickness = max(1.0, min(dim1, dim2))
            aspect_ratio = length / thickness

            # 선(Line/Wall) vs 꼬깔/점(Cone/Dot) 분리 판별
            # 선 조건: 길이가 길거나 얇고 긴 형태
            is_line = (
                (length >= 75 and aspect_ratio >= 2.3) or
                (aspect_ratio >= 3.5 and length >= 35) or
                (length >= 100) or
                (peri * peri / max(1.0, area) >= 32.0 and length >= 45)
            )

            if is_line:
                # [선 (Line) 처리] - 선의 궤적을 유지하며 좌우/상하 비등방 안전 팽창
                single_line_mask = np.zeros_like(raw_mask)
                cv2.drawContours(single_line_mask, [cnt], -1, 255, thickness=-1)

                if dx > 0 or dy > 0:
                    kw = max(1, int(dx * 2) + 1)
                    kh = max(1, int(dy * 2) + 1)
                    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw, kh))
                    dilated_line = cv2.dilate(single_line_mask, dilate_kernel)
                    obstacle_mask = cv2.bitwise_or(obstacle_mask, dilated_line)
                else:
                    obstacle_mask = cv2.bitwise_or(obstacle_mask, single_line_mask)

            else:
                # [꼬깔/점 (Cone/Point) 처리]
                # 1. 꼬깔의 맨 아래쪽 바닥 픽셀 (Bottom-most floor pixel)
                base_y = int(np.max(pts[:, 1]))
                bottom_pts_x = pts[pts[:, 1] >= base_y - 4, 0]
                if len(bottom_pts_x) > 0:
                    base_x = int(round(float(np.mean(bottom_pts_x))))
                else:
                    base_x = int(x + w / 2)

                cone_base_points.append((base_x, base_y))

                # 2. 꼬깔의 가장 윗 픽셀 (Top-most pixel)
                top_y = int(np.min(pts[:, 1]))
                top_pts_x = pts[pts[:, 1] <= top_y + 4, 0]
                if len(top_pts_x) > 0:
                    top_x = int(round(float(np.mean(top_pts_x))))
                else:
                    top_x = base_x

                # 3. 윗 픽셀부터 아랫 픽셀까지를 포함하는 타원(Ellipse)
                # 좌우(dx) / 상하(dy) 독립 팽창 적용
                cone_height = max(1, base_y - top_y)
                center_x = int(round((base_x + top_x) / 2.0))
                center_y = int(round((base_y + top_y) / 2.0))

                radius_x = max(1, int(dx))
                radius_y = max(1, int(round(cone_height / 2.0 + dy)))

                cv2.ellipse(
                    obstacle_mask,
                    (center_x, center_y),
                    (radius_x, radius_y),
                    0,
                    0,
                    360,
                    255,
                    -1
                )

        # 꼬깔 컨투어가 감지되지 않았으나 파란 픽셀이 있는 경우의 폴백 처리
        if np.count_nonzero(obstacle_mask) == 0 and np.count_nonzero(raw_mask) > 0:
            ys, xs = np.where(raw_mask > 0)
            if len(ys) > 0:
                min_y, max_y = int(np.min(ys)), int(np.max(ys))
                bottom_xs = xs[ys >= max_y - 4]
                top_xs = xs[ys <= min_y + 4]
                b_x = int(np.mean(bottom_xs)) if len(bottom_xs) > 0 else int(np.mean(xs))
                t_x = int(np.mean(top_xs)) if len(top_xs) > 0 else b_x
                c_x = int((b_x + t_x) / 2)
                c_y = int((min_y + max_y) / 2)
                r_x = max(1, int(dx))
                r_y = max(1, int((max_y - min_y) / 2.0 + dy))
                cv2.ellipse(obstacle_mask, (c_x, c_y), (r_x, r_y), 0, 0, 360, 255, -1)
                cone_base_points.append((b_x, max_y))

        return obstacle_mask, cone_base_points

    def extract_green_terrain_mask(
        self,
        hsv_map: np.ndarray,
        h_min: int = 35,
        h_max: int = 85,
        s_min: int = 40,
        v_min: int = 40
    ) -> np.ndarray:
        # 초록색 바닥 영역 마스킹 (주행 가능 지형)
        lower_green = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper_green = np.array([h_max, 255, 255], dtype=np.uint8)

        raw_mask = cv2.inRange(hsv_map, lower_green, upper_green)
        
        # 큰 커널로 닫기 연산을 수행하여 지형 틈새 연결
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closed_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
        closed_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, self.morph_kernel, iterations=1)

        # 주요 컨투어 내부 구멍 채우기
        contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = np.zeros_like(closed_mask)
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                cv2.drawContours(filled_mask, [cnt], -1, 255, -1)

        return filled_mask if np.count_nonzero(filled_mask) > 0 else closed_mask

    def reset(self):
        self.matrix = None
        self.inv_matrix = None
