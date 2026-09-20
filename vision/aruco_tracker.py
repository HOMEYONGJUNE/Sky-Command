import math
from typing import Optional, Tuple, Generator
import cv2
import numpy as np


class AdvancedArucoTracker:
    def __init__(
        self,
        dictionary_id=cv2.aruco.DICT_4X4_50,
        target_id: Optional[int] = None,
        smooth_window: int = 3,
        smooth_alpha: float = 0.90,
        max_coasting_frames: int = 3,
        fast_miss_after: int = 5,
        enhanced_scan_interval: int = 3,
    ):
        self.target_id = target_id

        self.last_valid_pos: Optional[Tuple[int, int]] = None
        self.last_valid_angle: float = 0.0
        self.last_corners: Optional[np.ndarray] = None
        self.miss_count: int = 999
        self.max_coasting_frames = max(0, int(max_coasting_frames))
        self.fast_miss_after = max(1, int(fast_miss_after))
        self.enhanced_scan_interval = max(1, int(enhanced_scan_interval))
        self.detect_count = 0

        # ArUco 사전 및 파라미터 로드
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)

        try:
            self.aruco_params = cv2.aruco.DetectorParameters()
            self._apply_parameters(self.aruco_params)
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            self.use_new_api = True
        except AttributeError:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self._apply_parameters(self.aruco_params)
            self.detector = None
            self.use_new_api = False

        # 전처리 필터
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self.sharpen_kernel = np.array([
            [0, -1,  0],
            [-1,  5, -1],
            [0, -1,  0]
        ], dtype=np.float32)

    def _apply_parameters(self, params):
        # 코너 정제 및 임계값 설정
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 30
        params.cornerRefinementMinAccuracy = 0.05

        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 25
        params.adaptiveThreshWinSizeStep = 4
        params.adaptiveThreshConstant = 7

        params.minMarkerPerimeterRate = 0.02
        params.maxMarkerPerimeterRate = 4.0
        params.polygonalApproxAccuracyRate = 0.03
        params.perspectiveRemoveIgnoredMarginPerCell = 0.13

    def _get_candidates(self, bgr_frame: np.ndarray) -> Generator:
        # 1: 그레이, 2: CLAHE, 3: 샤프닝, 4: 감마보정
        if len(bgr_frame.shape) == 3:
            gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = bgr_frame

        yield gray

        clahe_gray = self.clahe.apply(gray)
        yield clahe_gray

        yield cv2.filter2D(clahe_gray, -1, self.sharpen_kernel)

        inv_gamma = 1.0 / 1.4
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
        yield cv2.filter2D(cv2.LUT(gray, table), -1, self.sharpen_kernel)

    def detect(self, frame: np.ndarray) -> Tuple[Optional[Tuple[int, int]], float, Optional[np.ndarray], bool, Optional[np.ndarray]]:
        self.detect_count += 1
        detected_corners = None
        best_candidate = None

        # 장시간 미검출 시 고비용 보정 후보를 생략해 UI/제어 루프를 우선한다.
        candidates = self._get_candidates(frame)
        gray = next(candidates)
        detected_corners, _ = self._try_detect(gray)
        best_candidate = gray

        # 기본 grayscale에서 놓친 프레임에만 보정 영상을 주기적으로 적용한다.
        should_scan_enhanced = (
            detected_corners is None
            and self.miss_count < self.fast_miss_after
            and self.detect_count % self.enhanced_scan_interval == 0
        )
        if should_scan_enhanced:
            for candidate in candidates:
                detected_corners, _ = self._try_detect(candidate)
                best_candidate = candidate
                if detected_corners is not None:
                    break

        if detected_corners is not None:
            self.miss_count = 0

            # 중심점 좌표
            cx = int(round(float(np.mean(detected_corners[:, 0]))))
            cy = int(round(float(np.mean(detected_corners[:, 1]))))
            final_pos = (cx, cy)

            # 헤딩 각도 (0번->1번 코너 방향)
            p0, p1 = detected_corners[0], detected_corners[1]
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            angle_deg = (math.degrees(math.atan2(-dy, dx))) % 360.0

            self.last_valid_pos = final_pos
            self.last_valid_angle = angle_deg
            self.last_corners = detected_corners

            return final_pos, angle_deg, detected_corners, True, best_candidate

        # 미감지 시 설정된 프레임 수만큼 이전 위치를 유지
        self.miss_count += 1
        if self.miss_count <= self.max_coasting_frames and self.last_valid_pos is not None:
            return self.last_valid_pos, self.last_valid_angle, self.last_corners, True, best_candidate

        return None, 0.0, None, False, best_candidate

    def _try_detect(self, gray_img: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[int]]:
        if self.use_new_api:
            corners, ids, _ = self.detector.detectMarkers(gray_img)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray_img, self.aruco_dict, parameters=self.aruco_params
            )

        if ids is not None and len(corners) > 0:
            flat_ids = np.array(ids).flatten()
            if len(flat_ids) == 0:
                return None, None

            target_idx = 0
            if self.target_id is not None:
                matches = np.where(flat_ids == self.target_id)[0]
                if len(matches) == 0:
                    return None, None
                target_idx = int(matches[0])

            c = np.array(corners[target_idx])
            if c.ndim == 3 and c.shape[0] == 1:
                return c[0], int(flat_ids[target_idx])
            elif c.ndim == 2:
                return c, int(flat_ids[target_idx])

        return None, None

    def reset(self):
        # 상태 초기화
        self.last_valid_pos = None
        self.last_valid_angle = 0.0
        self.last_corners = None
        self.detect_count = 0
        self.miss_count = 999
