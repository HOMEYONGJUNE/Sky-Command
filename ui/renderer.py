import math
import os
import time
from typing import List, Optional, Tuple
import cv2
import numpy as np
import config
try:
    from PIL import Image, ImageDraw, ImageFont, ImageSequence
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

from .buttons import UIButtonManager

_UI_DIR = os.path.dirname(os.path.abspath(__file__))
_FONT_PATH = os.path.join(_UI_DIR, "DungGeunMo.ttf")
_FACE_GIF_PATH = os.path.join(_UI_DIR, "face.gif")
_CURSOR_PATH = os.path.join(_UI_DIR, "cursor.png")
_DRIVING_SWITCH_PATH = os.path.join(_UI_DIR, "driving_mode_switch_button.png")
_SUDONG_WINDOW_PATH = os.path.join(_UI_DIR, "Sudong_window.png")
_SUDONG_BUTTON_PATH = os.path.join(_UI_DIR, "Sudong_button.png")


class UIRenderer:
    def __init__(
        self,
        ui_image_path: str,
        window_width: int = 1280,
        window_height: int = 720,
        text_box_rect: Tuple[int, int, int, int] = (1090, 20, 1280, 520),
        button_manager: Optional[UIButtonManager] = None
    ):
        self.window_w = window_width
        self.window_h = window_height
        self.tb_x1, self.tb_y1, self.tb_x2, self.tb_y2 = text_box_rect
        self.tb_w = self.tb_x2 - self.tb_x1
        self.tb_h = self.tb_y2 - self.tb_y1

        self.button_manager = button_manager or UIButtonManager(
            ui_dir=_UI_DIR,
            window_w=self.window_w,
            window_h=self.window_h
        )

        self.ui_bgr, self.ui_alpha = self._load_ui_image(ui_image_path)

        # 주행 모드 전환 버튼 및 수동 안내창 이미지 로드 (1280x720 RGBA)
        self.driving_switch_bgr, self.driving_switch_alpha, self.driving_switch_bbox = self._load_overlay_layer(_DRIVING_SWITCH_PATH)
        self.sudong_window_bgr, self.sudong_window_alpha, self.sudong_window_bbox = self._load_overlay_layer(_SUDONG_WINDOW_PATH)
        self.sudong_button_bgr, self.sudong_button_alpha, self.sudong_button_bbox = self._load_overlay_layer(_SUDONG_BUTTON_PATH)

        # 마우스 커서 스케일 추가 50% 축소 (35% 스케일, 약 22x22px 실물 커서 크기)
        self.cursor_bgr, self.cursor_alpha, self.cursor_hotspot = self._load_cursor_image(_CURSOR_PATH, scale=0.35)

        # face.gif 로드 (중앙 x:1024, y:672)
        self.face_frames, self.face_durations, self.face_total_duration, self.face_bbox = self._load_face_gif(
            _FACE_GIF_PATH, center_x=1024, center_y=672
        )

        # HUD 텍스트 박스
        self.hud_x1 = int(1158 * self.window_w / 1920)
        self.hud_y1 = int(948 * self.window_h / 1080)
        self.hud_x2 = int(1412 * self.window_w / 1920)
        self.hud_y2 = int(1015 * self.window_h / 1080)
        self.hud_w = self.hud_x2 - self.hud_x1
        self.hud_h = self.hud_y2 - self.hud_y1
        self.hud_color_rgb = (0, 230, 25)
        self.hud_color_bgr = (25, 230, 0)

        # IP 텍스트 박스
        self.ip_x1 = int(884 * self.window_w / 1920)
        self.ip_y1 = int(1025 * self.window_h / 1080)
        self.ip_x2 = int(1075 * self.window_w / 1920)
        self.ip_y2 = int(1051 * self.window_h / 1080)
        self.ip_color_rgb = (255, 255, 255)
        self.ip_color_bgr = (255, 255, 255)

        self._pil_fonts = {}
        self._pil_available = _PIL_AVAILABLE and os.path.exists(_FONT_PATH)

    def _load_face_gif(self, path: str, center_x: int = 1024, center_y: int = 672):
        if not os.path.exists(path) or not _PIL_AVAILABLE:
            return [], [], 0.0, (0, 0, 0, 0)
        try:
            frames = []
            durations = []
            with Image.open(path) as im:
                w, h = im.size
                for frame in ImageSequence.Iterator(im):
                    rgba = frame.convert("RGBA")
                    arr = np.array(rgba)
                    bgr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
                    alpha = (arr[:, :, 3].astype(np.float32) / 255.0)[:, :, np.newaxis]
                    dur = frame.info.get("duration", 80) / 1000.0
                    if dur <= 0:
                        dur = 0.08
                    frames.append((bgr, alpha))
                    durations.append(dur)

            total_dur = sum(durations)
            x1 = center_x - w // 2
            y1 = center_y - h // 2
            x2 = x1 + w
            y2 = y1 + h
            return frames, durations, total_dur, (x1, y1, x2, y2)
        except Exception:
            return [], [], 0.0, (0, 0, 0, 0)

    def _get_pil_font(self, size: int):
        if size not in self._pil_fonts:
            self._pil_fonts[size] = ImageFont.truetype(_FONT_PATH, size)
        return self._pil_fonts[size]

    def _draw_textbox_fast(
        self,
        canvas: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        lines: List[str],
        font_size: int = 11,
        color_rgb: Tuple[int, int, int] = (255, 255, 255),
        line_spacing: int = 2
    ):
        sub_w = x2 - x1
        sub_h = y2 - y1
        if sub_w <= 0 or sub_h <= 0:
            return

        if not self._pil_available:
            draw_y = y1 + font_size + 2
            bgr_col = (color_rgb[2], color_rgb[1], color_rgb[0])
            for line in lines:
                cv2.putText(canvas, line, (x1 + 3, draw_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, bgr_col, 1, cv2.LINE_AA)
                draw_y += font_size + line_spacing
            return

        sub_canvas = canvas[y1:y2, x1:x2]
        pil_sub = Image.fromarray(cv2.cvtColor(sub_canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_sub)
        font = self._get_pil_font(font_size)

        cur_y = 2
        for line in lines:
            draw.text((3, cur_y), line, font=font, fill=color_rgb)
            bbox = font.getbbox(line)
            cur_y += (bbox[3] - bbox[1]) + line_spacing

        canvas[y1:y2, x1:x2] = cv2.cvtColor(np.array(pil_sub), cv2.COLOR_RGB2BGR)

    _HUD_FONT_SIZE = 11

    def _load_ui_image(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        if os.path.exists(path):
            try:
                img_array = np.fromfile(path, np.uint8)
                ui_img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
                if ui_img is not None:
                    ui_resized = cv2.resize(ui_img, (self.window_w, self.window_h))
                    if ui_resized.shape[2] == 4:
                        bgr = ui_resized[:, :, :3]
                        alpha = ui_resized[:, :, 3].astype(np.float32) / 255.0
                    else:
                        bgr = ui_resized
                        alpha = np.ones((self.window_h, self.window_w), dtype=np.float32)
                    return bgr, alpha
            except Exception:
                pass

        bgr = np.zeros((self.window_h, self.window_w, 3), dtype=np.uint8)
        alpha = np.zeros((self.window_h, self.window_w), dtype=np.float32)
        alpha[self.tb_y1:self.tb_y2, self.tb_x1:self.tb_x2] = 0.85
        bgr[self.tb_y1:self.tb_y2, self.tb_x1:self.tb_x2] = (25, 25, 30)
        alpha[620:720, 0:self.window_w] = 0.8
        bgr[620:720, 0:self.window_w] = (20, 20, 25)

        return bgr, alpha

    def _load_cursor_image(self, path: str, scale: float = 0.35) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Tuple[int, int]]:
        """마우스 커서 이미지(cursor.png) 35% 스케일(약 22x22px) 사전 로드 및 알파/핫스팟 계산"""
        if not os.path.exists(path):
            return None, None, (0, 0)
        try:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                return None, None, (0, 0)

            orig_h, orig_w = img.shape[:2]
            new_w = max(1, int(round(orig_w * scale)))
            new_h = max(1, int(round(orig_h * scale)))
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            if resized.shape[2] == 4:
                bgr = resized[:, :, :3]
                alpha = (resized[:, :, 3].astype(np.float32) / 255.0)[:, :, np.newaxis]
            else:
                bgr = resized
                alpha = np.ones((new_h, new_w, 1), dtype=np.float32)

            hotspot_x = int(round(7 * scale))
            hotspot_y = int(round(1 * scale))
            return bgr, alpha, (hotspot_x, hotspot_y)
        except Exception:
            return None, None, (0, 0)

    def _load_overlay_layer(self, path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """1280x720 오버레이 PNG(RGBA)를 로드하고 BGR, 알파(0~1), 유효 바운딩 박스를 반환합니다."""
        if not os.path.exists(path):
            return None, None, None
        try:
            img_array = np.fromfile(path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
            if img is None:
                return None, None, None
            if img.shape[0] != self.window_h or img.shape[1] != self.window_w:
                img = cv2.resize(img, (self.window_w, self.window_h))
            if img.shape[2] == 4:
                bgr = img[:, :, :3]
                alpha = img[:, :, 3].astype(np.float32) / 255.0
                alpha_raw = img[:, :, 3]
            else:
                bgr = img
                alpha = np.ones((self.window_h, self.window_w), dtype=np.float32)
                alpha_raw = np.full((self.window_h, self.window_w), 255, dtype=np.uint8)

            ys, xs = np.where(alpha_raw > 10)
            if len(xs) > 0:
                bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            else:
                bbox = None
            return bgr, alpha, bbox
        except Exception as e:
            print(f"[UI LOAD ERROR] {path}: {e}")
            return None, None, None

    def _blend_layer(
        self,
        canvas: np.ndarray,
        bgr: Optional[np.ndarray],
        alpha: Optional[np.ndarray],
        bbox: Optional[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        """바운딩 박스 영역만 고속 알파 블렌딩하여 캔버스에 합성합니다."""
        if bgr is None or alpha is None or bbox is None:
            return canvas
        x1, y1, x2, y2 = bbox
        sub_alpha = alpha[y1:y2, x1:x2, np.newaxis]
        sub_bgr = bgr[y1:y2, x1:x2]
        canvas[y1:y2, x1:x2] = (sub_bgr * sub_alpha + canvas[y1:y2, x1:x2] * (1.0 - sub_alpha)).astype(np.uint8)
        return canvas

    def is_driving_mode_switch_clicked(self, x: int, y: int) -> bool:
        """좌측 상단 driving_mode_switch_button.png의 투명하지 않은 영역 클릭 여부 판정"""
        if self.driving_switch_alpha is None:
            return False
        if 0 <= x < self.window_w and 0 <= y < self.window_h:
            return bool(self.driving_switch_alpha[y, x] > 0.05)
        return False

    def is_sudong_button_clicked(self, x: int, y: int) -> bool:
        """수동 전환 창(Sudong_button.png)의 투명하지 않은 영역 클릭 여부 판정"""
        if self.sudong_button_alpha is None:
            return False
        if 0 <= x < self.window_w and 0 <= y < self.window_h:
            return bool(self.sudong_button_alpha[y, x] > 0.05)
        return False

    def render_frame(
        self,
        main_view: np.ndarray,
        log_lines: List[str],
        robot_pos: Optional[Tuple[int, int]] = None,
        robot_angle_deg: float = 0.0,
        waypoints: Optional[List[Tuple[int, int]]] = None,
        current_wp_idx: int = 0,
        goal_pos: Optional[Tuple[int, int]] = None,
        obstacle_mask: Optional[np.ndarray] = None,
        detections: Optional[list] = None,
        ping_pos: Optional[Tuple[int, int]] = None,
        ping_start_time: float = 0.0,
        onboard_alerts: Optional[List[str]] = None,
        onboard_alert_start_time: float = 0.0,
        mouse_pos: Tuple[int, int] = (0, 0),
        status_text: str = "IDLE",
        speed_info: Tuple[int, int] = (0, 0),
        marker_corners: Optional[np.ndarray] = None,
        home_pos: Optional[Tuple[int, int]] = None,
        is_blind: bool = False,
        ip_address: str = "192.168.0.2",
        pi_cam_frame: Optional[np.ndarray] = None,
        show_lost_warning: bool = False,
        fps: float = 0.0,
    ) -> np.ndarray:
        CAM_W, CAM_H = 1090, 614

        # 1. 캔버스 초기화 및 카메라 프레임 배치
        canvas = np.zeros((self.window_h, self.window_w, 3), dtype=np.uint8)
        canvas[:] = (18, 18, 22)

        cam_frame = cv2.resize(main_view, (CAM_W, CAM_H))
        canvas[0:CAM_H, 0:CAM_W] = cam_frame
        cv2.putText(
            canvas,
            f"FPS: {fps:4.1f}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # 최신 장애물 검출 위치를 메인 카메라 화면에 표시한다.
        if detections:
            scale_x = CAM_W / self.window_w
            scale_y = CAM_H / self.window_h
            for detection in detections:
                x, y, width, height = detection.box
                p1 = (int(x * scale_x), int(y * scale_y))
                p2 = (int((x + width) * scale_x), int((y + height) * scale_y))
                cv2.rectangle(canvas[0:CAM_H, 0:CAM_W], p1, p2, (0, 165, 255), 2)
                cv2.putText(
                    canvas[0:CAM_H, 0:CAM_W],
                    f"{detection.label} {detection.confidence:.2f}",
                    (p1[0], max(16, p1[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 165, 255),
                    1,
                    cv2.LINE_AA,
                )

        def sc(pt):
            return (int(pt[0] * CAM_W / self.window_w), int(pt[1] * CAM_H / self.window_h))

        # 3. ArUco 위치/방향을 기준으로 직사각형 차체 표시
        car_half_length = 0
        car_half_width = 0
        center_screen = None

        if marker_corners is not None and len(marker_corners) >= 4:
            mc_pts = np.array(marker_corners, dtype=np.float32)
            c_center = np.mean(mc_pts, axis=0)
            center_screen = sc(c_center)

            side0 = float(np.linalg.norm(mc_pts[1] - mc_pts[0]))
            side1 = float(np.linalg.norm(mc_pts[2] - mc_pts[1]))
            side2 = float(np.linalg.norm(mc_pts[3] - mc_pts[2]))
            side3 = float(np.linalg.norm(mc_pts[0] - mc_pts[3]))
            avg_side = (side0 + side1 + side2 + side3) / 4.0

            car_half_width = max(
                10,
                int(avg_side * config.ROBOT_BODY_WIDTH_SCALE * CAM_W / self.window_w / 2.0),
            )
            car_half_length = max(
                car_half_width,
                int(avg_side * config.ROBOT_BODY_LENGTH_SCALE * CAM_H / self.window_h / 2.0),
            )

            # ArUco 마커 외곽선 (초록색 1px)
            scaled_marker = np.array([[sc(p)] for p in mc_pts], dtype=np.int32)
            cv2.polylines(canvas, [scaled_marker], True, (0, 255, 0), 1, cv2.LINE_AA)

        elif robot_pos is not None:
            center_screen = sc(robot_pos)

        # 헤딩 방향을 따라 회전한 직사각형 차체 렌더링
        if center_screen is not None and car_half_length > 0 and car_half_width > 0:
            rad = math.radians(robot_angle_deg)
            heading = np.array((math.cos(rad), -math.sin(rad)), dtype=np.float32)
            lateral = np.array((math.sin(rad), math.cos(rad)), dtype=np.float32)
            center = np.array(center_screen, dtype=np.float32)
            rectangle = np.array(
                [
                    center + heading * car_half_length + lateral * car_half_width,
                    center + heading * car_half_length - lateral * car_half_width,
                    center - heading * car_half_length - lateral * car_half_width,
                    center - heading * car_half_length + lateral * car_half_width,
                ],
                dtype=np.int32,
            )

            car_overlay = canvas.copy()
            cv2.fillConvexPoly(car_overlay, rectangle, (255, 200, 0), cv2.LINE_AA)
            canvas = cv2.addWeighted(canvas, 0.75, car_overlay, 0.25, 0)
            cv2.polylines(canvas, [rectangle], True, (255, 235, 50), 2, cv2.LINE_AA)

        # 로봇 중심점 및 헤딩 방향선 (슬림 1px)
        if robot_pos is not None:
            srp = sc(robot_pos)
            cv2.circle(canvas, srp, 3, (255, 255, 255), -1, cv2.LINE_AA)

            rad = math.radians(robot_angle_deg)
            arrow_len = max(
                24,
                car_half_length + 8 if center_screen is not None else 24,
            )
            ax = int(srp[0] + arrow_len * math.cos(rad))
            ay = int(srp[1] - arrow_len * math.sin(rad))
            cv2.line(canvas, srp, (ax, ay), (0, 255, 255), 2, cv2.LINE_AA)

        # 4. A* 경로 선 ((0,163,22), 남은 경로만)
        if waypoints and len(waypoints) > 0 and robot_pos is not None:
            remaining_wps = waypoints[current_wp_idx:] if current_wp_idx < len(waypoints) else []
            pts_to_draw = [sc(robot_pos)] + [sc(wp) for wp in remaining_wps]

            if len(pts_to_draw) >= 2:
                pts_arr = np.array(pts_to_draw, dtype=np.int32)
                cv2.polylines(canvas, [pts_arr], False, (22, 163, 0), 2, cv2.LINE_AA)

        # 5. 목표점 (심플한 점)
        if goal_pos is not None:
            sgp = sc(goal_pos)
            cv2.circle(canvas, sgp, 4, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, sgp, 4, (255, 255, 255), 1, cv2.LINE_AA)

        # 5-1. 홈 위치
        if home_pos is not None:
            shp = sc(home_pos)
            cv2.drawMarker(canvas, shp, (0, 255, 100), cv2.MARKER_DIAMOND, 10, 1, cv2.LINE_AA)

        # 6. 우클릭 핑 애니메이션
        if ping_pos is not None and ping_start_time > 0:
            elapsed = time.time() - ping_start_time
            if elapsed < 0.8:
                spp = sc(ping_pos)
                progress = elapsed / 0.8
                ring_radius = int(5 + progress * 24)
                ring_alpha = max(0.0, 1.0 - progress)
                overlay_ping = canvas.copy()
                cv2.circle(overlay_ping, spp, ring_radius, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.drawMarker(overlay_ping, spp, (0, 255, 0), cv2.MARKER_TILTED_CROSS, 10, 1)
                canvas = cv2.addWeighted(canvas, 1.0 - ring_alpha * 0.7, overlay_ping, ring_alpha * 0.7, 0)

        # 6-1. 미니맵 / 온보드 캠 (11, 583 ~ 245, 719)
        MM_X1, MM_Y1 = 11, 583
        MM_X2, MM_Y2 = 245, 719
        MM_W = MM_X2 - MM_X1
        MM_H = MM_Y2 - MM_Y1

        mode = self.button_manager.map_cam_mode if self.button_manager else "MAP"

        if mode == "CAM":
            # 라즈베리 파이 온보드 카메라 화면 렌더링
            if pi_cam_frame is not None and pi_cam_frame.size > 0:
                cam_view = cv2.resize(pi_cam_frame, (MM_W, MM_H))
                # 상단 헤더 바 및 상태 표시
                overlay_bar = cam_view.copy()
                cv2.rectangle(overlay_bar, (0, 0), (MM_W, 18), (0, 0, 0), -1)
                cam_view = cv2.addWeighted(overlay_bar, 0.65, cam_view, 0.35, 0)
                # LIVE 인디케이터 (초록색 원 + 텍스트)
                cv2.circle(cam_view, (10, 9), 4, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.putText(cam_view, "RC-CAM [CH1] LIVE", (20, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
                canvas[MM_Y1:MM_Y2, MM_X1:MM_X2] = cam_view
            else:
                # 카메라 신호 대기 / 연결 중 안내 화면
                no_signal = np.zeros((MM_H, MM_W, 3), dtype=np.uint8)
                no_signal[:] = (18, 18, 24)
                # 격자 배경
                for gy in range(0, MM_H, 20):
                    cv2.line(no_signal, (0, gy), (MM_W, gy), (30, 30, 38), 1)
                for gx in range(0, MM_W, 20):
                    cv2.line(no_signal, (gx, 0), (gx, MM_H), (30, 30, 38), 1)
                cv2.putText(no_signal, "RC-CAM [PORT 1]", (MM_W // 2 - 58, MM_H // 2 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)
                cv2.putText(no_signal, "NO SIGNAL / CONNECTING", (MM_W // 2 - 76, MM_H // 2 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 140, 255), 1, cv2.LINE_AA)
                canvas[MM_Y1:MM_Y2, MM_X1:MM_X2] = no_signal
        else:
            # 기본 미니맵 렌더링
            minimap = np.zeros((MM_H, MM_W, 3), dtype=np.uint8)

            def mm(pt):
                return (
                    int(pt[0] * MM_W / self.window_w),
                    int(pt[1] * MM_H / self.window_h)
                )

            # TFLite 장애물 영역을 미니맵에 표시
            if obstacle_mask is not None and np.count_nonzero(obstacle_mask) > 0:
                mini_mask = cv2.resize(obstacle_mask, (MM_W, MM_H), interpolation=cv2.INTER_NEAREST)
                minimap[mini_mask > 0] = (0, 0, 200)
                mini_contours, _ = cv2.findContours(mini_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(minimap, mini_contours, -1, (0, 165, 255), 1, cv2.LINE_AA)

            if waypoints and len(waypoints) > 0 and robot_pos is not None:
                remaining_wps = waypoints[current_wp_idx:] if current_wp_idx < len(waypoints) else []
                mm_pts = [mm(robot_pos)] + [mm(wp) for wp in remaining_wps]
                if len(mm_pts) >= 2:
                    mm_pts_arr = np.array(mm_pts, dtype=np.int32)
                    cv2.polylines(minimap, [mm_pts_arr], False, (22, 163, 0), 1, cv2.LINE_AA)

            if robot_pos is not None:
                mrp = mm(robot_pos)
                cv2.circle(minimap, mrp, 4, (0, 230, 25), -1, cv2.LINE_AA)
                cv2.circle(minimap, mrp, 6, (0, 180, 10), 1, cv2.LINE_AA)

                alerts = set(onboard_alerts or [])
                if alerts and onboard_alert_start_time > 0:
                    elapsed = time.time() - onboard_alert_start_time
                    if elapsed >= 0:
                        pulse = elapsed % 1.2
                        progress = pulse / 1.2
                        radius = int(7 + progress * 22)
                        alpha = 1.0 - progress
                        alert_overlay = minimap.copy()
                        if "claymore" in alerts:
                            color = (0, 0, 255)
                            cv2.circle(alert_overlay, mrp, radius, color, 2, cv2.LINE_AA)
                            cv2.drawMarker(
                                alert_overlay,
                                mrp,
                                color,
                                cv2.MARKER_TILTED_CROSS,
                                12,
                                2,
                                cv2.LINE_AA,
                            )
                        if "ally" in alerts:
                            color = (0, 255, 0)
                            cv2.circle(alert_overlay, mrp, max(5, radius - 5), color, 2, cv2.LINE_AA)
                            cv2.drawMarker(
                                alert_overlay,
                                mrp,
                                color,
                                cv2.MARKER_DIAMOND,
                                12,
                                2,
                                cv2.LINE_AA,
                            )
                        minimap = cv2.addWeighted(
                            minimap,
                            1.0 - alpha * 0.85,
                            alert_overlay,
                            alpha * 0.85,
                            0,
                        )

            canvas[MM_Y1:MM_Y2, MM_X1:MM_X2] = minimap

        # 7-1. face.gif 애니메이션 (ui.png 바로 뒤)
        if self.face_frames and self.face_total_duration > 0:
            t = time.time() % self.face_total_duration
            cur_t = 0.0
            frame_idx = 0
            for idx, dur in enumerate(self.face_durations):
                cur_t += dur
                if t < cur_t:
                    frame_idx = idx
                    break
            face_bgr, face_alpha = self.face_frames[frame_idx]
            fx1, fy1, fx2, fy2 = self.face_bbox
            sub_canvas = canvas[fy1:fy2, fx1:fx2]
            canvas[fy1:fy2, fx1:fx2] = (face_bgr * face_alpha + sub_canvas * (1.0 - face_alpha)).astype(np.uint8)

        # 8. UI 오버레이 합성
        for c in range(3):
            canvas[:, :, c] = (self.ui_bgr[:, :, c] * self.ui_alpha) + (canvas[:, :, c] * (1.0 - self.ui_alpha))

        # 8-1. 버튼 합성
        if self.button_manager is not None:
            canvas = self.button_manager.render_buttons(canvas)

        # 9. 콘솔 로그 텍스트
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.38
        line_h = 16
        max_lines = self.tb_h // line_h

        wrapped_lines = []
        for raw_line in log_lines:
            cur = ""
            for ch in raw_line:
                test = cur + ch
                size = cv2.getTextSize(test, font, font_scale, 1)[0]
                if size[0] > self.tb_w - 15:
                    wrapped_lines.append(cur)
                    cur = ch
                else:
                    cur = test
            if cur:
                wrapped_lines.append(cur)

        display_lines = wrapped_lines[-max_lines:]
        for i, line in enumerate(display_lines):
            draw_y = self.tb_y1 + (i + 1) * line_h
            color = (0, 255, 200) if i == len(display_lines) - 1 else (230, 230, 230)
            cv2.putText(canvas, line, (self.tb_x1 + 6, draw_y), font, font_scale, color, 1, cv2.LINE_AA)

        # 10. HUD 상태 텍스트
        vl, vr = speed_info
        if robot_pos is not None and goal_pos is not None:
            dist_px = math.hypot(goal_pos[0] - robot_pos[0], goal_pos[1] - robot_pos[1])
            dist_info = f"DIST: {dist_px:.0f}px"
        else:
            dist_info = "DIST: --"

        hud_lines = [
            f"ST: {status_text}",
            f"L: {vl:+4d} | R: {vr:+4d}",
            dist_info,
        ]

        self._draw_textbox_fast(
            canvas=canvas,
            x1=self.hud_x1, y1=self.hud_y1,
            x2=self.hud_x2, y2=self.hud_y2,
            lines=hud_lines,
            font_size=self._HUD_FONT_SIZE,
            color_rgb=self.hud_color_rgb,
            line_spacing=2
        )

        # 10-1. IP 주소 텍스트
        ip_lines = [f"IP: {ip_address}"]
        self._draw_textbox_fast(
            canvas=canvas,
            x1=self.ip_x1, y1=self.ip_y1,
            x2=self.ip_x2, y2=self.ip_y2,
            lines=ip_lines,
            font_size=10,
            color_rgb=self.ip_color_rgb,
            line_spacing=1
        )

        # 11. ArUco 5초 이상 미감지 시 수동 전환 창 및 버튼 오버레이 (Sudong_window.png + Sudong_button.png)
        if show_lost_warning:
            canvas = self._blend_layer(canvas, self.sudong_window_bgr, self.sudong_window_alpha, self.sudong_window_bbox)
            canvas = self._blend_layer(canvas, self.sudong_button_bgr, self.sudong_button_alpha, self.sudong_button_bbox)

        # 12. 좌측 상단 주행 모드 전환 버튼 오버레이 (driving_mode_switch_button.png, 관제 화면 최상위 레이어)
        canvas = self._blend_layer(canvas, self.driving_switch_bgr, self.driving_switch_alpha, self.driving_switch_bbox)

        # 13. 마우스 커서 렌더링 생략 (OS 기본 커서 사용)

        return canvas

    def render_manual_frame(
        self,
        pi_frame: Optional[np.ndarray],
        ip_address: str,
        port: int,
        speed_info: Tuple[int, int]
    ) -> np.ndarray:
        """수동 모드용: 라즈베리파이 카메라 전체화면 + HUD 오버레이 렌더링"""
        W, H = self.window_w, self.window_h
        
        if pi_frame is not None and pi_frame.size > 0:
            fh, fw = pi_frame.shape[:2]
            res_info = f"{fw}x{fh}"
            if (fw, fh) == (W, H):
                canvas = pi_frame.copy()
            else:
                canvas = cv2.resize(pi_frame, (W, H), interpolation=cv2.INTER_CUBIC)
        else:
            res_info = "--"
            canvas = np.zeros((H, W, 3), dtype=np.uint8)
            canvas[:] = (18, 18, 24)
            for gy in range(0, H, 30):
                cv2.line(canvas, (0, gy), (W, gy), (30, 30, 40), 1)
            for gx in range(0, W, 30):
                cv2.line(canvas, (gx, 0), (gx, H), (30, 30, 40), 1)
            cv2.putText(canvas, "RC-CAM: NO SIGNAL / CONNECTING...",
                        (W // 2 - 220, H // 2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas,
                        f"PI CAM  {ip_address}:{port}",
                        (W // 2 - 200, H // 2 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 120, 200), 1, cv2.LINE_AA)

        vl, vr = speed_info
        direction_label = "STOP"
        if vl > 0 and vr > 0: direction_label = "FORWARD"
        elif vl < 0 and vr < 0: direction_label = "BACKWARD"
        elif vl < 0 and vr > 0: direction_label = "LEFT"
        elif vl > 0 and vr < 0: direction_label = "RIGHT"

        # HUD 오버레이
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (W, 56), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)

        # 상단 HUD 텍스트 (좌측 0~200 영역은 driving_mode_switch_button이 위치하므로 x=215부터 출력)
        cv2.putText(canvas, "W:Forward  S:Back  A:Left  D:Right  |  P: Auto Mode",
                    (215, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)

        dir_colors = {
            "FORWARD":  (0, 255, 100),
            "BACKWARD": (0, 120, 255),
            "LEFT":     (255, 200, 0),
            "RIGHT":    (255, 200, 0),
            "STOP":     (100, 100, 100),
        }
        dir_color = dir_colors.get(direction_label, (200, 200, 200))
        cv2.putText(canvas, f"STATUS: {direction_label}",
                    (215, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, dir_color, 2, cv2.LINE_AA)

        # 좌측 상단 주행 모드 전환 버튼 오버레이 (driving_mode_switch_button.png, 수동 화면 최상위 레이어)
        canvas = self._blend_layer(canvas, self.driving_switch_bgr, self.driving_switch_alpha, self.driving_switch_bbox)

        cv2.putText(canvas, f"L:{vl:+4d}  R:{vr:+4d}",
                    (W - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 25), 1, cv2.LINE_AA)

        overlay2 = canvas.copy()
        cv2.rectangle(overlay2, (0, H - 36), (W, H), (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.55, canvas, 0.45, 0, canvas)
        cv2.putText(canvas,
                    f"RC-CAM LIVE [{res_info}]  |  PI: {ip_address}:{port}  |  ESC: Quit",
                    (16, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

        live_color = (0, 255, 0) if (pi_frame is not None) else (0, 0, 200)
        cv2.circle(canvas, (W - 20, H - 18), 7, live_color, -1, cv2.LINE_AA)

        return canvas
