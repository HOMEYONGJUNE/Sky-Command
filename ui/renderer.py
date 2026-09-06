import math
import os
import time
from typing import List, Optional, Tuple
import cv2
import numpy as np
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

    def render_frame(
        self,
        main_view: np.ndarray,
        log_lines: List[str],
        robot_pos: Optional[Tuple[int, int]] = None,
        robot_angle_deg: float = 0.0,
        waypoints: Optional[List[Tuple[int, int]]] = None,
        current_wp_idx: int = 0,
        goal_pos: Optional[Tuple[int, int]] = None,
        blue_mask: Optional[np.ndarray] = None,
        green_mask: Optional[np.ndarray] = None,
        ping_pos: Optional[Tuple[int, int]] = None,
        ping_start_time: float = 0.0,
        mouse_pos: Tuple[int, int] = (0, 0),
        status_text: str = "IDLE",
        speed_info: Tuple[int, int] = (0, 0),
        marker_corners: Optional[np.ndarray] = None,
        home_pos: Optional[Tuple[int, int]] = None,
        is_blind: bool = False,
        ip_address: str = "192.168.0.2",
        cone_base_points: Optional[List[Tuple[int, int]]] = None,
        pi_cam_frame: Optional[np.ndarray] = None
    ) -> np.ndarray:
        CAM_W, CAM_H = 1090, 614

        # 1. 캔버스 초기화 및 카메라 프레임 배치
        canvas = np.zeros((self.window_h, self.window_w, 3), dtype=np.uint8)
        canvas[:] = (18, 18, 22)

        cam_frame = cv2.resize(main_view, (CAM_W, CAM_H))
        canvas[0:CAM_H, 0:CAM_W] = cam_frame

        # 2. 장애물 마스크 오버레이 및 선명한 테두리선(Outline)
        if blue_mask is not None and np.count_nonzero(blue_mask) > 0:
            blue_mask_cam = cv2.resize(blue_mask, (CAM_W, CAM_H), interpolation=cv2.INTER_NEAREST)
            cam_region = canvas[0:CAM_H, 0:CAM_W].copy()
            cam_region[blue_mask_cam > 0] = (255, 60, 60)
            canvas[0:CAM_H, 0:CAM_W] = cv2.addWeighted(
                canvas[0:CAM_H, 0:CAM_W], 0.7, cam_region, 0.3, 0
            )

            # 파란색 장애물 영역 외곽 테두리선 추출 및 렌더링 (주황/빨강 2px + 외곽 강조)
            obs_contours, _ = cv2.findContours(blue_mask_cam, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas[0:CAM_H, 0:CAM_W], obs_contours, -1, (0, 70, 255), 2, cv2.LINE_AA)
            cv2.drawContours(canvas[0:CAM_H, 0:CAM_W], obs_contours, -1, (200, 230, 255), 1, cv2.LINE_AA)

        def sc(pt):
            return (int(pt[0] * CAM_W / self.window_w), int(pt[1] * CAM_H / self.window_h))

        # 2-1. 파란색 꼬깔의 맨 아래쪽 바닥 픽셀을 빨간 점(RED DOT)으로 시각화
        if cone_base_points:
            for bx, by in cone_base_points:
                sbp = sc((bx, by))
                # 채워진 빨간 원 (반경 5px) + 흰색 테두리(가시성 극대화) + 십자선 마커
                cv2.circle(canvas[0:CAM_H, 0:CAM_W], sbp, 5, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas[0:CAM_H, 0:CAM_W], sbp, 6, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.drawMarker(canvas[0:CAM_H, 0:CAM_W], sbp, (255, 255, 255), cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA)

        # 3. ArUco 마커 외곽선 및 RC카 원형 차체 영역 (원형 차체 1.6배 확장 크기)
        car_radius_screen = int(93 * CAM_W / self.window_w)
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

            # 원형 차체 반경 (마커 한 변 * 1.6)
            car_radius_px = avg_side * 1.6
            car_radius_screen = int(car_radius_px * CAM_W / self.window_w)

            # ArUco 마커 외곽선 (초록색 1px)
            scaled_marker = np.array([[sc(p)] for p in mc_pts], dtype=np.int32)
            cv2.polylines(canvas, [scaled_marker], True, (0, 255, 0), 1, cv2.LINE_AA)

        elif robot_pos is not None:
            center_screen = sc(robot_pos)

        # 원형 차체 렌더링 (반투명 채우기 + 2px 외곽선 원)
        if center_screen is not None and car_radius_screen > 0:
            car_overlay = canvas.copy()
            cv2.circle(car_overlay, center_screen, car_radius_screen, (255, 200, 0), -1, cv2.LINE_AA)
            canvas = cv2.addWeighted(canvas, 0.75, car_overlay, 0.25, 0)
            cv2.circle(canvas, center_screen, car_radius_screen, (255, 235, 50), 2, cv2.LINE_AA)

        # 로봇 중심점 및 헤딩 방향선 (슬림 1px)
        if robot_pos is not None:
            srp = sc(robot_pos)
            cv2.circle(canvas, srp, 3, (255, 255, 255), -1, cv2.LINE_AA)

            rad = math.radians(robot_angle_deg)
            arrow_len = max(24, car_radius_screen + 8 if center_screen is not None else 24)
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

            # 파란 장애물 → 어두운 파란색 및 테두리선
            if blue_mask is not None and np.count_nonzero(blue_mask) > 0:
                mini_mask = cv2.resize(blue_mask, (MM_W, MM_H), interpolation=cv2.INTER_NEAREST)
                minimap[mini_mask > 0] = (0, 0, 200)
                mini_contours, _ = cv2.findContours(mini_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(minimap, mini_contours, -1, (0, 150, 255), 1, cv2.LINE_AA)

            # 미니맵 꼬깔 바닥점 빨간 점
            if cone_base_points:
                for bx, by in cone_base_points:
                    mbp = mm((bx, by))
                    cv2.circle(minimap, mbp, 3, (0, 0, 255), -1, cv2.LINE_AA)

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

        # 11. 마우스 커서 렌더링 (70% 스케일, 고속 넘파이 블렌딩)
        mx, my = mouse_pos
        if self.cursor_bgr is not None and (0 <= mx < self.window_w and 0 <= my < self.window_h):
            hx, hy = self.cursor_hotspot
            cx1 = mx - hx
            cy1 = my - hy
            ch, cw = self.cursor_bgr.shape[:2]
            cx2 = cx1 + cw
            cy2 = cy1 + ch

            x1_clip = max(0, cx1)
            y1_clip = max(0, cy1)
            x2_clip = min(self.window_w, cx2)
            y2_clip = min(self.window_h, cy2)

            if x1_clip < x2_clip and y1_clip < y2_clip:
                src_x1 = x1_clip - cx1
                src_y1 = y1_clip - cy1
                src_x2 = src_x1 + (x2_clip - x1_clip)
                src_y2 = src_y1 + (y2_clip - y1_clip)

                c_bgr = self.cursor_bgr[src_y1:src_y2, src_x1:src_x2]
                c_alpha = self.cursor_alpha[src_y1:src_y2, src_x1:src_x2]
                sub_canvas = canvas[y1_clip:y2_clip, x1_clip:x2_clip]
                canvas[y1_clip:y2_clip, x1_clip:x2_clip] = (
                    c_bgr * c_alpha + sub_canvas * (1.0 - c_alpha)
                ).astype(np.uint8)
        else:
            cv2.drawMarker(canvas, (mx, my), (0, 255, 255), cv2.MARKER_CROSS, 12, 1)

        return canvas
