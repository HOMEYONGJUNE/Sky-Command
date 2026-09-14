import math
import sys
import os
import time
import threading
import cv2
import numpy as np
from typing import List, Optional, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import config
from control.motor_client import MotorClient
from control.navigator import StarcraftNavigator, NavState
from control.pi_cam_receiver import PiCamReceiver
from ui.logger import TerminalLogger
from ui.renderer import UIRenderer
from vision.aruco_tracker import AdvancedArucoTracker
from vision.map_transformer import MapTransformer

from pynput import keyboard as pynput_keyboard

MANUAL_PWM = 100 # 수동 모드 모터 속도 (0~255)




def load_hsv_settings(filepath: str, default_blue_hsv, default_green_hsv, default_blue_radius_x, default_blue_radius_y):
    """hsv_settings.txt 파일에서 HSV 및 좌우/상하 반경 설정값을 불러옵니다."""
    settings = {
        "blue_h_min": default_blue_hsv[0],
        "blue_h_max": default_blue_hsv[1],
        "blue_s_min": default_blue_hsv[2],
        "blue_v_min": default_blue_hsv[3],
        "blue_radius_x": default_blue_radius_x,
        "blue_radius_y": default_blue_radius_y,
        "green_h_min": default_green_hsv[0],
        "green_h_max": default_green_hsv[1],
        "green_s_min": default_green_hsv[2],
        "green_v_min": default_green_hsv[3],
    }
    legacy_radius = None
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip().lower()
                    if k == "blue_radius":
                        legacy_radius = int(v.strip())
                    elif k in settings:
                        settings[k] = int(v.strip())

            # 이전 버전 단일 blue_radius가 있고 새 키가 없을 때의 호환 처리
            if legacy_radius is not None:
                if "blue_radius_x" not in settings or settings["blue_radius_x"] == default_blue_radius_x:
                    settings["blue_radius_x"] = max(4, int(round(legacy_radius * 0.35)))
                if "blue_radius_y" not in settings or settings["blue_radius_y"] == default_blue_radius_y:
                    settings["blue_radius_y"] = max(10, int(round(legacy_radius * 1.3)))

            print(f"[설정 로드] {os.path.basename(filepath)}에서 HSV 및 좌우/상하 반경 설정을 불러왔습니다.")
        except Exception as e:
            print(f"[설정 로드 오류]: {e}")
    else:
        save_hsv_settings(
            filepath,
            [settings["blue_h_min"], settings["blue_h_max"], settings["blue_s_min"], settings["blue_v_min"]],
            [settings["green_h_min"], settings["green_h_max"], settings["green_s_min"], settings["green_v_min"]],
            settings["blue_radius_x"],
            settings["blue_radius_y"]
        )
    return settings


def save_hsv_settings(filepath: str, blue_hsv, green_hsv, blue_radius_x, blue_radius_y):
    """현재 HSV 및 좌우/상하 반경 설정값을 hsv_settings.txt 파일에 저장합니다."""
    try:
        content = (
            "# HSV 및 장애물 반경 설정 파일 (자동 저장/불러오기)\n"
            f"BLUE_H_MIN={int(blue_hsv[0])}\n"
            f"BLUE_H_MAX={int(blue_hsv[1])}\n"
            f"BLUE_S_MIN={int(blue_hsv[2])}\n"
            f"BLUE_V_MIN={int(blue_hsv[3])}\n"
            f"BLUE_RADIUS_X={int(blue_radius_x)}\n"
            f"BLUE_RADIUS_Y={int(blue_radius_y)}\n"
            f"BLUE_RADIUS={int(blue_radius_x)}\n"
            f"GREEN_H_MIN={int(green_hsv[0])}\n"
            f"GREEN_H_MAX={int(green_hsv[1])}\n"
            f"GREEN_S_MIN={int(green_hsv[2])}\n"
            f"GREEN_V_MIN={int(green_hsv[3])}\n"
        )
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        print(f"[설정 저장 오류]: {e}")


class StarcraftRCApp:
    def __init__(self):
        # 콘솔 로거 설정
        self.logger = TerminalLogger(max_lines=config.MAX_CONSOLE_LINES)
        self.logger.start_capture()

        print(f"[프로그램 시작] {config.WINDOW_TITLE}")

        # 제어 및 비전 모듈 초기화
        self.motor = MotorClient(ip=config.RASPBERRY_PI_IP, port=config.UDP_PORT)
        self.pi_cam = PiCamReceiver(ip=config.RASPBERRY_PI_IP, port=config.RASPBERRY_PI_CAM_PORT)
        self.pi_cam.start()
        self.tracker = AdvancedArucoTracker(
            dictionary_id=config.ARUCO_DICTIONARY_ID,
            target_id=config.TARGET_MARKER_ID,
            smooth_window=config.POSE_SMOOTH_WINDOW,
            smooth_alpha=0.90
        )
        self.map_trans = MapTransformer(
            map_width=config.WINDOW_WIDTH,
            map_height=config.WINDOW_HEIGHT
        )
        self.map_trans.matrix = np.eye(3, dtype=np.float32)

        self.nav = StarcraftNavigator(
            kp_angle=config.KP_ANGLE,
            kd_angle=config.KD_ANGLE,
            kp_dist=config.KP_DIST,
            kd_dist=config.KD_DIST,
            rot_threshold_deg=config.ROTATION_THRESHOLD_DEG,
            waypoint_dist_px=config.WAYPOINT_REACH_DIST_PX,
            final_goal_dist_px=config.FINAL_GOAL_REACH_DIST_PX,
            max_speed=config.MAX_PWM_SPEED,
            min_forward_pwm=config.MIN_FORWARD_PWM,
            min_rot_pwm=config.MIN_ROTATION_PWM,
            max_rot_pwm=config.MAX_ROTATION_PWM,
            diff_weight=config.DIFF_STEER_WEIGHT,
            robot_radius_px=config.ROBOT_RADIUS_PX,
            grid_size=config.GRID_CELL_SIZE,
            step_drive_enabled=config.STEP_DRIVE_ENABLED,
            step_move_sec=config.STEP_MOVE_SEC,
            step_pause_sec=config.STEP_PAUSE_SEC
        )
        self.renderer = UIRenderer(
            ui_image_path=config.UI_IMAGE_PATH,
            window_width=config.WINDOW_WIDTH,
            window_height=config.WINDOW_HEIGHT,
            text_box_rect=(config.TEXT_BOX_X1, config.TEXT_BOX_Y1, config.TEXT_BOX_X2, config.TEXT_BOX_Y2)
        )
        self.button_manager = self.renderer.button_manager

        # 카메라 열기
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_FRAME_HEIGHT)
        self.has_camera = self.cap.isOpened()

        if self.has_camera:
            print(f"[카메라] {config.CAMERA_INDEX}번 연결 완료")
            self._init_camera_focus()
        else:
            print("[카메라] 가상 모드로 동작")

        # 상태 변수
        self.is_running = True
        self.mouse_x, self.mouse_y = 0, 0
        self.home_pos = None
        self.last_robot_pos = None
        self.last_robot_angle = 0.0
        
        self.show_hsv_controls = False
        self.last_hsv_toggle_time = 0.0

        # txt 파일 기반 HSV 및 반경 설정 관리 (영구 유지)
        self.hsv_settings_file = os.path.join(CURRENT_DIR, "hsv_settings.txt")
        saved_settings = load_hsv_settings(
            self.hsv_settings_file,
            [config.DEFAULT_BLUE_H_MIN, config.DEFAULT_BLUE_H_MAX, config.DEFAULT_BLUE_S_MIN, config.DEFAULT_BLUE_V_MIN],
            [config.DEFAULT_GREEN_H_MIN, config.DEFAULT_GREEN_H_MAX, config.DEFAULT_GREEN_S_MIN, config.DEFAULT_GREEN_V_MIN],
            config.DEFAULT_BLUE_RADIUS_X,
            config.DEFAULT_BLUE_RADIUS_Y
        )
        self.blue_obstacle_radius_x = saved_settings["blue_radius_x"]
        self.blue_obstacle_radius_y = saved_settings["blue_radius_y"]
        self.blue_hsv = [
            saved_settings["blue_h_min"],
            saved_settings["blue_h_max"],
            saved_settings["blue_s_min"],
            saved_settings["blue_v_min"]
        ]
        self.green_hsv = [
            saved_settings["green_h_min"],
            saved_settings["green_h_max"],
            saved_settings["green_s_min"],
            saved_settings["green_v_min"]
        ]
        self.latest_blue_mask = None
        self.latest_green_mask = None
        self.latest_total_obstacle = None
        self.latest_cone_base_points = []
        
        # --- 수동 조작 모드 및 페이드 효과 상태 변수 ---
        self.manual_mode = False
        self.manual_vl = 0.0
        self.manual_vr = 0.0
        self.last_manual_key_time = 0.0
        self.last_aruco_seen_time = time.time()
        self.show_lost_warning = False
        self.aruco_popup_shown = False
        self.aruco_popup_result = None
        
        self.fade_progress = 0.0
        self.fade_start_time = 0.0
        self.is_fading = False
        self.fade_target = False # True = manual, False = auto

        # --- pynput OS 레벨 키 상태 추적 ---
        # cv2.waitKey는 OpenCV 윈도우에 포커스가 없으면 입력이 막혀 WASD가 작동 안 함
        # pynput으로 OS 레벨에서 현재 눌린 키 집합을 추적
        self._pressed_keys: set = set()
        self._key_listener = pynput_keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release
        )
        self._key_listener.daemon = True
        self._key_listener.start()

        # 윈도우 생성 및 마우스 콜백 등록
        cv2.namedWindow(config.WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(config.WINDOW_TITLE, config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        cv2.setMouseCallback(config.WINDOW_TITLE, self._on_mouse_event)

    def _get_total_obstacle_mask(
        self,
        blue_mask: Optional[np.ndarray],
        green_mask: Optional[np.ndarray],
        cone_points: Optional[List[Tuple[int, int]]] = None
    ) -> np.ndarray:
        h, w = config.WINDOW_HEIGHT, config.WINDOW_WIDTH
        if blue_mask is not None:
            base_obs = blue_mask.copy()
        else:
            base_obs = np.zeros((h, w), dtype=np.uint8)

        # 초록색 경기장 바닥이 맵 전체의 20% 이상 확실하게 검출될 때만 지형 경계 결합
        total_pixels = h * w
        if green_mask is not None and np.count_nonzero(green_mask) > total_pixels * 0.20:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            clean_green = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
            non_green = cv2.bitwise_not(clean_green)
            return cv2.bitwise_or(base_obs, non_green)

        return base_obs

    def _init_camera_focus(self):
        # 포커스를 흔들어 오토포커스 재기동 ('F' 키 기능)
        if not self.has_camera:
            print("[카메라] 연결된 카메라가 없습니다.")
            return
        try:
            print("[카메라] 'F' 키 입력: 초점(Autofocus) 자동 맞춤 실행 중...")
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.cap.set(cv2.CAP_PROP_FOCUS, 0)
            time.sleep(0.08)
            self.cap.set(cv2.CAP_PROP_FOCUS, 100)
            time.sleep(0.08)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            print("[카메라] 초점 맞춤 완료")
        except Exception as e:
            print(f"[카메라 포커스 오류]: {e}")

    def _on_key_press(self, key):
        """pynput: OS 레벨 키 누름 이벤트 - 눌린 키를 집합에 추가"""
        try:
            ch = key.char.lower() if hasattr(key, 'char') and key.char else None
            if ch:
                self._pressed_keys.add(ch)
        except Exception:
            pass

    def _on_key_release(self, key):
        """pynput: OS 레벨 키 뗌 이벤트 - 눌린 키를 집합에서 제거"""
        try:
            ch = key.char.lower() if hasattr(key, 'char') and key.char else None
            if ch:
                self._pressed_keys.discard(ch)
        except Exception:
            pass

    def _on_mouse_event(self, event, x, y, flags, param):
        self.mouse_x, self.mouse_y = x, y

        # 좌클릭 누름 (UI 버튼 및 옵션창 체크 - 단일 클릭만 처리하여 중복 토글 방지)
        if event == cv2.EVENT_LBUTTONDOWN:
            # 1. ArUco 5초 미감지 수동 전환 창의 버튼 클릭 체크 (Sudong_button.png 투명 제외 영역)
            if getattr(self, "show_lost_warning", False) and self.renderer.is_sudong_button_clicked(x, y):
                print("[UI] Sudong_button 클릭 감지 -> 수동 조작 모드로 전환합니다.")
                self.show_lost_warning = False
                self._enter_manual_mode()
                return

            # 2. 좌측 상단 주행 모드 전환 버튼 클릭 체크 (driving_mode_switch_button.png 투명 제외 영역)
            if self.renderer.is_driving_mode_switch_clicked(x, y):
                print("[UI] driving_mode_switch_button 클릭 감지 -> 주행 모드 전환")
                self._toggle_manual_mode()
                return

            # 옵션창 버튼 클릭 영역 (x: 981~1058, y: 609~630)
            if 981 <= x <= 1058 and 609 <= y <= 630:
                now = time.time()
                # 더블클릭으로 인해 열리자마자 바로 닫히는 현상 방지 (0.35초 디바운스)
                if now - self.last_hsv_toggle_time < 0.35:
                    return
                self.last_hsv_toggle_time = now

                is_open = False
                try:
                    if cv2.getWindowProperty("HSV Controls", cv2.WND_PROP_VISIBLE) >= 1:
                        is_open = True
                except Exception:
                    is_open = False

                if is_open:
                    self.show_hsv_controls = False
                    cv2.destroyWindow("HSV Controls")
                    print("[옵션] HSV 조절창을 닫았습니다.")
                else:
                    self.show_hsv_controls = True
                    self._setup_hsv_controls()
                    print("[옵션] 초록/파랑 HSV 및 반경 조절창을 열었습니다.")
                return

            clicked_btn = self.button_manager.handle_mouse_down(x, y)
            if clicked_btn is not None:
                if clicked_btn == "exit":
                    self.is_running = False
                elif clicked_btn == "stop":
                    self.motor.stop()
                    self.nav.reset()
                    print("[정지] STOP 버튼")
                elif clicked_btn == "recall":
                    target_home = self.home_pos if self.home_pos is not None else (640, 360)
                    if self.last_robot_pos is not None:
                        total_obs = self._get_total_obstacle_mask(
                            self.latest_blue_mask, self.latest_green_mask, self.latest_cone_base_points
                        )
                        self.nav.set_goal(self.last_robot_pos, target_home, total_obs)
                    else:
                        print("[오류] 마커 미인식")
                return

        # 좌클릭 뗌
        elif event == cv2.EVENT_LBUTTONUP:
            self.button_manager.handle_mouse_up(x, y)
            return

        # 우클릭 (또는 Ctrl+좌클릭) 이동 명령
        if event == cv2.EVENT_RBUTTONDOWN or (event == cv2.EVENT_LBUTTONDOWN and (flags & cv2.EVENT_FLAG_CTRLKEY)):
            CAM_W, CAM_H = 1090, 614
            if x > CAM_W or y > CAM_H:
                return

            cam_x = int(x * config.WINDOW_WIDTH / CAM_W)
            cam_y = int(y * config.WINDOW_HEIGHT / CAM_H)
            
            if self.last_robot_pos is not None:
                total_obs = self._get_total_obstacle_mask(
                    self.latest_blue_mask, self.latest_green_mask, self.latest_cone_base_points
                )
                self.nav.set_goal(self.last_robot_pos, (cam_x, cam_y), total_obs)
            else:
                print(f"[명령 대기] 마커 미인식 (목표: {cam_x}, {cam_y})")

    def _setup_hsv_controls(self):
        def nothing(val):
            pass

        cv2.namedWindow("HSV Controls", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("HSV Controls", 400, 480)
        # 파란색 장애물 조절
        cv2.createTrackbar("Blue H Min", "HSV Controls", self.blue_hsv[0], 179, nothing)
        cv2.createTrackbar("Blue H Max", "HSV Controls", self.blue_hsv[1], 179, nothing)
        cv2.createTrackbar("Blue S Min", "HSV Controls", self.blue_hsv[2], 255, nothing)
        cv2.createTrackbar("Blue V Min", "HSV Controls", self.blue_hsv[3], 255, nothing)
        cv2.createTrackbar("Blue Radius X (L/R)", "HSV Controls", self.blue_obstacle_radius_x, 150, nothing)
        cv2.createTrackbar("Blue Radius Y (U/D)", "HSV Controls", self.blue_obstacle_radius_y, 200, nothing)
        # 초록색 지형 조절
        cv2.createTrackbar("Green H Min", "HSV Controls", self.green_hsv[0], 179, nothing)
        cv2.createTrackbar("Green H Max", "HSV Controls", self.green_hsv[1], 179, nothing)
        cv2.createTrackbar("Green S Min", "HSV Controls", self.green_hsv[2], 255, nothing)
        cv2.createTrackbar("Green V Min", "HSV Controls", self.green_hsv[3], 255, nothing)

    def _update_hsv_from_trackbars(self):
        if not self.show_hsv_controls:
            return
        try:
            if cv2.getWindowProperty("HSV Controls", cv2.WND_PROP_VISIBLE) < 1:
                self.show_hsv_controls = False
                return

            bh0 = cv2.getTrackbarPos("Blue H Min", "HSV Controls")
            bh1 = cv2.getTrackbarPos("Blue H Max", "HSV Controls")
            bs0 = cv2.getTrackbarPos("Blue S Min", "HSV Controls")
            bv0 = cv2.getTrackbarPos("Blue V Min", "HSV Controls")
            br_x = cv2.getTrackbarPos("Blue Radius X (L/R)", "HSV Controls")
            br_y = cv2.getTrackbarPos("Blue Radius Y (U/D)", "HSV Controls")

            gh0 = cv2.getTrackbarPos("Green H Min", "HSV Controls")
            gh1 = cv2.getTrackbarPos("Green H Max", "HSV Controls")
            gs0 = cv2.getTrackbarPos("Green S Min", "HSV Controls")
            gv0 = cv2.getTrackbarPos("Green V Min", "HSV Controls")

            changed = (
                [bh0, bh1, bs0, bv0] != self.blue_hsv
                or br_x != self.blue_obstacle_radius_x
                or br_y != self.blue_obstacle_radius_y
                or [gh0, gh1, gs0, gv0] != self.green_hsv
            )

            if changed:
                self.blue_hsv = [bh0, bh1, bs0, bv0]
                self.blue_obstacle_radius_x = br_x
                self.blue_obstacle_radius_y = br_y
                self.green_hsv = [gh0, gh1, gs0, gv0]
                save_hsv_settings(
                    self.hsv_settings_file,
                    self.blue_hsv,
                    self.green_hsv,
                    self.blue_obstacle_radius_x,
                    self.blue_obstacle_radius_y
                )
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────────
    # 수동 조작 모드 진입/종료 (페이드 애니메이션 포함)
    # ──────────────────────────────────────────────────────────────────

    def _enter_manual_mode(self):
        """수동 조작 모드로 전환 (페이드 시작)."""
        if not self.manual_mode:
            self.manual_mode = True
            self.motor.stop()
            self.nav.reset()
            self.is_fading = True
            self.fade_start_time = time.time()
            self.fade_target = True  # 목표: 수동 화면
            print("[수동 모드] 수동 조작 모드로 전환 중... (P: 자동 모드 복귀 | W/A/S/D: 조작)")

    def _exit_manual_mode(self):
        """자동 조작 모드로 복귀 (페이드 시작)."""
        if self.manual_mode:
            self.manual_mode = False
            self.motor.stop()
            self.last_aruco_seen_time = time.time()
            self.aruco_popup_shown = False
            self.is_fading = True
            self.fade_start_time = time.time()
            self.fade_target = False  # 목표: 자동(탑뷰) 화면
            print("[자동 모드] 자동 조작 모드로 복귀 중...")

    def _toggle_manual_mode(self):
        """P키: 수동/자동 토글"""
        # 페이드 진행 중에는 토글 무시 (안정성)
        if self.is_fading:
            return
        if self.manual_mode:
            self._exit_manual_mode()
        else:
            self._enter_manual_mode()

    def _update_fade(self):
        """페이드 진행률 업데이트 (0.5초 기준)"""
        if self.is_fading:
            elapsed = time.time() - self.fade_start_time
            fade_duration = 0.5  # 0.5초 동안 전환
            if elapsed >= fade_duration:
                self.is_fading = False
                self.fade_progress = 1.0 if self.fade_target else 0.0
            else:
                ratio = elapsed / fade_duration
                # target=True(수동)이면 0->1, target=False(자동)이면 1->0
                self.fade_progress = ratio if self.fade_target else (1.0 - ratio)

    # ──────────────────────────────────────────────────────────────────
    # ArUco 7초 미검출 팝업 (별도 스레드에서 tkinter 실행)
    # ──────────────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────────────────────────
    # 메인 루프
    # ──────────────────────────────────────────────────────────────────

    def run(self):
        while self.is_running:
            # ── 키보드 입력 처리 ──────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == 27: # ESC
                break
            if key in [ord('p'), ord('P')]:
                self._toggle_manual_mode()

            # ── 상태 업데이트 ──────────────────────────────────────
            self._update_fade()

            # ── 영상 데이터 준비 ───────────────────────────────────
            # 메인 관제 카메라 프레임
            if self.has_camera:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    frame = np.zeros((config.WINDOW_HEIGHT, config.WINDOW_WIDTH, 3), dtype=np.uint8)
            else:
                frame = np.zeros((config.WINDOW_HEIGHT, config.WINDOW_WIDTH, 3), dtype=np.uint8)

            if frame.shape[0] != config.WINDOW_HEIGHT or frame.shape[1] != config.WINDOW_WIDTH:
                warped_map = cv2.resize(frame, (config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
            else:
                warped_map = frame.copy()

            # ── 자동 모드용 데이터 연산 (페이드 중이거나 자동 모드일 때) ──
            ui_output = None
            if not self.manual_mode or self.is_fading:
                self._update_hsv_from_trackbars()
                hsv_map = cv2.cvtColor(warped_map, cv2.COLOR_BGR2HSV)
                blue_mask, cone_base_points = self.map_trans.extract_blue_obstacle_mask(
                    hsv_map,
                    h_min=self.blue_hsv[0], h_max=self.blue_hsv[1],
                    s_min=self.blue_hsv[2], v_min=self.blue_hsv[3],
                    dilate_x=self.blue_obstacle_radius_x,
                    dilate_y=self.blue_obstacle_radius_y
                )
                green_mask = self.map_trans.extract_green_terrain_mask(
                    hsv_map,
                    h_min=self.green_hsv[0], h_max=self.green_hsv[1],
                    s_min=self.green_hsv[2], v_min=self.green_hsv[3]
                )
                self.latest_blue_mask = blue_mask
                self.latest_green_mask = green_mask
                self.latest_cone_base_points = cone_base_points
                self.latest_total_obstacle = self._get_total_obstacle_mask(blue_mask, green_mask, cone_base_points)

                robot_pos, robot_angle, corners, is_tracked, _ = self.tracker.detect(warped_map)
                
                if is_tracked and robot_pos is not None:
                    self.last_robot_pos = robot_pos
                    self.last_robot_angle = robot_angle
                    self.last_aruco_seen_time = time.time()

                    if corners is not None and len(corners) >= 4:
                        pts = np.array(corners, dtype=np.float32)
                        side0 = float(np.linalg.norm(pts[1] - pts[0]))
                        side1 = float(np.linalg.norm(pts[2] - pts[1]))
                        side2 = float(np.linalg.norm(pts[3] - pts[2]))
                        side3 = float(np.linalg.norm(pts[0] - pts[3]))
                        avg_side = (side0 + side1 + side2 + side3) / 4.0
                        self.nav.planner.robot_radius_px = max(60, int(avg_side * 1.6))

                    if self.home_pos is None:
                        self.home_pos = robot_pos
                        print(f"[홈 등록] 초기 위치: {self.home_pos}")

                # ArUco 5초 미감지 체크 (화면 안내창 플래그 설정)
                show_lost_warning = False
                if not self.manual_mode and not self.is_fading:
                    if time.time() - self.last_aruco_seen_time >= 5.0:
                        show_lost_warning = True
                self.show_lost_warning = show_lost_warning

                # 주행 제어 업데이트 (자동 모드일 때만 전송)
                v_left, v_right, nav_state = self.nav.update_control(robot_pos, robot_angle)
                if not self.manual_mode and not self.is_fading:
                    self.motor.send_speed(v_left, v_right)

                # 자동 모드 렌더링 준비
                auto_canvas = self.renderer.render_frame(
                    main_view=warped_map,
                    log_lines=self.logger.log_lines,
                    robot_pos=robot_pos,
                    robot_angle_deg=robot_angle,
                    waypoints=self.nav.waypoints,
                    current_wp_idx=self.nav.current_wp_idx,
                    goal_pos=self.nav.final_goal,
                    blue_mask=blue_mask,
                    green_mask=green_mask,
                    ping_pos=self.nav.ping_pos,
                    ping_start_time=self.nav.ping_start_time,
                    mouse_pos=(self.mouse_x, self.mouse_y),
                    status_text=nav_state.value if robot_pos is not None else "SEARCHING",
                    speed_info=(int(v_left), int(v_right)),
                    marker_corners=corners,
                    home_pos=self.home_pos,
                    is_blind=False,
                    ip_address=config.RASPBERRY_PI_IP,
                    cone_base_points=self.latest_cone_base_points,
                    pi_cam_frame=self.pi_cam.get_latest_frame(),
                    show_lost_warning=show_lost_warning
                )

            # ── 수동 모드 연산 및 제어 (페이드 중이거나 수동 모드일 때) ──
            if self.manual_mode or self.is_fading:
                # WASD 제어 - pynput으로 OS 레벨에서 현재 눌린 키를 실시간 확인
                vl, vr = 0, 0
                if self.manual_mode and not self.is_fading:
                    keys = self._pressed_keys  # 현재 눌린 키 집합 참조

                    if 'w' in keys:
                        vl, vr = MANUAL_PWM, MANUAL_PWM    # 직진
                    elif 's' in keys:
                        vl, vr = -MANUAL_PWM, -MANUAL_PWM  # 후진
                    elif 'a' in keys:
                        vl, vr = -MANUAL_PWM, MANUAL_PWM   # 왼쪽 회전
                    elif 'd' in keys:
                        vl, vr = MANUAL_PWM, -MANUAL_PWM   # 오른쪽 회전

                    if vl != 0 or vr != 0:
                        self.motor.send_speed(vl, vr)
                    else:
                        self.motor.stop()

                manual_canvas = self.renderer.render_manual_frame(
                    pi_frame=self.pi_cam.get_latest_frame(),
                    ip_address=config.RASPBERRY_PI_IP,
                    port=config.RASPBERRY_PI_CAM_PORT,
                    speed_info=(vl, vr)
                )

            # ── 화면 렌더링 선택/블렌딩 ─────────────────────────────
            if self.is_fading:
                ui_output = cv2.addWeighted(auto_canvas, 1.0 - self.fade_progress, manual_canvas, self.fade_progress, 0)
            elif self.manual_mode:
                ui_output = manual_canvas
            else:
                ui_output = auto_canvas

            cv2.imshow(config.WINDOW_TITLE, ui_output)
            
            if cv2.getWindowProperty(config.WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break
                
        self.cleanup()

    def cleanup(self):
        # 최종 HSV 및 반경 설정 저장
        save_hsv_settings(
            self.hsv_settings_file,
            self.blue_hsv,
            self.green_hsv,
            self.blue_obstacle_radius_x,
            self.blue_obstacle_radius_y
        )
        self.motor.stop()
        self.motor.close()
        self.pi_cam.stop()
        try:
            self._key_listener.stop()
        except Exception:
            pass
        if self.has_camera:
            self.cap.release()
        cv2.destroyAllWindows()
        self.logger.restore()
        print("[종료] 프로그램 종료 (HSV 설정 저장 완료)")


if __name__ == "__main__":
    try:
        app = StarcraftRCApp()
        app.run()
    except KeyboardInterrupt:
        print("\n[종료] 키보드 인터럽트")
    except Exception as e:
        print(f"\n[오류]: {e}")
