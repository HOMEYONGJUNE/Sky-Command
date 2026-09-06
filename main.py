import math
import sys
import os
import time
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

import atexit
import ctypes
import ctypes.util

# macOS 시스템 기본 커서 제어 (부하 없는 단일 상태 토글)
_cg_lib = None
try:
    if sys.platform == "darwin":
        _p = ctypes.util.find_library("ApplicationServices")
        if _p:
            _cg_lib = ctypes.cdll.LoadLibrary(_p)
except Exception:
    pass

_system_cursor_hidden = False


def hide_system_cursor():
    global _system_cursor_hidden
    if _cg_lib and not _system_cursor_hidden:
        try:
            _cg_lib.CGDisplayHideCursor(0)
            _system_cursor_hidden = True
        except Exception:
            pass


def show_system_cursor():
    global _system_cursor_hidden
    if _cg_lib and _system_cursor_hidden:
        try:
            _cg_lib.CGDisplayShowCursor(0)
            _system_cursor_hidden = False
        except Exception:
            pass


atexit.register(show_system_cursor)


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

    def _on_mouse_event(self, event, x, y, flags, param):
        self.mouse_x, self.mouse_y = x, y

        # 창 내부에 마우스가 위치하면 OS 기본 화살표 커서를 숨겨 이중 커서 방지 (단일 상태 토글)
        if 0 <= x < config.WINDOW_WIDTH and 0 <= y < config.WINDOW_HEIGHT:
            hide_system_cursor()
        else:
            show_system_cursor()

        # 좌클릭 누름 (UI 버튼 및 옵션창 체크 - 단일 클릭만 처리하여 중복 토글 방지)
        if event == cv2.EVENT_LBUTTONDOWN:
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

    def run(self):
        while self.is_running:
            # 1. 프레임 읽기 (메인 관제 카메라는 원본 그대로 유지)
            if self.has_camera:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    frame = np.zeros((config.WINDOW_HEIGHT, config.WINDOW_WIDTH, 3), dtype=np.uint8)
            else:
                frame = np.zeros((config.WINDOW_HEIGHT, config.WINDOW_WIDTH, 3), dtype=np.uint8)

            self._update_hsv_from_trackbars()

            # 2. 해상도 맞춤
            if frame.shape[0] != config.WINDOW_HEIGHT or frame.shape[1] != config.WINDOW_WIDTH:
                warped_map = cv2.resize(frame, (config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
            else:
                warped_map = frame.copy()

            # 3. 마스크 추출 (좌우/상하 팽창 반경 실시간 적용)
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

            # 파란색 장애물 + 꼬깔 바닥점 + 유효 초록색 외 영역 종합 마스크
            total_obstacle = self._get_total_obstacle_mask(blue_mask, green_mask, cone_base_points)
            self.latest_total_obstacle = total_obstacle

            # 4. ArUco 마커 검출
            robot_pos, robot_angle, corners, is_tracked, _ = self.tracker.detect(warped_map)

            if is_tracked and robot_pos is not None:
                self.last_robot_pos = robot_pos
                self.last_robot_angle = robot_angle

                # 마커 한 변 길이 측정 -> 1.6배 확장된 원형 차체 크기에 따른 A* 회피 반경 실시간 동기화
                if corners is not None and len(corners) >= 4:
                    pts = np.array(corners, dtype=np.float32)
                    side0 = float(np.linalg.norm(pts[1] - pts[0]))
                    side1 = float(np.linalg.norm(pts[2] - pts[1]))
                    side2 = float(np.linalg.norm(pts[3] - pts[2]))
                    side3 = float(np.linalg.norm(pts[0] - pts[3]))
                    avg_side = (side0 + side1 + side2 + side3) / 4.0
                    
                    # 원형 차체 반경: 기존 2배 크기 대비 1.6배 더 크게 설정 (반경 = 마커 한 변 * 1.6)
                    dynamic_car_radius = max(60, int(avg_side * 1.6))
                    self.nav.planner.robot_radius_px = dynamic_car_radius

                # 처음 감지된 위치를 홈 위치로 등록
                if self.home_pos is None:
                    self.home_pos = robot_pos
                    print(f"[홈 등록] 초기 위치: {self.home_pos}")

            # 5. 주행 제어 및 모터 명령 전송
            v_left, v_right, nav_state = self.nav.update_control(robot_pos, robot_angle)
            self.motor.send_speed(v_left, v_right)


            # 6. UI 프레임 합성
            ui_output = self.renderer.render_frame(
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
                pi_cam_frame=self.pi_cam.get_latest_frame()
            )

            cv2.imshow(config.WINDOW_TITLE, ui_output)

            # 7. 키보드 입력 처리
            key = cv2.waitKey(1) & 0xFF

            # 초점 맞춤
            if key in [ord('f'), ord('F')]:
                self._init_camera_focus()


            # 홈 복귀
            elif key in [ord('b'), ord('B'), ord('q'), ord('Q')]:
                target_home = self.home_pos if self.home_pos is not None else (640, 360)
                if self.last_robot_pos is not None:
                    total_obs = self._get_total_obstacle_mask(
                        self.latest_blue_mask, self.latest_green_mask, self.latest_cone_base_points
                    )
                    self.nav.set_goal(self.last_robot_pos, target_home, total_obs)
                else:
                    print("[오류] 마커 미인식")

            # HSV 트랙바 창 토글
            elif key in [ord('h'), ord('H')]:
                self.show_hsv_controls = not self.show_hsv_controls
                if self.show_hsv_controls:
                    self._setup_hsv_controls()
                else:
                    cv2.destroyWindow("HSV Controls")

            # 긴급 정지
            elif key in [ord('s'), ord('S')]:
                self.motor.stop()
                self.nav.reset()
                print("[정지] 비상 정지")

            # 홈 위치 재설정
            elif key in [ord('r'), ord('R')]:
                self.motor.stop()
                self.nav.reset()
                self.tracker.reset()
                if self.last_robot_pos is not None:
                    self.home_pos = self.last_robot_pos
                    print(f"[홈 재설정] {self.home_pos}")

            # ESC 종료
            elif key == 27:
                break

            if cv2.getWindowProperty(config.WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break

        self.cleanup()

    def cleanup(self):
        # 시스템 커서 복원
        show_system_cursor()
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
