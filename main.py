import math
import sys
import os
import shutil
import subprocess
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
from vision.yolo_obstacle_detector import YOLOObstacleDetector
from vision.yolo_obstacle_detector import Detection

from pynput import keyboard as pynput_keyboard

MANUAL_PWM = 100 # 수동 모드 모터 속도 (0~255)
OBSTACLE_MODEL_PATH = os.path.join(CURRENT_DIR, "best.pt")
ULTRASONIC_MAX_RANGE_CM = 50.0
ULTRASONIC_PIXELS_PER_METER = 500.0


class AudioPlayer:
    def __init__(self, ui_dir: str):
        self.paths = {
            "ally": os.path.join(ui_dir, "ally_alert.wav"),
            "claymore": os.path.join(ui_dir, "enemy_alert.wav"),
            "click": os.path.join(ui_dir, "click.wav"),
        }
        self.player = self._find_player()
        self.warned_missing_player = False
        self.process = None
        self.lock = threading.Lock()
        if self.player is None:
            print("[AUDIO ERROR] No WAV player found (afplay/ffplay/mpg123/mpg321).")
        for sound_name in ("ally", "claymore", "click"):
            if not os.path.isfile(self.paths[sound_name]):
                print(f"[AUDIO ERROR] Missing sound file: {self.paths[sound_name]}")

    @staticmethod
    def _find_player():
        if sys.platform == "darwin" and shutil.which("afplay"):
            return ["afplay"]
        for executable in ("ffplay", "mpg123", "mpg321"):
            if shutil.which(executable):
                return [executable, "-nodisp", "-autoexit"] if executable == "ffplay" else [executable]
        return None

    def play(self, sound_name: str):
        if self.player is None:
            return
        path = self.paths.get(sound_name)
        if path is None or not os.path.isfile(path):
            return
        try:
            with self.lock:
                if self.process is not None and self.process.poll() is None:
                    self.process.terminate()
                self.process = subprocess.Popen(
                    [*self.player, path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                print(f"[AUDIO] Playing {sound_name} alert.")
        except OSError:
            if not self.warned_missing_player:
                print("[AUDIO ERROR] Unable to start the audio player.")
                self.warned_missing_player = True

    def close(self):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
            self.process = None




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

            print(f"[CONFIG] Loaded HSV and radius settings from {os.path.basename(filepath)}.")
        except Exception as e:
            print(f"[CONFIG ERROR] Failed to load settings: {e}")
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
        print(f"[CONFIG ERROR] Failed to save settings: {e}")


class StarcraftRCApp:
    def __init__(self):
        # 콘솔 로거 설정
        self.logger = TerminalLogger(max_lines=config.MAX_CONSOLE_LINES)
        self.logger.start_capture()
        self.audio = AudioPlayer(os.path.join(CURRENT_DIR, "ui"))

        print(f"[STARTUP] {config.WINDOW_TITLE}")

        # 제어 및 비전 모듈 초기화
        self.motor = MotorClient(ip=config.RASPBERRY_PI_IP, port=config.UDP_PORT)
        self.motor.start_telemetry_listener()
        self.pi_cam = PiCamReceiver(ip=config.RASPBERRY_PI_IP, port=config.RASPBERRY_PI_CAM_PORT)
        self.pi_cam.start()
        self.tracker = AdvancedArucoTracker(
            dictionary_id=config.ARUCO_DICTIONARY_ID,
            target_id=config.TARGET_MARKER_ID,
            smooth_window=config.POSE_SMOOTH_WINDOW,
            smooth_alpha=0.90,
            fast_miss_after=config.ARUCO_FAST_MISS_AFTER,
            max_coasting_frames=config.MAX_COASTING_FRAMES,
            enhanced_scan_interval=config.ARUCO_ENHANCED_SCAN_INTERVAL
        )
        self.map_trans = MapTransformer(
            map_width=config.WINDOW_WIDTH,
            map_height=config.WINDOW_HEIGHT
        )
        self.map_trans.matrix = np.eye(3, dtype=np.float32)
        self.obstacle_detector = YOLOObstacleDetector(
            OBSTACLE_MODEL_PATH,
            image_size=320,
            box_scale=config.OBSTACLE_BOX_SCALE,
        )
        print(f"[MODEL] YOLO model loaded: {OBSTACLE_MODEL_PATH}")
        self.nav = StarcraftNavigator(
            kp_angle=config.KP_ANGLE,
            kd_angle=config.KD_ANGLE,
            kp_dist=config.KP_DIST,
            kd_dist=config.KD_DIST,
            rot_threshold_deg=config.ROTATION_THRESHOLD_DEG,
            forward_alignment_threshold_deg=config.FORWARD_ALIGNMENT_THRESHOLD_DEG,
            waypoint_dist_px=config.WAYPOINT_REACH_DIST_PX,
            final_goal_dist_px=config.FINAL_GOAL_REACH_DIST_PX,
            max_speed=config.MAX_PWM_SPEED,
            min_forward_pwm=config.MIN_FORWARD_PWM,
            min_rot_pwm=config.MIN_ROTATION_PWM,
            max_rot_pwm=config.MAX_ROTATION_PWM,
            diff_weight=config.DIFF_STEER_WEIGHT,
            robot_radius_px=config.ROBOT_RADIUS_PX,
            grid_size=config.GRID_CELL_SIZE,
            astar_extra_safety_margin_px=config.ASTAR_EXTRA_SAFETY_MARGIN_PX,
            waypoint_spacing_px=config.PATH_WAYPOINT_SPACING_PX,
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
            print(f"[CAMERA] Device {config.CAMERA_INDEX} connected")
            self._init_camera_focus()
        else:
            print("[CAMERA] Running in virtual mode")

        # 상태 변수
        self.is_running = True
        self.mouse_x, self.mouse_y = 0, 0
        self.home_pos = None
        self.last_robot_pos = None
        self.last_robot_angle = 0.0
        
        self.latest_obstacle_mask = None
        self.latest_detections = []
        self.onboard_alerts = set()
        self.onboard_alert_start_time = 0.0
        self.last_onboard_sound_time = {}
        self.frame_count = 0
        self.last_detection_mask = None
        self.last_detections = []
        self.last_detection_time = 0.0
        self.ultrasonic_reverse_until = 0.0
        self.ultrasonic_replan_pending = False
        self.ultrasonic_trigger_latched = False
        self.ultrasonic_objects = {}
        self.static_obstacle_mask = None
        self.static_detections = []
        self.auto_canvas = None
        self.fps = 0.0
        self.fps_window_start = time.monotonic()
        self.fps_window_frames = 0
        self._camera_lock = threading.Lock()
        self._camera_stop = threading.Event()
        self._latest_camera_frame = None
        self._camera_thread = None
        if self.has_camera:
            self._camera_thread = threading.Thread(
                target=self._camera_capture_loop,
                name="main-camera-capture",
                daemon=True,
            )
            self._camera_thread.start()

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
        self.fade_target = False

        self._pressed_keys: set = set()
        self._key_listener = pynput_keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release
        )
        self._key_listener.daemon = True
        self._key_listener.start()

        cv2.namedWindow(config.WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(config.WINDOW_TITLE, config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        cv2.setMouseCallback(config.WINDOW_TITLE, self._on_mouse_event)

    @staticmethod
    def _box_iou(box_a, box_b) -> float:
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b
        x1, y1 = max(ax, bx), max(ay, by)
        x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = aw * ah + bw * bh - intersection
        return intersection / union if union else 0.0

    def _remember_obstacles(self, detections, mask):
        if not config.OBSTACLE_MEMORY_ENABLED:
            return detections, mask
        if self.static_obstacle_mask is None or self.static_obstacle_mask.shape != mask.shape:
            self.static_obstacle_mask = np.zeros_like(mask)
        self.static_obstacle_mask = cv2.bitwise_or(self.static_obstacle_mask, mask)
        for detection in detections:
            if not any(
                self._box_iou(detection.box, old.box) >= 0.35
                for old in self.static_detections
            ):
                self.static_detections.append(detection)
        return self.static_detections, self.static_obstacle_mask

    def _camera_capture_loop(self):
        while not self._camera_stop.is_set():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self._camera_lock:
                    self._latest_camera_frame = frame

    def _get_latest_camera_frame(self):
        with self._camera_lock:
            if self._latest_camera_frame is None:
                return None
            return self._latest_camera_frame.copy()
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

    def _clear_ultrasonic_objects(self) -> None:
        self.ultrasonic_objects.clear()

    def _add_ultrasonic_obstacles(
        self,
        mask: np.ndarray,
        detections: List[Detection],
        robot_pos: Optional[Tuple[int, int]],
        robot_angle_deg: float,
        marker_size_px: float,
    ) -> None:
        """좌우 초음파 장애물을 센서별로 한 번 생성해 현재 맵에 반영한다."""
        if robot_pos is None:
            return

        reading = self.motor.get_ultrasonic_reading()
        height, width = mask.shape[:2]
        for side, detection in self.ultrasonic_objects.items():
            x, y, box_width, box_height = detection.box
            cv2.rectangle(mask, (x, y), (x + box_width, y + box_height), 255, -1)
            if not any(existing.label == detection.label for existing in detections):
                detections.append(detection)

        for side, distance_cm in (("LEFT", reading["left_cm"]), ("RIGHT", reading["right_cm"])):
            if side in self.ultrasonic_objects:
                continue
            if not 1.0 <= distance_cm <= config.ULTRASONIC_EMERGENCY_DISTANCE_CM:
                continue

            relative_angle = -30.0 if side == "LEFT" else 30.0
            distance_px = (distance_cm / 100.0) * ULTRASONIC_PIXELS_PER_METER
            heading = math.radians(robot_angle_deg + relative_angle)
            center_x = int(robot_pos[0] + math.cos(heading) * distance_px)
            center_y = int(robot_pos[1] - math.sin(heading) * distance_px)
            box_size = max(4, int(round(marker_size_px * 0.60)))
            x1 = int(np.clip(center_x - box_size // 2, 0, width - 1))
            y1 = int(np.clip(center_y - box_size // 2, 0, height - 1))
            x2 = int(np.clip(x1 + box_size, 0, width - 1))
            y2 = int(np.clip(y1 + box_size, 0, height - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            detection = Detection(
                box=(x1, y1, x2 - x1, y2 - y1),
                confidence=1.0,
                class_id=-1,
                label=f"ultrasonic {side.lower()}",
            )
            self.ultrasonic_objects[side] = detection
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            detections.append(detection)

    def _update_ultrasonic_emergency(self) -> bool:
        """5cm 이내 감지 시 0.7초 후진 후 현재 장애물 기준으로 재탐색한다."""
        if config.ENABLE_ULTRASONIC_REPLANNING != 1:
            return False

        reading = self.motor.get_ultrasonic_reading()
        closest_cm = min(reading["left_cm"], reading["right_cm"])
        now = time.monotonic()

        if now < self.ultrasonic_reverse_until:
            return True

        if closest_cm >= config.ULTRASONIC_REARM_DISTANCE_CM:
            self.ultrasonic_trigger_latched = False

        if (
            closest_cm <= config.ULTRASONIC_EMERGENCY_DISTANCE_CM
            and not self.ultrasonic_trigger_latched
        ):
            self.ultrasonic_trigger_latched = True
            self.ultrasonic_reverse_until = now + config.ULTRASONIC_REVERSE_SEC
            self.ultrasonic_replan_pending = True
            return True

        return False

    def _init_camera_focus(self):
        # 포커스를 흔들어 오토포커스 재기동 ('F' 키 기능)
        if not self.has_camera:
            print("[CAMERA] No camera connected.")
            return
        try:
            print("[CAMERA] Pressing 'F': running autofocus...")
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.cap.set(cv2.CAP_PROP_FOCUS, 0)
            time.sleep(0.08)
            self.cap.set(cv2.CAP_PROP_FOCUS, 100)
            time.sleep(0.08)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            print("[CAMERA] Autofocus complete")
        except Exception as e:
            print(f"[CAMERA FOCUS ERROR]: {e}")

    def _on_key_press(self, key):
        """pynput: OS 레벨 키 누름 이벤트 - 눌린 키를 집합에 추가"""
        try:
            ch = key.char.lower() if hasattr(key, 'char') and key.char else None
            if ch:
                if ch == "h" and ch not in self._pressed_keys:
                    if self.last_robot_pos is not None:
                        self.home_pos = tuple(self.last_robot_pos)
                        print(f"[HOME] Return position set to: {self.home_pos}")
                    else:
                        print("[HOME ERROR] Marker not detected; return position was not changed.")
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

        if event == cv2.EVENT_LBUTTONDOWN:
            self.audio.play("click")

        # 좌클릭 누름 (UI 버튼 및 옵션창 체크 - 단일 클릭만 처리하여 중복 토글 방지)
        if event == cv2.EVENT_LBUTTONDOWN:
            # 1. ArUco 5초 미감지 수동 전환 창의 버튼 클릭 체크 (Sudong_button.png 투명 제외 영역)
            if getattr(self, "show_lost_warning", False) and self.renderer.is_sudong_button_clicked(x, y):
                print("[UI] Manual-control button clicked; switching to manual mode.")
                self.show_lost_warning = False
                self._enter_manual_mode()
                return

            # 2. 좌측 상단 주행 모드 전환 버튼 클릭 체크 (driving_mode_switch_button.png 투명 제외 영역)
            if self.renderer.is_driving_mode_switch_clicked(x, y):
                print("[UI] Driving-mode button clicked; switching driving mode.")
                self._toggle_manual_mode()
                return

            clicked_btn = self.button_manager.handle_mouse_down(x, y)
            if clicked_btn is not None:
                if clicked_btn == "exit":
                    self.is_running = False
                elif clicked_btn == "stop":
                    self.motor.stop()
                    self.nav.reset()
                    print("[STOP] STOP button pressed")
                elif clicked_btn == "smoke":
                    self.motor.activate_smoke()
                    print("[SMOKE] Smoke button pressed; relay will run for 5 seconds.")
                elif clicked_btn == "recall":
                    target_home = self.home_pos if self.home_pos is not None else (640, 360)
                    if self.last_robot_pos is not None:
                        self._clear_ultrasonic_objects()
                        total_obs = self.latest_obstacle_mask
                        self.nav.set_goal(self.last_robot_pos, target_home, total_obs)
                    else:
                        print("[ERROR] Marker not detected")
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
                self._clear_ultrasonic_objects()
                total_obs = self.latest_obstacle_mask
                self.nav.set_goal(self.last_robot_pos, (cam_x, cam_y), total_obs)
            else:
                print(f"[WAITING] Marker not detected (target: {cam_x}, {cam_y})")

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
            print("[MODE] Switching to manual mode... (P: automatic mode | W/A/S/D: controls)")

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
            print("[MODE] Returning to automatic mode...")

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
            loop_start = time.monotonic()
            self.frame_count += 1
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
                frame = self._get_latest_camera_frame()
                if frame is None:
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
                now = time.monotonic()
                should_refresh_obstacles = (
                    self.last_detection_mask is None
                    or now - self.last_detection_time >= config.OBSTACLE_REFRESH_INTERVAL_SEC
                )
                if should_refresh_obstacles:
                    current_detections, current_mask = self.obstacle_detector.detect(warped_map)
                    self.last_detections = current_detections
                    self.last_detection_mask = current_mask
                    if config.OBSTACLE_MEMORY_ENABLED:
                        self._remember_obstacles(current_detections, current_mask)
                    else:
                        self.static_obstacle_mask = None
                        self.static_detections = []
                    self.last_detection_time = now
                    detections = current_detections
                    obstacle_mask = self.static_obstacle_mask if config.OBSTACLE_MEMORY_ENABLED else current_mask
                else:
                    detections = self.last_detections
                    obstacle_mask = (
                        self.static_obstacle_mask
                        if config.OBSTACLE_MEMORY_ENABLED and self.static_obstacle_mask is not None
                        else self.last_detection_mask.copy()
                    )
                self.latest_detections = detections
                self.latest_obstacle_mask = obstacle_mask

                robot_pos, robot_angle, corners, is_tracked, _ = self.tracker.detect(warped_map)
                marker_size_px = 0.0
                
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
                        marker_size_px = avg_side
                        self.nav.planner.robot_radius_px = max(
                            60,
                            int(avg_side * config.ROBOT_BODY_LENGTH_SCALE / 2.0),
                            int(avg_side * config.ROBOT_BODY_WIDTH_SCALE / 2.0),
                        )

                    if self.home_pos is None:
                        self.home_pos = robot_pos
                        print(f"[HOME] Initial position registered: {self.home_pos}")

                if config.ENABLE_ULTRASONIC_REPLANNING == 1:
                    self._add_ultrasonic_obstacles(
                        obstacle_mask,
                        detections,
                        robot_pos,
                        robot_angle,
                        marker_size_px,
                    )
                ultrasonic_reversing = self._update_ultrasonic_emergency()
                detected_alerts = {
                    item["label"]
                    for item in self.motor.get_onboard_detections()
                    if item.get("label") in {"claymore", "ally"}
                }
                now = time.monotonic()
                new_alerts = detected_alerts - self.onboard_alerts
                for alert in detected_alerts:
                    if (
                        alert in new_alerts
                        or now - self.last_onboard_sound_time.get(alert, 0.0) >= 3.0
                    ):
                        self.audio.play(alert)
                        self.last_onboard_sound_time[alert] = now
                if detected_alerts != self.onboard_alerts:
                    self.onboard_alerts = detected_alerts
                    self.onboard_alert_start_time = time.time() if detected_alerts else 0.0
                    if not detected_alerts:
                        self.last_onboard_sound_time.clear()
                self.latest_detections = detections
                self.latest_obstacle_mask = obstacle_mask

                # ArUco 5초 미감지 체크 (화면 안내창 플래그 설정)
                show_lost_warning = False
                if not self.manual_mode and not self.is_fading:
                    if time.time() - self.last_aruco_seen_time >= 5.0:
                        show_lost_warning = True
                self.show_lost_warning = show_lost_warning

                if (
                    self.ultrasonic_replan_pending
                    and not ultrasonic_reversing
                    and not self.manual_mode
                    and robot_pos is not None
                    and self.nav.final_goal is not None
                ):
                    self.nav.set_goal(
                        robot_pos,
                        self.nav.final_goal,
                        self.latest_obstacle_mask,
                    )
                    self.ultrasonic_replan_pending = False

                # YOLO 또는 초음파 가상 장애물이 현재 A* 경로를 침범하면 재탐색
                if (
                    not self.manual_mode
                    and robot_pos is not None
                    and self.nav.final_goal is not None
                    and self.nav.is_current_path_blocked(self.latest_obstacle_mask, robot_pos)
                ):
                    print("[A*] Obstacle blocks the current path; replanning.")
                    self.nav.set_goal(
                        robot_pos,
                        self.nav.final_goal,
                        self.latest_obstacle_mask,
                    )

                # 주행 제어 업데이트 (자동 모드일 때만 전송)
                if ultrasonic_reversing and not self.manual_mode and not self.is_fading:
                    v_left = -config.ULTRASONIC_REVERSE_PWM
                    v_right = -config.ULTRASONIC_REVERSE_PWM
                    nav_state = NavState.MOVING
                else:
                    v_left, v_right, nav_state = self.nav.update_control(robot_pos, robot_angle)
                if not self.manual_mode and not self.is_fading:
                    self.motor.send_speed(v_left, v_right)

                # 자동 모드 렌더링 준비
                if (
                    self.auto_canvas is None
                    or self.frame_count % max(1, config.RENDER_INTERVAL) == 0
                ):
                    self.auto_canvas = self.renderer.render_frame(
                    main_view=warped_map,
                    log_lines=self.logger.log_lines,
                    robot_pos=robot_pos,
                    robot_angle_deg=robot_angle,
                    waypoints=self.nav.waypoints,
                    current_wp_idx=self.nav.current_wp_idx,
                    goal_pos=self.nav.final_goal,
                    obstacle_mask=obstacle_mask,
                    detections=detections,
                    ping_pos=self.nav.ping_pos,
                    ping_start_time=self.nav.ping_start_time,
                    onboard_alerts=sorted(self.onboard_alerts),
                    onboard_alert_start_time=self.onboard_alert_start_time,
                    mouse_pos=(self.mouse_x, self.mouse_y),
                    status_text=nav_state.value if robot_pos is not None else "SEARCHING",
                    speed_info=(int(v_left), int(v_right)),
                    marker_corners=corners,
                    home_pos=self.home_pos,
                    is_blind=False,
                    ip_address=config.RASPBERRY_PI_IP,
                    pi_cam_frame=self.pi_cam.get_latest_frame(),
                    show_lost_warning=show_lost_warning,
                        fps=self.fps
                    )
                auto_canvas = self.auto_canvas

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
            self.fps_window_frames += 1
            elapsed = time.monotonic() - self.fps_window_start
            if elapsed >= 1.0:
                self.fps = self.fps_window_frames / elapsed
                self.fps_window_frames = 0
                self.fps_window_start = time.monotonic()
            
            if cv2.getWindowProperty(config.WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break
                
        self.cleanup()

    def cleanup(self):
        self.motor.stop()
        self.motor.close()
        self.pi_cam.stop()
        try:
            self._key_listener.stop()
        except Exception:
            pass
        if self.has_camera:
            self._camera_stop.set()
            if self._camera_thread is not None:
                self._camera_thread.join(timeout=1.0)
            self.cap.release()
        self.audio.close()
        cv2.destroyAllWindows()
        self.logger.restore()
        print("[SHUTDOWN] Program stopped (HSV settings saved).")


if __name__ == "__main__":
    try:
        app = StarcraftRCApp()
        app.run()
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Keyboard interrupt.")
    except Exception as e:
        print(f"\n[ERROR]: {e}")
