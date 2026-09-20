"""
설정 파일: RC카 자율주행, 영상 처리, 경로 계획, 통신, UI 파라미터 관리
"""
import os
import cv2

# 기본 경로 및 해상도
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.path.join(BASE_DIR, "ui")
UI_IMAGE_PATH = os.path.join(UI_DIR, "ui.png")

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
WINDOW_TITLE = "SKY Commander"

# 라즈베리파이 네트워크 및 카메라 설정
RASPBERRY_PI_IP = "192.168.0.10"
UDP_PORT = 8080
UDP_TIMEOUT_SEC = 0.2
RASPBERRY_PI_CAM_PORT = 8081  # 라즈베리파이 온보드 카메라 HTTP MJPEG 스트리밍 포트
RASPBERRY_PI_CAM_INDEX = 1    # 라즈베리파이 카메라 연결 포트/인덱스 (기본: 1)
PI_CAM_WIDTH = 640            # 온보드 카메라 기본 가로 해상도 (고화질: 1280)
PI_CAM_HEIGHT = 480           # 온보드 카메라 기본 세로 해상도 (고화질: 720)

# 관제 PC 상단 카메라 및 맵 설정
CAMERA_INDEX = 0
CAMERA_FRAME_WIDTH = 1280
CAMERA_FRAME_HEIGHT = 720
MAP_WIDTH = 1280
MAP_HEIGHT = 720

# ArUco 마커 설정
ARUCO_DICTIONARY_ID = cv2.aruco.DICT_4X4_50
TARGET_MARKER_ID = None

# 트래킹 파라미터
POSE_SMOOTH_WINDOW = 3
MAX_COASTING_FRAMES = 2
SMOOTH_ALPHA = 0.85

# 파란색 장애물 HSV 및 팽창 설정 (HSV 조절창에서 좌우/상하 실시간 조절 가능)
DEFAULT_BLUE_H_MIN = 95
DEFAULT_BLUE_H_MAX = 135
DEFAULT_BLUE_S_MIN = 70
DEFAULT_BLUE_V_MIN = 40
DEFAULT_BLUE_RADIUS_X = 16   # 파란색 장애물 좌우(X) 회피 팽창 기본값 (px)
DEFAULT_BLUE_RADIUS_Y = 60   # 파란색 장애물 상하(Y) 회피 팽창 기본값 (px)
BLUE_OBSTACLE_DILATE_PX = 48  # 기존 호환용 기본 반경

# 초록색 주행 가능 영역 HSV 기본값
DEFAULT_GREEN_H_MIN = 35
DEFAULT_GREEN_H_MAX = 85
DEFAULT_GREEN_S_MIN = 40
DEFAULT_GREEN_V_MIN = 40

# A* 경로 탐색 설정 (ArUco 가로 길이 기준 직사각형 차체 및 안전 회피 마진)
ROBOT_RADIUS_PX = 75
GRID_CELL_SIZE = 12
PATH_SMOOTHING_ENABLED = True
ASTAR_EXTRA_SAFETY_MARGIN_PX = 28
PATH_WAYPOINT_SPACING_PX = 24
YOLO_INFERENCE_INTERVAL = 3
OBSTACLE_REFRESH_INTERVAL_SEC = 1.0
OBSTACLE_BOX_SCALE = 1.5
RENDER_INTERVAL = 2
OBSTACLE_MEMORY_ENABLED = 1
# 차체 길이는 ArUco 마커 가로 길이의 3.2배, 폭은 1.6배
ROBOT_BODY_LENGTH_SCALE = 3.7
ROBOT_BODY_WIDTH_SCALE = 2.5
ARUCO_FAST_MISS_AFTER = 5
ARUCO_ENHANCED_SCAN_INTERVAL = 3

# 초음파 센서 가상 장애물 및 A* 재탐색: 1=활성화, 0=비활성화
ENABLE_ULTRASONIC_REPLANNING = 1

# 모터 PWM 설정
MAX_PWM_SPEED = 100
MIN_FORWARD_PWM = 70
MIN_ROTATION_PWM = 85    # 회전 최소 PWM: 85로 추가 하향 (회전 토크 완화)
MAX_ROTATION_PWM = 105   # 회전 최대 PWM: 105로 추가 하향

# 스텝 주행 설정
STEP_DRIVE_ENABLED = False
STEP_MOVE_SEC = 0.22
STEP_PAUSE_SEC = 0.09

# PID 및 주행 판정 기준 (회전력 및 게인 부드럽게 완화)
KP_ANGLE = 1.0          # 각도 비례 게인: 1.0으로 완화
KD_ANGLE = 0.16         # 각속도 댐핑 게인: 오실레이션 흡수 유지
KP_DIST = 0.65
KD_DIST = 0.08
DIFF_STEER_WEIGHT = 20.0 # 조향 편차 가중치: 20.0으로 완화

ROTATION_THRESHOLD_DEG = 30.0 # 호환용 회전 임계값
FORWARD_ALIGNMENT_THRESHOLD_DEG = 15.0 # 경로 방향 오차가 이 값 미만일 때만 전진
WAYPOINT_REACH_DIST_PX = 18.0
FINAL_GOAL_REACH_DIST_PX = 30.0

# UI 콘솔 텍스트 영역
TEXT_BOX_X1 = 1090
TEXT_BOX_Y1 = 20
TEXT_BOX_X2 = 1280
TEXT_BOX_Y2 = 520
MAX_CONSOLE_LINES = 50

# 우클릭 핑 애니메이션
PING_DURATION_SEC = 0.8
PING_MAX_RADIUS = 28
