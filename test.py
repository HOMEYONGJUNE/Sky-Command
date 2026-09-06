from collections import deque
import heapq
import math
import socket
import time
import cv2
import numpy as np

# ==========================================
# 0. 네트워크(UDP) & PID 제어 설정
# ==========================================
RASPBERRY_PI_IP = "172.20.10.3"
UDP_PORT = 8080
TARGET_ARUCO_ID = 0  # 추적할 로봇의 ArUco 마커 ID

# UDP 소켓 생성
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# PID 게인 설정
Kp_angle = 3.0
Kd_angle = 0.2
Kp_dist = 0.8
Kd_dist = 0.05

prev_angle_error = 0.0
prev_dist_error = 0.0


def normalize_angle(angle_rad):
    """각도 오차를 -pi ~ pi 범위로 정규화"""
    while angle_rad > math.pi:
        angle_rad -= 2 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2 * math.pi
    return angle_rad


def reset_pid_errors():
    global prev_angle_error, prev_dist_error
    prev_angle_error = 0.0
    prev_dist_error = 0.0


def send_motor_command(v_left, v_right):
    v_l = int(np.clip(v_left, -255, 255))
    v_r = int(np.clip(v_right, -255, 255))
    msg = f"{v_l},{v_r}".encode("utf-8")
    try:
        udp_sock.sendto(msg, (RASPBERRY_PI_IP, UDP_PORT))
    except Exception as e:
        print(f"[UDP 전송 오류]: {e}")


def compute_pid_control(robot_pt, robot_angle_deg, goal_pt, dt):
    global prev_angle_error, prev_dist_error

    if robot_pt is None or goal_pt is None or dt <= 0:
        return 0, 0

    rx, ry = robot_pt
    gx, gy = goal_pt

    dx = gx - rx
    dy = gy - ry

    distance_px = math.sqrt(dx**2 + dy**2)

    if distance_px < 15:
        reset_pid_errors()
        return 0, 0

    target_angle_deg = math.degrees(math.atan2(-dy, dx)) % 360.0

    angle_error_rad = math.radians(target_angle_deg - robot_angle_deg)
    angle_error_rad = normalize_angle(angle_error_rad)

    angle_deriv = (angle_error_rad - prev_angle_error) / dt
    w = (Kp_angle * angle_error_rad) + (Kd_angle * angle_deriv)
    prev_angle_error = angle_error_rad

    dist_deriv = (distance_px - prev_dist_error) / dt
    v = (Kp_dist * distance_px) + (Kd_dist * dist_deriv)
    prev_dist_error = distance_px

    if abs(math.degrees(angle_error_rad)) > 25:
        v = 0

    v_left = v - (w * 50)
    v_right = v + (w * 50)

    return v_left, v_right


# ==========================================
# 1. 포즈 스무더 (Pose Smoother) 클래스
# ==========================================
class PoseSmoother:

    def __init__(self, window_size=5):
        self.x_history = deque(maxlen=window_size)
        self.y_history = deque(maxlen=window_size)
        self.sin_history = deque(maxlen=window_size)
        self.cos_history = deque(maxlen=window_size)

    def update(self, x, y, angle_deg):
        self.x_history.append(x)
        self.y_history.append(y)

        rad = math.radians(angle_deg)
        self.sin_history.append(math.sin(rad))
        self.cos_history.append(math.cos(rad))

        avg_x = int(np.mean(self.x_history))
        avg_y = int(np.mean(self.y_history))

        avg_sin = np.mean(self.sin_history)
        avg_cos = np.mean(self.cos_history)
        avg_angle_deg = math.degrees(math.atan2(avg_sin, avg_cos)) % 360.0

        return (avg_x, avg_y), avg_angle_deg

    def reset(self):
        self.x_history.clear()
        self.y_history.clear()
        self.sin_history.clear()
        self.cos_history.clear()


# ==========================================
# A* 경로 계획 및 Waypoint 추출 클래스
# ==========================================
class AStarPlanner:

    def __init__(self, grid_size=10, robot_radius_px=25):
        self.grid_size = grid_size
        self.robot_radius_px = robot_radius_px

    def plan_path(self, start_pt, goal_pt, obstacle_mask):
        """obstacle_mask(2D binary array) 기반 A* 경로 생성"""
        h, w = obstacle_mask.shape[:2]

        # 1. 안전 여유 공간 확장을 위한 장애물 Dilate (원본 해상도에서 팽창)
        kernel_size = self.robot_radius_px * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        inflated_mask = cv2.dilate(obstacle_mask, kernel)

        # [수정 1] 시작점/목표점 주변은 강제로 통행 가능 처리
        # -> 팽창 반경이 넓으면 로봇이나 클릭 지점 자체가 막혀서 A*가
        #    무조건 실패하는 문제를 방지 (가장 흔한 "경로 없음" 원인)
        clear_radius = self.grid_size + 2
        cv2.circle(
            inflated_mask, (int(start_pt[0]), int(start_pt[1])), clear_radius, 0, -1
        )
        cv2.circle(
            inflated_mask, (int(goal_pt[0]), int(goal_pt[1])), clear_radius, 0, -1
        )

        # --- 디버그: 통행 가능 비율 + 마스크 시각화 ---
        free_ratio = 1.0 - (inflated_mask > 0).sum() / inflated_mask.size
        print(
            f"[A* Planner]: robot_radius_px={self.robot_radius_px}, "
            f"grid_size={self.grid_size}, 통행가능비율={free_ratio * 100:.1f}%"
        )
        cv2.imshow("A* Inflated Mask Debug (white=blocked)", inflated_mask)

        # 2. 픽셀 좌표 -> 그리드 좌표 변환
        start_node = (
            int(start_pt[1] // self.grid_size),
            int(start_pt[0] // self.grid_size),
        )
        goal_node = (
            int(goal_pt[1] // self.grid_size),
            int(goal_pt[0] // self.grid_size),
        )

        grid_h, grid_w = h // self.grid_size, w // self.grid_size

        if not (
            0 <= start_node[0] < grid_h and 0 <= start_node[1] < grid_w
        ) or not (0 <= goal_node[0] < grid_h and 0 <= goal_node[1] < grid_w):
            print("[A* Planner]: 시작점 또는 목표점이 맵 범위를 벗어났습니다.")
            return []

        # 8방향 이동 (대각선 포함)
        motion = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, 1.414),
            (-1, 1, 1.414),
            (1, -1, 1.414),
            (1, 1, 1.414),
        ]

        def cell_blocked(r, c):
            """그리드 셀(r, c) 중심 픽셀이 장애물(팽창된 마스크)에 있는지 확인"""
            if not (0 <= r < grid_h and 0 <= c < grid_w):
                return True
            py = r * self.grid_size + self.grid_size // 2
            px = c * self.grid_size + self.grid_size // 2
            if py >= h or px >= w:
                return True
            return inflated_mask[py, px] > 0

        open_set = []
        heapq.heappush(
            open_set,
            (
                0 + math.hypot(start_node[0] - goal_node[0], start_node[1] - goal_node[1]),
                start_node,
            ),
        )

        came_from = {}
        g_score = {start_node: 0}
        visited = set()

        path_found = False

        while open_set:
            _, current = heapq.heappop(open_set)
            if current in visited:
                continue
            visited.add(current)

            if current == goal_node:
                path_found = True
                break

            for dr, dc, cost in motion:
                neighbor = (current[0] + dr, current[1] + dc)

                if cell_blocked(neighbor[0], neighbor[1]):
                    continue

                # [수정 2] 대각선 이동 시 코너컷 방지
                # 대각선 양옆의 정방향 두 칸도 free여야 통과 허용
                # (둘 다 막혀 있으면 실제 로봇 몸체는 그 틈으로 못 지나감)
                if dr != 0 and dc != 0:
                    side1_blocked = cell_blocked(current[0] + dr, current[1])
                    side2_blocked = cell_blocked(current[0], current[1] + dc)
                    if side1_blocked or side2_blocked:
                        continue

                tentative_g = g_score[current] + cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h_cost = math.hypot(
                        neighbor[0] - goal_node[0], neighbor[1] - goal_node[1]
                    )
                    f_score = tentative_g + h_cost
                    heapq.heappush(open_set, (f_score, neighbor))

        if not path_found:
            return []

        grid_path = []
        curr = goal_node
        while curr in came_from:
            grid_path.append(curr)
            curr = came_from[curr]
        grid_path.append(start_node)
        grid_path.reverse()

        pixel_path = [
            (
                int(c * self.grid_size + self.grid_size // 2),
                int(r * self.grid_size + self.grid_size // 2),
            )
            for r, c in grid_path
        ]
        return pixel_path

    @staticmethod
    def simplify_path(path_pts):
        """직선상의 노드들을 제거하고 방향이 꺾이는 지점(Waypoint)만 추출"""
        if len(path_pts) <= 2:
            return path_pts

        waypoints = [path_pts[0]]
        for i in range(1, len(path_pts) - 1):
            p_prev = waypoints[-1]
            p_curr = path_pts[i]
            p_next = path_pts[i + 1]

            dir1 = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
            dir2 = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])

            angle1 = math.atan2(dir1[1], dir1[0])
            angle2 = math.atan2(dir2[1], dir2[0])

            if abs(normalize_angle(angle1 - angle2)) > math.radians(10):
                waypoints.append(p_curr)

        waypoints.append(path_pts[-1])
        return waypoints


# ==========================================
# 2. 전역 변수 및 콜백 함수
# ==========================================
def nothing(value):
    pass


latest_hsv_map = None
drawing = False
ix, iy = -1, -1
fx, fy = -1, -1
bbox_selected = False

matrix = None
MAP_WIDTH = 1000
MAP_HEIGHT = 1000
mouse_x, mouse_y = 0, 0
goal_pt = None

# Waypoint 관리용 전역 변수
planner = AStarPlanner(grid_size=15, robot_radius_px=30)
current_path = []
waypoints = []
current_wp_index = 0
replan_requested = False

square_px = 0.0
real_cm = 0.0
cm_scale = 0.0


def draw_rectangle(event, x, y, flags, param):
    global ix, iy, fx, fy, drawing, bbox_selected

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        bbox_selected = False

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            fx, fy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx, fy = x, y
        bbox_selected = True
        print(f"\n[선택 영역]: ({ix}, {iy}) -> ({fx}, {fy})")
        print("-> [Enter] 키를 누르면 영역 투시 변환을 진행합니다.")


def track_coordinates(event, x, y, flags, param):
    global mouse_x, mouse_y, goal_pt, latest_hsv_map, replan_requested

    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y

    elif event == cv2.EVENT_LBUTTONDOWN:
        goal_pt = (x, y)
        reset_pid_errors()
        replan_requested = True
        cart_gx = x
        cart_gy = MAP_HEIGHT - y
        print(f"[Goal 지정]: 화면 ({x}, {y}) | 카테시안 ({cart_gx}, {cart_gy}) px")

        if latest_hsv_map is not None:
            h, w = latest_hsv_map.shape[:2]
            if 0 <= x < w and 0 <= y < h:
                hv, sv, vv = latest_hsv_map[y, x]
                print(f" -> [클릭 지점 HSV]: H={int(hv)}, S={int(sv)}, V={int(vv)}")


def detect_reference_square(warped_img, red_mask):
    contours, _ = cv2.findContours(
        red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best_square_px = 0.0
    best_cnt = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.035 * peri, True)

        if 4 <= len(approx) <= 6:
            rect = cv2.minAreaRect(cnt)
            (w_rect, h_rect) = rect[1]

            if w_rect == 0 or h_rect == 0:
                continue

            aspect_ratio = float(w_rect) / h_rect
            if 0.7 <= aspect_ratio <= 1.4:
                avg_side = (w_rect + h_rect) / 2.0
                if avg_side > best_square_px:
                    best_square_px = avg_side
                    best_cnt = cnt

    return best_square_px, best_cnt


# ==========================================
# 3. 카메라 및 ArUco, UI 창 초기화
# ==========================================
CAMERA_INDEX = 0

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"{CAMERA_INDEX}번 카메라를 열 수 없습니다. 연결 상태를 확인해주세요.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

try:
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    use_new_api = True
except AttributeError:
    aruco_params = cv2.aruco.DetectorParameters_create()
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    use_new_api = False

smoother = PoseSmoother(window_size=5)

cv2.namedWindow("HSV Controls")
cv2.createTrackbar("Blue H Min", "HSV Controls", 90, 179, nothing)
cv2.createTrackbar("Blue H Max", "HSV Controls", 135, 179, nothing)
cv2.createTrackbar("Blue S Min", "HSV Controls", 70, 255, nothing)
cv2.createTrackbar("Blue V Min", "HSV Controls", 40, 255, nothing)

cv2.createTrackbar("Red H1 Max", "HSV Controls", 10, 179, nothing)
cv2.createTrackbar("Red H2 Min", "HSV Controls", 170, 179, nothing)
cv2.createTrackbar("Red S Min", "HSV Controls", 100, 255, nothing)
cv2.createTrackbar("Red V Min", "HSV Controls", 100, 255, nothing)

cv2.namedWindow("Drag Map Region", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Drag Map Region", 1280, 720)
cv2.setMouseCallback("Drag Map Region", draw_rectangle)

morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

print("=" * 60)
print(" [실행 안내]")
print(" 1. 카메라 화면에서 맵 영역을 드래그 후 [Enter]를 누르세요.")
print(" 2. [HSV Controls] 트랙바로 벽과 빨간색 상자를 튜닝하세요.")
print(" 3. 'p' 키: HSV 값 출력 | 'r' 키: 맵 영역 재설정 | 'q' 키: 프로그램 종료")
print("=" * 60)

last_loop_time = time.time()

# ==========================================
# 4. 실시간 메인 루프
# ==========================================
while True:
    current_time = time.time()
    dt = current_time - last_loop_time
    last_loop_time = current_time

    ret, frame = cap.read()
    if not ret:
        print("카메라 프레임을 불러올 수 없습니다.")
        break

    # [1단계] 영역 드래그
    if matrix is None:
        display_frame = frame.copy()

        if ix != -1 and fx != -1:
            cv2.rectangle(display_frame, (ix, iy), (fx, fy), (0, 255, 0), 2)

        cv2.putText(
            display_frame,
            "Drag map area & Press [Enter]",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Drag Map Region", display_frame)

    # [2단계] Top-View 변환 및 주행 제어
    else:
        warped_map = cv2.warpPerspective(frame, matrix, (MAP_WIDTH, MAP_HEIGHT))
        display_img = warped_map.copy()

        hsv_map = cv2.cvtColor(warped_map, cv2.COLOR_BGR2HSV)
        latest_hsv_map = hsv_map.copy()

        # --- 파란색 마스크 ---
        blue_h_min = cv2.getTrackbarPos("Blue H Min", "HSV Controls")
        blue_h_max = cv2.getTrackbarPos("Blue H Max", "HSV Controls")
        blue_s_min = cv2.getTrackbarPos("Blue S Min", "HSV Controls")
        blue_v_min = cv2.getTrackbarPos("Blue V Min", "HSV Controls")
        blue_h_min = min(blue_h_min, blue_h_max)

        lower_blue = np.array([blue_h_min, blue_s_min, blue_v_min], dtype=np.uint8)
        upper_blue = np.array([blue_h_max, 255, 255], dtype=np.uint8)

        blue_mask = cv2.inRange(hsv_map, lower_blue, upper_blue)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, morph_kernel, iterations=1)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, morph_kernel, iterations=2)

        # --- 빨간색 마스크 ---
        red_h1_max = cv2.getTrackbarPos("Red H1 Max", "HSV Controls")
        red_h2_min = cv2.getTrackbarPos("Red H2 Min", "HSV Controls")
        red_s_min = cv2.getTrackbarPos("Red S Min", "HSV Controls")
        red_v_min = cv2.getTrackbarPos("Red V Min", "HSV Controls")

        lower_red1 = np.array([0, red_s_min, red_v_min], dtype=np.uint8)
        upper_red1 = np.array([red_h1_max, 255, 255], dtype=np.uint8)

        lower_red2 = np.array([red_h2_min, red_s_min, red_v_min], dtype=np.uint8)
        upper_red2 = np.array([179, 255, 255], dtype=np.uint8)

        mask1 = cv2.inRange(hsv_map, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_map, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, morph_kernel, iterations=1)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, morph_kernel, iterations=2)

        overlay = display_img.copy()
        overlay[blue_mask > 0] = (0, 0, 255)
        display_img = cv2.addWeighted(display_img, 0.7, overlay, 0.3, 0)

        detected_px, cnt = detect_reference_square(warped_map, red_mask)
        if detected_px > 0:
            square_px = detected_px
            if real_cm > 0:
                cm_scale = real_cm / square_px

            cv2.drawContours(display_img, [cnt], -1, (0, 255, 0), 2)
            cv2.putText(
                display_img,
                f"Red Ref Sq: {square_px:.1f}px",
                (cnt[0][0][0], max(15, cnt[0][0][1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
            )

        # --- ArUco 마커 검출 (특정 ID 지정) ---
        gray = cv2.cvtColor(warped_map, cv2.COLOR_BGR2GRAY)
        if use_new_api:
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, aruco_dict, parameters=aruco_params
            )

        robot_screen_pt = None
        robot_angle_deg = 0.0

        if ids is not None and len(corners) > 0:
            ids_flat = ids.flatten()
            if TARGET_ARUCO_ID in ids_flat:
                idx = np.where(ids_flat == TARGET_ARUCO_ID)[0][0]
                marker_corners = corners[idx][0]

                raw_cx = int(np.mean(marker_corners[:, 0]))
                raw_cy = int(np.mean(marker_corners[:, 1]))

                p0, p1 = marker_corners[0], marker_corners[1]
                dx_m, dy_m = p1[0] - p0[0], p1[1] - p0[1]
                angle_rad = np.arctan2(-dy_m, dx_m)
                raw_angle_deg = np.degrees(angle_rad) % 360

                robot_screen_pt, robot_angle_deg = smoother.update(
                    raw_cx, raw_cy, raw_angle_deg
                )

                center_x, center_y = robot_screen_pt
                cart_rx = center_x
                cart_ry = MAP_HEIGHT - center_y

                cv2.circle(display_img, robot_screen_pt, 8, (255, 0, 0), -1)
                arrow_len = 35
                arrow_end_x = int(
                    center_x + arrow_len * np.cos(np.radians(robot_angle_deg))
                )
                arrow_end_y = int(
                    center_y - arrow_len * np.sin(np.radians(robot_angle_deg))
                )
                cv2.arrowedLine(
                    display_img,
                    robot_screen_pt,
                    (arrow_end_x, arrow_end_y),
                    (255, 0, 0),
                    2,
                    tipLength=0.3,
                )

                info_text = f"ROBOT: ({cart_rx}, {cart_ry}) | {robot_angle_deg:.1f}deg"
                cv2.putText(
                    display_img,
                    info_text,
                    (center_x - 70, center_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

        # ==========================================
        # 1. A* 경로 탐색 (클릭 시 1회만 트리거)
        # ==========================================
        if replan_requested and goal_pt is not None and robot_screen_pt is not None:
            current_path = planner.plan_path(robot_screen_pt, goal_pt, blue_mask)
            waypoints = planner.simplify_path(current_path)
            current_wp_index = 0
            replan_requested = False

            if len(waypoints) > 0:
                print(f"[A* Planner]: 경로 생성 완료! Waypoints 수: {len(waypoints)}")
            else:
                print("[A* Planner]: 경로를 찾을 수 없습니다.")

        # ==========================================
        # 2. Target Waypoint 지정 및 거리 계산
        # ==========================================
        active_target_pt = None

        if robot_screen_pt is not None and waypoints and current_wp_index < len(waypoints):
            active_target_pt = waypoints[current_wp_index]

            if active_target_pt is not None:
                wp_dx = active_target_pt[0] - robot_screen_pt[0]
                wp_dy = active_target_pt[1] - robot_screen_pt[1]
                wp_dist = math.sqrt(wp_dx**2 + wp_dy**2)

                if wp_dist < 15:
                    current_wp_index += 1
                    reset_pid_errors()
                    if current_wp_index < len(waypoints):
                        print(f"-> Waypoint {current_wp_index} 도달! 다음 목표: {waypoints[current_wp_index]}")
                    else:
                        print("-> 최종 Goal 지점 도달 완료!")

        # --- PID 제어 계산 및 명령 전송 ---
        if active_target_pt is not None and robot_screen_pt is not None:
            v_l, v_r = compute_pid_control(
                robot_screen_pt, robot_angle_deg, active_target_pt, dt
            )
            send_motor_command(v_l, v_r)
        else:
            send_motor_command(0, 0)

        # ==========================================
        # 경로 및 Waypoint 시각화
        # ==========================================
        path_pts_to_draw = []
        if robot_screen_pt is not None:
            path_pts_to_draw.append(robot_screen_pt)
        if waypoints:
            path_pts_to_draw.extend(waypoints)
        elif current_path:
            path_pts_to_draw.extend(current_path)

        if len(path_pts_to_draw) >= 2:
            pts_arr = np.array(path_pts_to_draw, dtype=np.int32)
            # 글로우 이중선 효과
            overlay_path = display_img.copy()
            cv2.polylines(overlay_path, [pts_arr], False, (255, 230, 0), 5, cv2.LINE_AA)
            display_img = cv2.addWeighted(display_img, 0.7, overlay_path, 0.3, 0)
            cv2.polylines(display_img, [pts_arr], False, (255, 255, 120), 2, cv2.LINE_AA)

        for idx, wp in enumerate(waypoints):
            color = (0, 255, 0) if idx == current_wp_index else (0, 220, 255)
            cv2.circle(display_img, wp, 5, color, -1, cv2.LINE_AA)
            cv2.putText(
                display_img, f"W{idx}", (wp[0] + 6, wp[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA
            )

        # UI 오버레이
        cartesian_x = mouse_x
        cartesian_y = MAP_HEIGHT - mouse_y

        cv2.drawMarker(
            display_img, (mouse_x, mouse_y), (0, 0, 255),
            markerType=cv2.MARKER_CROSS, markerSize=15, thickness=1,
        )

        text_x = mouse_x + 15 if mouse_x + 160 < MAP_WIDTH else mouse_x - 160
        text_y = mouse_y - 10 if mouse_y - 20 > 0 else mouse_y + 20
        cv2.putText(
            display_img, f"({cartesian_x}, {cartesian_y}) px", (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
        )

        scale_info = f"Scale: {cm_scale:.4f} cm/px" if cm_scale > 0 else "Scale: N/A"
        cv2.putText(
            display_img, scale_info, (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA,
        )

        if goal_pt is not None:
            cv2.circle(display_img, goal_pt, 8, (0, 0, 255), -1)
            cv2.putText(
                display_img, "GOAL", (goal_pt[0] + 10, goal_pt[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA,
            )

            if robot_screen_pt is not None:
                dist_px = np.sqrt(
                    (goal_pt[0] - robot_screen_pt[0]) ** 2
                    + (goal_pt[1] - robot_screen_pt[1]) ** 2
                )
                mid_x = (robot_screen_pt[0] + goal_pt[0]) // 2
                mid_y = (robot_screen_pt[1] + goal_pt[1]) // 2

                dist_text = (
                    f"Dist: {dist_px * cm_scale:.1f}cm" if cm_scale > 0
                    else f"Dist: {dist_px:.1f}px"
                )
                cv2.putText(
                    display_img, dist_text, (mid_x, mid_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA,
                )

        cv2.imshow("Warped Top-View Map", display_img)
        cv2.imshow("Blue Maze Mask", blue_mask)
        cv2.imshow("Red Square Mask", red_mask)

    # ----------------------------------------------------
    # 키보드 입력 처리
    # ----------------------------------------------------
    key = cv2.waitKey(20) & 0xFF

    if key in [13, 32] and bbox_selected and matrix is None:
        x1, x2 = min(ix, fx), max(ix, fx)
        y1, y2 = min(iy, fy), max(iy, fy)

        box_w = x2 - x1
        box_h = y2 - y1

        if box_w > 10 and box_h > 10:
            MAP_HEIGHT = int(MAP_WIDTH * (box_h / box_w))

            src_pts = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            dst_pts = np.float32(
                [[0, 0], [MAP_WIDTH, 0], [MAP_WIDTH, MAP_HEIGHT], [0, MAP_HEIGHT]]
            )

            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

            print("\n" + "=" * 50)
            try:
                user_input = input("기준 빨간색 사각형의 실제 한 변 길이(cm)를 입력하세요: ")
                real_cm = float(user_input)
                print(f"-> 설정 완료: {real_cm} cm")
            except ValueError:
                print("-> 입력 오류. 픽셀 단위를 유지합니다.")
                real_cm = 0.0
            print("=" * 50 + "\n")

            cv2.destroyWindow("Drag Map Region")
            cv2.namedWindow("Warped Top-View Map", cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                "Warped Top-View Map", 1000, int(1000 * (MAP_HEIGHT / MAP_WIDTH))
            )
            cv2.setMouseCallback("Warped Top-View Map", track_coordinates)

    elif key == ord("p"):
        print("\n===== [현재 파란색 HSV 범위] =====")
        print(f"H: {blue_h_min} ~ {blue_h_max} | S: {blue_s_min} | V: {blue_v_min}")
        print("===== [현재 빨간색 HSV 범위] =====")
        print(
            f"H1: 0 ~ {red_h1_max} | H2: {red_h2_min} ~ 179 | S: {red_s_min} | V: {red_v_min}"
        )
        print("==================================\n")

    elif key == ord("r"):
        send_motor_command(0, 0)
        matrix = None
        bbox_selected = False
        ix, iy, fx, fy = -1, -1, -1, -1
        goal_pt = None
        real_cm, square_px, cm_scale = 0.0, 0.0, 0.0
        current_path = []
        waypoints = []
        current_wp_index = 0
        replan_requested = False

        smoother.reset()
        reset_pid_errors()
        cv2.destroyAllWindows()

        cv2.namedWindow("HSV Controls")
        cv2.createTrackbar("Blue H Min", "HSV Controls", 90, 179, nothing)
        cv2.createTrackbar("Blue H Max", "HSV Controls", 135, 179, nothing)
        cv2.createTrackbar("Blue S Min", "HSV Controls", 70, 255, nothing)
        cv2.createTrackbar("Blue V Min", "HSV Controls", 40, 255, nothing)

        cv2.createTrackbar("Red H1 Max", "HSV Controls", 10, 179, nothing)
        cv2.createTrackbar("Red H2 Min", "HSV Controls", 170, 179, nothing)
        cv2.createTrackbar("Red S Min", "HSV Controls", 100, 255, nothing)
        cv2.createTrackbar("Red V Min", "HSV Controls", 100, 255, nothing)

        cv2.namedWindow("Drag Map Region", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Drag Map Region", 1280, 720)
        cv2.setMouseCallback("Drag Map Region", draw_rectangle)

    elif key == ord("q"):
        send_motor_command(0, 0)
        break

# 자원 해제
send_motor_command(0, 0)
udp_sock.close()
cap.release()
cv2.destroyAllWindows()