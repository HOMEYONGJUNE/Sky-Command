import socket
import json
import time
import threading
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("[WARNING] OpenCV(cv2) unavailable - camera streaming disabled")
    CV2_AVAILABLE = False

try:
    from gpiozero import LED, PWMLED
    GPIO_AVAILABLE = True
except ImportError:
    print("[WARNING] gpiozero unavailable (simulation mode)")
    GPIO_AVAILABLE = False

try:
    from gpiozero import DistanceSensor
    ULTRASONIC_AVAILABLE = True
except ImportError:
    DistanceSensor = None
    ULTRASONIC_AVAILABLE = False

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False


# ==========================================
# 라즈베리 파이 카메라 HTTP MJPEG 스트리머
# ==========================================
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class StreamingHandler(BaseHTTPRequestHandler):
    streamer = None

    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path in ("/", "/stream.mjpg", "/video_feed"):
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                self.wfile.flush()
                while True:
                    frame = None
                    if StreamingHandler.streamer is not None:
                        frame = StreamingHandler.streamer.get_jpeg_frame()

                    if frame is not None:
                        self.wfile.write(b"--frame\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(frame)))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                    time.sleep(0.033)  # ~30 FPS
            except Exception:
                pass
        else:
            self.send_error(404)
            self.end_headers()


class PiCameraStreamer:
    """라즈베리 파이 5 카메라(Picamera2 / CSI / USB) 영상 HTTP MJPEG 스트리밍 송출"""
    def __init__(self, camera_index: int = 0, http_port: int = 8081, width: int = 640, height: int = 480, quality: int = 80):
        self.camera_index = camera_index
        self.http_port = http_port
        self.width = width
        self.height = height
        self.quality = max(10, min(100, quality))
        self.running = False
        self.latest_jpeg = None
        self.lock = threading.Lock()
        self.server = None
        self.thread = None
        self.server_thread = None
        self.picam2 = None
        self.cap = None

        # 기본 대기 화면 생성
        self._init_placeholder_frame()

    def _init_placeholder_frame(self):
        if CV2_AVAILABLE:
            import numpy as np
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            img[:] = (20, 20, 28)
            cv2.putText(img, "RC-CAM INITIALIZING...", (20, self.height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 25), 1, cv2.LINE_AA)
            _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
            self.latest_jpeg = enc.tobytes()

    def _init_camera(self):
        # 1. 라즈베리 파이 5 전용 Picamera2 시도 (CSI 포트 ov5647 등 공식 카메라)
        if PICAMERA2_AVAILABLE:
            try:
                print("[CAMERA] Initializing Picamera2...")
                p2 = Picamera2()
                cfg = p2.create_video_configuration(main={"size": (self.width, self.height), "format": "BGR888"})
                p2.configure(cfg)
                p2.start()
                time.sleep(0.5)
                test_arr = p2.capture_array()
                if test_arr is not None and test_arr.size > 0:
                    print(f"[CAMERA] Picamera2 connected: {self.width}x{self.height}")
                    self.picam2 = p2
                    return "picam2"
                p2.stop()
                p2.close()
            except Exception as e:
                print(f"[CAMERA] Picamera2 failed ({e}); trying OpenCV VideoCapture")

        # 2. OpenCV VideoCapture 시도 (USB 카메라 또는 V4L2)
        if CV2_AVAILABLE:
            for idx in [self.camera_index, 1, 0, 2]:
                try:
                    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
                    if not cap.isOpened():
                        cap = cv2.VideoCapture(idx)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        ret, test_f = cap.read()
                        if ret and test_f is not None:
                            print(f"[CAMERA] OpenCV device {idx} connected: {self.width}x{self.height}")
                            self.cap = cap
                            return "opencv"
                        cap.release()
                except Exception:
                    pass

        print("[CAMERA WARNING] No usable camera found; retrying")
        return None

    def _capture_loop(self):
        cam_type = self._init_camera()
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]

        while self.running:
            if cam_type == "picam2" and self.picam2 is not None:
                try:
                    frame = self.picam2.capture_array()
                    if frame is not None and frame.size > 0:
                        # 1. 180도 회전 (뒤집힌 온보드 카메라 화면 정상화)
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                        # 2. 색상 보정: Picamera2 RGB -> BGR 변환 (하늘색 피부 -> 정상 피부색, 주황색 꼬깔 -> 파란색 정상화)
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                        _, jpeg = cv2.imencode(".jpg", frame, encode_param)
                        with self.lock:
                            self.latest_jpeg = jpeg.tobytes()
                        time.sleep(0.02)
                        continue
                except Exception as e:
                    print(f"[CAMERA Picamera2 ERROR]: {e}")
                    cam_type = None

            elif cam_type == "opencv" and self.cap is not None:
                try:
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        if frame.shape[1] != self.width or frame.shape[0] != self.height:
                            frame = cv2.resize(frame, (self.width, self.height))
                        # 1. 180도 회전
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                        # 2. 색상 보정
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                        _, jpeg = cv2.imencode(".jpg", frame, encode_param)
                        with self.lock:
                            self.latest_jpeg = jpeg.tobytes()
                        time.sleep(0.02)
                        continue
                except Exception as e:
                    print(f"[CAMERA OpenCV ERROR]: {e}")
                    cam_type = None

            # 카메라가 끊겼거나 없는 경우 2초마다 재시도
            time.sleep(2.0)
            if self.running and cam_type is None:
                cam_type = self._init_camera()

    def get_jpeg_frame(self):
        with self.lock:
            return self.latest_jpeg

    def start(self):
        if not CV2_AVAILABLE:
            return
        self.running = True
        StreamingHandler.streamer = self

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

        try:
            self.server = ThreadedHTTPServer(("0.0.0.0", self.http_port), StreamingHandler)
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()
            print(f"[CAMERA] Streamer started: http://0.0.0.0:{self.http_port}/stream.mjpg")
        except Exception as e:
            print(f"[CAMERA SERVER ERROR]: {e}")

    def stop(self):
        self.running = False
        if self.server:
            try:
                self.server.shutdown()
                self.server.server_close()
            except Exception:
                pass
        if self.picam2:
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception:
                pass
        if self.cap and self.cap.isOpened():
            self.cap.release()
        print("[CAMERA] Streamer stopped")


# GPIO 핀 설정
if GPIO_AVAILABLE:
    # 앞바퀴
    FRONT_LEFT_DIR = LED(18)
    FRONT_LEFT_PWM = PWMLED(19)
    FRONT_RIGHT_DIR = LED(20)
    FRONT_RIGHT_PWM = PWMLED(21)

    # 뒷바퀴
    BACK_LEFT_DIR = LED(22)
    BACK_LEFT_PWM = PWMLED(23)
    BACK_RIGHT_DIR = LED(24)
    BACK_RIGHT_PWM = PWMLED(25)

    FRONT_RIGHT_BOOST = 1.2  # 오른쪽 앞바퀴 출력 보정
    FRONT_LEFT_BOOST = 1.2  # 왼쪽 앞바퀴 출력 보정


    ALL_DEVICES = [
        FRONT_LEFT_PWM, FRONT_RIGHT_PWM, BACK_LEFT_PWM, BACK_RIGHT_PWM,
        FRONT_LEFT_DIR, FRONT_RIGHT_DIR, BACK_LEFT_DIR, BACK_RIGHT_DIR
    ]
else:
    FRONT_RIGHT_BOOST = 1.2
    FRONT_LEFT_BOOST = 1.2  # 왼쪽 앞바퀴 출력 보정

    ALL_DEVICES = []


UDP_IP = "0.0.0.0"
UDP_PORT = 8080
TELEMETRY_PORT = 8082
WATCHDOG_TIMEOUT = 0.25
# 전진/일반 주행 출력 게인 (기존 0.96에서 10% 하향)
MOTOR_GAIN = 0.864
# 좌우 모터가 반대 방향으로 도는 제자리 회전 출력 게인
# 기존 0.80에서 15% 상향
ROTATION_MOTOR_GAIN = 0.92
ULTRASONIC_POLL_INTERVAL = 0.05
ULTRASONIC_STOP_DISTANCE_M = 0.07
ultrasonic_sensors = []
ultrasonic_thread = None
ultrasonic_running = False
last_client_addr = None
last_client_lock = threading.Lock()
telemetry_sock = None


def _ultrasonic_log_loop():
    """센서 데이터를 관제 PC로 전송한다."""
    while ultrasonic_running:
        try:
            left = ultrasonic_sensors[0].distance * ultrasonic_sensors[0].max_distance
            right = ultrasonic_sensors[1].distance * ultrasonic_sensors[1].max_distance
            blocked = min(left, right) <= ULTRASONIC_STOP_DISTANCE_M

            with last_client_lock:
                client_addr = last_client_addr
            if client_addr is not None:
                telemetry = json.dumps({
                    "left_cm": left * 100.0,
                    "right_cm": right * 100.0,
                    "blocked": blocked,
                }).encode("utf-8")
                if telemetry_sock is not None:
                    try:
                        telemetry_sock.sendto(telemetry, (client_addr[0], TELEMETRY_PORT))
                    except OSError:
                        pass
        except Exception:
            pass
        time.sleep(ULTRASONIC_POLL_INTERVAL)


def start_ultrasonic_monitor():
    global ultrasonic_thread, ultrasonic_running
    if not ULTRASONIC_AVAILABLE:
        return
    try:
        ultrasonic_sensors.extend([
            DistanceSensor(echo=17, trigger=4, max_distance=0.5, queue_len=5),
            DistanceSensor(echo=15, trigger=14, max_distance=0.5, queue_len=5),
        ])
        ultrasonic_running = True
        ultrasonic_thread = threading.Thread(
            target=_ultrasonic_log_loop,
            name="ultrasonic-monitor",
            daemon=True,
        )
        ultrasonic_thread.start()
    except Exception:
        ultrasonic_sensors.clear()


def stop_ultrasonic_monitor():
    global ultrasonic_running
    ultrasonic_running = False
    if ultrasonic_thread is not None:
        ultrasonic_thread.join(timeout=1.5)
    for sensor in ultrasonic_sensors:
        try:
            sensor.close()
        except Exception:
            pass
    ultrasonic_sensors.clear()


def stop_all():
    if GPIO_AVAILABLE:
        FRONT_LEFT_PWM.value = 0.0
        FRONT_RIGHT_PWM.value = 0.0
        BACK_LEFT_PWM.value = 0.0
        BACK_RIGHT_PWM.value = 0.0


def set_motors_raw(v_left: float, v_right: float):
    v_left = max(-255.0, min(255.0, float(v_left)))
    v_right = max(-255.0, min(255.0, float(v_right)))
    # 좌우 명령의 부호가 반대이면 제자리 회전으로 판단한다.
    is_rotation = v_left != 0.0 and v_right != 0.0 and (v_left * v_right < 0.0)
    output_gain = ROTATION_MOTOR_GAIN if is_rotation else MOTOR_GAIN
    v_left *= output_gain
    v_right *= output_gain

    left_forward = (v_left >= 0)
    right_forward = (v_right >= 0)

    speed_left = abs(v_left) / 255.0
    speed_right = abs(v_right) / 255.0

    if not GPIO_AVAILABLE:
        return

    # 왼쪽 모터 방향
    if left_forward:
        FRONT_LEFT_DIR.on()
        BACK_LEFT_DIR.on()
    else:
        FRONT_LEFT_DIR.off()
        BACK_LEFT_DIR.off()

    # 오른쪽 모터 방향 (반전)
    if right_forward:
        FRONT_RIGHT_DIR.off()
        BACK_RIGHT_DIR.off()
    else:
        FRONT_RIGHT_DIR.on()
        BACK_RIGHT_DIR.on()

    # PWM 출력
    FRONT_LEFT_PWM.value = BACK_LEFT_PWM.value = speed_left
    FRONT_RIGHT_PWM.value = min(1.0, speed_right * FRONT_RIGHT_BOOST)
    BACK_RIGHT_PWM.value = speed_right


def main():
    global last_client_addr, telemetry_sock
    parser = argparse.ArgumentParser(description="라즈베리 파이 RC카 모터 & 온보드 카메라 서버")
    parser.add_argument("--width", type=int, default=640, help="카메라 가로 해상도 (기본: 640, 고화질 720p: 1280)")
    parser.add_argument("--height", type=int, default=480, help="카메라 세로 해상도 (기본: 480, 고화질 720p: 720)")
    parser.add_argument("--quality", type=int, default=80, help="JPEG 압축 화질 1~100 (기본: 80)")
    parser.add_argument("--cam-index", type=int, default=1, help="카메라 장치 인덱스 (기본: 1)")
    parser.add_argument("--cam-port", type=int, default=8081, help="카메라 HTTP 스트리밍 포트 (기본: 8081)")
    args = parser.parse_args()

    # 1. 라즈베리 파이 카메라 스트리머 시작 (기본: 640x480 @ 80% 화질)
    print(f"[CAMERA] Settings: {args.width}x{args.height}, quality: {args.quality}%, port: {args.cam_port}")
    camera_streamer = PiCameraStreamer(
        camera_index=args.cam_index,
        http_port=args.cam_port,
        width=args.width,
        height=args.height,
        quality=args.quality
    )
    camera_streamer.start()
    start_ultrasonic_monitor()

    # 2. 모터 제어 UDP 서버 시작 (포트 8080)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    telemetry_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(WATCHDOG_TIMEOUT)

    print(f"[MOTOR] Server started on port {UDP_PORT}")
    stop_all()
    last_print_time = 0

    try:
        while True:
            try:
                data, addr = sock.recvfrom(1024)
                with last_client_lock:
                    last_client_addr = addr
                msg = data.decode("utf-8").strip()

                if "," in msg:
                    parts = msg.split(",")
                    v_l = float(parts[0])
                    v_r = float(parts[1])
                else:
                    payload = json.loads(msg)
                    v_l = float(payload.get("v_left", 0))
                    v_r = float(payload.get("v_right", 0))

                set_motors_raw(v_l, v_r)

                now = time.time()
                if now - last_print_time > 0.5:
                    last_print_time = now
                    print(f"\r[RECEIVING] Left: {v_l:+6.1f} | Right: {v_r:+6.1f} | From: {addr[0]}", end="", flush=True)

            except socket.timeout:
                stop_all()

            except Exception as e:
                print(f"\n[ERROR]: {e}")
                stop_all()

    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Stopping server.")
    finally:
        stop_all()
        stop_ultrasonic_monitor()
        sock.close()
        telemetry_sock.close()
        camera_streamer.stop()
        for device in ALL_DEVICES:
            try:
                device.close()
            except Exception:
                pass
        print("[DONE] Motors stopped; GPIO and camera resources released.")


if __name__ == "__main__":
    main()