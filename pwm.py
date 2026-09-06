import socket
import json
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("[경고] OpenCV(cv2) 없음 - 카메라 스트리밍 비활성화")
    CV2_AVAILABLE = False

try:
    from gpiozero import LED, PWMLED
    GPIO_AVAILABLE = True
except ImportError:
    print("[경고] gpiozero 없음 (시뮬레이션 모드)")
    GPIO_AVAILABLE = False


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
    def __init__(self, camera_index: int = 0, http_port: int = 8081, width: int = 320, height: int = 240):
        self.camera_index = camera_index
        self.http_port = http_port
        self.width = width
        self.height = height
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
            _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
            self.latest_jpeg = enc.tobytes()

    def _init_camera(self):
        # 1. 라즈베리 파이 5 전용 Picamera2 시도 (CSI 포트 ov5647 등 공식 카메라)
        if PICAMERA2_AVAILABLE:
            try:
                print("[카메라] Picamera2 초기화 시도...")
                p2 = Picamera2()
                cfg = p2.create_video_configuration(main={"size": (self.width, self.height), "format": "BGR888"})
                p2.configure(cfg)
                p2.start()
                time.sleep(0.5)
                test_arr = p2.capture_array()
                if test_arr is not None and test_arr.size > 0:
                    print(f"[카메라] Picamera2 연결 성공! 해상도: {self.width}x{self.height}")
                    self.picam2 = p2
                    return "picam2"
                p2.stop()
                p2.close()
            except Exception as e:
                print(f"[카메라] Picamera2 실패 ({e}) -> OpenCV VideoCapture 시도")

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
                            print(f"[카메라] OpenCV {idx}번 포트 연결 성공! 해상도: {self.width}x{self.height}")
                            self.cap = cap
                            return "opencv"
                        cap.release()
                except Exception:
                    pass

        print("[카메라 경고] 사용 가능한 카메라를 찾지 못했습니다. (재시도 대기)")
        return None

    def _capture_loop(self):
        cam_type = self._init_camera()
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 65]

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
                    print(f"[카메라 Picamera2 오류]: {e}")
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
                    print(f"[카메라 OpenCV 오류]: {e}")
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
            print(f"[카메라 스트리머 시작] http://0.0.0.0:{self.http_port}/stream.mjpg")
        except Exception as e:
            print(f"[카메라 스트리머 서버 오류]: {e}")

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
        print("[카메라 스트리머 종료 완료]")


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

    ALL_DEVICES = [
        FRONT_LEFT_PWM, FRONT_RIGHT_PWM, BACK_LEFT_PWM, BACK_RIGHT_PWM,
        FRONT_LEFT_DIR, FRONT_RIGHT_DIR, BACK_LEFT_DIR, BACK_RIGHT_DIR
    ]
else:
    FRONT_RIGHT_BOOST = 1.2
    ALL_DEVICES = []


UDP_IP = "0.0.0.0"
UDP_PORT = 8080
WATCHDOG_TIMEOUT = 0.25


def stop_all():
    if GPIO_AVAILABLE:
        FRONT_LEFT_PWM.value = 0.0
        FRONT_RIGHT_PWM.value = 0.0
        BACK_LEFT_PWM.value = 0.0
        BACK_RIGHT_PWM.value = 0.0


def set_motors_raw(v_left: float, v_right: float):
    v_left = max(-255.0, min(255.0, float(v_left)))
    v_right = max(-255.0, min(255.0, float(v_right)))

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
    # 1. 라즈베리 파이 1번 포트 카메라 스트리머 시작 (포트 8081)
    camera_streamer = PiCameraStreamer(camera_index=1, http_port=8081, width=320, height=240)
    camera_streamer.start()

    # 2. 모터 제어 UDP 서버 시작 (포트 8080)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(WATCHDOG_TIMEOUT)

    print(f"[모터 서버 시작] 포트: {UDP_PORT}")
    stop_all()
    last_print_time = 0

    try:
        while True:
            try:
                data, addr = sock.recvfrom(1024)
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
                    print(f"\r[수신중] Left: {v_l:+6.1f} | Right: {v_r:+6.1f} | From: {addr[0]}", end="", flush=True)

            except socket.timeout:
                stop_all()

            except Exception as e:
                print(f"\n[오류 발생]: {e}")
                stop_all()

    except KeyboardInterrupt:
        print("\n[종료] 서버를 종료합니다.")
    finally:
        stop_all()
        sock.close()
        camera_streamer.stop()
        for device in ALL_DEVICES:
            try:
                device.close()
            except Exception:
                pass
        print("[완료] 모터 정지 및 GPIO/카메라 자원 해제 완료.")


if __name__ == "__main__":
    main()