import socket
import json
import time
import threading
import argparse
from pathlib import Path
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("[WARNING] OpenCV(cv2) unavailable - camera streaming disabled")
    CV2_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ort = None
    ONNXRUNTIME_AVAILABLE = False

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

ONBOARD_MODEL_PATH = Path(__file__).with_name("onboard.onnx")
ONBOARD_MODEL_INPUT_SIZE = 640
ONBOARD_MODEL_CONFIDENCE = 0.65
ONBOARD_MODEL_NMS_THRESHOLD = 0.45
ONBOARD_DEFAULT_CLASS_NAMES = ("ally", "claymore")
ONBOARD_CLASS_COLORS = {
    0: (0, 255, 0),  # green in BGR
    1: (0, 0, 255),  # red in BGR
}


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
    def __init__(
        self,
        camera_index: int = 0,
        http_port: int = 8081,
        width: int = 960,
        height: int = 540,
        quality: int = 80,
        camera_zoom: float = 1.0,
        fps: int = 20,
    ):
        self.camera_index = camera_index
        self.http_port = http_port
        self.width = width
        self.height = height
        self.quality = max(10, min(100, quality))
        self.camera_zoom = max(0.5, min(1.0, float(camera_zoom)))
        self.fps = max(5, min(30, int(fps)))
        self.capture_size = (960, 720)
        # IMX219 full sensor mode (8 MP, 4:3). The output is cropped to 16:9
        # later, after the low-resolution capture, to keep streaming responsive.
        self.sensor_size = (1640, 1232)
        self.running = False
        self.latest_jpeg = None
        self.lock = threading.Lock()
        self.server = None
        self.thread = None
        self.server_thread = None
        self.picam2 = None
        self.cap = None
        self.onboard_net = None
        self.onboard_session = None
        self.onboard_input_name = None
        self.onboard_backend = None
        self.onboard_input_size = ONBOARD_MODEL_INPUT_SIZE
        self.onboard_labels = self._load_onboard_labels()
        self.onboard_detections = []
        self.onboard_detection_lock = threading.Lock()
        self.inference_frame = None
        self.inference_lock = threading.Lock()
        self.inference_event = threading.Event()
        self.inference_thread = None
        self.inference_running = False
        self.inference_interval = 0.25
        self.last_inference_submit = 0.0
        self.onboard_error_reported = False

        # 기본 대기 화면 생성
        self._init_placeholder_frame()

    @staticmethod
    def _load_onboard_labels():
        for filename in ("onboard.names", "onboard.txt", "classes.txt"):
            path = Path(__file__).with_name(filename)
            if path.exists():
                return [
                    line.strip()
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
        return list(ONBOARD_DEFAULT_CLASS_NAMES)

    def _load_onboard_model(self):
        if not ONBOARD_MODEL_PATH.exists():
            print(f"[ONBOARD] Model not found: {ONBOARD_MODEL_PATH}")
            return
        try:
            if not ONNXRUNTIME_AVAILABLE:
                print(
                    "[ONBOARD ERROR] onnxruntime is not installed; "
                    "OpenCV DNN fallback is disabled for this YOLOv8 model."
                )
                return

            self.onboard_session = ort.InferenceSession(
                str(ONBOARD_MODEL_PATH),
                providers=["CPUExecutionProvider"],
            )
            input_info = self.onboard_session.get_inputs()[0]
            self.onboard_input_name = input_info.name
            input_shape = input_info.shape
            if (
                len(input_shape) == 4
                and isinstance(input_shape[2], int)
                and isinstance(input_shape[3], int)
            ):
                self.onboard_input_size = (input_shape[3], input_shape[2])
            self.onboard_backend = "onnxruntime"
            print(
                f"[ONBOARD] Model loaded with ONNX Runtime: {ONBOARD_MODEL_PATH} "
                f"(input: {self.onboard_input_size[0]}x{self.onboard_input_size[1]})"
            )
        except (cv2.error, OSError, ValueError, RuntimeError) as exc:
            self.onboard_net = None
            self.onboard_session = None
            self.onboard_backend = None
            print(f"[ONBOARD ERROR] Failed to load model: {exc}")

    @staticmethod
    def _clip_box(box, width, height):
        x1, y1, x2, y2 = box
        return (
            int(np.clip(x1, 0, width - 1)),
            int(np.clip(y1, 0, height - 1)),
            int(np.clip(x2, 0, width - 1)),
            int(np.clip(y2, 0, height - 1)),
        )

    def _detect_onboard_objects(self, frame):
        if self.onboard_backend is None:
            return []

        height, width = frame.shape[:2]
        if self.onboard_backend == "onnxruntime":
            resized = cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                self.onboard_input_size,
            )
            input_tensor = resized.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
            outputs = self.onboard_session.run(
                None,
                {self.onboard_input_name: input_tensor},
            )
            output = np.asarray(outputs[0])
        else:
            blob = cv2.dnn.blobFromImage(
                frame,
                scalefactor=1.0 / 255.0,
                size=(ONBOARD_MODEL_INPUT_SIZE, ONBOARD_MODEL_INPUT_SIZE),
                swapRB=True,
                crop=False,
            )
            self.onboard_net.setInput(blob)
            output = np.asarray(self.onboard_net.forward())
        if output.ndim == 3:
            output = output[0]
        if output.ndim != 2:
            raise ValueError(f"Unsupported ONNX output shape: {output.shape}")

        # This model returns YOLOv8 output as [channels, candidates]:
        # [1, 6, 5376] -> [5376, 6] after transpose.
        if output.shape[0] <= 256 and output.shape[1] > output.shape[0]:
            output = output.transpose(1, 0)

        boxes = []
        scores = []
        class_ids = []
        for row in output:
            if row.size >= 6:
                # YOLOv8 output: [center_x, center_y, width, height, class scores...].
                class_scores = row[4:]
                class_id = int(np.argmax(class_scores))
                score = float(class_scores[class_id])
                center_x, center_y, box_width, box_height = row[:4]
                if max(abs(float(value)) for value in (center_x, center_y, box_width, box_height)) <= 2.0:
                    center_x *= width
                    center_y *= height
                    box_width *= width
                    box_height *= height
                else:
                    center_x *= width / self.onboard_input_size[0]
                    center_y *= height / self.onboard_input_size[1]
                    box_width *= width / self.onboard_input_size[0]
                    box_height *= height / self.onboard_input_size[1]
                coords = (
                    center_x - box_width / 2.0,
                    center_y - box_height / 2.0,
                    center_x + box_width / 2.0,
                    center_y + box_height / 2.0,
                )
            elif row.size == 5:
                # Single-class YOLO output: [center_x, center_y, width, height, score].
                class_id = 0
                score = float(row[4])
                center_x, center_y, box_width, box_height = row[:4]
                if max(abs(float(value)) for value in (center_x, center_y, box_width, box_height)) <= 2.0:
                    center_x *= width
                    center_y *= height
                    box_width *= width
                    box_height *= height
                else:
                    center_x *= width / self.onboard_input_size[0]
                    center_y *= height / self.onboard_input_size[1]
                    box_width *= width / self.onboard_input_size[0]
                    box_height *= height / self.onboard_input_size[1]
                coords = (
                    center_x - box_width / 2.0,
                    center_y - box_height / 2.0,
                    center_x + box_width / 2.0,
                    center_y + box_height / 2.0,
                )
            elif row.size == 4:
                continue
            elif row.size == 6:
                x1, y1, x2, y2, score, class_id = row.tolist()
                coords = (x1, y1, x2, y2)
                if max(abs(value) for value in coords) <= 2.0:
                    coords = tuple(
                        value * scale
                        for value, scale in zip(coords, (width, height, width, height))
                    )
            else:
                continue

            score = float(score)
            if score < ONBOARD_MODEL_CONFIDENCE:
                continue
            x1, y1, x2, y2 = self._clip_box(coords, width, height)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(score)
            class_ids.append(int(class_id))

        if not boxes:
            return []
        keep = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            ONBOARD_MODEL_CONFIDENCE,
            ONBOARD_MODEL_NMS_THRESHOLD,
        )
        detections = []
        for index in np.asarray(keep).reshape(-1):
            index = int(index)
            x, y, box_width, box_height = boxes[index]
            class_id = class_ids[index]
            label = ONBOARD_DEFAULT_CLASS_NAMES[class_id] if class_id in (0, 1) else (
                self.onboard_labels[class_id]
                if 0 <= class_id < len(self.onboard_labels)
                else f"class_{class_id}"
            )
            detections.append((x, y, box_width, box_height, scores[index], label))
        return detections

    def _draw_onboard_objects(self, frame):
        with self.onboard_detection_lock:
            detections = list(self.onboard_detections)
        for x, y, box_width, box_height, score, label in detections:
            class_id = ONBOARD_DEFAULT_CLASS_NAMES.index(label) if label in ONBOARD_DEFAULT_CLASS_NAMES else -1
            color = ONBOARD_CLASS_COLORS.get(class_id, (0, 255, 0))
            cv2.rectangle(
                frame,
                (x, y),
                (x + box_width, y + box_height),
                color,
                2,
            )
            cv2.putText(
                frame,
                f"{label} {score:.2f}",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        return frame

    def _inference_loop(self):
        while self.inference_running:
            self.inference_event.wait(timeout=0.5)
            self.inference_event.clear()
            if not self.inference_running:
                break
            with self.inference_lock:
                frame = self.inference_frame
                self.inference_frame = None
            if frame is None or self.onboard_backend is None:
                continue
            try:
                detections = self._detect_onboard_objects(frame)
                with self.onboard_detection_lock:
                    self.onboard_detections = detections
            except Exception as exc:
                self.onboard_net = None
                self.onboard_session = None
                self.onboard_backend = None
                with self.onboard_detection_lock:
                    self.onboard_detections = []
                if not self.onboard_error_reported:
                    print(
                        f"[ONBOARD ERROR] Inference stopped; camera stream continues: {exc}. "
                        "Install onnxruntime if this is an OpenCV DNN compatibility error."
                    )
                    self.onboard_error_reported = True

    def _submit_inference_frame(self, frame):
        if self.onboard_backend is None:
            return
        now = time.monotonic()
        if now - self.last_inference_submit < self.inference_interval:
            return
        self.last_inference_submit = now
        with self.inference_lock:
            self.inference_frame = frame.copy()
        self.inference_event.set()

    def get_onboard_detections(self):
        with self.onboard_detection_lock:
            return [
                {"label": label, "confidence": float(score)}
                for _, _, _, _, score, label in self.onboard_detections
            ]

    def _safe_annotate_frame(self, frame):
        self._submit_inference_frame(frame)
        return self._draw_onboard_objects(frame)

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
        # 1. IMX219 CSI camera through Picamera2.
        if PICAMERA2_AVAILABLE:
            try:
                print("[CAMERA] Initializing Picamera2...")
                p2 = Picamera2()
                cfg = p2.create_video_configuration(
                    main={"size": self.capture_size, "format": "BGR888"},
                    sensor={"output_size": self.sensor_size},
                )
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
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_size[0])
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_size[1])
                        cap.set(cv2.CAP_PROP_FPS, self.fps)
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

    def _format_camera_frame(self, frame):
        """Center-crop the sensor image to the requested output aspect ratio."""
        if frame is None or frame.size == 0:
            return frame

        if self.camera_zoom < 1.0:
            frame = cv2.resize(
                frame,
                (
                    max(1, int(round(frame.shape[1] * self.camera_zoom))),
                    max(1, int(round(frame.shape[0] * self.camera_zoom))),
                ),
                interpolation=cv2.INTER_AREA,
            )

        source_h, source_w = frame.shape[:2]
        output_aspect = self.width / self.height
        source_aspect = source_w / source_h

        if source_aspect > output_aspect:
            crop_w = int(round(source_h * output_aspect))
            crop_h = source_h
        else:
            crop_w = source_w
            crop_h = int(round(source_w / output_aspect))

        crop_w = min(source_w, max(1, crop_w))
        crop_h = min(source_h, max(1, crop_h))
        left = (source_w - crop_w) // 2
        top = (source_h - crop_h) // 2
        cropped = frame[top:top + crop_h, left:left + crop_w]
        return cv2.resize(
            cropped,
            (self.width, self.height),
            interpolation=cv2.INTER_AREA,
        )

    def _capture_loop(self):
        cam_type = self._init_camera()
        if cam_type is not None:
            self._load_onboard_model()
            if self.onboard_backend is not None:
                self.inference_running = True
                self.inference_thread = threading.Thread(
                    target=self._inference_loop,
                    name="onboard-inference",
                    daemon=True,
                )
                self.inference_thread.start()
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
                        frame = self._format_camera_frame(frame)

                        frame = self._safe_annotate_frame(frame)
                        _, jpeg = cv2.imencode(".jpg", frame, encode_param)
                        with self.lock:
                            self.latest_jpeg = jpeg.tobytes()
                        time.sleep(1.0 / self.fps)
                        continue
                except Exception as e:
                    print(f"[CAMERA Picamera2 ERROR]: {e}")
                    cam_type = None

            elif cam_type == "opencv" and self.cap is not None:
                try:
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        if frame.shape[1] != self.capture_size[0] or frame.shape[0] != self.capture_size[1]:
                            frame = cv2.resize(frame, self.capture_size)
                        # 1. 180도 회전
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                        # 2. 색상 보정
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        frame = self._format_camera_frame(frame)

                        frame = self._safe_annotate_frame(frame)
                        _, jpeg = cv2.imencode(".jpg", frame, encode_param)
                        with self.lock:
                            self.latest_jpeg = jpeg.tobytes()
                        time.sleep(1.0 / self.fps)
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
        self.inference_running = False
        self.inference_event.set()
        if self.inference_thread is not None:
            self.inference_thread.join(timeout=1.0)
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
ULTRASONIC_LOG_INTERVAL = 1.0
RELAY_GPIO = 26
# This relay turns on when GPIO 26 is HIGH.
RELAY_ACTIVE_HIGH = True
SMOKE_DURATION_SEC = 5.0
ultrasonic_sensors = []
ultrasonic_thread = None
ultrasonic_running = False
smoke_timer = None
smoke_lock = threading.Lock()
smoke_generation = 0
last_client_addr = None
last_client_lock = threading.Lock()
telemetry_sock = None
camera_streamer_instance = None


if GPIO_AVAILABLE:
    SMOKE_RELAY = LED(
        RELAY_GPIO,
        active_high=RELAY_ACTIVE_HIGH,
        initial_value=False,
    )
    SMOKE_RELAY.off()
else:
    SMOKE_RELAY = None


def activate_smoke():
    """Turn the smoke relay on and switch it off automatically after five seconds."""
    global smoke_timer, smoke_generation
    if SMOKE_RELAY is None:
        print("[SMOKE ERROR] GPIO is unavailable; relay was not activated.")
        return

    with smoke_lock:
        if smoke_timer is not None:
            smoke_timer.cancel()
        smoke_generation += 1
        generation = smoke_generation
        SMOKE_RELAY.on()
        smoke_timer = threading.Timer(
            SMOKE_DURATION_SEC,
            _expire_smoke,
            args=(generation,),
        )
        smoke_timer.daemon = True
        smoke_timer.start()
    print("[SMOKE] Relay activated for 5 seconds.")


def _expire_smoke(generation):
    with smoke_lock:
        if generation != smoke_generation:
            return
        if SMOKE_RELAY is not None:
            SMOKE_RELAY.off()
        global smoke_timer
        smoke_timer = None


def deactivate_smoke():
    global smoke_timer, smoke_generation
    with smoke_lock:
        smoke_generation += 1
        if smoke_timer is not None:
            smoke_timer.cancel()
        if SMOKE_RELAY is not None:
            SMOKE_RELAY.off()
        smoke_timer = None


def _ultrasonic_log_loop():
    """Read the ultrasonic sensors and send telemetry to the control PC."""
    last_log_time = 0.0
    last_detection_payload = None
    while ultrasonic_running:
        left = right = 50.0
        try:
            if len(ultrasonic_sensors) >= 2:
                left = ultrasonic_sensors[0].distance * ultrasonic_sensors[0].max_distance * 100.0
                right = ultrasonic_sensors[1].distance * ultrasonic_sensors[1].max_distance * 100.0
        except Exception:
            # A missing ultrasonic echo must not prevent object telemetry.
            pass

        blocked = min(left, right) <= ULTRASONIC_STOP_DISTANCE_M * 100.0
        now = time.monotonic()
        if now - last_log_time >= ULTRASONIC_LOG_INTERVAL:
            print(
                f"[ULTRASONIC] Left: {left:5.1f} cm | "
                f"Right: {right:5.1f} cm | "
                f"Blocked: {'YES' if blocked else 'NO'}",
                flush=True,
            )
            last_log_time = now

        detections = (
            camera_streamer_instance.get_onboard_detections()
            if camera_streamer_instance is not None
            else []
        )
        detection_payload = tuple(
            (item.get("label"), round(float(item.get("confidence", 0.0)), 3))
            for item in detections
        )
        if detection_payload != last_detection_payload:
            if detection_payload:
                print(f"[ONBOARD] Detection telemetry: {detection_payload}", flush=True)
            last_detection_payload = detection_payload

        with last_client_lock:
            client_addr = last_client_addr
        if client_addr is not None and telemetry_sock is not None:
            telemetry = json.dumps({
                "left_cm": left,
                "right_cm": right,
                "blocked": blocked,
                "onboard_detections": detections,
            }).encode("utf-8")
            try:
                telemetry_sock.sendto(telemetry, (client_addr[0], TELEMETRY_PORT))
            except OSError:
                pass
        time.sleep(ULTRASONIC_POLL_INTERVAL)


def start_ultrasonic_monitor():
    global ultrasonic_thread, ultrasonic_running
    if ULTRASONIC_AVAILABLE:
        try:
            ultrasonic_sensors.extend([
                DistanceSensor(echo=17, trigger=4, max_distance=0.5, queue_len=5),
                DistanceSensor(echo=15, trigger=14, max_distance=0.5, queue_len=5),
            ])
        except Exception as exc:
            ultrasonic_sensors.clear()
            print(f"[ULTRASONIC WARNING] Sensors unavailable: {exc}")

    # This loop also carries onboard detections, so it must run without sensors.
    ultrasonic_running = True
    ultrasonic_thread = threading.Thread(
        target=_ultrasonic_log_loop,
        name="telemetry-monitor",
        daemon=True,
    )
    ultrasonic_thread.start()


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
    global last_client_addr, telemetry_sock, camera_streamer_instance
    parser = argparse.ArgumentParser(description="라즈베리 파이 RC카 모터 & 온보드 카메라 서버")
    parser.add_argument("--width", type=int, default=960, help="카메라 송신 가로 해상도 (기본: 960)")
    parser.add_argument("--height", type=int, default=540, help="카메라 송신 세로 해상도 (기본: 540)")
    parser.add_argument("--quality", type=int, default=75, help="JPEG 압축 화질 1~100 (기본: 75)")
    parser.add_argument("--fps", type=int, default=20, help="카메라 송신 FPS (기본: 20)")
    parser.add_argument("--cam-index", type=int, default=1, help="카메라 장치 인덱스 (기본: 1)")
    parser.add_argument("--cam-port", type=int, default=8081, help="카메라 HTTP 스트리밍 포트 (기본: 8081)")
    parser.add_argument("--camera-zoom", type=float, default=1.0, help="추가 디지털 축소 비율 (기본: 1.0, 센서 전체 화각 사용)")
    args = parser.parse_args()

    # 1. 라즈베리 파이 카메라 스트리머 시작 (기본: 960x540 @ 75%, 20 FPS)
    print(
        f"[CAMERA] Settings: {args.width}x{args.height}, "
        f"quality: {args.quality}%, fps: {args.fps}, "
        f"port: {args.cam_port}, zoom: {args.camera_zoom}"
    )
    camera_streamer = PiCameraStreamer(
        camera_index=args.cam_index,
        http_port=args.cam_port,
        width=args.width,
        height=args.height,
        quality=args.quality,
        camera_zoom=args.camera_zoom,
        fps=args.fps,
    )
    camera_streamer.start()
    camera_streamer_instance = camera_streamer
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
                    if payload.get("command") == "smoke":
                        activate_smoke()
                        continue
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
        deactivate_smoke()
        stop_ultrasonic_monitor()
        sock.close()
        telemetry_sock.close()
        camera_streamer.stop()
        for device in ALL_DEVICES:
            try:
                device.close()
            except Exception:
                pass
        if SMOKE_RELAY is not None:
            try:
                SMOKE_RELAY.close()
            except Exception:
                pass
        print("[DONE] Motors stopped; GPIO and camera resources released.")


if __name__ == "__main__":
    main()