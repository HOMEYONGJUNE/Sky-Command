"""
라즈베리 파이 온보드 카메라 비디오 스트림 수신기 (pi_cam_receiver.py)
- HTTP MJPEG 스트림 비동기 수신
- 최신 1프레임 원자적 버퍼링 (Zero-latency)
- 자동 재연결 및 안전한 스레드 종료 지원
"""
import threading
import time
from typing import Optional
import urllib.request
import cv2
import numpy as np


class PiCamReceiver:
    def __init__(self, ip: str = "192.168.0.10", port: int = 8081, stream_path: str = "/stream.mjpg"):
        self.ip = ip
        self.port = port
        self.stream_path = stream_path
        self.url = f"http://{self.ip}:{self.port}{self.stream_path}"
        
        self.running = False
        self.is_connected = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.last_frame_time = 0.0

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()
        print(f"[라즈베리파이 캠] 수신 스레드 시작 ({self.url})")

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.is_connected = False
        print("[라즈베리파이 캠] 수신 스레드 종료")

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.latest_frame is not None:
                # 2초 이상 새 프레임이 없으면 연결 끊김으로 간주
                if time.time() - self.last_frame_time > 2.0:
                    self.is_connected = False
                    return None
                return self.latest_frame.copy()
            return None

    def _receive_loop(self):
        while self.running:
            stream = None
            try:
                # 타임아웃 2.5초로 연결
                req = urllib.request.Request(self.url, headers={"User-Agent": "RC-GCS-Client"})
                stream = urllib.request.urlopen(req, timeout=2.5)
                self.is_connected = True
                print(f"[라즈베리파이 캠] 연결 성공: {self.url}")

                bytes_buffer = b""
                while self.running:
                    chunk = stream.read(4096)
                    if not chunk:
                        break
                    bytes_buffer += chunk

                    # JPEG 이미지 시작(\xff\xd8) 및 종료(\xff\xd9) 탐색
                    a = bytes_buffer.find(b"\xff\xd8")
                    b = bytes_buffer.find(b"\xff\xd9")
                    if a != -1 and b != -1 and b > a:
                        jpg = bytes_buffer[a:b + 2]
                        bytes_buffer = bytes_buffer[b + 2:]

                        # 이미지 디코딩
                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None:
                            with self.lock:
                                self.latest_frame = frame
                                self.last_frame_time = time.time()
                                self.is_connected = True

            except Exception:
                self.is_connected = False
                with self.lock:
                    self.latest_frame = None

            finally:
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass

            # 연결 실패 또는 끊김 시 1.5초 후 재시도
            if self.running:
                time.sleep(1.5)
