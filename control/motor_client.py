import socket
import json
import threading
from typing import Tuple
import numpy as np


class MotorClient:
    def __init__(self, ip: str = "192.168.0.2", port: int = 8080, telemetry_port: int = 8082):
        self.ip = ip
        self.port = port
        self.telemetry_port = telemetry_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.telemetry_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.telemetry_sock.settimeout(0.5)
        self.telemetry_running = False
        self.telemetry_thread = None
        self.ultrasonic_reading = {"left_cm": 50.0, "right_cm": 50.0}
        self.ultrasonic_state_lock = threading.Lock()
        self.last_sent: Tuple[int, int] = (0, 0)
        self.is_connected = True

    def send_speed(self, v_left: float, v_right: float):
        # -255 ~ 255 범위로 클리핑 후 UDP 전송
        vl = int(np.clip(v_left, -255, 255))
        vr = int(np.clip(v_right, -255, 255))
        
        msg = f"{vl},{vr}".encode("utf-8")
        try:
            self.sock.sendto(msg, (self.ip, self.port))
            self.last_sent = (vl, vr)
            self.is_connected = True
        except Exception as e:
            self.is_connected = False
            print(f"[UDP 통신 오류]: {e}")

    def start_telemetry_listener(self):
        if self.telemetry_running:
            return
        try:
            self.telemetry_sock.bind(("0.0.0.0", self.telemetry_port))
        except OSError as e:
            print(f"[초음파 로그 수신 오류] UDP {self.telemetry_port} 포트를 열 수 없습니다: {e}")
            return
        self.telemetry_running = True
        self.telemetry_thread = threading.Thread(
            target=self._telemetry_loop,
            name="ultrasonic-telemetry",
            daemon=True,
        )
        self.telemetry_thread.start()
        print(f"[초음파 로그] 라즈베리파이 센서 수신 대기 (UDP {self.telemetry_port})")

    def _telemetry_loop(self):
        while self.telemetry_running:
            try:
                data, _ = self.telemetry_sock.recvfrom(1024)
                payload = json.loads(data.decode("utf-8"))
                with self.ultrasonic_state_lock:
                    self.ultrasonic_reading = {
                        "left_cm": float(payload["left_cm"]),
                        "right_cm": float(payload["right_cm"]),
                    }
                print(
                    f"[ULTRASONIC/PI] left={float(payload['left_cm']):5.1f}cm "
                    f"right={float(payload['right_cm']):5.1f}cm",
                    flush=True,
                )
            except socket.timeout:
                continue
            except (OSError, ValueError, KeyError) as e:
                if self.telemetry_running:
                    print(f"[초음파 로그 수신 오류] {e}")

    def get_ultrasonic_reading(self):
        with self.ultrasonic_state_lock:
            return dict(self.ultrasonic_reading)

    def stop_telemetry_listener(self):
        self.telemetry_running = False
        if self.telemetry_thread is not None:
            self.telemetry_thread.join(timeout=1.0)

    def stop(self):
        self.send_speed(0, 0)

    def close(self):
        try:
            self.stop_telemetry_listener()
            self.stop()
            self.sock.close()
            self.telemetry_sock.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
