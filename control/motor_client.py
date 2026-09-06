import socket
from typing import Tuple
import numpy as np


class MotorClient:
    def __init__(self, ip: str = "192.168.0.2", port: int = 8080):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
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

    def stop(self):
        self.send_speed(0, 0)

    def close(self):
        try:
            self.stop()
            self.sock.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
