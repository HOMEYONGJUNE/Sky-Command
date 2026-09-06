import os
import time
from typing import Dict, List, Optional
import cv2
import numpy as np


class UIButtonManager:
    def __init__(self, ui_dir: str, window_w: int = 1280, window_h: int = 720):
        self.ui_dir = ui_dir
        self.window_w = window_w
        self.window_h = window_h

        # 맵/캠 토글 상태 (기본: MAP)
        self.map_cam_mode = "MAP"
        self.last_toggle_time = 0.0

        # 누르고 있는 버튼 상태
        self.pressed_button: Optional[str] = None
        self.action_names = ["exit", "heal", "move", "recall", "smoke", "stop"]

        # 이미지 캐시
        self.images: Dict[str, Dict] = {}
        self._load_all_buttons()

    def _load_img(self, filename: str) -> Optional[Dict]:
        path = os.path.join(self.ui_dir, filename)
        if not os.path.exists(path):
            return None
        try:
            img_arr = np.fromfile(path, np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_UNCHANGED)
            if img is None:
                return None
            if img.shape[0] != self.window_h or img.shape[1] != self.window_w:
                img = cv2.resize(img, (self.window_w, self.window_h))
            
            if img.shape[2] == 4:
                bgr = img[:, :, :3]
                alpha_2d = img[:, :, 3]
                mask = (alpha_2d > 15)
                # 클릭 감도 향상용 확장 마스크 (여백 5px 보정)
                hit_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
                hit_mask = cv2.dilate(mask.astype(np.uint8), hit_kernel) > 0
                alpha_f = alpha_2d.astype(np.float32) / 255.0
            else:
                bgr = img
                mask = np.ones((self.window_h, self.window_w), dtype=bool)
                hit_mask = mask
                alpha_f = np.ones((self.window_h, self.window_w), dtype=np.float32)

            # 불투명 영역 bbox 계산
            if np.any(mask):
                ys, xs = np.where(mask)
                bbox = (int(ys.min()), int(ys.max() + 1), int(xs.min()), int(xs.max() + 1))
            else:
                bbox = (0, 0, 0, 0)

            return {
                "bgr": bgr,
                "alpha_f": alpha_f,
                "mask": mask,
                "hit_mask": hit_mask,
                "bbox": bbox
            }
        except Exception:
            return None

    def _load_all_buttons(self):
        # 맵/캠 토글 버튼
        self.images["map_clicked"] = self._load_img("map_clicked_button.png")
        self.images["cam_clicked"] = self._load_img("cam_clicked_button.png")

        # 일반 액션 버튼 6개
        for name in self.action_names:
            self.images[f"{name}_normal"] = self._load_img(f"{name}_normal.png")
            self.images[f"{name}_clicked"] = self._load_img(f"{name}_clicked.png")

    def handle_mouse_down(self, x: int, y: int) -> Optional[str]:
        if not (0 <= x < self.window_w and 0 <= y < self.window_h):
            return None

        # 1. 맵/캠 토글 버튼 클릭 확인
        cur_map_cam_key = "map_clicked" if self.map_cam_mode == "MAP" else "cam_clicked"
        alt_map_cam_key = "cam_clicked" if self.map_cam_mode == "MAP" else "map_clicked"
        map_cam_data = self.images.get(cur_map_cam_key)
        alt_map_cam_data = self.images.get(alt_map_cam_key)

        is_map_cam_hit = False
        if map_cam_data and map_cam_data.get("hit_mask", map_cam_data["mask"])[y, x]:
            is_map_cam_hit = True
        elif alt_map_cam_data and alt_map_cam_data.get("hit_mask", alt_map_cam_data["mask"])[y, x]:
            is_map_cam_hit = True

        if is_map_cam_hit:
            now = time.time()
            if now - self.last_toggle_time < 0.28:
                return f"map_cam_{self.map_cam_mode.lower()}"
            self.last_toggle_time = now
            self.map_cam_mode = "CAM" if self.map_cam_mode == "MAP" else "MAP"
            print(f"[UI] 모드 전환: {self.map_cam_mode}")
            return f"map_cam_{self.map_cam_mode.lower()}"

        # 2. 액션 버튼 클릭 확인
        for name in self.action_names:
            normal_data = self.images.get(f"{name}_normal")
            clicked_data = self.images.get(f"{name}_clicked")
            is_hit = False
            if normal_data and normal_data.get("hit_mask", normal_data["mask"])[y, x]:
                is_hit = True
            elif clicked_data and clicked_data.get("hit_mask", clicked_data["mask"])[y, x]:
                is_hit = True

            if is_hit:
                self.pressed_button = name
                print(f"[UI] {name.upper()} 버튼 클릭")
                return name

        return None

    def handle_mouse_up(self, x: int, y: int) -> Optional[str]:
        released_btn = self.pressed_button
        if self.pressed_button is not None:
            self.pressed_button = None
        return released_btn

    def render_buttons(self, canvas: np.ndarray) -> np.ndarray:
        # 1. 맵/캠 버튼 합성
        cur_map_cam_key = "map_clicked" if self.map_cam_mode == "MAP" else "cam_clicked"
        mc_data = self.images.get(cur_map_cam_key)
        if mc_data:
            self._blend_layer_fast(canvas, mc_data)

        # 2. 액션 버튼 6개 합성
        for name in self.action_names:
            key = f"{name}_clicked" if self.pressed_button == name else f"{name}_normal"
            btn_data = self.images.get(key)
            if btn_data:
                self._blend_layer_fast(canvas, btn_data)

        return canvas

    @staticmethod
    def _blend_layer_fast(canvas: np.ndarray, layer_data: Dict):
        y1, y2, x1, x2 = layer_data["bbox"]
        if y1 >= y2 or x1 >= x2:
            return

        sub_mask = layer_data["mask"][y1:y2, x1:x2]
        sub_bgr = layer_data["bgr"][y1:y2, x1:x2]
        sub_alpha = layer_data["alpha_f"][y1:y2, x1:x2, np.newaxis]
        sub_canvas = canvas[y1:y2, x1:x2]

        # BBox 내부의 마스크 영역만 고속 벡터 연산
        blended = (sub_bgr * sub_alpha + sub_canvas * (1.0 - sub_alpha)).astype(np.uint8)
        canvas[y1:y2, x1:x2] = np.where(sub_mask[:, :, np.newaxis], blended, sub_canvas)
