import cv2
import numpy as np
import sys
import io

WINDOW_W, WINDOW_H = 1280, 720

TEXT_BOX_X1, TEXT_BOX_Y1 = 1090, 20
TEXT_BOX_X2, TEXT_BOX_Y2 = 1280, 520
TEXT_BOX_W = TEXT_BOX_X2 - TEXT_BOX_X1  
TEXT_BOX_H = TEXT_BOX_Y2 - TEXT_BOX_Y1  

class TerminalLogger(io.StringIO):
    def __init__(self):
        super().__init__()
        self.log_lines = []

    def write(self, s):
        sys.__stdout__.write(s)
        if s.strip('\r\n'):
            for line in s.splitlines():
                if line.strip():
                    self.log_lines.append(line)
                    if len(self.log_lines) > 50:
                        self.log_lines.pop(0)

logger = TerminalLogger()
sys.stdout = logger

#1280x720 규격 UI 로드
ui_path = r'/Users/hong-yongjun/Desktop/Project/ui.png'
try:
    img_array = np.fromfile(ui_path, np.uint8)
    orig_ui = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
except Exception as e:
    sys.stdout = sys.__stdout__
    print(f"[에러] UI 이미지를 읽는 도중 오류 발생: {e}")
    sys.exit()

if orig_ui is None:
    sys.stdout = sys.__stdout__
    print("[에러] UI 이미지를 로드하지 못했습니다. 경로를 확인하세요.")
    sys.exit()

ui_img = cv2.resize(orig_ui, (WINDOW_W, WINDOW_H))

if ui_img.shape[2] == 4:
    ui_bgr = ui_img[:, :, :3]
    ui_alpha = ui_img[:, :, 3] / 255.0  
else:
    ui_bgr = ui_img
    ui_alpha = np.ones((WINDOW_H, WINDOW_W), dtype=np.float32)

#카메라 설정
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_H)
camera_connected = cap.isOpened()

if not camera_connected:
    print("Camera initialized in 1280x720 Dummy Mode.")
else:
    print("Camera standard stream 1280x720 connected successfully.")

cv2.namedWindow("RC Car Control System", cv2.WINDOW_AUTOSIZE)
frame_count = 0

while True:
    frame_count += 1
    if frame_count % 30 == 0:  # 더 자주 로그가 찍히도록 주기를 30으로 단축
        print(f"Frame Count: {frame_count}")
        print("Signal Packets Status Standard OK.")

    if camera_connected:
        ret, frame = cap.read()
        if not ret or frame is None:
            frame = np.zeros((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)
    else:
        frame = np.zeros((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)
        cv2.putText(frame, "No Camera Signal (1280x720)", (40, WINDOW_H // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # [단계 1] 베이스 카메라 화면 깔기
    display_output = frame.copy()

    # [단계 2] 1280x720 UI 레이어를 먼저 오버레이 (텍스트보다 밑에 깔리게)
    for c in range(0, 3):
        display_output[:, :, c] = (ui_bgr[:, :, c] * ui_alpha) + (display_output[:, :, c] * (1.0 - ui_alpha))

    #UI 레이어 위에 콘솔 텍스트 출력하기
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_color = (255, 255, 255)  # 흰색
    line_height = 15  
    max_lines = TEXT_BOX_H // line_height
    
    wrapped_lines = []
    for raw_line in logger.log_lines:
        current_phrase = ""
        for char in raw_line:
            test_phrase = current_phrase + char
            text_size = cv2.getTextSize(test_phrase, font, font_scale, 1)[0]
            if text_size[0] > TEXT_BOX_W - 10:  
                wrapped_lines.append(current_phrase)
                current_phrase = char
            else:
                current_phrase = test_phrase
        if current_phrase:
            wrapped_lines.append(current_phrase)

    display_lines = wrapped_lines[-max_lines:]

    # 이제 UI 가림막 위 최전방에 글씨를 뿌립니다
    for i, line in enumerate(display_lines):
        draw_y = TEXT_BOX_Y1 + (i + 1) * line_height
        cv2.putText(display_output, line, (TEXT_BOX_X1 + 5, draw_y), font, font_scale, font_color, 1, cv2.LINE_AA)

    #일반 계기판 데이터 매핑
    cv2.putText(display_output, "18 km/h", (200, 650), font, 0.6, (0, 255, 0), 2)
    
    cv2.imshow("RC Car Control System", display_output)

    cv2.imshow("RC Car Control System", display_output)


    key = cv2.waitKey(1) & 0xFF
    

    if key == ord('q') or key == ord('Q') or key == 27:
        break

    if cv2.getWindowProperty("RC Car Control System", cv2.WND_PROP_VISIBLE) < 1:
        break

sys.stdout = sys.__stdout__
if camera_connected:
    cap.release()
cv2.destroyAllWindows()