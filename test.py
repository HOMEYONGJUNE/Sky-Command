import os
import onnxruntime as ort
import numpy as np

# 1. 맥 바탕화면의 onboard.onnx 파일 경로 설정
desktop_path = os.path.expanduser("~/Desktop")
model_path = os.path.join(desktop_path, "onboard.onnx")

# 파일 존재 여부 확인
if not os.path.exists(model_path):
    print(f"Error: 파일을 찾을 수 없습니다 -> {model_path}")
    exit()

print(f"모델 불러오는 중: {model_path}")

# 2. ONNX Runtime 세션 생성
session = ort.InferenceSession(model_path)

# 3. 모델의 입력 및 출력 정보 확인
print("\n=== 모델 입력 정보 ===")
input_details = []
for inp in session.get_inputs():
    print(f"이름: {inp.name}, 모양(Shape): {inp.shape}, 타입: {inp.type}")
    input_details.append(inp)

print("\n=== 모델 출력 정보 ===")
for out in session.get_outputs():
    print(f"이름: {out.name}, 모양(Shape): {out.shape}, 타입: {out.type}")

# 4. 테스트용 가상 입력 데이터 생성 (첫 번째 입력 기준)
first_input = input_details[0]

# dynamic shape(가변 차원) 처리: None 또는 문자열일 경우 1로 대체
input_shape = [dim if isinstance(dim, int) else 1 for dim in first_input.shape]

# 데이터 타입에 맞춰 무작위 배열 생성 (기본적으로 float32 사용)
dummy_input = np.random.randn(*input_shape).astype(np.float32)

# 5. 추론 실행
inputs = {first_input.name: dummy_input}
outputs = session.run(None, inputs)

print("\n=== 추론 성공! ===")
print("입력 데이터 Shape:", dummy_input.shape)
print("출력 결과 Shape:", outputs[0].shape)
print("출력 결과 일부:", outputs[0])