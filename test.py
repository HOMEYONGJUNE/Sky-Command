from gpiozero import DistanceSensor
from time import sleep

# GPIO 14(Trig), GPIO 15(Echo) 설정
# (1.8k 오옴 저항으로 분압된 Echo 신호선이 GPIO 15에 연결되어 있어야 합니다)
sensor = DistanceSensor(echo=15, trigger=14, max_distance=2.0)

print("Ultrasonic distance measurement started (exit: Ctrl+C)")
print("-" * 35)

try:
    while True:
        # DistanceSensor는 기본적으로 미터(m) 단위로 반환하므로 100을 곱해 cm로 변환
        distance_cm = sensor.distance * 100
        print(f"Distance: {distance_cm:.1f} cm")
        sleep(0.5)

except KeyboardInterrupt:
    print("\nMeasurement stopped.")