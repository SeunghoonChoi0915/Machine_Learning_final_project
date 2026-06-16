import cv2
import os
from datetime import datetime
import re
# =========================
# 설정
# =========================

CLASS_NAME = "p5"     # 여기만 b1, b2, s1, l1, p1 등으로 바꿔서 촬영
BACKGROUND = "green"   # 배경(세션): desk, gray = 학습용 / wood = 평가용(val+test)
# SAVE_DIR = f"/home/choi/cnn_pictures/dataset/{BACKGROUND}/{CLASS_NAME}"
SAVE_DIR = f"/home/choi/cnn_pictures/dataset/{CLASS_NAME}/{BACKGROUND}"

def last_index(save_dir, class_name):
    pat = re.compile(rf"^{class_name}_(\d+)\.jpg$")
    nums = [int(m.group(1)) for f in os.listdir(save_dir)
            if (m := pat.match(f))]
    return max(nums, default=0)

# count = last_index(SAVE_DIR, CLASS_NAME)
os.makedirs(SAVE_DIR, exist_ok=True)


CAMERA_INDEX = 2
WIDTH = 640
HEIGHT = 480

# =========================
# 카메라 열기
# =========================
cap = cv2.VideoCapture(CAMERA_INDEX)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

print(f"저장 폴더: {SAVE_DIR}")
print("s 또는 space: 사진 저장")
print("q: 종료")

count = len([f for f in os.listdir(SAVE_DIR) if f.endswith(".jpg")])

while True:
    ret, frame = cap.read()

    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    display = frame.copy()

    cv2.putText(display, f"Class: {CLASS_NAME}  /  BG: {BACKGROUND}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(display, f"Saved: {count}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(display, "Press s/space to save, q to quit", (10, 455),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Halli Galli Dataset Capture", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("s") or key == 32:
        count += 1

        filename = f"{CLASS_NAME}_{count:04d}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)

        resized = cv2.resize(frame, (WIDTH, HEIGHT))
        cv2.imwrite(filepath, resized)

        print(f"저장됨: {filepath}")

cap.release()
cv2.destroyAllWindows()