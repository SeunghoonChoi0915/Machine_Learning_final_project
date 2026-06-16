import cv2
import os
import time
import re

# =========================
# 설정
# =========================

CLASS_NAME = "p5"      # b1, b2, s1, l1, p1, none 등으로 바꿔서 촬영
BACKGROUND = "blue"   # 배경(세션): desk, gray = 학습용 / wood = 평가용(val+test)
SAVE_DIR = f"/home/choi/cnn_pictures/dataset/{CLASS_NAME}/{BACKGROUND}"

RECORD_SECONDS = 30    # 한 번 녹화할 시간(초)
FPS_SAVE = 3           # 1초당 저장할 사진 수
SAVE_INTERVAL = 1.0 / FPS_SAVE

CAMERA_INDEX = 2
WIDTH = 640
HEIGHT = 480


def last_index(save_dir, class_name):
    pat = re.compile(rf"^{class_name}_(\d+)\.jpg$")
    nums = [int(m.group(1)) for f in os.listdir(save_dir)
            if (m := pat.match(f))]
    return max(nums, default=0)


os.makedirs(SAVE_DIR, exist_ok=True)

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
print(f"space: {RECORD_SECONDS}초 녹화 시작/중지 (1초당 {FPS_SAVE}장 자동 저장)")
print("s: 단발 사진 저장")
print("q: 종료")

# 기존 파일 번호 이어서 저장 (len() 대신 최대 번호 기준이라 중간에 지워도 안전)
count = last_index(SAVE_DIR, CLASS_NAME)

recording = False
record_start = 0.0
last_save = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    now = time.time()

    # =========================
    # 녹화 중이면 일정 간격으로 자동 저장
    # =========================
    if recording:
        if now - last_save >= SAVE_INTERVAL:
            count += 1
            filename = f"{CLASS_NAME}_{count:04d}.jpg"
            filepath = os.path.join(SAVE_DIR, filename)

            resized = cv2.resize(frame, (WIDTH, HEIGHT))
            cv2.imwrite(filepath, resized)
            last_save = now
            print(f"저장됨: {filepath}")

        # 설정 시간이 지나면 자동 종료
        if now - record_start >= RECORD_SECONDS:
            recording = False
            print(f"녹화 종료 (총 {count}장까지 저장)")

    # =========================
    # 화면 표시
    # =========================
    display = frame.copy()

    cv2.putText(display, f"Class: {CLASS_NAME}  /  BG: {BACKGROUND}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(display, f"Saved: {count}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if recording:
        remain = RECORD_SECONDS - (now - record_start)
        cv2.putText(display, f"REC {remain:4.1f}s", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        # 녹화 중 표시용 빨간 원
        cv2.circle(display, (WIDTH - 30, 30), 10, (0, 0, 255), -1)

    cv2.putText(display, "space: record  /  s: single shot  /  q: quit", (10, 455),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Halli Galli Dataset Capture", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == 32:  # space: 녹화 시작/중지 토글
        if not recording:
            recording = True
            record_start = time.time()
            last_save = 0.0  # 시작하자마자 첫 장 저장
            print(f"녹화 시작: {RECORD_SECONDS}초 동안 1초당 {FPS_SAVE}장 저장")
        else:
            recording = False
            print(f"녹화 조기 종료 (총 {count}장까지 저장)")

    elif key == ord("s"):  # 단발 촬영
        count += 1
        filename = f"{CLASS_NAME}_{count:04d}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)

        resized = cv2.resize(frame, (WIDTH, HEIGHT))
        cv2.imwrite(filepath, resized)
        print(f"저장됨: {filepath}")

cap.release()
cv2.destroyAllWindows()

# CLASS_NAME = "p1"      # b1, b2, s1, l1, p1, none 등으로 바꿔서 촬영
# BACKGROUND = "desk"   # 배경(세션): desk, gray = 학습용 / wood = 평가용(val+test)
# SAVE_DIR = f"/home/choi/cnn_pictures/dataset/{CLASS_NAME}/{BACKGROUND}"

# RECORD_SECONDS = 20    # 한 번 녹화할 시간(초)
# FPS_SAVE = 3           # 1초당 저장할 사진 수
# SAVE_INTERVAL = 1.0 / FPS_SAVE

# CAMERA_INDEX = 0
# WIDTH = 640
# HEIGHT = 480


# def last_index(save_dir, class_name):
#     pat = re.compile(rf"^{class_name}_(\d+)\.jpg$")
#     nums = [int(m.group(1)) for f in os.listdir(save_dir)
#             if (m := pat.match(f))]
#     return max(nums, default=0)


# os.makedirs(SAVE_DIR, exist_ok=True)

# # =========================
# # 카메라 열기
# # =========================
# cap = cv2.VideoCapture(CAMERA_INDEX)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# if not cap.isOpened():
#     print("웹캠을 열 수 없습니다.")
#     exit()

# print(f"저장 폴더: {SAVE_DIR}")
# print(f"space: {RECORD_SECONDS}초 녹화 시작/중지 (1초당 {FPS_SAVE}장 자동 저장)")
# print("s: 단발 사진 저장")
# print("q: 종료")

# # 기존 파일 번호 이어서 저장 (len() 대신 최대 번호 기준이라 중간에 지워도 안전)
# count = last_index(SAVE_DIR, CLASS_NAME)

# recording = False
# record_start = 0.0
# last_save = 0.0

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("프레임을 읽을 수 없습니다.")
#         break

#     now = time.time()

#     # =========================
#     # 녹화 중이면 일정 간격으로 자동 저장
#     # =========================
#     if recording:
#         if now - last_save >= SAVE_INTERVAL:
#             count += 1
#             filename = f"{CLASS_NAME}_{count:04d}.jpg"
#             filepath = os.path.join(SAVE_DIR, filename)

#             resized = cv2.resize(frame, (WIDTH, HEIGHT))
#             cv2.imwrite(filepath, resized)
#             last_save = now
#             print(f"저장됨: {filepath}")

#         # 설정 시간이 지나면 자동 종료
#         if now - record_start >= RECORD_SECONDS:
#             recording = False
#             print(f"녹화 종료 (총 {count}장까지 저장)")

#     # =========================
#     # 화면 표시
#     # =========================
#     display = frame.copy()

#     cv2.putText(display, f"Class: {CLASS_NAME}  /  BG: {BACKGROUND}", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#     cv2.putText(display, f"Saved: {count}", (10, 65),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#     if recording:
#         remain = RECORD_SECONDS - (now - record_start)
#         cv2.putText(display, f"REC {remain:4.1f}s", (10, 100),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
#         # 녹화 중 표시용 빨간 원
#         cv2.circle(display, (WIDTH - 30, 30), 10, (0, 0, 255), -1)

#     cv2.putText(display, "space: record  /  s: single shot  /  q: quit", (10, 455),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#     cv2.imshow("Halli Galli Dataset Capture", display)

#     key = cv2.waitKey(1) & 0xFF

#     if key == ord("q"):
#         break

#     elif key == 32:  # space: 녹화 시작/중지 토글
#         if not recording:
#             recording = True
#             record_start = time.time()
#             last_save = 0.0  # 시작하자마자 첫 장 저장
#             print(f"녹화 시작: {RECORD_SECONDS}초 동안 1초당 {FPS_SAVE}장 저장")
#         else:
#             recording = False
#             print(f"녹화 조기 종료 (총 {count}장까지 저장)")

#     elif key == ord("s"):  # 단발 촬영
#         count += 1
#         filename = f"{CLASS_NAME}_{count:04d}.jpg"
#         filepath = os.path.join(SAVE_DIR, filename)

#         resized = cv2.resize(frame, (WIDTH, HEIGHT))
#         cv2.imwrite(filepath, resized)
#         print(f"저장됨: {filepath}")

# cap.release()
# cv2.destroyAllWindows()