"""
할리갈리 카드 실시간 webcam 분류
=====================================================
- best_model.pth (ResNet18, from-scratch 학습) 로드
- webcam 프레임마다 추론 → 클래스 + 확신도 표시
- confidence threshold: 최대 softmax 확률이 낮으면 "NO CARD" 처리
  (모델은 20개 클래스만 알기 때문에, 카드가 없어도 뭔가를 출력함.
   확신도가 낮을 때 거르는 것이 최소한의 안전장치)
- 최근 N프레임 다수결(smoothing)로 예측 깜빡임 완화

조작:
  q : 종료
  t : threshold 조절 (0.5 → 0.7 → 0.9 순환)
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, Counter
from torchvision import models, transforms

# =========================================================
# 설정
# =========================================================
MODEL_PATH   = "/home/choi/cnn_pictures/best_model_v2.pth"
CAMERA_INDEX = 0          # saving.py와 동일한 카메라
WIDTH, HEIGHT = 640, 480

CONF_THRESHOLD = 0.7      # 이 미만이면 "NO CARD / UNCERTAIN"
SMOOTH_WINDOW  = 5        # 최근 N프레임 다수결 (1이면 smoothing 끔)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# !! 중요 !!
# 아래 transform은 augmentation.py의 val_transform과 "반드시" 동일해야 합니다.
# (Resize 크기, Normalize 평균/표준편차가 다르면 정확도가 크게 떨어짐)
INFER_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# =========================================================
# 모델 로드 (train.py의 build_model과 구조 동일해야 함)
# =========================================================
def load_model(path):
    ckpt = torch.load(path, map_location=DEVICE)
    classes = ckpt["classes"]

    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, len(classes)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()

    print(f"모델 로드 완료: epoch {ckpt.get('epoch', '?')}, "
          f"val_acc {ckpt.get('val_acc', 0):.2f}%")
    print(f"클래스 {len(classes)}개: {classes}")
    return model, classes


# =========================================================
# 한 프레임 추론
# =========================================================
@torch.no_grad()
def predict(model, frame_bgr):
    # OpenCV는 BGR → 학습은 RGB(PIL) 기준이므로 반드시 변환
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    x = INFER_TRANSFORM(rgb).unsqueeze(0).to(DEVICE)

    logits = model(x)
    probs = F.softmax(logits, dim=1)[0]
    conf, idx = probs.max(0)
    return idx.item(), conf.item(), probs.cpu()


# =========================================================
# 메인 루프
# =========================================================
def main():
    global CONF_THRESHOLD

    model, classes = load_model(MODEL_PATH)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    print("q: 종료  /  t: threshold 변경")

    recent = deque(maxlen=SMOOTH_WINDOW)   # 최근 프레임 예측 (다수결용)
    thresholds = [0.5, 0.7, 0.9]
    t_idx = thresholds.index(CONF_THRESHOLD) if CONF_THRESHOLD in thresholds else 1

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        idx, conf, probs = predict(model, frame)

        # ---- threshold + smoothing ----
        if conf >= CONF_THRESHOLD:
            recent.append(idx)
        else:
            recent.append(-1)              # -1 = 불확실

        vote, _ = Counter(recent).most_common(1)[0]

        display = frame.copy()
        if vote == -1:
            label_text = "NO CARD / UNCERTAIN"
            color = (0, 0, 255)            # 빨강
        else:
            label_text = f"{classes[vote]}"
            color = (0, 255, 0)            # 초록

        # ---- 화면 표시 ----
        cv2.putText(display, label_text, (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
        cv2.putText(display, f"conf: {conf*100:.1f}%  (th: {CONF_THRESHOLD:.1f})",
                    (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # 상위 3개 후보 표시 (디버깅에 유용)
        top3 = torch.topk(probs, 3)
        for i, (p, ci) in enumerate(zip(top3.values, top3.indices)):
            cv2.putText(display, f"{classes[ci]}: {p*100:.1f}%",
                        (10, 120 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.putText(display, "q: quit  /  t: threshold", (10, HEIGHT - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Halli Galli Live Classification", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("t"):
            t_idx = (t_idx + 1) % len(thresholds)
            CONF_THRESHOLD = thresholds[t_idx]
            print(f"threshold → {CONF_THRESHOLD}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()