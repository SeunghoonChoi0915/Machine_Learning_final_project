import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from torchvision import models, transforms
from torchvision.models import resnet18

# =========================================================
# 설정
# =========================================================
DETECTOR_PATH   = "/home/choi/cnn_pictures/detector_best.pth"
CLASSIFIER_PATH = "/home/choi/cnn_pictures/best_model_crops_v3.pth"
CAMERA_INDEX    = 2
WIDTH, HEIGHT   = 640, 480

OBJ_THRESH  = 0.5
CLS_THRESH  = 0.7
NMS_IOU     = 0.4            # 이 이상 겹치면 같은 카드로 보고 제거
MAX_CARDS   = 10            # 한 프레임 최대 인식 카드 수

IMG_W, IMG_H    = 640, 480
STRIDE          = 32
GRID_W, GRID_H  = IMG_W // STRIDE, IMG_H // STRIDE
NUM_ANCHORS     = 5

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], np.float32)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSIFY_TF = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 과일별 박스 색 (BGR)
FRUIT_COLORS = {
    "b": (0, 215, 255),   # 바나나 - 노랑  (BGR)
    "s": (0, 0, 220),     # 딸기   - 빨강
    "l": (50, 205, 50),   # 라임   - 연두
    "p": (180, 0, 180),   # 자두   - 보라
}
DEFAULT_COLOR = (200, 200, 200)  # 알 수 없는 클래스

# 합계 패널 표기용 이름 (cv2.putText 는 한글이 깨지므로 영어)
FRUIT_NAMES = {
    "b": "BANANA",
    "s": "STRAWBERRY",
    "l": "LIME",
    "p": "PLUM",
}

BELL_TARGET = 5   # 할리갈리: 같은 과일 합이 정확히 이 값이면 종!


# =========================================================
# anchor grid / 모델 / quad decode
# =========================================================
def build_anchor_grid(anchor_wh):
    ys, xs = np.meshgrid(np.arange(GRID_H), np.arange(GRID_W), indexing="ij")
    cx = (xs + 0.5) * STRIDE
    cy = (ys + 0.5) * STRIDE
    K  = len(anchor_wh)
    grid = np.zeros((GRID_H, GRID_W, K, 4), np.float32)
    grid[..., 0] = cx[..., None]
    grid[..., 1] = cy[..., None]
    grid[..., 2] = anchor_wh[None, None, :, 0]
    grid[..., 3] = anchor_wh[None, None, :, 1]
    return grid.reshape(-1, 4)


class CardDetector(nn.Module):
    def __init__(self, num_anchors=NUM_ANCHORS):
        super().__init__()
        backbone = resnet18(weights=None)
        self.body = nn.Sequential(*list(backbone.children())[:-2])
        self.head = nn.Conv2d(512, num_anchors * 13, kernel_size=1)
        self.K    = num_anchors

    def forward(self, x):
        f   = self.body(x)
        out = self.head(f)
        B, _, Hf, Wf = out.shape
        out = out.view(B, self.K, 13, Hf, Wf)
        out = out.permute(0, 3, 4, 1, 2).reshape(B, -1, 13)
        return out[..., 0], out[..., 1:5], out[..., 5:13]


def decode_quad(offsets8, anchor):
    off = offsets8.view(4, 2)
    cx, cy = anchor[0], anchor[1]
    w,  h  = anchor[2], anchor[3]
    x = off[:, 0] * w + cx
    y = off[:, 1] * h + cy
    return torch.stack([x, y], dim=1)


def load_detector(path):
    ckpt       = torch.load(path, map_location=DEVICE)
    anchor_wh  = np.array(ckpt["anchors_wh"], np.float32)
    anchors_t  = torch.from_numpy(build_anchor_grid(anchor_wh)).to(DEVICE)
    model = CardDetector().to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    # print(f"[검출기] epoch={ckpt.get('epoch','?')}  mIoU={ckpt.get('miou',0):.3f}  "
    #       f"recall={ckpt.get('recall',0):.3f}  꼭짓점오차={ckpt.get('corner_err',0):.1f}px")
    return model, anchors_t


def load_classifier(path):
    ckpt    = torch.load(path, map_location=DEVICE)
    classes = ckpt["classes"]
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.fc.in_features, len(classes)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()
    # print(f"[분류기] epoch={ckpt.get('epoch','?')}  val_acc={ckpt.get('val_acc',0):.2f}%  "
    #       f"클래스 {len(classes)}개")
    return model, classes


# =========================================================
# NMS (axis-aligned box 기준)
# =========================================================
def iou_aabb(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def nms(dets, iou_thresh):
    """dets: list of dict(quad, aabb, score) → 중복 제거된 list"""
    dets = sorted(dets, key=lambda d: d["score"], reverse=True)
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if iou_aabb(best["aabb"], d["aabb"]) < iou_thresh]
    return keep


# =========================================================
# 검출 : 프레임 → 여러 개 quad
# =========================================================
def frame_to_tensor(frame_bgr):
    img = frame_bgr.astype(np.float32)[..., ::-1] / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(img.copy()).permute(2, 0, 1).unsqueeze(0).to(DEVICE)


@torch.no_grad()
def detect_multi(detector, anchors_t, frame_bgr, obj_thresh):
    inp = frame_to_tensor(frame_bgr)
    obj_logit, _, quad_off = detector(inp)
    scores = torch.sigmoid(obj_logit[0])             # (A,)

    keep = scores >= obj_thresh
    if keep.sum() == 0:
        return []

    cand_idx = torch.nonzero(keep, as_tuple=False).squeeze(1).tolist()

    dets = []
    for i in cand_idx:
        quad = decode_quad(quad_off[0, i], anchors_t[i])       # (4,2)
        q = quad.cpu().numpy().astype(np.float32)
        q[:, 0] = q[:, 0].clip(0, IMG_W - 1)
        q[:, 1] = q[:, 1].clip(0, IMG_H - 1)
        x1, y1 = q[:, 0].min(), q[:, 1].min()
        x2, y2 = q[:, 0].max(), q[:, 1].max()
        # 면적이 너무 작은 건 제거 (노이즈)
        if (x2 - x1) * (y2 - y1) < (IMG_W * IMG_H) * 0.01:
            continue
        dets.append({"quad": q,
                     "aabb": np.array([x1, y1, x2, y2], np.float32),
                     "score": float(scores[i])})

    return nms(dets, NMS_IOU)[:MAX_CARDS]


def warp_card(frame_bgr, quad_np):
    src = quad_np.astype(np.float32)                 # tl,tr,br,bl
    dst = np.float32([[0, 0], [224, 0], [224, 224], [0, 224]])
    M   = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame_bgr, M, (224, 224))


@torch.no_grad()
def classify(classifier, crop_bgr):
    rgb   = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    x     = CLASSIFY_TF(rgb).unsqueeze(0).to(DEVICE)
    probs = F.softmax(classifier(x), dim=1)[0]
    conf, idx = probs.max(0)
    return int(idx), float(conf)


# =========================================================
# 메인 루프
# =========================================================
def main():
    global OBJ_THRESH, CLS_THRESH
    print(f"디바이스: {DEVICE}")
    detector,  anchors_t = load_detector(DETECTOR_PATH)
    classifier, classes  = load_classifier(CLASSIFIER_PATH)
    print()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    print("q: 종료  /  t: 분류기 threshold  /  o: 검출기 threshold")

    cls_options  = [0.5, 0.7, 0.9]
    det_options  = [0.3, 0.5, 0.7]
    c_idx, d_idx = 1, 1
    CLS_THRESH   = cls_options[c_idx]
    OBJ_THRESH   = det_options[d_idx]

    frame_no  = 0       # 배너 깜빡임용 카운터
    prev_bell = False   # 비프음을 종이 "처음" 울릴 때만 내기 위한 직전 상태

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        display = frame.copy()
        frame_no += 1

        dets = detect_multi(detector, anchors_t, frame, OBJ_THRESH)

        card_count = 0
        fruit_sums = Counter()          # 과일별 개수 합 누적

        for d in dets:
            crop = warp_card(frame, d["quad"])
            idx, cls_conf = classify(classifier, crop)

            if cls_conf < CLS_THRESH:    # 분류기가 거부 → 카드 아님
                continue

            label     = classes[idx]     # 예: "s4"
            fruit_key = label[0]         # "s"
            try:
                count = int(label[1:])   # 4
            except ValueError:
                count = 0
            fruit_sums[fruit_key] += count   # 합계 누적

            color = FRUIT_COLORS.get(fruit_key, DEFAULT_COLOR)
            pts   = d["quad"].astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(display, [pts], isClosed=True, color=color, thickness=2)
            
            # x1, y1, x2, y2 = d["aabb"].astype(np.int32)
            # cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

            # 카드 라벨 (박스 좌상단 위에)
            x1, y1 = int(d["aabb"][0]), int(d["aabb"][1])
            text = f"{label} {cls_conf*100:.0f}%"
            cv2.putText(display, text, (x1, max(y1 - 8, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            card_count += 1

        # ── 할리갈리 룰: 같은 과일 합이 정확히 5 → 종! ───────
        bell_fruits = [k for k, v in fruit_sums.items() if v == BELL_TARGET]
        ring_bell   = len(bell_fruits) > 0

        # 종이 "처음" 울리는 순간에만 비프음 (매 프레임 X)
        if ring_bell and not prev_bell:
            print("\a")   # 시스템 비프음. wav 재생 원하면 아래로 교체:
            # import os; os.system("aplay /home/choi/cnn_pictures/bell.wav &")
        prev_bell = ring_bell

        # ── HUD ───────────────────────────────────────────
        head_color = (0, 255, 0) if card_count > 0 else (0, 0, 255)
        head_text  = f"{card_count} CARD(S)" if card_count > 0 else "NO CARD"
        cv2.putText(display, head_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, head_color, 3)
        cv2.putText(display,
                    f"det_th={OBJ_THRESH:.1f}  cls_th={CLS_THRESH:.1f}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        # ── 과일별 합계 패널 ───────────────────────────────
        y0 = 100
        for i, (key, name) in enumerate(FRUIT_NAMES.items()):
            s     = fruit_sums.get(key, 0)
            hit   = (s == BELL_TARGET)
            col   = FRUIT_COLORS.get(key, DEFAULT_COLOR)
            thick = 2 if hit else 1
            line  = f"{name}: {s}/{BELL_TARGET}" + ("  <- BELL!" if hit else "")
            cv2.putText(display, line, (10, y0 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, thick)

        # ── "종 쳐!" 배너 (깜빡임) ─────────────────────────
        if ring_bell and (frame_no // 8) % 2 == 0:
            banner = "RING THE BELL!"
            scale, th = 1.6, 4
            (tw, tht), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
            cx = (WIDTH - tw) // 2
            cy = HEIGHT // 2
            cv2.rectangle(display, (cx - 20, cy - tht - 20),
                          (cx + tw + 20, cy + 20), (0, 0, 0), -1)
            cv2.putText(display, banner, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), th)

        cv2.putText(display, "q: quit  t: cls_th  o: det_th",
                    (10, HEIGHT - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Halli Galli Demo (multi)", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("t"):
            c_idx = (c_idx + 1) % len(cls_options)
            CLS_THRESH = cls_options[c_idx]
            print(f"분류기 threshold → {CLS_THRESH}")
        elif key == ord("o"):
            d_idx = (d_idx + 1) % len(det_options)
            OBJ_THRESH = det_options[d_idx]
            print(f"검출기 threshold → {OBJ_THRESH}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()