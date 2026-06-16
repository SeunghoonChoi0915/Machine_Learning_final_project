# """
# webcam_demo.py
# ==============
# End-to-end 파이프라인:
#   1. 웹캠 프레임 → CardDetector (detector_best.pth, anchor 기반)
#   2. 검출된 quad (tl→tr→br→bl) → perspective warp (224×224)
#   3. warp 크롭 → best_model_crops.pth (ResNet18 분류기)

# 조작:
#   q : 종료
#   t : 분류기 threshold 순환 (0.5→0.7→0.9)
#   o : 검출기 threshold 순환 (0.3→0.5→0.7)
# """

# import cv2
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from collections import deque, Counter
# from torchvision import models, transforms
# from torchvision.models import resnet18

# # =========================================================
# # 설정
# # =========================================================
# DETECTOR_PATH   = "/home/choi/cnn_pictures/detector_best.pth"
# CLASSIFIER_PATH = "/home/choi/cnn_pictures/best_model_crops.pth"
# CAMERA_INDEX    = 2           # 필요 시 2로 변경
# WIDTH, HEIGHT   = 640, 480

# OBJ_THRESH  = 0.5             # 검출기 objectness threshold
# CLS_THRESH  = 0.7             # 분류기 confidence threshold
# SMOOTH_WIN  = 5               # smoothing 프레임 수

# # train_detector_v2.py 와 반드시 동일해야 함
# IMG_W, IMG_H    = 640, 480
# STRIDE          = 32
# GRID_W, GRID_H  = IMG_W // STRIDE, IMG_H // STRIDE   # 20 × 15
# NUM_ANCHORS     = 5

# IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
# IMAGENET_STD  = np.array([0.229, 0.224, 0.225], np.float32)

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CLASSIFY_TF = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]),
# ])


# # =========================================================
# # anchor grid  (train_detector_v2.py : build_anchor_grid 동일)
# # =========================================================
# def build_anchor_grid(anchor_wh: np.ndarray) -> np.ndarray:
#     ys, xs = np.meshgrid(np.arange(GRID_H), np.arange(GRID_W), indexing="ij")
#     cx = (xs + 0.5) * STRIDE
#     cy = (ys + 0.5) * STRIDE
#     K  = len(anchor_wh)
#     grid = np.zeros((GRID_H, GRID_W, K, 4), np.float32)
#     grid[..., 0] = cx[..., None]
#     grid[..., 1] = cy[..., None]
#     grid[..., 2] = anchor_wh[None, None, :, 0]
#     grid[..., 3] = anchor_wh[None, None, :, 1]
#     return grid.reshape(-1, 4)   # (GRID_H × GRID_W × K, 4) = (1500, 4)


# # =========================================================
# # CardDetector  (train_detector_v2.py : CardDetector 동일)
# # =========================================================
# class CardDetector(nn.Module):
#     def __init__(self, num_anchors=NUM_ANCHORS):
#         super().__init__()
#         backbone = resnet18(weights=None)
#         self.body = nn.Sequential(*list(backbone.children())[:-2])
#         self.head = nn.Conv2d(512, num_anchors * 13, kernel_size=1)
#         self.K    = num_anchors

#     def forward(self, x):
#         f   = self.body(x)
#         out = self.head(f)
#         B, _, Hf, Wf = out.shape
#         out = out.view(B, self.K, 13, Hf, Wf)
#         out = out.permute(0, 3, 4, 1, 2).reshape(B, -1, 13)  # (B, A, 13)
#         return out[..., 0], out[..., 1:5], out[..., 5:13]     # obj, box, quad


# # =========================================================
# # quad decode  (train_detector_v2.py : decode_quad_t 동일)
# # =========================================================
# def decode_quad(offsets8: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
#     """
#     offsets8 : (8,) — 학습 시 encode_quad 반대
#     anchor   : (4,) — cxcywh
#     반환      : (4, 2) 픽셀좌표, 순서 tl→tr→br→bl
#     """
#     off = offsets8.view(4, 2)
#     cx, cy = anchor[0], anchor[1]
#     w,  h  = anchor[2], anchor[3]
#     x = off[:, 0] * w + cx
#     y = off[:, 1] * h + cy
#     return torch.stack([x, y], dim=1)   # (4, 2)


# # =========================================================
# # 모델 로드
# # =========================================================
# def load_detector(path):
#     ckpt       = torch.load(path, map_location=DEVICE)
#     anchor_wh  = np.array(ckpt["anchors_wh"], np.float32)
#     anchors_np = build_anchor_grid(anchor_wh)          # (1500, 4) numpy
#     anchors_t  = torch.from_numpy(anchors_np).to(DEVICE)

#     model = CardDetector().to(DEVICE)
#     model.load_state_dict(ckpt["model"])
#     model.eval()

#     print(f"[검출기] epoch={ckpt.get('epoch','?')}  "
#           f"mIoU={ckpt.get('miou',0):.3f}  "
#           f"recall={ckpt.get('recall',0):.3f}  "
#           f"꼭짓점오차={ckpt.get('corner_err',0):.1f}px")
#     return model, anchors_t


# def load_classifier(path):
#     ckpt    = torch.load(path, map_location=DEVICE)
#     classes = ckpt["classes"]

#     model = models.resnet18(weights=None)
#     model.fc = nn.Sequential(
#         nn.Dropout(p=0.3),
#         nn.Linear(model.fc.in_features, len(classes)),
#     )
#     model.load_state_dict(ckpt["model_state"])
#     model.to(DEVICE).eval()

#     print(f"[분류기] epoch={ckpt.get('epoch','?')}  "
#           f"val_acc={ckpt.get('val_acc',0):.2f}%  "
#           f"클래스 {len(classes)}개")
#     return model, classes


# # =========================================================
# # 검출 : 프레임 → quad + objectness score
# # =========================================================
# def frame_to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
#     """BGR ndarray → (1,3,H,W) normalized tensor"""
#     img = frame_bgr.astype(np.float32)[..., ::-1] / 255.0   # BGR→RGB
#     img = (img - IMAGENET_MEAN) / IMAGENET_STD
#     return torch.from_numpy(img.copy()).permute(2, 0, 1).unsqueeze(0).to(DEVICE)


# @torch.no_grad()
# def detect(detector, anchors_t, frame_bgr, obj_thresh):
#     """
#     반환: (warped_224 BGR, quad_pts (4,2) int32, obj_score)
#          카드 없으면 (None, None, best_score)
#     """
#     inp                        = frame_to_tensor(frame_bgr)
#     obj_logit, _, quad_off     = detector(inp)               # (1,A) / (1,A,4) / (1,A,8)
#     scores                     = torch.sigmoid(obj_logit[0]) # (A,)
#     best_idx                   = int(scores.argmax())
#     best_score                 = float(scores[best_idx])

#     if best_score < obj_thresh:
#         return None, None, best_score

#     # quad 디코드 → (4,2) 픽셀좌표 tl→tr→br→bl
#     quad = decode_quad(quad_off[0, best_idx], anchors_t[best_idx])  # (4,2) tensor
#     quad_np = quad.cpu().numpy().astype(np.float32)

#     # 화면 밖으로 나가는 좌표 클리핑
#     quad_np[:, 0] = quad_np[:, 0].clip(0, IMG_W - 1)
#     quad_np[:, 1] = quad_np[:, 1].clip(0, IMG_H - 1)

#     # perspective warp → 224×224
#     src = quad_np                                             # tl,tr,br,bl
#     dst = np.float32([[0, 0], [224, 0], [224, 224], [0, 224]])
#     M   = cv2.getPerspectiveTransform(src, dst)
#     warped = cv2.warpPerspective(frame_bgr, M, (224, 224))

#     return warped, quad_np.astype(np.int32), best_score


# # =========================================================
# # 분류
# # =========================================================
# @torch.no_grad()
# def classify(classifier, crop_bgr):
#     rgb    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
#     x      = CLASSIFY_TF(rgb).unsqueeze(0).to(DEVICE)
#     probs  = F.softmax(classifier(x), dim=1)[0]
#     conf, idx = probs.max(0)
#     return int(idx), float(conf), probs.cpu()


# # =========================================================
# # 메인 루프
# # =========================================================
# def main():
#     print(f"디바이스: {DEVICE}")
#     detector,  anchors_t = load_detector(DETECTOR_PATH)
#     classifier, classes  = load_classifier(CLASSIFIER_PATH)
#     print()

#     cap = cv2.VideoCapture(CAMERA_INDEX)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
#     if not cap.isOpened():
#         print("웹캠을 열 수 없습니다.")
#         return

#     print("q: 종료  /  t: 분류기 threshold  /  o: 검출기 threshold")

#     recent        = deque(maxlen=SMOOTH_WIN)
#     cls_options   = [0.5, 0.7, 0.9]
#     det_options   = [0.3, 0.5, 0.7]
#     c_idx, d_idx  = 1, 1
#     cls_thresh    = cls_options[c_idx]
#     det_thresh    = det_options[d_idx]

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         display = frame.copy()

#         # ── 검출 ──────────────────────────────────────────
#         crop, quad_pts, det_score = detect(detector, anchors_t, frame, det_thresh)

#         cls_conf = 0.0
#         probs    = None

#         # if crop is not None:
#         #     # ── 분류 ──────────────────────────────────────
#         #     idx, cls_conf, probs = classify(classifier, crop)
#         #     recent.append(idx if cls_conf >= cls_thresh else -1)

#         #     # quad 윤곽선 (노란색)
#         #     pts_draw = quad_pts.reshape((-1, 1, 2))
#         #     cv2.polylines(display, [pts_draw], isClosed=True,
#         #                   color=(0, 255, 255), thickness=2)

#         #     # 우상단 warp 크롭 미리보기
#         #     preview = cv2.resize(crop, (150, 150))
#         #     display[0:150, WIDTH - 150:WIDTH] = preview
#         #     cv2.rectangle(display, (WIDTH - 150, 0), (WIDTH, 150), (0, 255, 255), 1)
#         # else:
#         #     recent.append(-1)
#         if crop is not None:
#             # ── 분류 ──────────────────────────────────────
#             idx, cls_conf, probs = classify(classifier, crop)
#             recent.append(idx if cls_conf >= cls_thresh else -1)

#             if cls_conf >= cls_thresh:   # ← 이 줄 추가
#                 # quad 윤곽선 (노란색)
#                 pts_draw = quad_pts.reshape((-1, 1, 2))
#                 cv2.polylines(display, [pts_draw], isClosed=True,
#                             color=(0, 255, 255), thickness=2)

#                 # 우상단 warp 크롭 미리보기
#                 preview = cv2.resize(crop, (150, 150))
#                 display[0:150, WIDTH - 150:WIDTH] = preview
#                 cv2.rectangle(display, (WIDTH - 150, 0), (WIDTH, 150), (0, 255, 255), 1)
#         else:
#             recent.append(-1)
#         # ── smoothing ─────────────────────────────────────
#         vote, _ = Counter(recent).most_common(1)[0]

#         if vote == -1:
#             label_text, color = "NO CARD", (0, 0, 255)
#         else:
#             label_text, color = classes[vote], (0, 255, 0)

#         # ── HUD ───────────────────────────────────────────
#         cv2.putText(display, label_text, (10, 45),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
#         cv2.putText(display,
#                     f"det {det_score*100:.0f}%  cls {cls_conf*100:.1f}%",
#                     (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
#         cv2.putText(display,
#                     f"det_th={det_thresh:.1f}  cls_th={cls_thresh:.1f}",
#                     (10, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

#         if probs is not None:
#             top3 = torch.topk(probs, 3)
#             for i, (p, ci) in enumerate(zip(top3.values, top3.indices)):
#                 cv2.putText(display, f"{classes[ci]}: {p*100:.1f}%",
#                             (10, 135 + i * 26),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 2)

#         cv2.putText(display, "q: quit  t: cls_th  o: det_th",
#                     (10, HEIGHT - 12),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#         cv2.imshow("Halli Galli Demo", display)

#         key = cv2.waitKey(1) & 0xFF
#         if key == ord("q"):
#             break
#         elif key == ord("t"):
#             c_idx     = (c_idx + 1) % len(cls_options)
#             cls_thresh = cls_options[c_idx]
#             print(f"분류기 threshold → {cls_thresh}")
#         elif key == ord("o"):
#             d_idx     = (d_idx + 1) % len(det_options)
#             det_thresh = det_options[d_idx]
#             print(f"검출기 threshold → {det_thresh}")

#     cap.release()
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()


"""
webcam_demo_multi.py
====================
여러 장 카드 동시 인식 버전
---------------------------------------------------
single 버전과 차이:
  - scores.argmax() (1개)  →  threshold 넘는 anchor 전부 선택
  - NMS 로 같은 카드 중복 검출 제거
  - 살아남은 박스마다 warp + 분류 반복

  ※ 프레임 간 smoothing(다수결)은 카드 정체성 추적이 필요해
    multi 모드에서는 생략. 매 프레임 독립 분류.

조작:
  q : 종료
  t : 분류기 threshold 순환 (0.5→0.7→0.9)
  o : 검출기 threshold 순환 (0.3→0.5→0.7)
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

# 카드마다 다른 박스 색 (BGR)
# BOX_COLORS = [
#     (0, 255, 255), (0, 255, 0), (255, 128, 0), (255, 0, 255),
#     (0, 128, 255), (255, 255, 0), (128, 0, 255), (0, 200, 128),
#     (200, 0, 0), (0, 0, 200),
# ]
FRUIT_COLORS = {
    "b": (0, 215, 255),   # 바나나 - 노랑  (BGR)
    "s": (0, 0, 220),     # 딸기   - 빨강
    "l": (50, 205, 50),   # 라임   - 연두
    "p": (180, 0, 180),   # 자두   - 보라
}
DEFAULT_COLOR = (200, 200, 200)  # 알 수 없는 클래스

# =========================================================
# anchor grid / 모델 / quad decode  (single 버전과 동일)
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
    print(f"[검출기] epoch={ckpt.get('epoch','?')}  mIoU={ckpt.get('miou',0):.3f}  "
          f"recall={ckpt.get('recall',0):.3f}  꼭짓점오차={ckpt.get('corner_err',0):.1f}px")
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
    print(f"[분류기] epoch={ckpt.get('epoch','?')}  val_acc={ckpt.get('val_acc',0):.2f}%  "
          f"클래스 {len(classes)}개")
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

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        display = frame.copy()

        dets = detect_multi(detector, anchors_t, frame, OBJ_THRESH)

        card_count = 0
        for d in dets:
            crop = warp_card(frame, d["quad"])
            idx, cls_conf = classify(classifier, crop)

            if cls_conf < CLS_THRESH:        # 분류기가 거부 → 카드 아님
                continue

            # color = BOX_COLORS[card_count % len(BOX_COLORS)]
            fruit_key = classes[idx][0]   # 'b1' → 'b', 's3' → 's' 등
            color = FRUIT_COLORS.get(fruit_key, DEFAULT_COLOR)
            pts   = d["quad"].astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(display, [pts], isClosed=True, color=color, thickness=2)

            # 카드 라벨 (박스 좌상단 위에)
            x1, y1 = int(d["aabb"][0]), int(d["aabb"][1])
            text = f"{classes[idx]} {cls_conf*100:.0f}%"
            cv2.putText(display, text, (x1, max(y1 - 8, 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            card_count += 1

        # ── HUD ───────────────────────────────────────────
        head_color = (0, 255, 0) if card_count > 0 else (0, 0, 255)
        head_text  = f"{card_count} CARD(S)" if card_count > 0 else "NO CARD"
        cv2.putText(display, head_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, head_color, 3)
        cv2.putText(display,
                    f"det_th={OBJ_THRESH:.1f}  cls_th={CLS_THRESH:.1f}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
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