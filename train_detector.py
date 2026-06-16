"""
train_detector_v2.py — 카드 검출기 v2: bbox + quad(꼭짓점 4점) 동시 회귀
=====================================================
[v1 → v2 변경]
  anchor당 출력 5 → 13채널:
    objectness(1) + bbox 보정(4) + quad 꼭짓점 오프셋(8)
  - quad 꼭짓점 인코딩: (corner − anchor중심) / anchor크기
  - 라벨의 quad는 tl→tr→br→bl 순서 고정 → 꼭짓점 대응 일관
  - 학습 데이터: conf low(=audit 강등 포함) 또는 quad 없는 라벨 제외
  - 좌우/상하 반전 augmentation 시 quad 좌표·꼭짓점 순서 동기 변환
  - 평가에 "꼭짓점 평균 오차(px)" 추가 — 보라색 윤곽의 정밀도 직접 측정

추론 시: best anchor의 8개 오프셋을 디코드하면 그대로 카드 윤곽 quad.
quad로 perspective warp → 분류기 입력 (make_crops_v2와 일관).

사용법:
    python3 train_detector_v2.py
"""

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights

# =========================================================
# 설정
# =========================================================
LABELS_JSON  = Path("/home/choi/cnn_pictures/labels.json")
NONE_DIR     = Path("/home/choi/cnn_pictures/dataset/none")  # 없으면 무시됨
ANCHORS_JSON = Path("/home/choi/cnn_pictures/anchors.json")
CKPT_PATH    = Path("/home/choi/cnn_pictures/detector_best.pth")

IMG_W, IMG_H = 640, 480
STRIDE       = 32
GRID_W, GRID_H = IMG_W // STRIDE, IMG_H // STRIDE   # 20 x 15

NUM_ANCHORS  = 5
SKIP_CONF    = {"low"}    # audit WRITE_BACK으로 강등된 항목 자동 제외

POS_IOU, NEG_IOU = 0.5, 0.4
NEG_RATIO    = 3
QUAD_LOSS_W  = 1.0        # quad 회귀 손실 가중치

BLOCK        = 30         # 유사 세션 블록 크기
VAL_RATIO    = 0.15
BATCH_SIZE   = 16
EPOCHS       = 20
LR           = 1e-4
SEED         = 42

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], np.float32)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =========================================================
# 1) anchor k-means (IoU 거리)
# =========================================================
def iou_wh(wh, centers):
    inter = (np.minimum(wh[:, None, 0], centers[None, :, 0])
             * np.minimum(wh[:, None, 1], centers[None, :, 1]))
    union = wh[:, None].prod(2) + centers[None, :].prod(2) - inter
    return inter / union


def kmeans_anchors(annotations, k=NUM_ANCHORS, iters=100):
    wh = np.array([[a["bbox"][2] - a["bbox"][0], a["bbox"][3] - a["bbox"][1]]
                   for a in annotations], np.float32)
    rng = np.random.default_rng(SEED)
    centers = wh[rng.choice(len(wh), k, replace=False)]
    for _ in range(iters):
        assign = iou_wh(wh, centers).argmax(1)
        new = np.array([np.median(wh[assign == i], 0) if (assign == i).any()
                        else centers[i] for i in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    centers = centers[np.argsort(centers.prod(1))]
    best_iou = iou_wh(wh, centers).max(1).mean()
    print(f"anchors (w,h): {centers.round(0).tolist()}")
    print(f"평균 best-IoU: {best_iou:.3f}  (0.7 이상이면 충분)")
    return centers


# =========================================================
# 2) anchor 그리드 / 인코딩-디코딩 / IoU
# =========================================================
def build_anchor_grid(anchor_wh):
    ys, xs = np.meshgrid(np.arange(GRID_H), np.arange(GRID_W), indexing="ij")
    cx = (xs + 0.5) * STRIDE
    cy = (ys + 0.5) * STRIDE
    K = len(anchor_wh)
    grid = np.zeros((GRID_H, GRID_W, K, 4), np.float32)
    grid[..., 0] = cx[..., None]
    grid[..., 1] = cy[..., None]
    grid[..., 2] = anchor_wh[None, None, :, 0]
    grid[..., 3] = anchor_wh[None, None, :, 1]
    return grid.reshape(-1, 4)


def cxcywh_to_xyxy(b):
    return np.stack([b[..., 0] - b[..., 2] / 2, b[..., 1] - b[..., 3] / 2,
                     b[..., 0] + b[..., 2] / 2, b[..., 1] + b[..., 3] / 2], -1)


def iou_xyxy(a, b):
    x1 = np.maximum(a[:, 0], b[0]); y1 = np.maximum(a[:, 1], b[1])
    x2 = np.minimum(a[:, 2], b[2]); y2 = np.minimum(a[:, 3], b[3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def encode_box(gt_cxcywh, anchors):
    t = np.zeros_like(anchors)
    t[:, 0] = (gt_cxcywh[0] - anchors[:, 0]) / anchors[:, 2]
    t[:, 1] = (gt_cxcywh[1] - anchors[:, 1]) / anchors[:, 3]
    t[:, 2] = np.log(gt_cxcywh[2] / anchors[:, 2])
    t[:, 3] = np.log(gt_cxcywh[3] / anchors[:, 3])
    return t


def encode_quad(quad, anchors):
    """quad: (4,2) 픽셀좌표 → (A,8) anchor 상대 오프셋"""
    A = len(anchors)
    t = np.zeros((A, 8), np.float32)
    for i in range(4):
        t[:, 2 * i]     = (quad[i, 0] - anchors[:, 0]) / anchors[:, 2]
        t[:, 2 * i + 1] = (quad[i, 1] - anchors[:, 1]) / anchors[:, 3]
    return t


def decode_quad_t(offsets8, anchors):
    """torch. offsets8: (...,8), anchors: (...,4) cxcywh → (...,4,2) 픽셀"""
    off = offsets8.view(*offsets8.shape[:-1], 4, 2)
    cx, cy = anchors[..., 0:1], anchors[..., 1:2]
    w,  h  = anchors[..., 2:3], anchors[..., 3:4]
    x = off[..., 0] * w + cx
    y = off[..., 1] * h + cy
    return torch.stack([x, y], -1)


# =========================================================
# 3) 데이터셋 (블록 단위 split, wood→val, none 음성)
# =========================================================
def block_split(annotations):
    groups = {}
    for a in annotations:
        p = Path(a["path"])
        groups.setdefault((a["class"], p.parent.name), []).append(a)

    rng = random.Random(SEED)
    train, val = [], []
    for key in sorted(groups):
        items = sorted(groups[key], key=lambda a: a["path"])
        if key[1] == "wood":              # wood는 통째로 val (train 미사용 철학 유지)
            val.extend(items)
            continue
        blocks = [items[i:i + BLOCK] for i in range(0, len(items), BLOCK)]
        rng.shuffle(blocks)
        n_val = max(1, round(len(blocks) * VAL_RATIO))
        for i, blk in enumerate(blocks):
            (val if i < n_val else train).extend(blk)
    return train, val


def collect_none_images():
    if not NONE_DIR.exists():
        return []
    items = []
    for p in sorted(NONE_DIR.rglob("*")):
        if p.suffix.lower() in IMG_EXTS:
            items.append({"path": str(p), "bbox": None, "quad": None})
    print(f"none(음성) 이미지: {len(items)}장 포함")
    return items


def hflip_quad(quad):
    """좌우 반전 + tl,tr,br,bl 순서 복원"""
    m = [[IMG_W - x, y] for x, y in quad]
    return [m[1], m[0], m[3], m[2]]


def vflip_quad(quad):
    """상하 반전 + 순서 복원"""
    m = [[x, IMG_H - y] for x, y in quad]
    return [m[3], m[2], m[1], m[0]]


class CardDetDataset(Dataset):
    def __init__(self, items, anchors_xyxy, anchors_cxcywh, train=True):
        self.items = items
        self.anchors_xyxy = anchors_xyxy
        self.anchors_cxcywh = anchors_cxcywh
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        a = self.items[idx]
        img = cv2.imread(a["path"])
        if img.shape[:2] != (IMG_H, IMG_W):
            sy, sx = IMG_H / img.shape[0], IMG_W / img.shape[1]
            img = cv2.resize(img, (IMG_W, IMG_H))
        else:
            sx = sy = 1.0

        bbox = None
        quad = None
        if a["bbox"] is not None:
            x1, y1, x2, y2 = a["bbox"]
            bbox = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
            quad = [[px * sx, py * sy] for px, py in a["quad"]]

        if self.train:
            if random.random() < 0.5:                 # 좌우 반전
                img = img[:, ::-1]
                if bbox:
                    x1, y1, x2, y2 = bbox
                    bbox = [IMG_W - x2, y1, IMG_W - x1, y2]
                    quad = hflip_quad(quad)
            if random.random() < 0.5:                 # 상하 반전
                img = img[::-1]
                if bbox:
                    x1, y1, x2, y2 = bbox
                    bbox = [x1, IMG_H - y2, x2, IMG_H - y1]
                    quad = vflip_quad(quad)
            alpha = 1.0 + random.uniform(-0.2, 0.2)   # 밝기/대비 (색조 금지)
            beta  = random.uniform(-20, 20)
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255)

        img = img.astype(np.float32)[..., ::-1] / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = torch.from_numpy(img.copy()).permute(2, 0, 1)

        A = len(self.anchors_xyxy)
        obj_t  = np.zeros(A, np.float32)
        box_t  = np.zeros((A, 4), np.float32)
        quad_t = np.zeros((A, 8), np.float32)

        if bbox is not None:
            gt = np.array(bbox, np.float32)
            ious = iou_xyxy(self.anchors_xyxy, gt)
            pos = ious > POS_IOU
            pos[ious.argmax()] = True
            ignore = (~pos) & (ious > NEG_IOU)
            obj_t[pos] = 1.0
            obj_t[ignore] = -1.0
            gt_cxcywh = np.array([(gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2,
                                  gt[2] - gt[0], gt[3] - gt[1]], np.float32)
            box_t  = encode_box(gt_cxcywh, self.anchors_cxcywh).astype(np.float32)
            quad_t = encode_quad(np.array(quad, np.float32), self.anchors_cxcywh)

        gt_box  = torch.tensor(bbox if bbox is not None else [-1] * 4,
                               dtype=torch.float32)
        gt_quad = torch.tensor(np.array(quad, np.float32).reshape(-1)
                               if quad is not None else [-1] * 8,
                               dtype=torch.float32)
        return (img, torch.from_numpy(obj_t), torch.from_numpy(box_t),
                torch.from_numpy(quad_t), gt_box, gt_quad)


# =========================================================
# 4) 모델: anchor당 13채널 (obj 1 + box 4 + quad 8)
# =========================================================
class CardDetector(nn.Module):
    def __init__(self, num_anchors=NUM_ANCHORS):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.body = nn.Sequential(*list(backbone.children())[:-2])
        self.head = nn.Conv2d(512, num_anchors * 13, kernel_size=1)
        nn.init.normal_(self.head.weight, std=0.01)
        bias = self.head.bias.view(num_anchors, 13)
        with torch.no_grad():
            bias.zero_()
            bias[:, 0].fill_(-4.0)     # objectness 음성 우세 사전확률
        self.K = num_anchors

    def forward(self, x):
        f = self.body(x)
        out = self.head(f)
        B, _, Hf, Wf = out.shape
        out = out.view(B, self.K, 13, Hf, Wf)
        out = out.permute(0, 3, 4, 1, 2).reshape(B, -1, 13)   # (B, A, 13)
        return out[..., 0], out[..., 1:5], out[..., 5:13]     # obj, box, quad


# =========================================================
# 5) 손실
# =========================================================
def detection_loss(obj_logit, box_off, quad_off, obj_t, box_t, quad_t):
    pos = obj_t == 1
    neg = obj_t == 0

    obj_loss_all = F.binary_cross_entropy_with_logits(
        obj_logit, pos.float(), reduction="none")
    pos_loss = obj_loss_all[pos].sum()

    n_pos = int(pos.sum().item())
    n_neg = max(NEG_RATIO * max(n_pos, 1), 16)
    neg_losses = obj_loss_all[neg]
    if neg_losses.numel() > n_neg:
        neg_losses, _ = neg_losses.topk(n_neg)
    neg_loss = neg_losses.sum()

    if n_pos > 0:
        box_loss  = F.smooth_l1_loss(box_off[pos],  box_t[pos],  reduction="sum")
        quad_loss = F.smooth_l1_loss(quad_off[pos], quad_t[pos], reduction="sum")
    else:
        box_loss = quad_loss = obj_logit.sum() * 0.0

    norm = max(n_pos, 1)
    return ((pos_loss + neg_loss) / norm,
            box_loss / norm,
            QUAD_LOSS_W * quad_loss / norm)


# =========================================================
# 6) 평가: bbox mIoU / recall@0.5 / quad 꼭짓점 평균 오차 / none 오검출
# =========================================================
@torch.no_grad()
def evaluate(model, loader, anchors_t, obj_thresh=0.5):
    model.eval()
    ious, corner_errs, n_neg_img, n_neg_fp = [], [], 0, 0
    for img, obj_t, box_t, quad_t, gt_box, gt_quad in loader:
        img = img.to(DEVICE)
        obj_logit, box_off, quad_off = model(img)
        scores = torch.sigmoid(obj_logit)
        best = scores.argmax(1)
        for b in range(img.size(0)):
            has_card = gt_box[b, 0] >= 0
            if not has_card:
                n_neg_img += 1
                if scores[b, best[b]] > obj_thresh:
                    n_neg_fp += 1
                continue
            a_sel = anchors_t[best[b]]
            # quad 디코드 → 꼭짓점 평균 오차
            pq = decode_quad_t(quad_off[b, best[b]].float().cpu(), a_sel)  # (4,2)
            gq = gt_quad[b].view(4, 2)
            corner_errs.append(float((pq - gq).norm(dim=1).mean()))
            # quad의 AABB로 IoU (검출 recall 측정)
            x1, y1 = pq[:, 0].min(), pq[:, 1].min()
            x2, y2 = pq[:, 0].max(), pq[:, 1].max()
            pred = np.array([x1, y1, x2, y2], np.float32)
            ious.append(float(iou_xyxy(pred[None], gt_box[b].numpy())[0]))
    miou = float(np.mean(ious)) if ious else 0.0
    recall = float(np.mean(np.array(ious) > 0.5)) if ious else 0.0
    cerr = float(np.mean(corner_errs)) if corner_errs else 0.0
    fp_rate = n_neg_fp / n_neg_img if n_neg_img else None
    return miou, recall, cerr, fp_rate


# =========================================================
# 메인
# =========================================================
def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    data = json.loads(LABELS_JSON.read_text())
    anns = [a for a in data["annotations"]
            if a["conf"] not in SKIP_CONF and a["quad"] is not None]
    print(f"라벨 {len(anns)}장 (conf low / quad 없음 제외)")

    anchor_wh = kmeans_anchors(anns)
    ANCHORS_JSON.write_text(json.dumps(
        {"anchors_wh": anchor_wh.tolist(), "stride": STRIDE,
         "img_w": IMG_W, "img_h": IMG_H}, indent=2))
    print(f"→ {ANCHORS_JSON} 저장 (추론 시 재사용)\n")

    anchors_cxcywh = build_anchor_grid(anchor_wh)
    anchors_xyxy   = cxcywh_to_xyxy(anchors_cxcywh)
    anchors_t      = torch.from_numpy(anchors_cxcywh)

    train_items, val_items = block_split(anns)
    none_items = collect_none_images()
    rng = random.Random(SEED)
    rng.shuffle(none_items)
    n_val_none = round(len(none_items) * VAL_RATIO)
    train_items += none_items[n_val_none:]
    val_items   += none_items[:n_val_none]
    print(f"train {len(train_items)} / val {len(val_items)}"
          f" (블록={BLOCK}, wood는 전량 val)\n")

    train_ds = CardDetDataset(train_items, anchors_xyxy, anchors_cxcywh, train=True)
    val_ds   = CardDetDataset(val_items,   anchors_xyxy, anchors_cxcywh, train=False)
    train_ld = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=4,
                          pin_memory=True, drop_last=True)
    val_ld   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=4,
                          pin_memory=True)

    model = CardDetector().to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)

    best_score = -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tot = {"obj": 0.0, "box": 0.0, "quad": 0.0}
        n_batch = 0
        for img, obj_t, box_t, quad_t, _, _ in train_ld:
            img    = img.to(DEVICE)
            obj_t  = obj_t.to(DEVICE)
            box_t  = box_t.to(DEVICE)
            quad_t = quad_t.to(DEVICE)
            obj_logit, box_off, quad_off = model(img)
            l_obj, l_box, l_quad = detection_loss(
                obj_logit, box_off, quad_off, obj_t, box_t, quad_t)
            loss = l_obj + l_box + l_quad
            optim.zero_grad(); loss.backward(); optim.step()
            tot["obj"] += l_obj.item(); tot["box"] += l_box.item()
            tot["quad"] += l_quad.item(); n_batch += 1
        sched.step()

        miou, recall, cerr, fp = evaluate(model, val_ld, anchors_t)
        fp_str = f"  none오검출 {fp*100:.1f}%" if fp is not None else ""
        print(f"[{epoch:2d}/{EPOCHS}] obj {tot['obj']/n_batch:.3f}  "
              f"box {tot['box']/n_batch:.3f}  quad {tot['quad']/n_batch:.3f}  |  "
              f"val mIoU {miou:.3f}  recall@0.5 {recall:.3f}  "
              f"꼭짓점오차 {cerr:.1f}px{fp_str}")

        # 선택 기준: recall 우선, 동률이면 꼭짓점 오차 작은 쪽
        score = recall - cerr / 1000.0
        if score > best_score:
            best_score = score
            torch.save({"model": model.state_dict(),
                        "anchors_wh": anchor_wh.tolist(),
                        "epoch": epoch, "miou": miou,
                        "recall": recall, "corner_err": cerr}, CKPT_PATH)
            print(f"   ↳ best 갱신 → {CKPT_PATH}")

    print(f"\n완료. 꼭짓점 평균 오차가 한 자릿수 px대로 내려오면")
    print("데모의 보라색 윤곽이 라벨 수준으로 정밀하게 나온다는 뜻입니다.")


if __name__ == "__main__":
    main()