"""
자동 라벨링 v9: 오검출 방지 + 기울어진 카드 구제
=====================================================
[v8 → v9 변경]
  1. 기울어진(원근 심한) 카드 구제:
     - _shape_check에 relaxed 모드 추가 (aspect ≤ 3.0, fill ≥ 0.65)
     - 일반 검사(직접→hull) 전부 실패 시에만 relaxed로 재시도
     - relaxed로 들어온 후보는 ring 대비가 STRICT 마진을 통과해야만 수용,
       conf는 mid가 상한 → 모양이 느슨한 만큼 광도 증거를 강하게 요구
  2. sat_otsu를 RESCUE 전략으로 강등:
     - 과노출 나무(저채도)와 카드가 한 덩어리로 붙는 오검출이 확인됨
     - RESCUE 전략의 conf는 mid가 상한 → 다른 전략이 찾으면 절대 못 이김
  3. 후보 선택 규칙 변경: conf → fill(직사각형 충실도) → 면적
     - 기존 "conf 같으면 면적 큰 쪽"은 카드+배경 합체 blob이
       정확한 검출을 이기는 버그의 원인이었음
  4. 승리 전략 기록: JSON "strategy" 필드 + 미리보기/디버그 표시
     - 이상 검출이 나오면 어느 전 ㅂ    략이 범인인지 바로 추적 가능

[v7 → v8 요약]
  - refine_quad: quad 각 변을 실제 카드 테두리(강한 에지)에 스냅
  - extract_quad: 근사 → 스냅 → 패딩 순서, gray 인자 추가
  - mask_sat_otsu 전략 추가 (v9에서 RESCUE로 강등)

[v6.1 → v7 요약]
  - quad(꼭짓점 4점) 추가, 미리보기 3색, fill_holes 버그 수정 등

사용법:
    python3 auto_label.py            # 전체 라벨링
    python3 auto_label.py debug 이미지경로
    python3 auto_label.py demo
"""

import json
import random
import shutil
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

# =========================================================
# 설정
# =========================================================
SRC_DIR     = Path("/home/choi/cnn_pictures/dataset")
OUT_JSON    = Path("/home/choi/cnn_pictures/labels.json")
PREVIEW_DIR = Path("/home/choi/cnn_pictures/labeling_preview")
DEBUG_DIR   = Path("/home/choi/cnn_pictures/labeling_debug")

PREVIEW_PER_CLASS = 3

# ---- 후보 검증 기준 ----
MIN_AREA_RATIO  = 0.015      # 프레임의 1.5% 이상
MAX_AREA_RATIO  = 0.60       # 프레임의 60% 이하
ASPECT_RANGE    = (1.0, 2.1) # 카드 긴변:짧은변 비율 (회전 보정 후)
                             # 하한 1.0: 비스듬히 내려찍은 카드는 원근 압축으로
                             # 거의 정사각형(~1.0)까지 투영될 수 있음
MIN_RECT_FILL   = 0.80       # 윤곽 면적 / minAreaRect 면적

# ---- 기울어진 카드 구제용 완화 기준 (v9) ----
# 짧은 변 방향 원근 압축은 종횡비를 늘리고(1.51/cos θ),
# 사다리꼴 투영은 minAreaRect 충실도를 떨어뜨린다.
# 완화 기준으로 들어온 후보는 STRICT 대비를 통과해야만 수용됨.
ASPECT_HI_RELAXED   = 3.0
MIN_RECT_FILL_RELAXED = 0.65

BORDER_MARGIN   = 3          # 이 px 이내면 "경계에 닿음"으로 간주
MAX_BORDER_TOUCH = 1         # 닿은 변이 이 수를 초과하면 거부 (배경 덩어리)

BOX_PAD_RATIO = 0.04         # 최종 박스를 변 길이의 4%씩 확장 (프레임 안으로 클립)

# ---- 국소(ring) 대비 검증: 카드 내부 vs 윤곽 바로 바깥 띠 ----
RING_WIDTH = 25
RING_KERNEL = np.ones((RING_WIDTH, RING_WIDTH), np.uint8)
STRICT_MARGINS  = {"bright": 10, "sat": 15, "warm": 12}
RELAXED_MARGINS = {"bright": 4,  "sat": 7,  "warm": 6}

COOL_FIXED_THRESH = 113      # signed_warm < 113 (= R−B < −15) 이면 "차가움"

# 단독 신뢰가 낮은 전략: 후보를 내도 conf 상한이 mid.
# 다른 전략이 뭐라도 찾으면 절대 이기지 못하는 "구조 전용" 전략.
RESCUE_STRATEGIES = {"sat_otsu"}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))


# =========================================================
# 채널 헬퍼
# =========================================================
def signed_warm(img):
    """R−B를 부호 보존한 채 128 중심 uint8로.
       차가움(카드) < 128 < 따뜻함(나무).
       주의: cv2.subtract(uint8)는 음수를 0으로 포화시키므로 쓰면 안 됨."""
    d = img[:, :, 2].astype(np.int16) - img[:, :, 0].astype(np.int16)
    return np.clip(d + 128, 0, 255).astype(np.uint8)


def fill_holes(mask):
    """마스크 내부의 구멍(과일 그림, 끊긴 윤곽 내부)을 메움.
       주의: (0,0)에서 바로 flood fill하면 모서리가 전경(255)일 때
       배경을 못 찾아 전체가 뒤집히는 버그가 있음 → 검은 1px 패딩을
       둘러서 배경 시드를 보장한다."""
    h, w = mask.shape
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff = padded.copy()
    cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 255)
    holes = cv2.bitwise_not(ff)[1:-1, 1:-1]
    return mask | holes


# =========================================================
# 이진화 전략  (fn(blur_gray, img) → mask)
# =========================================================
def mask_otsu(blur, img):
    _, m = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m

def mask_adaptive(blur, img):
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, 51, -5)

def mask_canny(blur, img):
    edges = cv2.Canny(blur, 40, 120)
    return cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)

def mask_cool_otsu(blur, img):
    w8 = cv2.GaussianBlur(signed_warm(img), (5, 5), 0)
    _, m = cv2.threshold(w8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return m

def mask_cool_fixed(blur, img):
    w8 = cv2.GaussianBlur(signed_warm(img), (5, 5), 0)
    _, m = cv2.threshold(w8, COOL_FIXED_THRESH, 255, cv2.THRESH_BINARY_INV)
    return m

def mask_sat_otsu(blur, img):
    """[RESCUE] 밝은 나무 배경 대응. 카드는 저채도, 나무는 중간 채도.
       주의: 과노출로 하얗게 날아간 나무도 저채도라 카드와 한 덩어리로
       붙는 오검출 사례가 확인됨 → RESCUE_STRATEGIES로 강등."""
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
    sat = cv2.GaussianBlur(sat, (5, 5), 0)
    _, m = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return m

STRATEGIES = [
    ("otsu",       mask_otsu),
    ("adaptive",   mask_adaptive),
    ("canny",      mask_canny),
    ("cool_otsu",  mask_cool_otsu),
    ("cool_fixed", mask_cool_fixed),
    ("sat_otsu",   mask_sat_otsu),   # RESCUE: conf 상한 mid
]


# =========================================================
# quad 추출: 윤곽 → 꼭짓점 4점 (원근 사다리꼴 밀착)
# =========================================================
def _order_quad(pts):
    """tl → tr → br → bl 순서로 정렬 (getPerspectiveTransform 호환).
       중심 기준 각도 정렬 → 항상 자기 교차 없는 볼록 사각형."""
    pts = pts.astype(np.float32)
    c = pts.mean(0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]            # 이미지 좌표(y↓)에서 시계방향
    start = int(np.argmin(pts.sum(1)))    # 좌상단(x+y 최소)에서 시작하도록 회전
    return np.roll(pts, -start, axis=0)


def _line_intersect(p1, d1, p2, d2):
    """점 p + t*d 형태 두 직선의 교점. 평행이면 None."""
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]], np.float64)
    if abs(np.linalg.det(A)) < 1e-8:
        return None
    t = np.linalg.solve(A, (p2 - p1).astype(np.float64))[0]
    return p1 + t * d1


def refine_quad(gray, quad, search=8, n_samples=25):   # ★ search 14 → 8
    """quad의 각 변을 따라 법선 방향 ±search px에서 그래디언트 최대 지점
       (= 카드 실제 테두리)을 찾아 cv2.fitLine으로 변을 재피팅하고,
       인접 변의 교점을 새 꼭짓점으로 삼는다.

       - 그림자 경계는 에지가 약하고 카드 테두리는 날카로우므로
         마스크가 그림자를 포함해 quad가 부풀었어도 카드에 달라붙음
       - 변의 중간 70%(t=0.15~0.85)만 샘플링 → 둥근 모서리 영향 배제
       - [v9.1] 에지 '방향'이 변의 법선과 정렬된 샘플만 채택
         → 카드 가장자리 근처의 과일 윤곽(곡선)·비스듬한 나뭇결에
           스냅이 끌려가 꼭짓점이 어긋나던 문제 수정
       - 그래디언트가 약한 샘플은 버리고, 유효 샘플이 부족하거나
         꼭짓점이 search*2px 이상 튀면 해당 변/quad는 원본 유지(안전장치)"""
    h, w = gray.shape
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_floor = float(np.median(mag)) * 3.0   # 이보다 약한 에지는 무시

    q = np.array(quad, np.float32)
    lines = []   # 변마다 (anchor, direction)

    for i in range(4):
        p1, p2 = q[i], q[(i + 1) % 4]
        d = p2 - p1
        length = float(np.linalg.norm(d))
        if length < 10:
            return quad
        d = d / length
        n = np.array([-d[1], d[0]], np.float32)   # 법선

        snapped = []
        for t in np.linspace(0.15, 0.85, n_samples):
            base = p1 + d * (t * length)
            best_pt, best_m = None, mag_floor
            for s in np.linspace(-search, search, 2 * search + 1):
                x, y = base + n * s
                xi, yi = int(round(x)), int(round(y))
                if not (0 <= xi < w and 0 <= yi < h):       # ★ 여기부터
                    continue
                m = mag[yi, xi]
                if m <= best_m:
                    continue
                g = np.array([gx[yi, xi], gy[yi, xi]], np.float32)
                # 에지 방향 검사: 카드 테두리의 그래디언트는 변의 법선과
                # 평행하지만, 과일 윤곽·비스듬한 나뭇결은 방향이 어긋남
                if abs(float(g @ n)) / (np.linalg.norm(g) + 1e-6) < 0.7:
                    continue
                best_m = m
                best_pt = (x, y)                            # ★ 여기까지 교체
            if best_pt is not None:
                snapped.append(best_pt)

        if len(snapped) < n_samples // 3:
            # 이 변은 에지를 못 찾음(저대비) → 원래 변 유지
            lines.append((p1, d))
        else:
            pts = np.array(snapped, np.float32)
            vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            lines.append((np.array([x0, y0], np.float32),
                          np.array([vx, vy], np.float32)))

    new_q = []
    for i in range(4):
        # 꼭짓점 i = 변 (i-1) 과 변 i 의 교점
        inter = _line_intersect(lines[i - 1][0], lines[i - 1][1],
                                lines[i][0], lines[i][1])
        if inter is None or np.linalg.norm(inter - q[i]) > search * 2:
            new_q.append(q[i])                  # 너무 튀면 원본 유지
        else:
            new_q.append(inter.astype(np.float32))
    return [[round(float(x), 1), round(float(y), 1)] for x, y in new_q]

def extract_quad(cnt, w, h, gray):
    """convex hull을 4점으로 근사한 뒤 refine_quad로 실제 카드 테두리에
       스냅하고, 마지막에 BOX_PAD_RATIO만큼 소폭 확장한다."""
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    quad = None
    for eps in np.linspace(0.01, 0.08, 15):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2)
            break
        if len(approx) < 4:
            break
    if quad is None:
        return None
    quad = _order_quad(quad)

    hull_area = cv2.contourArea(hull)
    quad_area = cv2.contourArea(quad.astype(np.int32))
    if (len(np.unique(quad.round(0), axis=0)) < 4
            or quad_area < 0.8 * hull_area
            or quad_area > 1.6 * hull_area):
        return None

    # ---- 에지 스냅 (패딩 전!) ----
    quad = np.array(refine_quad(gray, quad), np.float32)

    c = quad.mean(0)
    quad = c + (quad - c) * (1 + 2 * BOX_PAD_RATIO)
    quad[:, 0] = np.clip(quad[:, 0], 0, w - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, h - 1)
    return [[round(float(x), 1), round(float(y), 1)] for x, y in quad]


# =========================================================
# 박스 패딩: bbox/rbox를 변 길이의 BOX_PAD_RATIO씩 확장
# =========================================================
def _pad_boxes(bbox, rbox, w, h):
    x1, y1, x2, y2 = bbox
    px = (x2 - x1) * BOX_PAD_RATIO
    py = (y2 - y1) * BOX_PAD_RATIO
    bbox = (max(0, int(x1 - px)), max(0, int(y1 - py)),
            min(w, int(x2 + px)), min(h, int(y2 + py)))
    cx, cy, rw, rh, angle = rbox
    scale = 1 + 2 * BOX_PAD_RATIO
    rbox = (cx, cy, round(rw * scale, 2), round(rh * scale, 2), angle)
    return bbox, rbox


# =========================================================
# rbox 정규화: 항상 rw = 긴 변, angle ∈ [0, 180)
# =========================================================
def normalize_rbox(cx, cy, rw, rh, angle):
    if rw < rh:
        rw, rh = rh, rw
        angle += 90.0
    angle %= 180.0
    return (round(cx, 2), round(cy, 2), round(rw, 2), round(rh, 2), round(angle, 2))


# =========================================================
# 후보 검증
# =========================================================
def _shape_check(cnt, frame_area, h, w, relaxed=False):
    """모양 게이트. 통과 시 (area, rect, bbox, fill) 반환, 실패 시 None.
       relaxed=True: 기울어진 카드용 완화 기준
         - 종횡비 상한 2.1 → 3.0 (짧은 변 원근 압축은 종횡비를 늘림)
         - fill 하한 0.80 → 0.65 (사다리꼴은 minAreaRect를 꽉 못 채움)"""
    aspect_hi = ASPECT_HI_RELAXED   if relaxed else ASPECT_RANGE[1]
    min_fill  = MIN_RECT_FILL_RELAXED if relaxed else MIN_RECT_FILL

    area = cv2.contourArea(cnt)
    if not (MIN_AREA_RATIO * frame_area <= area <= MAX_AREA_RATIO * frame_area):
        return None
    rect = cv2.minAreaRect(cnt)
    rw, rh = rect[1]
    if min(rw, rh) == 0:
        return None
    aspect = max(rw, rh) / min(rw, rh)
    fill   = area / (rw * rh)
    if not (ASPECT_RANGE[0] <= aspect <= aspect_hi) or fill < min_fill:
        return None
    x, y, bw, bh = cv2.boundingRect(cnt)
    # 가장자리에 걸친 카드 → 보통 1변 / 배경 덩어리 → 보통 2변 이상
    touched = sum([
        x <= BORDER_MARGIN,
        y <= BORDER_MARGIN,
        x + bw >= w - BORDER_MARGIN,
        y + bh >= h - BORDER_MARGIN,
    ])
    if touched > MAX_BORDER_TOUCH:
        return None
    return area, rect, (x, y, x + bw, y + bh), fill


def validate(cnt, gray, sat, warm, frame_area, h, w):
    """통과 시 (bbox, rbox, area, conf, contour, fill) 반환, 실패 시 None.
       시도 순서: 직접 → hull → [완화: 직접 → hull]
       완화(loose) 경로로 들어온 후보는 STRICT 대비를 통과해야만 수용,
       conf 상한은 mid. (모양 증거가 약한 만큼 광도 증거를 강하게 요구)
       conf = 2(high) / 1(mid) / 0(low)"""
    shape, use, loose = _shape_check(cnt, frame_area, h, w), cnt, False
    if shape is None:
        hull = cv2.convexHull(cnt)
        shape, use = _shape_check(hull, frame_area, h, w), hull
    if shape is None:
        # 기울어진 카드 구제: 모양 완화 + (아래에서) 대비 엄격
        shape, use, loose = _shape_check(cnt, frame_area, h, w, relaxed=True), cnt, True
        if shape is None:
            hull = cv2.convexHull(cnt)
            shape, use = _shape_check(hull, frame_area, h, w, relaxed=True), hull
    if shape is None:
        return None
    area, rect, bbox, fill = shape

    # 대비 단서: 카드 내부 vs 윤곽 바로 바깥 띠(ring), 중앙값 비교
    inner = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(inner, [use], -1, 255, -1)
    ring = cv2.subtract(cv2.dilate(inner, RING_KERNEL), inner)
    ring_px = ring > 0
    if not ring_px.any():
        return None
    in_px = inner > 0

    d_bright = float(np.median(gray[in_px])) - float(np.median(gray[ring_px]))
    d_sat    = float(np.median(sat[in_px]))  - float(np.median(sat[ring_px]))
    d_warm   = float(np.median(warm[in_px])) - float(np.median(warm[ring_px]))

    def passes(m):
        return (d_bright >= m["bright"]      # (a) 주변보다 밝음
                or d_sat <= -m["sat"]        # (b) 주변보다 무채색 (보조)
                or d_warm <= -m["warm"])     # (c) 주변보다 차가움 (가장 신뢰)

    if passes(STRICT_MARGINS):
        conf = 2
    elif passes(RELAXED_MARGINS):
        conf = 1
    else:
        conf = 0   # 대비 단서 없음 — 모양만으로 저신뢰 수용

    if loose:
        if not passes(STRICT_MARGINS):
            return None      # 모양도 애매 + 대비도 애매 → 거부
        conf = 1             # 구제된 후보는 mid가 상한

    rw, rh = rect[1]
    rbox = normalize_rbox(rect[0][0], rect[0][1], rw, rh, rect[2])
    return bbox, rbox, area, conf, use, fill


def detect_card(img, return_debug=False):
    """(bbox, rbox, quad, conf, strategy) 반환.
       실패 시 (None, None, None, None, None).
       quad = [[x,y] x4] (tl,tr,br,bl) 또는 근사 실패 시 None
       conf: "high" / "mid" / "low"
       strategy: 승리한 이진화 전략 이름 (오검출 추적용)
       return_debug=True 면 (..., {전략: 마스크}) 반환.

       후보 선택: conf → fill(직사각형 충실도) → 면적 순.
       (면적 우선은 카드+배경 합체 blob이 정답을 이기는 버그의 원인이었음)"""
    h, w = img.shape[:2]
    frame_area = h * w

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sat  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
    warm = img[:, :, 2].astype(np.int16) - img[:, :, 0].astype(np.int16)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    best = None   # (conf, fill, area, bbox, rbox, contour, strategy_name)
    debug_masks = {}

    for name, fn in STRATEGIES:
        mask = fn(blur, img)
        debug_masks[name] = mask
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, CLOSE_KERNEL)
        # filled = fill_holes(closed)
        # contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # for cnt in contours:
        #     result = validate(cnt, gray, sat, warm, frame_area, h, w)
        #     if result is None:
        #         continue
        #     bbox, rbox, area, conf, used, fill = result
        #     if name in RESCUE_STRATEGIES:
        #         conf = min(conf, 1)   # RESCUE 전략은 mid가 상한
        #     cand = (conf, round(fill, 3), area, bbox, rbox, used, name)
        #     if best is None or cand[:3] > best[:3]:
        #         best = cand
        filled = fill_holes(closed)
        if name == "canny":
            # mask_canny의 dilate(5x5 x2)로 부푼 윤곽을 같은 양만큼 침식해 원복
            filled = cv2.erode(filled, np.ones((5, 5), np.uint8), iterations=2)
        contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            result = validate(cnt, gray, sat, warm, frame_area, h, w)
            if result is None:
                continue
            bbox, rbox, area, conf, used, fill = result
            if name in RESCUE_STRATEGIES:
                conf = min(conf, 1)   # RESCUE 전략은 mid가 상한
            cand = (conf, round(fill, 3), area, bbox, rbox, used, name)
            if best is None or cand[:3] > best[:3]:
                best = cand

    if best is None:
        out = (None, None, None, None, None)
    else:
        bbox, rbox = _pad_boxes(best[3], best[4], w, h)
        quad = extract_quad(best[5], w, h, gray)
        out = (bbox, rbox, quad,
               {2: "high", 1: "mid", 0: "low"}[best[0]],
               best[6])

    if return_debug:
        return out + (debug_masks,)
    return out


# =========================================================
# 그리기 헬퍼: 최종 윤곽 하나만 표시
#   quad(마젠타) → 없으면 rbox(주황) → 없으면 bbox(초록)
#   (JSON에는 세 가지 모두 저장됨)
# =========================================================
def draw_boxes(img, bbox, rbox, quad=None):
    if quad is not None:
        pts = np.array(quad, np.int32)
        cv2.polylines(img, [pts], True, (255, 0, 255), 3)
    elif rbox is not None:
        cx, cy, rw, rh, angle = rbox
        pts = cv2.boxPoints(((cx, cy), (rw, rh), angle)).astype(int)
        cv2.polylines(img, [pts], True, (0, 165, 255), 3)
    elif bbox is not None:
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)


# =========================================================
# 디버그: 한 장의 중간 과정을 이미지로 저장
# =========================================================
def debug_one(img_path):
    DEBUG_DIR.mkdir(exist_ok=True)
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"이미지를 읽을 수 없습니다: {img_path}")
        return
    bbox, rbox, quad, conf, strategy, masks = detect_card(img, return_debug=True)

    vis = img.copy()
    if bbox:
        draw_boxes(vis, bbox, rbox, quad)
        print(f"conf={conf}  strategy={strategy}")
        print(f"rbox: cx={rbox[0]} cy={rbox[1]} w={rbox[2]} h={rbox[3]} angle={rbox[4]}°")
        print(f"quad: {quad}")
        cv2.putText(vis, f"{conf} [{strategy}]", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
    else:
        cv2.putText(vis, "FAILED", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    panels = [vis] + [cv2.cvtColor(m, cv2.COLOR_GRAY2BGR) for m in masks.values()]
    labels = ["result"] + list(masks.keys())
    for p, t in zip(panels, labels):
        cv2.putText(p, t, (10, p.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)

    # result + 전략 6개 = 7패널 → 3열 그리드 (빈 칸은 검정 패딩)
    while len(panels) % 3 != 0:
        panels.append(np.zeros_like(panels[0])); labels.append("")
    rows = [np.hstack(panels[i:i+3]) for i in range(0, len(panels), 3)]
    grid = np.vstack(rows)

    out = DEBUG_DIR / f"debug_{Path(img_path).stem}.png"
    cv2.imwrite(str(out), grid)
    print(f"저장: {out}")
    print("→ 3열 그리드: result / otsu / adaptive // canny / cool_otsu / cool_fixed // sat_otsu")
    print("→ 카드가 깨끗한 흰 덩어리로 나오는 전략이 있는지 확인하세요")


# =========================================================
# 메인 (전체 라벨링)
# =========================================================
def main():
    # 이전 실행의 미리보기가 섞이지 않도록 폴더를 비우고 시작
    if PREVIEW_DIR.exists():
        shutil.rmtree(PREVIEW_DIR)
    PREVIEW_DIR.mkdir(parents=True)

    classes = sorted(
        d.name for d in SRC_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    print(f"클래스 {len(classes)}개: {classes}\n")

    annotations = []
    stats = defaultdict(lambda: {"ok": 0, "fail": 0})
    conf_count = {"high": 0, "mid": 0, "low": 0}
    strategy_count = defaultdict(int)
    card_ws, card_hs, card_angles = [], [], []
    preview_pool = defaultdict(lambda: {"ok": [], "fail": []})

    for cls in classes:
        for bg_dir in sorted((SRC_DIR / cls).iterdir()):
            if not bg_dir.is_dir() or bg_dir.name.startswith("."):
                continue
            bg = bg_dir.name

            for img_path in sorted(bg_dir.iterdir()):
                if img_path.suffix.lower() not in IMG_EXTS:
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                bbox, rbox, quad, conf, strategy = detect_card(img)
                key = (cls, bg)
                if bbox is None:
                    stats[key]["fail"] += 1
                    preview_pool[cls]["fail"].append(
                        (img_path, None, None, None, None, None))
                else:
                    stats[key]["ok"] += 1
                    conf_count[conf] += 1
                    strategy_count[strategy] += 1
                    annotations.append({
                        "path":     str(img_path),
                        "class":    cls,
                        "bbox":     list(bbox),   # (x1, y1, x2, y2)
                        "rbox":     list(rbox),   # (cx, cy, rw, rh, angle), rw=긴 변
                        "quad":     quad,         # [[x,y] x4] tl,tr,br,bl / 실패 시 null
                        "conf":     conf,         # high / mid / low — low는 검수 권장
                        "strategy": strategy,     # 승리 전략 — 오검출 추적용
                    })
                    card_ws.append(rbox[2])
                    card_hs.append(rbox[3])
                    card_angles.append(rbox[4])
                    preview_pool[cls]["ok"].append(
                        (img_path, bbox, rbox, quad, conf, strategy))

    with open(OUT_JSON, "w") as f:
        json.dump({"classes": classes, "annotations": annotations}, f, indent=2)

    for cls in classes:
        samples = (random.sample(preview_pool[cls]["ok"],
                                 min(PREVIEW_PER_CLASS - 1, len(preview_pool[cls]["ok"])))
                   + random.sample(preview_pool[cls]["fail"],
                                   min(1, len(preview_pool[cls]["fail"]))))
        for img_path, bbox, rbox, quad, conf, strategy in samples:
            img = cv2.imread(str(img_path))
            if bbox is not None:
                draw_boxes(img, bbox, rbox, quad)
                cv2.putText(img, f"conf: {conf} [{strategy}]", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
                tag = f"ok_{conf}"
            else:
                cv2.putText(img, "DETECTION FAILED", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                tag = "fail"
            out_name = f"{cls}_{img_path.parent.name}_{tag}_{img_path.name}"
            cv2.imwrite(str(PREVIEW_DIR / out_name), img)

    print(f"{'class':>5} {'bg':>8} | {'ok':>5} {'fail':>5} | {'성공률':>7}")
    print("-" * 42)
    total_ok = total_fail = 0
    for (cls, bg), s in sorted(stats.items()):
        n = s["ok"] + s["fail"]
        rate = s["ok"] / n * 100 if n else 0
        flag = "  ←확인" if rate < 90 else ""
        print(f"{cls:>5} {bg:>8} | {s['ok']:>5} {s['fail']:>5} | {rate:>6.1f}%{flag}")
        total_ok += s["ok"]; total_fail += s["fail"]

    n = total_ok + total_fail
    print("-" * 42)
    print(f"전체: {total_ok}/{n}  ({total_ok/n*100:.1f}%)  → {OUT_JSON}")
    print(f"신뢰도: high {conf_count['high']}  /  mid {conf_count['mid']}"
          f"  /  low {conf_count['low']}  ← low는 미리보기/JSON에서 검수 권장")

    strat_str = "  /  ".join(f"{k} {v}" for k, v in
                             sorted(strategy_count.items(), key=lambda x: -x[1]))
    print(f"승리 전략 분포: {strat_str}")
    print("  ← RESCUE(sat_otsu)나 평소 안 보이던 전략의 비중이 갑자기 크면 검수 필요")

    if card_ws:
        ws, hs, angs = np.array(card_ws), np.array(card_hs), np.array(card_angles)
        print(f"\n[카드 크기 분포 — 회전 보정(rbox) 기준, anchor 설계 근거]")
        print(f"  긴 변 : 중앙값 {np.median(ws):.0f}px  (5~95%: {np.percentile(ws,5):.0f}~{np.percentile(ws,95):.0f})")
        print(f"  짧은 변: 중앙값 {np.median(hs):.0f}px  (5~95%: {np.percentile(hs,5):.0f}~{np.percentile(hs,95):.0f})")
        print(f"  종횡비(긴/짧): 중앙값 {np.median(ws/hs):.2f}")
        print(f"  각도: 중앙값 {np.median(angs):.0f}°  (5~95%: {np.percentile(angs,5):.0f}~{np.percentile(angs,95):.0f}°)")

    print(f"\n미리보기: {PREVIEW_DIR}  (마젠타=quad 최종 윤곽, 주황/초록은 quad 근사 실패 시 폴백)")


# =========================================================
# 실시간 데모
# =========================================================
def demo_webcam(camera_index=2, width=640, height=480):
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    print("q: 종료")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        bbox, rbox, quad, conf, strategy = detect_card(frame)
        if bbox is not None:
            draw_boxes(frame, bbox, rbox, quad)
            x1, y1 = bbox[0], bbox[1]
            cv2.putText(frame,
                        f"card {rbox[2]:.0f}x{rbox[3]:.0f}px {rbox[4]:.0f}deg "
                        f"[{conf}/{strategy}]",
                        (x1, max(y1 - 10, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "no card detected", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imshow("Card Detection Demo", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_webcam()
    elif len(sys.argv) > 2 and sys.argv[1] == "debug":
        debug_one(sys.argv[2])
    else:
        main()