"""
make_crops_v2.py — quad로 perspective warp 크롭 생성 (분류기 재학습용)
=====================================================
[v1 → v2 변경]
  bbox 사각 크롭 → quad 기반 perspective warp 크롭
  - 카드를 평평하게 펴서 저장 → 분류기 입력에서 원근/회전 변수가 제거됨
  - 추론 파이프라인(검출기가 quad 예측 → warp → 분류)과 일관된 입력 분포

- 입력 : labels.json (conf low 제외, quad 없는 항목 제외)
- 출력 : dataset_crops/{class}/{bg}/{원본파일명}
- 크롭 크기: 카드 실측 비율 유지를 위해 quad 변 길이에서 계산,
  분류기 입력이 어차피 Resize(224)이므로 크기 자체는 중요하지 않음

사용법:
    python3 make_crops_v2.py
"""

import json
from pathlib import Path

import cv2
import numpy as np

LABELS_JSON = Path("/home/choi/cnn_pictures/labels.json")
OUT_DIR     = Path("/home/choi/cnn_pictures/dataset_crops")

SKIP_CONF = {"low"}     # audit에서 강등된 항목 포함 자동 제외
MIN_SIDE  = 64          # warp 결과가 이보다 작으면 스킵 (이상 라벨 안전장치)
MAX_SIDE  = 600


def warp_card(img, quad):
    """quad(tl,tr,br,bl)를 직사각형으로 펴서 반환. 실패 시 None."""
    q = np.array(quad, np.float32)
    w_top    = np.linalg.norm(q[1] - q[0])
    w_bottom = np.linalg.norm(q[2] - q[3])
    h_left   = np.linalg.norm(q[3] - q[0])
    h_right  = np.linalg.norm(q[2] - q[1])
    out_w = int(round((w_top + w_bottom) / 2))
    out_h = int(round((h_left + h_right) / 2))
    if not (MIN_SIDE <= out_w <= MAX_SIDE and MIN_SIDE <= out_h <= MAX_SIDE):
        return None
    dst = np.array([[0, 0], [out_w - 1, 0],
                    [out_w - 1, out_h - 1], [0, out_h - 1]], np.float32)
    M = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(img, M, (out_w, out_h))


def main():
    data = json.loads(LABELS_JSON.read_text())
    anns = data["annotations"]

    n_ok = n_skip = 0
    for a in anns:
        if a["conf"] in SKIP_CONF or a["quad"] is None:
            n_skip += 1
            continue

        src = Path(a["path"])
        img = cv2.imread(str(src))
        if img is None:
            n_skip += 1
            continue

        crop = warp_card(img, a["quad"])
        if crop is None:
            n_skip += 1
            continue

        bg = src.parent.name
        out = OUT_DIR / a["class"] / bg / src.name
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), crop)
        n_ok += 1

    print(f"warp 크롭 저장: {n_ok}장  /  스킵: {n_skip}장  →  {OUT_DIR}")
    print("분류기 재학습 시 데이터 경로만 dataset_crops로 변경.")
    print("주의: 크롭이 카드로 꽉 차므로 augmentation에서 crop 계열 금지 원칙이")
    print("      더 중요해지고, 회전/반전은 그대로 유효합니다.")


if __name__ == "__main__":
    main()