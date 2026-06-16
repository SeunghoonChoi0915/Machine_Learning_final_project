"""
add_blue_test.py — blue 배경(새 OOD 배경)을 test 세트에만 추가
=====================================================
기존 train/val/test 크롭·labels.json 을 전혀 건드리지 않고,
dataset/{class}/blue/ 만 라벨(detect_card) → warp(warp_card) →
dataset_crops_split/test/{class}/blue_{name} 로 저장한다.

- conf=low 또는 검출 실패 또는 warp 크기 이상은 make_crops 와 동일하게 제외.
- 실행 후 evaluate.py 를 돌리면 blue 가 test 정확도에 반영됨.
"""

from pathlib import Path

import cv2

from bounding_box import detect_card
from make_crops import warp_card

SRC_DIR  = Path("/home/choi/cnn_pictures/dataset")
TEST_DIR = Path("/home/choi/cnn_pictures/dataset_crops_split/test")
BG       = "blue"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    classes = sorted(d.name for d in SRC_DIR.iterdir()
                     if d.is_dir() and not d.name.startswith("."))

    n_ok = n_fail = n_low = n_warpfail = 0
    per_cls = {}
    for cls in classes:
        blue_dir = SRC_DIR / cls / BG
        if not blue_dir.is_dir():
            continue
        out_dir = TEST_DIR / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        cnt = 0
        for img_path in sorted(blue_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                n_fail += 1
                continue
            bbox, rbox, quad, conf, strategy = detect_card(img)
            if bbox is None or quad is None:
                n_fail += 1
                continue
            if conf == "low":
                n_low += 1
                continue
            crop = warp_card(img, quad)
            if crop is None:
                n_warpfail += 1
                continue
            cv2.imwrite(str(out_dir / f"{BG}_{img_path.name}"), crop)
            n_ok += 1
            cnt += 1
        per_cls[cls] = cnt

    print(f"{'class':>6} | {'blue_ok':>7}")
    print("-" * 18)
    for cls in classes:
        print(f"{cls:>6} | {per_cls.get(cls, 0):>7}")
    print("-" * 18)
    print(f"\nblue test 추가: {n_ok}장 저장")
    print(f"제외 → 검출실패 {n_fail} / conf low {n_low} / warp실패 {n_warpfail}")
    print(f"저장 위치: {TEST_DIR}/<class>/blue_*.jpg")
    print("이제 evaluate.py 를 실행하면 blue 가 test 정확도에 반영됩니다.")


if __name__ == "__main__":
    main()
