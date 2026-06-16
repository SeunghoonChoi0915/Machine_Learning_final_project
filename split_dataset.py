"""
데이터셋 세션(배경) 단위 train/val/test 분할 v2
=====================================================
[v1 → v2 변경]
  green 배경을 train 전용에서 → 시간순 분할로 변경
  - train 70% / val 15% / test 15% (GAP으로 경계 분리)
  - 이유: green이 train에만 있으면 "green에서 잘 되는가"를
    val/test 지표로 확인할 방법이 없음

[배경별 전략]
  desk, gray → train 전용 (변경 없음)
  green      → 시간순 70/15/15 분할 → train + val + test
  wood       → 시간순 절반 분할    → val + test (변경 없음)

[결과 구조]
  dataset_split/
    train/  b1~s5  (desk 전부 + gray 전부 + green 앞 70%)
    val/    b1~s5  (green 중간 15%        + wood 앞 절반)
    test/   b1~s5  (green 뒤  15%        + wood 뒤 절반)
"""

import os
import shutil
from pathlib import Path

# =========================================================
# 설정
# =========================================================
# 바운딩 박스 없는 버전의 경우 이것을 수정할 것
# SRC_DIR = Path("/home/choi/cnn_pictures/dataset")
# OUT_DIR = Path("/home/choi/cnn_pictures/dataset_split")

SRC_DIR = Path("/home/choi/cnn_pictures/dataset_crops")
OUT_DIR = Path("/home/choi/cnn_pictures/dataset_crops_split")

TRAIN_ONLY_BG = ["desk", "gray"]   # train 전용 배경
SPLIT_BG      = ["green", "wood"]  # train/val/test 분할 배경
TEST_ONLY_BG  = ["floor"]          # 새로 촬영한 '처음 보는 배경' → 전량 test
                                   #   예: ["floor"] (capture_video.py 의 BACKGROUND 와 동일 이름)
                                   #   train/val 에 전혀 안 들어가는 진짜 held-out OOD test

# green: 70 / 15 / 15 비율
# wood : 0  / 50 / 50 비율 (기존과 동일 — wood는 train에 안 씀)
SPLIT_RATIO = {
    "green": (0.70, 0.15, 0.15),
    "wood" : (0.00, 0.50, 0.50),
}

GAP      = 3   # val/test 경계에서 버릴 프레임 수
GAP2     = 3   # train/val 경계에서 버릴 프레임 수 (green 전용)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_classes(src_dir: Path):
    return sorted(
        e.name for e in os.scandir(src_dir)
        if e.is_dir() and not e.name.startswith(".")
    )


def list_images(folder: Path):
    return sorted(f for f in folder.glob("*") if f.suffix.lower() in IMG_EXTS)


def split_by_ratio(imgs, train_r, val_r, gap=3, gap2=3):
    """시간순 정렬된 imgs를 (train, val, test) 로 분할.
       경계마다 gap 장씩 버려 인접 프레임 누출 방지."""
    n = len(imgs)
    n_train = int(n * train_r)
    n_val   = int(n * val_r)

    train = imgs[:n_train]
    # train/val 경계 gap 제거
    val_start = n_train + gap2
    val_end   = val_start + n_val
    val       = imgs[val_start:val_end]
    # val/test 경계 gap 제거
    test_start = val_end + gap
    test      = imgs[test_start:]
    return train, val, test


# =========================================================
# 분할 실행
# =========================================================
def split():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
        print(f"기존 분할 폴더 삭제: {OUT_DIR}")

    classes = list_classes(SRC_DIR)
    print(f"클래스 {len(classes)}개: {classes}")
    print(f"train 전용 배경 : {TRAIN_ONLY_BG}")
    print(f"분할 배경       : {SPLIT_BG}")
    print(f"test 전용 배경  : {TEST_ONLY_BG}")
    print(f"  green 비율: train {SPLIT_RATIO['green'][0]:.0%} / "
          f"val {SPLIT_RATIO['green'][1]:.0%} / test {SPLIT_RATIO['green'][2]:.0%}")
    print(f"  wood  비율: val {SPLIT_RATIO['wood'][1]:.0%} / "
          f"test {SPLIT_RATIO['wood'][2]:.0%}\n")

    total = {"train": 0, "val": 0, "test": 0}

    for cls in classes:
        # train 전용 배경은 반드시 있어야 함 (없으면 학습 데이터 누락)
        for bg in TRAIN_ONLY_BG:
            bg_path = SRC_DIR / cls / bg
            if not bg_path.is_dir():
                raise FileNotFoundError(f"배경 폴더 없음: {bg_path}")

        pairs = {"train": [], "val": [], "test": []}

        # train 전용 배경 → 전부 train
        for bg in TRAIN_ONLY_BG:
            for img in list_images(SRC_DIR / cls / bg):
                pairs["train"].append((bg, img))

        # 분할 배경 → 비율대로 쪼개기 (해당 클래스에 없으면 건너뜀)
        for bg in SPLIT_BG:
            bg_path = SRC_DIR / cls / bg
            if not bg_path.is_dir():
                print(f"  ⚠ {cls}/{bg} 없음 — 건너뜀")
                continue
            imgs = list_images(bg_path)
            tr, val_r, te = SPLIT_RATIO[bg]
            tr_imgs, val_imgs, te_imgs = split_by_ratio(
                imgs, tr, val_r, gap=GAP, gap2=GAP2
            )
            for img in tr_imgs:  pairs["train"].append((bg, img))
            for img in val_imgs: pairs["val"].append((bg, img))
            for img in te_imgs:  pairs["test"].append((bg, img))

        # test 전용 배경 → 전부 test (해당 클래스에 없으면 건너뜀)
        for bg in TEST_ONLY_BG:
            bg_path = SRC_DIR / cls / bg
            if not bg_path.is_dir():
                print(f"  ⚠ {cls}/{bg} 없음 — 건너뜀")
                continue
            for img in list_images(bg_path):
                pairs["test"].append((bg, img))

        # 파일 복사 (배경 접두어로 파일명 충돌 방지)
        for split_name, split_pairs in pairs.items():
            dst = OUT_DIR / split_name / cls
            dst.mkdir(parents=True, exist_ok=True)
            for bg, img in split_pairs:
                shutil.copy2(img, dst / f"{bg}_{img.name}")
            total[split_name] += len(split_pairs)

        g_tr = sum(1 for bg, _ in pairs["train"]  if bg == "green")
        g_va = sum(1 for bg, _ in pairs["val"]    if bg == "green")
        g_te = sum(1 for bg, _ in pairs["test"]   if bg == "green")
        print(f"  {cls:>4}  train {len(pairs['train']):>4} (green {g_tr:>3})"
              f"  /  val {len(pairs['val']):>4} (green {g_va:>3})"
              f"  /  test {len(pairs['test']):>4} (green {g_te:>3})")

    print(f"\n전체  →  train {total['train']}장  /  "
          f"val {total['val']}장  /  test {total['test']}장")
    print(f"저장 위치: {OUT_DIR}")
    print("\n완료! train.py 재학습 → evaluate.py 로 test 성능 확인")
    print("→ val/test 의 green 행을 보면 '새 배경' 일반화 성능을 바로 확인 가능")


if __name__ == "__main__":
    split()