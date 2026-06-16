"""
augmentation_crops.py
=====================
dataset_crops_split/ (perspective warp 크롭, 배경 기반 분할) 전용 DataLoader
- [변경] 랜덤 85/15 분할 → split_dataset.py 가 만든 train/val/test 폴더 사용
  * 이유: 연속 프레임이 train/val 에 동시에 들어가는 누수 방지
  * desk/gray = train 전용, green/wood = val/test → '새 배경' 일반화 측정 가능
- 크롭은 이미 카드가 정면으로 펴진 상태라서 scale 범위를 좁히고 rotation만 유지
"""

import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

SPLIT_DIR   = "/home/choi/cnn_pictures/dataset_crops_split"
TRAIN_DIR   = f"{SPLIT_DIR}/train"
VAL_DIR     = f"{SPLIT_DIR}/val"
TEST_DIR    = f"{SPLIT_DIR}/test"
BATCH_SIZE  = 32
NUM_WORKERS = 4

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ----- 학습용 증강 -----
# hue=0.0 고정 (과일 색 = 클래스 정보라 변경 불가)
# scale=(0.8,1.0): 크롭이 이미 정규화됐으므로 원본(0.5,1.0)보다 좁게
TRAIN_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(
        degrees=180,
        scale=(0.8, 1.0),
        fill=0,
    ),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.0),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ----- 검증/테스트용 (증강 없음) -----
VAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class CleanImageFolder(datasets.ImageFolder):
    """숨김 폴더(.) 를 클래스에서 제외 — 정렬 순서 일관성 보장."""
    def find_classes(self, directory):
        classes = sorted(
            entry.name for entry in os.scandir(directory)
            if entry.is_dir() and not entry.name.startswith(".")
        )
        if not classes:
            raise FileNotFoundError(f"클래스 폴더를 찾을 수 없습니다: {directory}")
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


def build_dataloaders():
    train_set = CleanImageFolder(TRAIN_DIR, transform=TRAIN_TF)
    val_set   = CleanImageFolder(VAL_DIR,   transform=VAL_TF)

    assert train_set.classes == val_set.classes, \
        "train/val 클래스 목록이 다릅니다! split_dataset.py 를 다시 실행하세요."
    assert len(train_set.classes) == 20, \
        f"클래스가 20개여야 하는데 {len(train_set.classes)}개 감지됨"

    print(f"감지된 클래스 {len(train_set.classes)}개: {train_set.classes}")
    print(f"학습 {len(train_set)}장 / 검증 {len(val_set)}장")

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    return train_loader, val_loader, train_set.classes


if __name__ == "__main__":
    tl, vl, cls = build_dataloaders()
    imgs, labels = next(iter(tl))
    print(f"배치 shape: {imgs.shape}  레이블 샘플: {labels[:8].tolist()}")
    print("DataLoader 정상")
