"""
할리갈리 카드 분류 - 데이터 전처리 및 Augmentation
=====================================================
[수정] 데이터 누수 방지를 위해 train/val 폴더를 물리적으로 분리한 구조 사용
  dataset_split/
    train/  b1~s5
    val/    b1~s5
"""

import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

TRAIN_DIR   = "/home/choi/cnn_pictures/dataset_split/train"
VAL_DIR     = "/home/choi/cnn_pictures/dataset_split/val"
IMG_SIZE    = 224
BATCH_SIZE  = 32
NUM_WORKERS = 4

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# train_transform = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.RandomRotation(180),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomVerticalFlip(p=0.5),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
# ])
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomAffine(degrees=180, scale=(0.5, 1.0), fill=0),  # ← 교체
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

class CleanImageFolder(datasets.ImageFolder):
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
    train_set = CleanImageFolder(TRAIN_DIR, transform=train_transform)
    val_set   = CleanImageFolder(VAL_DIR,   transform=val_transform)

    assert train_set.classes == val_set.classes, \
        "train/val 클래스 목록이 다릅니다! split_dataset.py 를 다시 실행하세요."
    assert len(train_set.classes) == 20, \
        f"클래스가 20개여야 하는데 {len(train_set.classes)}개 감지됨"

    print(f"감지된 클래스 {len(train_set.classes)}개: {train_set.classes}")
    print(f"학습 {len(train_set)}장 / 검증 {len(val_set)}장")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader, train_set.classes

if __name__ == "__main__":
    train_loader, val_loader, classes = build_dataloaders()
    images, labels = next(iter(train_loader))
    print(f"배치 텐서 shape: {images.shape}")
    print(f"라벨 예시: {labels[:8].tolist()}")
