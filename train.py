"""
할리갈리 카드 분류 - ResNet18 처음부터 학습 (From Scratch)
=====================================================
- 모델   : ResNet18 (가중치 없이 처음부터)
           * from-scratch 에서는 큰 모델일수록 배경 등 지름길을
             외우기 쉬워, 데이터 규모에 맞는 ResNet18 선택
- 최적화 : Adam (lr=0.001, weight_decay=1e-4)
- 손실   : CrossEntropyLoss
- 스케줄 : StepLR (10 epoch마다 lr × 0.1)
- 클래스 : 20개 (b1~b5, l1~l5, p1~p5, s1~s5)
"""

import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import models
from tqdm import tqdm

from augmentation import build_dataloaders

# =========================================================
# 1. 설정
# =========================================================
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS  = 30
NUM_CLASSES = 20
SAVE_PATH   = "/home/choi/cnn_pictures/best_model_v2.pth"

print(f"사용 디바이스: {DEVICE}")


# =========================================================
# 2. 모델 정의 - ResNet18 처음부터 학습 (From Scratch)
# =========================================================
def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)

    in_features = model.fc.in_features          # 512
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )

    return model.to(DEVICE)


# =========================================================
# 3. 학습 함수
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [Train]", leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            "loss": f"{running_loss/total:.4f}",
            "acc" : f"{correct/total*100:.1f}%",
        })

    return running_loss / total, correct / total * 100


# =========================================================
# 4. 검증 함수
# =========================================================
@torch.no_grad()
def validate(model, loader, criterion, epoch):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [ Val ]", leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({
            "loss": f"{running_loss/total:.4f}",
            "acc" : f"{correct/total*100:.1f}%",
        })

    return running_loss / total, correct / total * 100


# =========================================================
# 5. 메인 학습 루프
# =========================================================
def main():
    train_loader, val_loader, classes = build_dataloaders()

    model     = build_model(NUM_CLASSES)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    # lr 낮추고 스케줄러 더 촘촘하게
    optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.3)

    best_val_acc = 0.0
    epoch_times  = []

    print(f"\n{'epoch':>6} | {'train_loss':>10} | {'train_acc':>9} | {'val_loss':>8} | {'val_acc':>7} | {'lr':>8} | {'epoch_time':>10} | {'남은시간':>10}")
    print("-" * 95)

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss,   val_acc   = validate(model, val_loader, criterion, epoch)
        scheduler.step()

        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        avg_time      = sum(epoch_times) / len(epoch_times)
        remaining     = avg_time * (NUM_EPOCHS - epoch)
        remaining_str = f"{int(remaining//60)}분 {int(remaining%60)}초"
        elapsed_str   = f"{elapsed:.1f}초"
        current_lr    = scheduler.get_last_lr()[0]

        print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>8.2f}% | {val_loss:>8.4f} | {val_acc:>6.2f}% | {current_lr:>8.6f} | {elapsed_str:>10} | {remaining_str:>10}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch"      : epoch,
                "model_state": copy.deepcopy(model.state_dict()),
                "classes"    : classes,
                "val_acc"    : best_val_acc,
            }, SAVE_PATH)
            print(f"         → 최고 모델 저장 (val_acc={best_val_acc:.2f}%)")

    print(f"\n학습 완료! 최고 검증 정확도: {best_val_acc:.2f}%")
    print(f"모델 저장 위치: {SAVE_PATH}")


if __name__ == "__main__":
    main()