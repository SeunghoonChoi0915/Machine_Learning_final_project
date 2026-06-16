
import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import models
from tqdm import tqdm

from augmentation_crops import build_dataloaders

# =========================================================
# 설정
# =========================================================
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS  = 30
NUM_CLASSES = 20
SAVE_PATH   = "/home/choi/cnn_pictures/best_model_crops_v3_fixed.pth"

print(f"사용 디바이스: {DEVICE}")


# =========================================================
# 모델
# =========================================================
def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)          # scratch 학습
    in_features = model.fc.in_features             # 512
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model.to(DEVICE)


# =========================================================
# 학습 / 검증 루프
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [Train]", leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * imgs.size(0)
        correct  += out.max(1)[1].eq(labels).sum().item()
        total    += labels.size(0)
        pbar.set_postfix(loss=f"{loss_sum/total:.4f}",
                         acc=f"{correct/total*100:.1f}%")
    return loss_sum / total, correct / total * 100


@torch.no_grad()
def validate(model, loader, criterion, epoch):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [ Val ]", leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out  = model(imgs)
        loss = criterion(out, labels)
        loss_sum += loss.item() * imgs.size(0)
        correct  += out.max(1)[1].eq(labels).sum().item()
        total    += labels.size(0)
        pbar.set_postfix(loss=f"{loss_sum/total:.4f}",
                         acc=f"{correct/total*100:.1f}%")
    return loss_sum / total, correct / total * 100


# =========================================================
# main
# =========================================================
def main():
    train_loader, val_loader, classes = build_dataloaders()

    model     = build_model(NUM_CLASSES)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    best_val_acc = 0.0
    times = []

    hdr = (f"{'epoch':>6} | {'train_loss':>10} | {'train_acc':>9} | "
           f"{'val_loss':>8} | {'val_acc':>7} | {'lr':>8} | "
           f"{'epoch_time':>10} | {'남은시간':>10}")
    print(f"\n{hdr}")
    print("-" * len(hdr))

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        vl, va = validate(model, val_loader, criterion, epoch)
        scheduler.step()

        elapsed = time.time() - t0
        times.append(elapsed)
        remain  = sum(times) / len(times) * (NUM_EPOCHS - epoch)
        lr      = scheduler.get_last_lr()[0]

        print(f"{epoch:>6} | {tl:>10.4f} | {ta:>8.2f}% | "
              f"{vl:>8.4f} | {va:>6.2f}% | {lr:>8.6f} | "
              f"{elapsed:>9.1f}초 | {int(remain//60)}분 {int(remain%60)}초")

        if va > best_val_acc:
            best_val_acc = va
            torch.save({
                "epoch":       epoch,
                "model_state": copy.deepcopy(model.state_dict()),
                "classes":     classes,
                "val_acc":     best_val_acc,
            }, SAVE_PATH)
            print(f"         → 최고 모델 저장 (val_acc={best_val_acc:.2f}%)")

    print(f"\n학습 완료! 최고 검증 정확도: {best_val_acc:.2f}%")
    print(f"모델 저장 위치: {SAVE_PATH}")


if __name__ == "__main__":
    main()