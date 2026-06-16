"""
독립 test 구간으로 진짜 성능 확인 (크롭 파이프라인)
=====================================================
- split_dataset.py 가 만든 dataset_crops_split/test/ 는 학습/검증에 전혀 쓰이지
  않은 영상 뒷부분 구간(green 뒤 15% + wood 절반)입니다.
  여기서 나오는 정확도가 '새 배경' 실전 성능에 가장 가깝습니다.
- best_model_crops_v3.pth(학습이 val 기준으로 저장한 모델)를 불러와 test 정확도와
  클래스별 정확도를 출력합니다.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from augmentation_crops import CleanImageFolder, VAL_TF, TEST_DIR, BATCH_SIZE, NUM_WORKERS

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/home/choi/cnn_pictures/best_model_crops_v3_fixed.pth"


def build_model(num_classes: int) -> nn.Module:
    # train_crop.py 의 build_model 과 동일한 구조여야 state_dict 가 맞습니다
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model.to(DEVICE)


@torch.no_grad()
def main():
    ckpt    = torch.load(MODEL_PATH, map_location=DEVICE)
    classes = ckpt["classes"]
    print(f"불러온 모델: epoch={ckpt['epoch']}, 저장 당시 val_acc={ckpt['val_acc']:.2f}%")
    print(f"클래스 {len(classes)}개\n")

    test_set = CleanImageFolder(TEST_DIR, transform=VAL_TF)
    assert test_set.classes == classes, "test 클래스가 모델 클래스와 다릅니다!"
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    model = build_model(len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    correct, total = 0, 0
    per_correct = [0] * len(classes)
    per_total   = [0] * len(classes)

    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        preds = model(images).argmax(1)
        correct += preds.eq(labels).sum().item()
        total   += labels.size(0)
        for label, pred in zip(labels, preds):
            per_total[label]   += 1
            per_correct[label] += int(pred == label)

    print(f"{'class':>6} | {'acc':>7} | {'n':>5}")
    print("-" * 26)
    for i, cls in enumerate(classes):
        acc = per_correct[i] / per_total[i] * 100 if per_total[i] else 0.0
        print(f"{cls:>6} | {acc:>6.2f}% | {per_total[i]:>5}")

    print("-" * 26)
    print(f"\n전체 test 정확도: {correct/total*100:.2f}%  ({correct}/{total})")
    print("→ 이 숫자가 학습 중 val_acc 보다 많이 낮으면 여전히 누수/과적합이 있는 것입니다.")


if __name__ == "__main__":
    main()
