"""
confusion_l1.py — 모델이 test에서 무엇을 무엇으로 착각하는지 분석
- 클래스별 오분류 분포, 배경(blue/green/wood)별 정확도
- 특히 l1 의 오답이 어디로 새는지 집중 분석
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from augmentation_crops import CleanImageFolder, VAL_TF, TEST_DIR, BATCH_SIZE, NUM_WORKERS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "/home/choi/cnn_pictures/best_model_crops_v3_fixed.pth"


def build_model(n):
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, n))
    return m.to(DEVICE)


@torch.no_grad()
def main():
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    classes = ckpt["classes"]
    print(f"모델: {Path(MODEL_PATH).name}  epoch={ckpt['epoch']}  val_acc={ckpt['val_acc']:.2f}%\n")

    ds = CleanImageFolder(TEST_DIR, transform=VAL_TF)
    paths = [p for p, _ in ds.samples]                      # shuffle=False 라 순서 일치
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    model = build_model(len(classes)); model.load_state_dict(ckpt["model_state"]); model.eval()

    preds_all, labels_all = [], []
    for imgs, labels in loader:
        preds_all += model(imgs.to(DEVICE)).argmax(1).cpu().tolist()
        labels_all += labels.tolist()

    def bg_of(path):
        name = Path(path).name
        return name.split("_", 1)[0]   # blue / green / wood

    # 1) 배경별 전체 정확도
    bg_tot, bg_ok = defaultdict(int), defaultdict(int)
    for p, t, pr in zip(paths, labels_all, preds_all):
        bg = bg_of(p); bg_tot[bg] += 1; bg_ok[bg] += int(t == pr)
    print("=== 배경별 전체 정확도 ===")
    for bg in sorted(bg_tot):
        print(f"  {bg:>6}: {bg_ok[bg]/bg_tot[bg]*100:6.2f}%  ({bg_ok[bg]}/{bg_tot[bg]})")

    # 2) l1 집중 분석: 오답이 어디로 가나 (배경별)
    l1 = classes.index("l1")
    print("\n=== l1(라임1) 분석 ===")
    for bg in sorted(bg_tot):
        wrong = Counter()
        tot = ok = 0
        for p, t, pr in zip(paths, labels_all, preds_all):
            if t == l1 and bg_of(p) == bg:
                tot += 1; ok += int(pr == l1)
                if pr != l1:
                    wrong[classes[pr]] += 1
        if tot:
            miss = "  ".join(f"{c}:{n}" for c, n in wrong.most_common())
            print(f"  {bg:>6}: acc {ok/tot*100:6.2f}% ({ok}/{tot})  오답→ {miss or '없음'}")

    # 3) 전체에서 가장 흔한 혼동쌍 top 10
    conf = Counter()
    for t, pr in zip(labels_all, preds_all):
        if t != pr:
            conf[(classes[t], classes[pr])] += 1
    print("\n=== 가장 흔한 혼동 (정답→예측) top 10 ===")
    for (t, pr), n in conf.most_common(10):
        print(f"  {t} → {pr} : {n}회")


if __name__ == "__main__":
    main()
