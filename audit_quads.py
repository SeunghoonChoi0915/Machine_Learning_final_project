"""
audit_quads.py — labels.json에서 기하학적으로 수상한 quad 자동 검출
=====================================================
모션 블러 프레임 등에서 마스크가 일그러져 quad가 비틀리는 케이스를
사람 눈 대신 두 가지 지표로 걸러낸다:

  1. 면적 일치: quad 면적 / rbox 면적 — 정상이면 ~1.0 근처
     (둘 다 같은 윤곽에서 나오므로 크게 어긋나면 quad가 비틀린 것)
  2. 꼭짓점 각도: 원근이 있어도 카드 quad의 내각은 90°±35° 안쪽.
     그보다 뾰족/뭉툭한 꼭짓점은 한 변이 엉뚱한 데 스냅된 것

플래그된 항목은 미리보기 이미지로 저장 + 목록 출력.
검토 후 처리 방법:
  - 몇십 장 수준이면: 그대로 둬도 됨 (bbox 크롭에는 영향 미미)
  - 확실히 깨진 것들은: 아래 WRITE_BACK=True 로 바꿔 재실행하면
    해당 항목의 conf를 "low"로 강등 → make_crops의 SKIP_CONF로 자동 제외

사용법:
    python3 audit_quads.py
"""

import json
from pathlib import Path

import cv2
import numpy as np

LABELS_JSON = Path("/home/choi/cnn_pictures/labels.json")
OUT_DIR     = Path("/home/choi/cnn_pictures/quad_audit")

AREA_RATIO_RANGE = (0.80, 1.10)   # quad면적/rbox면적 허용 범위
ANGLE_RANGE      = (55.0, 125.0)  # 꼭짓점 내각 허용 범위 (도)

WRITE_BACK = True   # True면 플래그 항목의 conf를 "low"로 바꿔 저장


def quad_area(q):
    x, y = q[:, 0], q[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def corner_angles(q):
    angs = []
    for i in range(4):
        v1 = q[i - 1] - q[i]
        v2 = q[(i + 1) % 4] - q[i]
        c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        angs.append(float(np.degrees(np.arccos(np.clip(c, -1, 1)))))
    return angs


def main():
    data = json.loads(LABELS_JSON.read_text())
    flagged = []

    for a in data["annotations"]:
        if a["quad"] is None:
            continue
        q = np.array(a["quad"], np.float32)
        _, _, rw, rh, _ = a["rbox"]
        ratio = quad_area(q) / (rw * rh + 1e-9)
        angs = corner_angles(q)

        if not (AREA_RATIO_RANGE[0] <= ratio <= AREA_RATIO_RANGE[1]) \
                or min(angs) < ANGLE_RANGE[0] or max(angs) > ANGLE_RANGE[1]:
            flagged.append((a, ratio, min(angs), max(angs)))

    n = len(data["annotations"])
    print(f"전체 {n}장 중 플래그 {len(flagged)}장 ({len(flagged)/n*100:.2f}%)\n")

    if flagged:
        if OUT_DIR.exists():
            for f in OUT_DIR.glob("*"):
                f.unlink()
        OUT_DIR.mkdir(exist_ok=True)

        for a, ratio, amin, amax in sorted(flagged, key=lambda x: x[1]):
            print(f"  ratio {ratio:4.2f}  각도 {amin:3.0f}~{amax:3.0f}°  "
                  f"[{a.get('strategy','?'):>10}]  {a['path']}")
            img = cv2.imread(a["path"])
            if img is None:
                continue
            pts = np.array(a["quad"], np.int32)
            cv2.polylines(img, [pts], True, (255, 0, 255), 3)
            x1, y1, x2, y2 = a["bbox"]                          # ★ 추가
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # ★ 추가
            cv2.putText(img, f"ratio {ratio:.2f} ang {amin:.0f}-{amax:.0f}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
            name = f"{a['class']}_{Path(a['path']).parent.name}_{Path(a['path']).name}"
            cv2.imwrite(str(OUT_DIR / name), img)

        print(f"\n미리보기 저장: {OUT_DIR}")

    if WRITE_BACK and flagged:
        flagged_paths = {a["path"] for a, *_ in flagged}
        for a in data["annotations"]:
            if a["path"] in flagged_paths:
                a["conf"] = "low"
        LABELS_JSON.write_text(json.dumps(data, indent=2))
        print(f"플래그 {len(flagged)}장의 conf를 'low'로 강등하여 저장 완료")
        print("→ make_crops.py가 SKIP_CONF로 자동 제외합니다")


if __name__ == "__main__":
    main()