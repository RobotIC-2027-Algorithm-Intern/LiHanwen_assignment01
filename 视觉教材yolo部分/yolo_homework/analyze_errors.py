import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path("/home/zhulu/github_clone/Assignment-1-Computer-Vision/视觉教材yolo部分")
WEIGHTS = ROOT / "runs/detect/train2/weights/best.pt"
VAL_IMGS = ROOT / "dataset/images/val"
VAL_LBLS = ROOT / "dataset/labels/val"
OUT = Path(__file__).resolve().parent / "out"
IOU_T = 0.45
CONF_T = 0.25


def load_gt(txt, w, h):
    rows = []
    if not txt.exists():
        return rows
    for line in txt.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        c, xc, yc, bw, bh = int(p[0]), *map(float, p[1:5])
        rows.append([c, (xc - bw / 2) * w, (yc - bh / 2) * h,
                     (xc + bw / 2) * w, (yc + bh / 2) * h])
    return np.array(rows, dtype=float).reshape(-1, 5)


def iou(a, b):
    x1 = np.maximum(a[0], b[:, 0]); y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2]); y2 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return np.where(ua > 0, inter / ua, 0.0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "miss").mkdir(exist_ok=True)
    (OUT / "false").mkdir(exist_ok=True)
    (OUT / "confuse").mkdir(exist_ok=True)

    model = YOLO(str(WEIGHTS))
    names = model.names
    stat = defaultdict(lambda: defaultdict(int))
    per_img = []

    for r in model.predict(source=str(VAL_IMGS), conf=CONF_T, iou=IOU_T,
                           stream=True, verbose=False):
        stem = Path(r.path).stem
        h, w = r.orig_shape
        gt = load_gt(VAL_LBLS / f"{stem}.txt", w, h)
        pred = np.column_stack([r.boxes.xyxy.cpu().numpy(),
                                r.boxes.conf.cpu().numpy(),
                                r.boxes.cls.cpu().numpy()]).reshape(-1, 6)

        n_tp = n_fp_bg = n_fp_dup = n_fp_cls = n_fn = 0
        matched = np.zeros(len(gt), dtype=bool)
        order = np.argsort(-pred[:, 4]) if len(pred) else []
        for i in order:
            if len(gt) == 0:
                n_fp_bg += 1
                continue
            ov = iou(pred[i, :4], gt[:, 1:5]) * (pred[i, 5] == gt[:, 0])
            k = int(np.argmax(ov))
            if ov[k] > IOU_T and not matched[k]:
                matched[k] = True
                n_tp += 1
            else:
                ov_all = iou(pred[i, :4], gt[:, 1:5])
                k2 = int(np.argmax(ov_all)) if len(ov_all) else -1
                if k2 >= 0 and ov_all[k2] > IOU_T and gt[k2, 0] != pred[i, 5]:
                    n_fp_cls += 1
                elif k2 >= 0 and ov_all[k2] > IOU_T:
                    n_fp_dup += 1
                else:
                    n_fp_bg += 1
        n_fn = int((~matched).sum())

        for j, g in enumerate(gt):
            stat[int(g[0])]["gt"] += 1
            if matched[j]:
                stat[int(g[0])]["tp"] += 1
            else:
                stat[int(g[0])]["fn"] += 1
        for p in pred:
            stat[int(p[5])]["pred"] += 1

        per_img.append(dict(img=stem, tp=n_tp, fp=n_fp_bg + n_fp_dup + n_fp_cls,
                            fp_bg=n_fp_bg, fp_dup=n_fp_dup, fp_cls=n_fp_cls, fn=n_fn,
                            gt=len(gt)))

        if n_fn or n_fp_bg or n_fp_dup or n_fp_cls:
            vis = r.orig_img.copy()
            for g in gt:
                cv2.rectangle(vis, (int(g[1]), int(g[2])), (int(g[3]), int(g[4])), (0, 255, 0), 3)
            for p in pred:
                cv2.rectangle(vis, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), (0, 0, 255), 2)
                cv2.putText(vis, f"{names[int(p[5])]} {p[4]:.2f}", (int(p[0]), int(p[1]) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if n_fn and not (n_fp_bg or n_fp_dup or n_fp_cls):
                cv2.imwrite(str(OUT / "miss" / f"{stem}.jpg"), vis)
            elif n_fp_bg or n_fp_dup or n_fp_cls:
                cv2.imwrite(str(OUT / "false" / f"{stem}.jpg"), vis)

    rows = []
    print(f"{'类别':<10}{'GT':>6}{'预测':>6}{'TP':>6}{'FP':>6}{'FN':>6}{'P':>8}{'R':>8}")
    for c in sorted(stat):
        s = stat[c]
        tp, fn, pr = s["tp"], s["fn"], s["pred"]
        P = tp / pr if pr else 0
        R = tp / (tp + fn) if (tp + fn) else 0
        print(f"{names[c]:<10}{s['gt']:>6}{pr:>6}{tp:>6}{pr - tp:>6}{fn:>6}{P:>8.3f}{R:>8.3f}")
        rows.append(dict(cls=names[c], gt=s["gt"], pred=pr, tp=tp, fp=pr - tp, fn=fn, P=P, R=R))

    tot = dict(tp=sum(x["tp"] for x in per_img), fn=sum(x["fn"] for x in per_img),
               fp=sum(x["fp"] for x in per_img), fp_bg=sum(x["fp_bg"] for x in per_img),
               fp_dup=sum(x["fp_dup"] for x in per_img), fp_cls=sum(x["fp_cls"] for x in per_img))
    print("\n合计:", tot)
    print("误检构成: 背景误检 %d / 重复框 %d / 类别错 %d" % (tot["fp_bg"], tot["fp_dup"], tot["fp_cls"]))

    worst_fn = sorted(per_img, key=lambda x: -x["fn"])[:8]
    worst_fp = sorted(per_img, key=lambda x: -x["fp"])[:8]
    print("\n漏检最多:", [(x["img"], x["fn"], x["gt"]) for x in worst_fn])
    print("误检最多:", [(x["img"], x["fp"], x["fp_bg"], x["fp_dup"]) for x in worst_fp])

    (OUT / "summary.json").write_text(json.dumps(
        dict(classes=rows, total=tot, per_img=per_img), ensure_ascii=False, indent=1))
    print("\n图片已存到:", OUT)


if __name__ == "__main__":
    main()
