"""[NEW] Quantitative evaluation (the original code had none)."""
import csv
from pathlib import Path

import cv2
import numpy as np

from .copy_move import detect_copy_move

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def pixel_prf(pred, gt):
    p, g = pred > 0, gt > 127
    tp = np.logical_and(p, g).sum()
    prec = tp / p.sum() if p.sum() else 0.0
    rec = tp / g.sum() if g.sum() else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def _find_gt(stem, gt_dir):
    for name in (stem + "_gt", stem + "_mask"):
        for ext in IMG_EXT:
            f = Path(gt_dir) / (name + ext)
            if f.exists():
                return f
    return None


def evaluate_copy_move(forged_dir, gt_dir=None, authentic_dir=None, limit=100, cfg=None,
                       out_csv="results/copy_move_eval.csv"):
    rows, tp_img, n_forged = [], 0, 0
    files = sorted(p for p in Path(forged_dir).iterdir()
                   if p.suffix.lower() in IMG_EXT and not p.stem.endswith(("_gt", "_mask")))[:limit]
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        r = detect_copy_move(img, cfg)
        n_forged += 1
        tp_img += int(r.detected)
        P = R = F = ""
        gt_f = _find_gt(f.stem, gt_dir) if gt_dir else None
        if gt_f is not None:
            gt = cv2.imread(str(gt_f), 0)
            gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            P, R, F = pixel_prf(r.mask, gt)
        rows.append([f.name, int(r.detected), r.n_matches, r.n_inliers, P, R, F])

    fp, n_auth = 0, 0
    if authentic_dir:
        for f in sorted(p for p in Path(authentic_dir).iterdir() if p.suffix.lower() in IMG_EXT)[:limit]:
            img = cv2.imread(str(f))
            if img is None:
                continue
            n_auth += 1
            fp += int(detect_copy_move(img, cfg).detected)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "detected", "matches", "inliers", "precision", "recall", "f1"])
        w.writerows(rows)

    scored = [r for r in rows if r[6] != ""]
    print(f"Image-level detection rate (TPR): {tp_img}/{n_forged} = {tp_img / max(n_forged, 1):.2%}")
    if n_auth:
        print(f"False-positive rate on authentic: {fp}/{n_auth} = {fp / n_auth:.2%}")
    if scored:
        print(f"Pixel-level (mean over {len(scored)}): "
              f"P={np.mean([r[4] for r in scored]):.3f} "
              f"R={np.mean([r[5] for r in scored]):.3f} "
              f"F1={np.mean([r[6] for r in scored]):.3f}")
    print(f"Per-image results -> {out_csv}")
