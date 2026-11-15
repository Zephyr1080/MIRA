"""Turns the detector outputs into one tampering likelihood for the web app."""
import math
from pathlib import Path

from .config import CopyMoveConfig
from .pipeline import analyze


def copy_move_score(cm):
    """0 when nothing is found. A detection starts at 0.70 and rises toward 1.0
    as the number of RANSAC inliers grows (about 0.77 at 8 inliers, 0.95 at 100).
    This is a hand-made mapping, not a trained probability."""
    if not cm.detected:
        return 0.0
    return 0.70 + 0.30 * (1 - math.exp(-cm.n_inliers / 30))


def combine(cm_score, splice_prob):
    """The image counts as tampered if either detector fires (noisy-OR)."""
    if splice_prob is None:
        return cm_score
    return 1 - (1 - cm_score) * (1 - splice_prob)


def label_for(p, likely=0.5, most_likely=0.75):
    """Three outcomes. A copy-move detection scores 0.77 or more, so it always
    lands in the top tier; splicing alone needs a probability of 0.75 or more."""
    if p < likely:
        return "Likely authentic"
    if p < most_likely:
        return "Likely tampered"
    return "Most likely tampered"


def assess(image_path, model_path="models/splicing.joblib", cfg=None,
           out_dir="results"):
    """Runs the full pipeline on one image and returns everything the UI shows."""
    cfg = cfg or CopyMoveConfig()
    r = analyze(image_path, model_path, cm_cfg=cfg, out_dir=out_dir)
    cm = r["copy_move"]
    sp = r["splicing_prob"]
    cm_s = copy_move_score(cm)
    p = combine(cm_s, sp)
    return {
        "likelihood": p,
        "label": label_for(p),
        "copy_move_score": cm_s,
        "splicing_prob": sp,
        "model_found": Path(model_path).exists(),
        "verdict": r["verdict"],
        "n_keypoints": cm.n_keypoints,
        "n_matches": cm.n_matches,
        "n_inliers": cm.n_inliers,
        "detected_copy_move": cm.detected,
        "overlay": cm.overlay,   # BGR array
        "mask": cm.mask,         # 0/255 array
    }
