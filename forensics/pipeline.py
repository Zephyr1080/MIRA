"""[NEW] Unified entry point: one call runs copy-move + splicing and returns a verdict."""
from pathlib import Path

import cv2

from .copy_move import detect_copy_move
from .splicing import predict_splicing


def analyze(image_path, model_path=None, cm_cfg=None, out_dir="results", splice_threshold=0.5):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    cm = detect_copy_move(img, cm_cfg)
    splice_prob = None
    if model_path and Path(model_path).exists():
        splice_prob = predict_splicing(img, model_path)

    if cm.detected:
        verdict = "FORGED (copy-move)"
    elif splice_prob is not None and splice_prob >= splice_threshold:
        verdict = "FORGED (splicing suspected)"
    else:
        verdict = "NO FORGERY DETECTED"

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    cv2.imwrite(str(out / f"{stem}_overlay.png"), cm.overlay)
    cv2.imwrite(str(out / f"{stem}_mask.png"), cm.mask)

    return {"verdict": verdict, "copy_move": cm, "splicing_prob": splice_prob,
            "overlay_path": str(out / f"{stem}_overlay.png")}
