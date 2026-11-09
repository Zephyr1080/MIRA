"""[FIX] Original demo pasted a flat-colour square (no texture -> SIFT finds nothing).
This builds a textured image so the detector actually has something to match."""
from pathlib import Path

import cv2
import numpy as np


def make_demo(out_dir="data/demo", seed=0, rotate=15, scale=1.1):
    rng = np.random.default_rng(seed)
    base = np.zeros((480, 640, 3), np.float32)
    for sigma, amp in ((1, 60), (4, 90), (12, 80)):          # multi-scale noise = natural-ish texture
        n = rng.normal(0, 1, base.shape).astype(np.float32)
        base += amp * cv2.GaussianBlur(n, (0, 0), sigma)
    base = cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    for _ in range(25):                                       # random shapes for structure
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        cv2.circle(base, (int(rng.integers(0, 640)), int(rng.integers(0, 480))),
                   int(rng.integers(8, 35)), c, -1)
    authentic = base.copy()

    # copy a patch, rotate + scale it, paste elsewhere
    y0, y1, x0, x1 = 80, 200, 80, 200
    patch = base[y0:y1, x0:x1]
    M = cv2.getRotationMatrix2D((60, 60), rotate, scale)
    patch2 = cv2.warpAffine(patch, M, (120, 120), borderMode=cv2.BORDER_REFLECT)
    forged = base.copy()
    ty, tx = 280, 400
    forged[ty:ty + 120, tx:tx + 120] = patch2
    gt = np.zeros(base.shape[:2], np.uint8)
    gt[y0:y1, x0:x1] = 255
    gt[ty:ty + 120, tx:tx + 120] = 255

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "authentic.png"), authentic)
    cv2.imwrite(str(out / "forged.png"), forged)
    cv2.imwrite(str(out / "forged_gt.png"), gt)
    return out / "forged.png"


if __name__ == "__main__":
    print("Created", make_demo())
