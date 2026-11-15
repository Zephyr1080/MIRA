"""[NEW] Splicing detection with classical texture features + SVM / Random Forest (no deep learning).

Features per channel of YCrCb (chroma channels expose splice boundaries better than luma alone):
    LBP histogram (10)  +  GLCM statistics (10)  +  Gabor responses (8, optional)
"""
from functools import lru_cache
from pathlib import Path

import cv2
import joblib
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

LBP_P, LBP_R = 8, 1
N_LBP, N_GLCM, N_GABOR, N_RES = LBP_P + 2, 10, 8, 27
DEFAULT_MAX_SIDE = 1024      # was 512; set to None to never resize
IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _lbp_hist(ch):
    lbp = local_binary_pattern(ch, LBP_P, LBP_R, method="uniform")
    h, _ = np.histogram(lbp.ravel(), bins=N_LBP, range=(0, N_LBP), density=True)
    return h


def _glcm_feats(ch, levels=32):
    q = (ch // (256 // levels)).astype(np.uint8)
    g = graycomatrix(q, distances=[1, 2], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                     levels=levels, symmetric=True, normed=True)
    f = []
    for p in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation"):
        v = graycoprops(g, p)
        f += [v.mean(), v.std()]
    return np.array(f)


def _gabor_feats(ch):
    """[ENHANCEMENT] Gabor filter bank (Unit 2): mean/std response at 4 orientations."""
    f = []
    x = ch.astype(np.float32)
    for theta in np.arange(0, np.pi, np.pi / 4):
        k = cv2.getGaborKernel((21, 21), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        r = cv2.filter2D(x, cv2.CV_32F, k)
        f += [r.mean(), r.std()]
    return np.array(f)


def _residual_feats(ch):
    """[NEW] Noise-residual features. A pasted region usually has different noise and
    compression traces than its surroundings, which texture features alone miss.
    Residual = pixel - 3x3 median; quantised to 5 levels; co-occurrence of neighbours."""
    ch = np.ascontiguousarray(ch)
    res = ch.astype(np.float32) - cv2.medianBlur(ch, 3).astype(np.float32)
    q = (np.clip(np.round(res), -2, 2) + 2).astype(np.int64)          # 0..4
    h = (q[:, :-1] * 5 + q[:, 1:]).ravel()                            # horizontal pairs
    v = (q[:-1, :] * 5 + q[1:, :]).ravel()                            # vertical pairs
    co = (np.bincount(h, minlength=25) / h.size + np.bincount(v, minlength=25) / v.size) / 2
    return np.concatenate([co, [res.std(), np.abs(res).mean()]])


def extract_features(img_bgr, use_gabor=True, max_side=DEFAULT_MAX_SIDE, use_residual=True):
    h, w = img_bgr.shape[:2]
    if max_side and max(h, w) > max_side:                       # speed; keeps aspect ratio
        s = max_side / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    feats = []
    for c in range(3):
        ch = ycrcb[..., c]
        parts = [_lbp_hist(ch), _glcm_feats(ch)]
        if use_gabor:
            parts.append(_gabor_feats(ch))
        if use_residual:
            parts.append(_residual_feats(ch))
        feats.append(np.concatenate(parts))
    return np.concatenate(feats)


def feature_groups(use_gabor=True, use_residual=True):
    """Column indices per feature family (used for the ablation study)."""
    per = N_LBP + N_GLCM + (N_GABOR if use_gabor else 0) + (N_RES if use_residual else 0)
    g = {"lbp": [], "glcm": [], "gabor": [], "residual": []}
    for c in range(3):
        o = c * per
        g["lbp"] += range(o, o + N_LBP)
        g["glcm"] += range(o + N_LBP, o + N_LBP + N_GLCM)
        k = o + N_LBP + N_GLCM
        if use_gabor:
            g["gabor"] += range(k, k + N_GABOR)
            k += N_GABOR
        if use_residual:
            g["residual"] += range(k, k + N_RES)
    return g


def build_dataset(root, n_per_class=300, use_gabor=True, seed=42, use_residual=True):
    """root/Au = authentic (label 0), root/Tp = tampered (label 1)  (CASIA 2.0 layout)."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for label, folder in ((0, "Au"), (1, "Tp")):
        files = sorted(p for p in (Path(root) / folder).iterdir() if p.suffix.lower() in IMG_EXT)
        files = [files[i] for i in rng.permutation(len(files))[:n_per_class]]
        for i, p in enumerate(files):
            img = cv2.imread(str(p))
            if img is None:
                continue
            X.append(extract_features(img, use_gabor, use_residual=use_residual))
            y.append(label)
            if (i + 1) % 50 == 0:
                print(f"  {folder}: {i + 1}/{len(files)}")
    return np.array(X), np.array(y)


def train(root, n_per_class=300, use_gabor=True, out="models/splicing.joblib", seed=42, use_residual=True):
    X, y = build_dataset(root, n_per_class, use_gabor, seed, use_residual)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

    models = {
        "svm": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale",
                                                   probability=True, class_weight="balanced",
                                                   random_state=seed)),
        "rf": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     random_state=seed, n_jobs=-1),
    }
    best = None
    for name, m in models.items():
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        f1 = f1_score(yte, pred)
        print(f"\n=== {name.upper()}  F1={f1:.3f} ===")
        print(classification_report(yte, pred, target_names=["authentic", "tampered"]))
        print(confusion_matrix(yte, pred))
        # [NEW] 5-fold cross-validation: a steadier number than one 80/20 split
        cv = cross_val_score(m, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                             scoring="f1", n_jobs=1)
        print(f"5-fold CV F1: {cv.mean():.3f} +/- {cv.std():.3f}")
        score = cv.mean()
        if best is None or score > best[0]:
            best = (score, name, m)

    # [NEW] Ablation: which feature family helps? (good material for the report)
    print("\n=== Ablation (Random Forest, F1) ===")
    groups = feature_groups(use_gabor, use_residual)
    combos = {"LBP": ["lbp"], "GLCM": ["glcm"], "LBP+GLCM": ["lbp", "glcm"]}
    if use_gabor:
        combos["LBP+GLCM+Gabor"] = ["lbp", "glcm", "gabor"]
    if use_residual:
        combos["Residual only"] = ["residual"]
        combos["LBP+GLCM+Residual"] = ["lbp", "glcm", "residual"]
        if use_gabor:
            combos["All features"] = ["lbp", "glcm", "gabor", "residual"]
    for label, fams in combos.items():
        cols = sorted(c for f in fams for c in groups[f])
        rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                    random_state=seed, n_jobs=-1).fit(Xtr[:, cols], ytr)
        print(f"{label:16s} {f1_score(yte, rf.predict(Xte[:, cols])):.3f}")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    best[2].fit(X, y)   # final model uses all data (scores above came from held-out data)
    joblib.dump({"model": best[2], "name": best[1], "use_gabor": use_gabor,
                 "use_residual": use_residual, "max_side": DEFAULT_MAX_SIDE}, out)
    print(f"\nSaved best model ({best[1]}) -> {out}")


@lru_cache(maxsize=2)
def _load(model_path):
    return joblib.load(model_path)


def predict_splicing(img_bgr, model_path):
    """Returns probability that the image is spliced."""
    bundle = _load(str(model_path))
    x = extract_features(img_bgr, bundle["use_gabor"],
                         max_side=bundle.get("max_side", 512),
                         use_residual=bundle.get("use_residual", False)).reshape(1, -1)
    return float(bundle["model"].predict_proba(x)[0, 1])
