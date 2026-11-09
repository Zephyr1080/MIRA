"""Copy-move forgery detection.

Pipeline:  CLAHE -> SIFT -> self-matching -> ratio test -> spatial filter
           -> iterative RANSAC affine verification -> forgery mask.

Original code used DBSCAN to filter matches. Changes are marked in brackets:
  [FIX]          bug / weakness in the original
  [ENHANCEMENT]  existing step improved
  [NEW]          step that did not exist before
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import CopyMoveConfig


@dataclass
class CopyMoveResult:
    detected: bool
    mask: np.ndarray                  # [NEW] binary forgery mask (0/255), original only drew lines
    overlay: np.ndarray               # visualisation
    n_keypoints: int
    n_matches: int                    # matches surviving ratio test + spatial filter
    n_inliers: int                    # [NEW] matches surviving RANSAC
    transforms: list = field(default_factory=list)  # [NEW] 2x3 affine matrix per copied region


def preprocess(img_bgr, cfg):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if cfg.use_clahe:  # [ENHANCEMENT] contrast enhancement so SIFT finds keypoints in smooth areas
        clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(cfg.clahe_tile,) * 2)
        gray = clahe.apply(gray)
    return gray


def self_match(gray, cfg):
    """SIFT descriptors matched against themselves. Returns (keypoints, src_pts, dst_pts)."""
    sift = cv2.SIFT_create(contrastThreshold=cfg.sift_contrast)
    kps, desc = sift.detectAndCompute(gray, None)
    empty = np.empty((0, 2), np.float32)
    if desc is None or len(kps) < 10:
        return kps, empty, empty

    # [ENHANCEMENT] exact brute-force matcher: FLANN is approximate, which can miss the true copy
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc, desc, k=3)

    src, dst = [], []
    for m in matches:
        # [FIX] original assumed m[0] is always the self-match; drop it explicitly by index instead
        cand = [x for x in m if x.trainIdx != x.queryIdx]
        if len(cand) < 2:
            continue
        best, second = cand[0], cand[1]
        p1, p2 = kps[best.queryIdx].pt, kps[best.trainIdx].pt
        if np.hypot(p1[0] - p2[0], p1[1] - p2[1]) <= cfg.min_spatial_dist:
            continue
        if best.distance < cfg.ratio * second.distance:  # Lowe's ratio test
            src.append(p1)
            dst.append(p2)
    return kps, np.float32(src).reshape(-1, 2), np.float32(dst).reshape(-1, 2)


def ransac_rounds(src, dst, cfg):
    """[NEW] Replaces DBSCAN. Keeps only matches consistent with ONE affine transform.
    Repeats on the leftovers to find additional copied regions (iterative RANSAC)."""
    groups = []
    s, d = src.copy(), dst.copy()
    for _ in range(cfg.max_rounds):
        if len(s) < cfg.min_inliers:
            break
        M, inl = cv2.estimateAffine2D(
            s, d, method=cv2.RANSAC, ransacReprojThreshold=cfg.ransac_thresh,
            maxIters=5000, confidence=0.99)
        if M is None or inl is None:
            break
        inl = inl.ravel().astype(bool)
        if inl.sum() < cfg.min_inliers:
            break
        if abs(np.linalg.det(M[:, :2])) > 0.2:   # reject degenerate (collapsed) transforms
            groups.append((M, s[inl], d[inl]))
        s, d = s[~inl], d[~inl]
    return groups


def build_mask(shape, groups, cfg):
    """[NEW] Convex hull of inlier points (source + copy) -> morphological closing -> dilation."""
    mask = np.zeros(shape[:2], np.uint8)
    for _, s, d in groups:
        for pts in (s, d):
            p = np.unique(np.round(pts).astype(np.int32), axis=0)
            if len(p) >= 3:
                cv2.fillConvexPoly(mask, cv2.convexHull(p), 255)
    if mask.any():
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_kernel,) * 2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cfg.dilate_px + 1,) * 2)
        mask = cv2.dilate(mask, k2)
    return mask


def draw_overlay(img, mask, groups):
    out = img.copy()
    tint = np.zeros_like(out)
    tint[..., 2] = 255                                   # [NEW] red tint over detected region
    region = mask > 0
    out[region] = cv2.addWeighted(out, 0.6, tint, 0.4, 0)[region]
    for _, s, d in groups:                               # same visual style as original
        for p1, p2 in zip(s, d):
            a, b = tuple(map(int, p1)), tuple(map(int, p2))
            cv2.line(out, a, b, (255, 0, 0), 1)
            cv2.circle(out, a, 3, (0, 0, 255), -1)
            cv2.circle(out, b, 3, (0, 255, 0), -1)
    return out


def detect_copy_move(img_bgr, cfg=None):
    cfg = cfg or CopyMoveConfig()
    gray = preprocess(img_bgr, cfg)
    kps, src, dst = self_match(gray, cfg)
    groups = ransac_rounds(src, dst, cfg)
    mask = build_mask(img_bgr.shape, groups, cfg)
    return CopyMoveResult(
        detected=len(groups) > 0,
        mask=mask,
        overlay=draw_overlay(img_bgr, mask, groups),
        n_keypoints=len(kps),
        n_matches=len(src),
        n_inliers=sum(len(g[1]) for g in groups),
        transforms=[g[0] for g in groups],
    )
