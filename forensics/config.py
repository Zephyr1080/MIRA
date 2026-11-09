"""Central configuration. [NEW] All tunable parameters live here instead of being hard-coded."""
from dataclasses import dataclass


@dataclass
class CopyMoveConfig:
    # --- Pre-processing (Unit 1: Histogram Processing / Image Enhancement) ---
    use_clahe: bool = True            # [ENHANCEMENT] CLAHE before SIFT -> more keypoints in flat regions
    clahe_clip: float = 2.0
    clahe_tile: int = 8

    # --- SIFT + matching (Unit 2: SIFT, DoG, Scale-space) ---
    sift_contrast: float = 0.02       # [ENHANCEMENT] lower than OpenCV default (0.04) -> more keypoints
    ratio: float = 0.6                # Lowe's ratio test (same as original match_threshold)
    min_spatial_dist: int = 20        # ignore matches between neighbouring keypoints (same as original)

    # --- Geometric verification (Unit 3: Homography, RANSAC, inliers/outliers) ---
    ransac_thresh: float = 3.0        # [NEW] reprojection error (px) for an inlier
    min_inliers: int = 8              # [NEW] fewer inliers than this -> not a forgery
    max_rounds: int = 4               # [NEW] iterative RANSAC -> finds several copied regions

    # --- Mask generation (Unit 3: Segmentation) ---
    morph_kernel: int = 15            # [NEW] closing kernel to merge hulls into a solid region
    dilate_px: int = 7                # [NEW] hull of keypoints under-covers the patch; grow it a little
