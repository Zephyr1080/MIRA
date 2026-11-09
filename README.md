# Image Forensics: Copy-Move & Splicing Detection

Course project for **AML 502 – Computer Vision**. Classical CV only (no deep learning).

- **Copy-move**: CLAHE → SIFT → self-matching → ratio test → **RANSAC affine verification** → forgery mask
- **Splicing**: LBP + GLCM + Gabor texture features on YCrCb → SVM / Random Forest
- **Evaluation**: pixel-level precision/recall/F1, image-level detection rate, false-positive rate, feature ablation

## Structure
```
image-forensics/
├── README.md
├── requirements.txt
├── run.py                  # CLI: analyze | train | eval
├── make_demo.py            # builds a textured forged/authentic demo + ground-truth mask
├── forensics/
│   ├── config.py           # all tunable parameters
│   ├── copy_move.py        # SIFT + RANSAC detector, mask, overlay
│   ├── splicing.py         # LBP/GLCM/Gabor features, SVM/RF training, prediction
│   ├── pipeline.py         # analyze(image) -> unified verdict
│   └── metrics.py          # precision / recall / F1, TPR / FPR
├── data/                   # put CASIA 2.0 here (Au/, Tp/) + demo images
├── models/                 # trained splicing model (.joblib)
└── results/                # overlays, masks, CSVs
```

## Usage
```bash
pip install -r requirements.txt
python make_demo.py                                   # textured demo images
python run.py analyze data/demo/forged.png            # copy-move works without a model
python run.py train --root data/CASIA2 --n 300        # needs CASIA2/Au and CASIA2/Tp
python run.py analyze some_image.jpg                  # now includes splicing probability
python run.py eval --forged data/forged_cm --gt data/gt --authentic data/Au --limit 100
```
Ground-truth masks are matched by name: `<image>_gt.png` or `<image>_mask.png`.

In Colab: upload/unzip the folder, `!pip install -r requirements.txt`, then call
`from forensics import analyze; analyze("img.jpg")` or run the CLI with `!python run.py ...`.

## Changes vs. the original script  (`[FIX]` / `[ENHANCEMENT]` / `[NEW]`)

| Tag | Where | What |
|---|---|---|
| [ENHANCEMENT] | copy_move.preprocess | CLAHE contrast enhancement before SIFT (Unit 1: histogram processing) |
| [ENHANCEMENT] | copy_move.self_match | Lower SIFT contrast threshold (0.02) -> more keypoints in smooth areas |
| [ENHANCEMENT] | copy_move.self_match | Exact BFMatcher instead of approximate FLANN |
| [FIX] | copy_move.self_match | Self-match removed by index, not assumed to be `m[0]` |
| [NEW] | copy_move.ransac_rounds | **RANSAC affine verification replaces DBSCAN**: matches must obey one geometric transform (Unit 3: homography, RANSAC, inliers/outliers) |
| [NEW] | copy_move.ransac_rounds | Iterative RANSAC finds multiple copied regions; rejects degenerate transforms |
| [NEW] | copy_move.build_mask | Binary forgery mask via convex hull + morphological closing + dilation (Unit 3: segmentation) |
| [NEW] | copy_move.detect_copy_move | Returns estimated affine transform (rotation / scale / translation) per region |
| [NEW] | copy_move.draw_overlay | Red-tinted region on top of the original match lines |
| [FIX] | (removed) | Unused `matplotlib` import; fragile `i*2` DBSCAN index bookkeeping |
| [FIX] | make_demo.py | Demo used a flat-colour square (no SIFT keypoints). Now a textured, rotated + scaled copy |
| [NEW] | splicing.py | LBP + GLCM features on YCrCb channels -> SVM and Random Forest |
| [ENHANCEMENT] | splicing._gabor_feats | Gabor filter bank features (Unit 2) |
| [NEW] | splicing.train | Train/test split, classification report, confusion matrix, best-model selection |
| [NEW] | splicing.train | Feature ablation: LBP vs GLCM vs LBP+GLCM vs +Gabor |
| [NEW] | pipeline.py | Unified `analyze()` returning a single verdict |
| [NEW] | metrics.py | Pixel P/R/F1, image-level TPR, false-positive rate, CSV export |
| [NEW] | config.py | All parameters in one dataclass |

## Syllabus mapping
| Topic | Syllabus unit | Used in |
|---|---|---|
| Histogram processing / enhancement | Unit 1 (Labs 2-3) | CLAHE |
| Texture, feature vectors, preprocessing | Unit 2 | LBP, GLCM, scaling, SVM/RF |
| Gabor filters | Unit 2 | splicing features |
| SIFT / DoG | Unit 2 (Labs 5-7) | keypoints |
| Affine / projective transforms | Unit 1, 3 | RANSAC transform model |
| RANSAC, inliers/outliers | Unit 3 (Labs 8-9) | match verification |
| Segmentation | Unit 3 | mask generation |

## Known limitations (put these in the report)
- Smooth or uniform regions give few SIFT keypoints, so copies there can be missed.
- Repetitive structures (windows, tiles) can still produce false positives.
- Heavy JPEG compression or blur lowers keypoint repeatability.
- The hull-of-keypoints mask under-covers patch edges.
- Splicing detection is texture-statistics based, so accuracy is limited versus deep-learning methods.

## Suggested future work
Harris/SURF comparison, image pyramids for strong scaling, region growing / Graph-Cut mask refinement, DWT features.
