# MIRA: Manipulated Image Recognition & Analysis

Detects copy-move forgeries and splicing in images using classical computer vision and machine learning (no deep learning). Includes a small web app: upload an image and get a verdict of **Likely authentic**, **Likely tampered**, or **Most likely tampered**.


## How it works

**Copy-move detection** (a region duplicated within the same image)

1. CLAHE contrast enhancement
2. SIFT keypoints and descriptors
3. Self-matching with Lowe's ratio test
4. Iterative RANSAC with an affine transform model, so only geometrically consistent matches survive
5. Forgery mask from the convex hull of inliers, cleaned with morphology

This also reports the transform (rotation, scale, translation) of each copied region. Because a copy is identical to its source, both regions are marked and the detector cannot tell which one is the original.

**Splicing detection** (content pasted in from another image)

- Features on each YCrCb channel: LBP histograms, GLCM statistics, Gabor responses, and noise-residual co-occurrence features
- Classifier: SVM or Random Forest, whichever scores better in 5-fold cross-validation
- Output: an image-level probability. It does not localize the splice.

**Verdict**

The two detectors are combined into one score:

| Result | When |
|---|---|
| Most likely tampered | A copy-move forgery is detected, or the splicing probability is 0.75 or higher |
| Likely tampered | The splicing probability is between 0.5 and 0.75 |
| Likely authentic | Everything else |

## Project structure
=======
- **Copy-move**: CLAHE → SIFT → self-matching → ratio test → **RANSAC affine verification** → forgery mask
- **Splicing**: LBP + GLCM + Gabor texture features on YCrCb → SVM / Random Forest
- **Evaluation**: pixel-level precision/recall/F1, image-level detection rate, false-positive rate, feature ablation


```
MIRA/
├── app.py                  # Streamlit web app
├── run.py                  # CLI: analyze | train | eval
├── make_demo.py            # builds a textured forged/authentic demo + ground-truth mask
├── requirements.txt
├── forensics/
│   ├── config.py           # copy-move parameters
│   ├── copy_move.py        # SIFT + RANSAC detector, mask, overlay
│   ├── splicing.py         # features, training, prediction
│   ├── pipeline.py         # analyze(image) -> verdict
│   ├── scoring.py          # combines both detectors into one score and label
│   └── metrics.py          # precision / recall / F1, TPR / FPR
├── models/
│   └── splicing.joblib     # trained splicing model
├── data/
│   └── demo/               # demo images
└── results/                # overlays and masks written by run.py
```

## Setup

Python 3.10 to 3.12 is recommended.

```
pip install -r requirements.txt
```

## Usage

**Web app**

```
streamlit run app.py
```

**Command line**

```
python make_demo.py                                    # create demo images
python run.py analyze data/demo/forged.png             # copy-move works without a model
python run.py train --root data/CASIA2 --n 500         # needs Au/ and Tp/ folders
python run.py analyze some_image.jpg                   # now includes splicing probability
python run.py eval --forged data/forged_cm --gt data/gt --authentic data/Au --limit 100
```

Ground-truth masks are matched by name: `<image>_gt.png` or `<image>_mask.png`.

`python run.py train` takes `--no-gabor` and `--no-residual` to leave out a feature family.

## Results

Splicing classifier, trained on 500 images per class from a CASIA 2.0 derived dataset. Random Forest F1 by feature set on a held-out 20% split (200 images):

| Features | F1 |
|---|---|
| LBP | 0.670 |
| GLCM | 0.608 |
| LBP + GLCM | 0.715 |
| LBP + GLCM + Gabor | 0.701 |
| Residual only | 0.747 |
| LBP + GLCM + Residual | 0.756 |
| All features | 0.758 |

5-fold cross-validation F1 of the selected model: _fill in from your training output_

Noise-residual features were the largest single improvement over texture features alone. With a 200-image test set, differences of a few points are within noise.

Copy-move detection was checked on the synthetic demo and on individual real examples. Add your `eval` results here if you run them on a labeled set.

## Limitations

- Smooth or uniform regions give few SIFT keypoints, so copies there can be missed.
- Repetitive structures (windows, tiles) can cause false positives.
- Heavy JPEG compression or blur lowers keypoint repeatability.
- The keypoint-hull mask under-covers the edges of copied regions.
- Copy-move marks both the source and the copy.
- Splicing detection gives only an image-level estimate, and its accuracy is far below deep-learning methods.
- The splicing model was trained on an amplified, re-compressed copy of CASIA 2.0. Authentic and tampered images came from different JPEG quality levels, so part of what it learned may be compression differences. Real-world accuracy may be lower.
- Retouching and AI-generated images are not covered.

This is a screening tool, not proof of tampering.

## Possible future work

Patch-based features, the original CASIA 2.0 images, Harris/SURF comparison, image pyramids for strong scaling, region growing or Graph-Cut mask refinement, DWT features.
