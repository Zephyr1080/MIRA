"""MIRA web app. Run from the repo root:  streamlit run app.py"""
import io
import tempfile
from pathlib import Path

import cv2
import streamlit as st
from PIL import Image

from forensics.config import CopyMoveConfig
from forensics.scoring import assess

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "splicing.joblib"

st.set_page_config(page_title="MIRA: image tampering check", layout="centered")
st.title("Is this image authentic?")
st.write("Upload a photo. MIRA looks for copied regions and splicing, "
         "then gives a tampering likelihood.")

with st.sidebar:
    st.subheader("Verdict thresholds")
    low = st.slider("Below this: likely authentic", 0.05, 0.50, 0.35, 0.05)
    high = st.slider("Above this: likely tampered", 0.50, 0.95, 0.65, 0.05)
    with st.expander("Copy-move detector settings"):
        ratio = st.slider("Match ratio (lower is stricter)", 0.3, 0.9, 0.6, 0.05)
        ransac = st.slider("RANSAC tolerance (px)", 1.0, 10.0, 3.0, 0.5)
        min_inl = st.slider("Minimum inliers", 4, 40, 8, 1)
    st.caption("The splicing score comes from the trained classifier "
               "(about 69% accuracy in testing), so treat it as a weak signal. "
               "The copy-move score is a formula based on RANSAC inliers.")

if not MODEL.exists():
    st.warning("Splicing model not found (models/splicing.joblib). "
               "Only copy-move detection will run.")

file = st.file_uploader("Choose an image",
                        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"])

if file is not None:
    try:
        pil = Image.open(io.BytesIO(file.getvalue())).convert("RGB")
    except Exception:
        st.error("That file could not be read as an image.")
        st.stop()
    st.image(pil, caption="Uploaded image", use_container_width=True)

    cfg = CopyMoveConfig(ratio=ratio, ransac_thresh=ransac, min_inliers=min_inl)
    with st.spinner("Analyzing..."):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "upload.png"
            pil.save(path)  # PNG, so odd formats never trouble OpenCV
            res = assess(str(path), str(MODEL), cfg, out_dir=tmp, low=low, high=high)

    p = res["likelihood"]
    st.subheader(res["label"])
    st.metric("Tampering likelihood", f"{p * 100:.0f}%")
    st.progress(min(max(p, 0.0), 1.0))

    c1, c2 = st.columns(2)
    c1.metric("Copy-move score", f"{res['copy_move_score'] * 100:.0f}%",
              help=f"{res['n_inliers']} inlier matches ({res['n_matches']} matches, "
                   f"{res['n_keypoints']} keypoints)")
    sp = res["splicing_prob"]
    c2.metric("Splicing score", "n/a" if sp is None else f"{sp * 100:.0f}%")

    if res["detected_copy_move"]:
        with st.expander("Show duplicated regions", expanded=True):
            st.image(cv2.cvtColor(res["overlay"], cv2.COLOR_BGR2RGB),
                     caption="Red tint: regions that match each other. "
                             "Blue lines: matched points.",
                     use_container_width=True)
            st.image(res["mask"],
                     caption="Mask of the source and the copy "
                             "(either one may be the original).",
                     use_container_width=True)

st.divider()
st.caption("A screening tool, not proof. It can miss retouching and AI-generated "
           "edits, and may flag images with repetitive patterns.")
