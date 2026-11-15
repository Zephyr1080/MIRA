"""MIRA web app. Run from the repo root:  streamlit run app.py"""
import io
import tempfile
from pathlib import Path

import cv2
import streamlit as st
from PIL import Image

from forensics.scoring import assess

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "splicing.joblib"

st.set_page_config(page_title="MIRA: image tampering check", layout="centered")
st.title("Is this image authentic?")
st.write("Upload a photo and MIRA will check it for copied regions and splicing.")

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
    st.image(pil, use_container_width=True)

    with st.spinner("Analyzing..."):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "upload.png"
            pil.save(path)  # PNG, so odd formats never trouble OpenCV
            res = assess(str(path), str(MODEL), out_dir=tmp)

    label = res["label"]
    if label == "Likely authentic":
        st.success(label)
    elif label == "Likely tampered":
        st.warning(label)
    else:
        st.error(label)

    if res["detected_copy_move"]:
        with st.expander("Show duplicated regions"):
            st.image(cv2.cvtColor(res["overlay"], cv2.COLOR_BGR2RGB),
                     caption="Red tint: regions that match each other.",
                     use_container_width=True)

st.divider()
st.caption("A screening tool, not proof. It can miss retouching and AI-generated "
           "edits, and may flag images with repetitive patterns.")
