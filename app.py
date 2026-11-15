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

st.set_page_config(page_title="MIRA: image tampering check", page_icon="🔎",
                   layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Public+Sans:wght@400;500;600&display=swap');
html, body, .stApp, .stApp p, .stApp label, .stApp button { font-family: 'Public Sans', sans-serif; }
.block-container { max-width: 760px; padding-top: 3.5rem; padding-bottom: 3rem; }
#MainMenu, footer { visibility: hidden; }

.mira-title { font-family: 'Newsreader', serif; font-size: 4.2rem; font-weight: 600;
              letter-spacing: -0.02em; line-height: 1; margin: 0; color: #14213D; }
.mira-sub { font-family: 'Newsreader', serif; font-size: 1.4rem; color: #3D4B66;
            margin: 0.5rem 0 1rem; }
.mira-intro { color: #4A5568; max-width: 54ch; margin: 0 0 1.8rem; line-height: 1.55; }

.stImage img { border-radius: 10px; }

.verdict { border-radius: 14px; padding: 1.7rem 1.9rem 1.5rem; margin: 1.6rem 0 1rem; }
.verdict-label { font-family: 'Newsreader', serif; font-size: 2.3rem; font-weight: 600;
                 line-height: 1.1; margin: 0 0 0.4rem; }
.verdict-note { margin: 0; font-size: 1rem; line-height: 1.5; }
.scale { margin-top: 1.4rem; font-size: 0; }
.scale i { display: inline-block; width: 32%; margin-right: 2%; height: 8px; border-radius: 4px;
           background: rgba(20,33,61,0.12); }
.scale i:last-child, .scale-names span:last-child { margin-right: 0; }
.scale-names { margin-top: 0.45rem; font-size: 0; }
.scale-names span { display: inline-block; width: 32%; margin-right: 2%; font-size: 0.8rem;
                    color: #5A6678; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="mira-title">MIRA</h1>', unsafe_allow_html=True)
st.markdown('<p class="mira-sub">Manipulated Image Recognition &amp; Analysis</p>',
            unsafe_allow_html=True)
st.markdown('<p class="mira-intro">Upload a photo and MIRA checks it for copied regions '
            'and for signs of splicing, then tells you how likely it is that the image '
            'has been tampered with.</p>', unsafe_allow_html=True)

if not MODEL.exists():
    st.warning("Splicing model not found (models/splicing.joblib). "
               "Only copy-move detection will run.")

STYLE = {  # level, background, text, bar
    "Likely authentic":     (1, "#E6F2EC", "#1D5A41", "#2F7D5B"),
    "Likely tampered":      (2, "#FBF0DC", "#7A4B0B", "#C98A1B"),
    "Most likely tampered": (3, "#F9E3E0", "#8C241B", "#C8372D"),
}


def verdict_html(label, note):
    level, bg, fg, bar = STYLE[label]
    segs = "".join(f'<i style="background:{bar}"></i>' if n <= level else "<i></i>"
                   for n in (1, 2, 3))
    return (f'<div class="verdict" style="background:{bg};color:{fg}">'
            f'<div class="verdict-label">{label}</div>'
            f'<p class="verdict-note">{note}</p>'
            f'<div class="scale">{segs}</div>'
            f'<div class="scale-names"><span>Authentic</span><span>Likely tampered</span>'
            f'<span>Most likely tampered</span></div></div>')


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
        note = "No copied regions were found, and the splicing check looks normal."
    elif res["detected_copy_move"]:
        note = "Parts of this image appear to be duplicated elsewhere in the same image."
    elif label == "Most likely tampered":
        note = "The splicing check strongly suggests content was added from another image."
    else:
        note = "The splicing check found some signs of tampering. This is a weaker signal."
    st.markdown(verdict_html(label, note), unsafe_allow_html=True)

    if res["detected_copy_move"]:
        with st.expander("Show duplicated regions"):
            st.image(cv2.cvtColor(res["overlay"], cv2.COLOR_BGR2RGB),
                     caption="The red tint marks regions that match each other. "
                             "Either one may be the original.",
                     use_container_width=True)

with st.expander("How it works"):
    st.write("**Copy-move check:** finds distinctive points in the image and looks for "
             "groups that match another part of the same image, which happens when a "
             "region has been copied and pasted.")
    st.write("**Splicing check:** a classifier trained on texture and noise patterns "
             "estimates whether parts of the image came from somewhere else. It is a "
             "weaker signal than the copy-move check.")

st.divider()
st.caption("A screening tool, not proof. It can miss retouching and AI-generated "
           "edits, and may flag images with repetitive patterns.")
