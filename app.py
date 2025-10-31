import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import pandas as pd
import io
import os
import time
from typing import Tuple, Optional

st.set_page_config(page_title="DermalScan AI - Futuristic", layout="centered")

# ---- CSS: Hide sidebar, improve UI, readable tables ----
st.markdown("""
<style>
[data-testid="stSidebar"] { display: none !important; }
body, .main { background: linear-gradient(113deg,#191e29 0,#232e47 100%)!important; }
.app-main-heading {
    font-size:2.18em;
    font-weight:900;
    color:#051a31;
    padding:18px 0 18px 0;
    text-align:center;
    letter-spacing:1.3px;
    border-radius:18px;
    background: linear-gradient(90deg, #47e7ed 0%, #0877fa 100%);
    box-shadow: 0 6px 32px 0 rgba(8,119,250,.11);
    margin-bottom: 20px;
    margin-top: 18px;
}
.glass-card {
    background: rgba(255,255,255,0.13);
    border-radius: 30px;
    padding: 34px 22px 10px 24px;
    margin: 34px auto 26px auto;
    box-shadow:0 8px 42px #51fffb22, 0 1.8px 8px #1b328d33;
    backdrop-filter: blur(10px);
    border:1px solid #138eb777;
    max-width: 570px;
}
.neon-badge {
    font-size: 1.34em; font-weight: 900; border-radius:19px;
    margin:12px 0 17px 0; padding:9px 33px 8px 33px;
    display:inline-block; color:#fff; 
    background: linear-gradient(91deg,#0ffdc1 0%,#2124fd 100%);
    box-shadow: 0 1px 20px #62c2fff5, 0 2px 24px #57ffd8cc;
    letter-spacing:1.4px; filter: blur(.2px);
    border: 1.5px solid #00e9fa99;
    text-shadow: 0 0 .4em #aaf0ffcc,0 2px 18px #0007;
}
.age-neon {
    font-size: 1.26em; font-weight: 800; 
    border-radius:18px; margin-left:18px;
    padding:7px 23px 6px 23px; color:#fff;
    background: linear-gradient(91deg,#fee6fb 0%,#10d3f8 98%);
    box-shadow:0 0 8px #bfeaecbd;
    border: 2px solid #80c3ffaa;
    text-shadow:0 0 6px #fff,0 2px 20px #07e3de99;
}
.processing {
    font-size: 1.08em; color:#42e7f3;
    background: linear-gradient(92deg,#202641 0%,#183c48 90%);
    border-radius: 12px; padding:7px 17px;margin-top:13px;
    box-shadow:0 0 10px #00aaff44;
    font-weight:500; letter-spacing:.8px;display:inline-block
}
.res-card, .download-btn {
    background: rgba(255,255,255,0.04)!important;
    border-radius:18px!important;
    box-shadow: 0 2px 24px #bdebe709;
    padding: 15px 20px 8px 20px;
    margin: 16px 0 20px 0;
    border:1.6px solid #66aaff40;
}
.stDataFrame {background:#fafbff!important;}
th, td { color: #222!important; font-weight: 600!important; }
</style>
""", unsafe_allow_html=True)

# ---- MAIN HEADING EXACTLY AS IN YOUR IMAGE ----
st.markdown("""
<div class="app-main-heading">
DermalScan:AI_Facial Skin Aging Detection App
</div>
""", unsafe_allow_html=True)

FEATURE_CLASSES = ["clear_face", "dark_spots", "puffy_eyes", "wrinkles"]
AGE_CLASSES = ["0-9", "10-29", "30-59", "60+"]
AGE_MIDPOINTS = {"0-9": 5, "10-29": 20, "30-59": 45, "60+": 70}

def load_model(path: str):
    return tf.keras.models.load_model(path)

model = None
model_load_error = None
MODEL_PATH = "best_model.h5"
for candidate in ["best_model.h5","age_estimator.h5","model.h5"]:
    if os.path.exists(candidate):
        try:
            model = load_model(candidate)
            model_name = candidate
            break
        except Exception as e:
            model_load_error = str(e)
            model = None
if model is None:
    st.error(f"Model could not be loaded. Error: {model_load_error}")
    st.stop()

def preprocess(img: Image.Image, mode="0-1") -> np.ndarray:
    img = img.convert("RGB").resize((224,224))
    arr = np.array(img).astype("float32")/255.0
    return np.expand_dims(arr, axis=0)

def to_probs(arr: np.ndarray) -> np.ndarray:
    arr = np.array(arr, dtype=float).reshape(-1)
    s = np.sum(arr)
    if s > 0 and abs(s - 1.0) <= 1e-3:
        return arr
    ex = np.exp(arr - np.max(arr))
    return ex / np.sum(ex)

def parse_model_output(preds) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float], str]:
    feature_probs = None
    age_probs = None
    age_reg = None
    if isinstance(preds, (list, tuple)):
        for item in preds:
            arr = np.array(item).reshape(-1)
            if arr.shape[0] == len(FEATURE_CLASSES) and feature_probs is None:
                feature_probs = to_probs(arr)
            elif arr.shape[0] == len(AGE_CLASSES) and age_probs is None:
                age_probs = to_probs(arr)
            elif arr.shape[0] == 1 and age_reg is None:
                age_reg = float(arr[0])
        return feature_probs, age_probs, age_reg, "multi-output parsed"
    arr = np.array(preds).reshape(-1)
    N = arr.shape[0]
    if N == len(FEATURE_CLASSES) + len(AGE_CLASSES):
        f = arr[:len(FEATURE_CLASSES)]
        a = arr[len(FEATURE_CLASSES):]
        return to_probs(f), to_probs(a), None, "concatenated F+A parsed"
    if N == len(FEATURE_CLASSES):
        return to_probs(arr), None, None, "only features parsed"
    if N == len(AGE_CLASSES):
        return None, to_probs(arr), None, "only age categories parsed"
    if N == 1:
        return None, None, float(arr[0]), "regression scalar parsed"
    if N > len(FEATURE_CLASSES):
        head = arr[:len(FEATURE_CLASSES)]
        tail = arr[len(FEATURE_CLASSES):]
        if tail.shape[0] == len(AGE_CLASSES):
            return to_probs(head), to_probs(tail), None, "heuristic head-tail parsed"
    return (to_probs(arr) if N == len(FEATURE_CLASSES) else None,
            to_probs(arr) if N == len(AGE_CLASSES) else None,
            None,
            "fallback parsed")

def age_from_probs_or_reg(age_probs: Optional[np.ndarray], age_reg: Optional[float]) -> Tuple[Optional[int], str]:
    if age_reg is not None:
        a = int(round(float(age_reg)))
        a = max(0, min(120, a))
        return a, "regression"
    if age_probs is not None:
        midpoints = np.array([AGE_MIDPOINTS[c] for c in AGE_CLASSES], dtype=float)
        val = float(np.sum(age_probs * midpoints))
        val = int(round(val))
        val = max(0, min(120, val))
        return val, "categorical_weighted"
    return None, "none"

def draw_label_on_image(img: Image.Image, box: Tuple[int,int,int,int], label: str, font: ImageFont.ImageFont):
    draw = ImageDraw.Draw(img)
    draw.rectangle(box, outline="#10e9b6", width=3)
    bbox = draw.textbbox((0,0), label, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = box[0] + 12
    ty = box[1] - th - 18
    if ty < 0:
        ty = box[3] + 12
    grad_rect_col = (29, 44, 108, 230)
    draw.rectangle([(tx-11, ty-7), (tx+tw+24, ty+th+8)], fill=grad_rect_col)
    draw.text((tx, ty), label, font=font, fill="#09daea")
    return img

uploaded = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility='collapsed')
if uploaded is None:
    st.markdown("<div class='glass-card'><b style='font-size:1.4em;'>Upload a face image to start futuristic analysis!</b></div>", unsafe_allow_html=True)
    st.stop()

image = Image.open(uploaded).convert("RGB")
st.markdown('<div class="glass-card" style="padding:19px 20px;">', unsafe_allow_html=True)
st.image(image, caption="👤 Uploaded Image", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

with st.spinner("⚡ Running AI inference..."):
    arr = preprocess(image)
    start = time.perf_counter()
    raw_preds = model.predict(arr)
    latency = round(time.perf_counter() - start, 3)

feature_probs, age_probs, age_reg, parse_method = parse_model_output(raw_preds)
if feature_probs is None and age_probs is None and age_reg is None:
    st.error("Model output could not be interpreted.")
    st.stop()

feat_label = None
feat_conf = None
if feature_probs is not None:
    idx = int(np.argmax(feature_probs))
    feat_label = FEATURE_CLASSES[idx]
    feat_conf = float(feature_probs[idx] * 100.0)

age_value, age_method = age_from_probs_or_reg(age_probs, age_reg)
if age_value is None and feature_probs is not None:
    FEATURE_TO_AGE = {"clear_face": 25, "dark_spots": 45, "puffy_eyes": 35, "wrinkles": 55}
    vals = np.array([FEATURE_TO_AGE[f] for f in FEATURE_CLASSES], dtype=float)
    age_value = int(round(np.sum(feature_probs * vals)))
    age_method = "fallback_from_features"

st.markdown(f'<div class="neon-badge">Feature: {feat_label} <span style="color:#fefea9">({feat_conf:.1f}%)</span></div>', unsafe_allow_html=True)
st.markdown(f'<div class="age-neon">Predicted Age: {age_value} years</div>', unsafe_allow_html=True)
st.markdown(f'<div class="processing">Processing Time: <b>{latency} seconds</b></div>', unsafe_allow_html=True)

annotated = image.copy()
draw = ImageDraw.Draw(annotated)
w, h = annotated.size
box = (int(w*0.25), int(h*0.25), int(w*0.75), int(h*0.75))
label_text = f"{feat_label or 'N/A'} ({feat_conf:.1f}%) | Age: {age_value if age_value is not None else 'N/A'}"
try:
    font = ImageFont.truetype("arial.ttf", 24)
except Exception:
    font = ImageFont.load_default()
annotated = draw_label_on_image(annotated, box, label_text, font)
max_display_width = 540
display_img = annotated.copy()
if display_img.width > max_display_width:
    ratio = max_display_width / display_img.width
    display_img = display_img.resize((max_display_width, int(display_img.height * ratio)))
st.markdown('<div class="glass-card" style="padding:18px 6px;">', unsafe_allow_html=True)
st.image(display_img, caption="🟦 Advanced AI Annotated Result", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

if feature_probs is not None:
    feat_df = pd.DataFrame({
        "Feature": FEATURE_CLASSES,
        "Confidence (%)": [f"{p*100:.2f}" for p in feature_probs]
    })
    st.markdown('<div class="res-card" style="margin-top:11px"><span style="font-weight:600;color:#26dbe7;">Feature Probabilities</span></div>', unsafe_allow_html=True)
    st.dataframe(
        feat_df.style.set_properties(
            **{
                'background-color': '#fafbff',
                'color': '#222',
                'border-radius': '10px',
                'font-size': '1em',
                'font-weight': '500'
            }
        ),
        use_container_width=True
    )

if age_probs is not None:
    age_df = pd.DataFrame({
        "Age bin": AGE_CLASSES,
        "Confidence (%)": [f"{p*100:.2f}" for p in age_probs]
    })
    st.markdown('<div class="res-card" style="margin-top:11px"><span style="font-weight:600;color:#102be7;">Age Probabilities</span></div>', unsafe_allow_html=True)
    st.dataframe(
        age_df.style.set_properties(
            **{
                'background-color': '#f6fcff',
                'color': '#222',
                'border-radius': '10px',
                'font-size': '1em',
                'font-weight': '500'
            }
        ),
        use_container_width=True
    )

buf = io.BytesIO()
annotated.save(buf, format="PNG")
st.markdown('<div class="download-btn" style="text-align:center;">', unsafe_allow_html=True)
st.download_button("Download annotated image", buf.getvalue(), "annotated.png", "image/png")
st.markdown('</div>', unsafe_allow_html=True)
