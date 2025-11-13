# dermalscan_futuristic_fast.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import pandas as pd
import io
import os
import time
import json
from datetime import datetime

# Optional: attempt to import model-specific preprocessors
try:
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
except Exception:
    mobilenet_preprocess = None

try:
    from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
except Exception:
    efficientnet_preprocess = None

# Slight performance policy (optional)
try:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")
except Exception:
    pass  # ignore if not supported

# ---------------- Streamlit page config ----------------
st.set_page_config(page_title="DermalScan AI - Futuristic", layout="wide", page_icon="🧠")

# ---------------- CSS (improved contrast for header & table) ----------------
st.markdown("""
<style>
[data-testid="stSidebar"] { display: none !important; }
body, .main { background: linear-gradient(113deg,#10141f 0,#1c243a 100%)!important; color: #eaf6ff; }
.app-main-heading {
    font-size:2.2em; font-weight:900; color:#ffffff;
    padding:18px 0; text-align:center; letter-spacing:1.3px;
    border-radius:18px;
    background: linear-gradient(90deg, #47e7ed 0%, #0877fa 100%);
    box-shadow: 0 6px 32px 0 rgba(8,119,250,.15);
    margin-bottom: 20px; margin-top: 18px;
    text-shadow: 0 1px 6px rgba(0,0,0,0.45);
}
.glass-card {
    background: rgba(255,255,255,0.06);
    border-radius: 20px;
    padding: 18px 18px 12px 18px;
    margin: 14px auto 18px auto;
    box-shadow:0 8px 42px #51fffb22, 0 1.8px 8px #1b328d33;
    backdrop-filter: blur(8px);
    border:1px solid #26c9f322;
    max-width: 720px;
}
table {
    border-collapse: collapse;
    width: 100%;
    border: 2px solid #00e6ff !important;
    border-radius: 12px;
    overflow: hidden;
}
thead tr {
    background: linear-gradient(90deg, #00f0ff 0%, #0072ff 100%);
    color: #ffffff !important;
    font-weight: 800;
    font-size: 1.1em;
}
tbody tr {
    background-color: rgba(255,255,255,0.03);
    color: #f0f8ff !important;
    font-weight: 600;
}
tbody tr:nth-child(even) {
    background-color: rgba(255,255,255,0.06);
}
tbody tr:hover {
    background-color: rgba(0,255,255,0.16);
    color: #000 !important;
    font-weight: 700;
}
.stDataFrame {
    background: rgba(255,255,255,0.02);
    border-radius: 12px;
    color: #ffffff;
}
th, td { text-align:left; padding:8px; color: #eaf6ff; }
.download-button {
    color: #002b3a;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown("""
<div class="app-main-heading">
DermalScan:AI_Facial Skin Aging Detection App
</div>
""", unsafe_allow_html=True)

# ---------------- Constants ----------------
MODEL_PATH = "best_model.h5"
FALLBACK_MODELS = ["model.h5", "age_estimator.h5"]
MODEL_CANDIDATES = [MODEL_PATH] + FALLBACK_MODELS

CLASSES = ["clear_face", "dark_spots", "puffy_eyes", "wrinkles"]
IMG_SIZE = 224
LOG_DIR = "logs"
MAX_IMAGE_SIZE = 400
CONFIDENCE_THRESHOLD = 0.40
os.makedirs(LOG_DIR, exist_ok=True)

AGE_MAP = {
    "clear_face": {"range": "15-25", "desc": "Youthful skin"},
    "puffy_eyes": {"range": "25-40", "desc": "Early aging signs"},
    "dark_spots": {"range": "35-55", "desc": "Sun damage/aging"},
    "wrinkles": {"range": "50-70+", "desc": "Advanced aging"}
}

# ---------------- Utility functions ----------------
def resize_image_cv(img, max_size=MAX_IMAGE_SIZE):
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR), scale
    return img, 1.0

def pad_to_square(img, fill=(0, 0, 0)):
    """Pad rectangular image to square (centered)."""
    h, w = img.shape[:2]
    if h == w:
        return img
    size = max(h, w)
    top = (size - h) // 2
    left = (size - w) // 2
    padded = np.zeros((size, size, 3), dtype=img.dtype)
    padded[:] = fill
    padded[top:top+h, left:left+w] = img
    return padded

def preprocess_for_model(img_bgr, model_info=None):
    """
    Convert BGR->RGB, pad to square externally, resize to IMG_SIZE,
    apply model-specific preprocess if available, else scale 0..1.
    """
    # convert BGR->RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # ensure square before resizing (call pad_to_square earlier if desired)
    resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype("float32")
    arr = np.expand_dims(arr, axis=0)  # shape (1, H, W, 3)

    # model-aware preprocessing
    if model_info and "path" in model_info:
        name = model_info["path"].lower()
        if "mobilenet" in name and mobilenet_preprocess is not None:
            arr = mobilenet_preprocess(arr)
        elif "efficientnet" in name and efficientnet_preprocess is not None:
            arr = efficientnet_preprocess(arr)
        else:
            arr = arr / 255.0
    else:
        arr = arr / 255.0

    return arr.astype(np.float32)

def estimate_age(label, confidence):
    info = AGE_MAP.get(label, {"range": "25-40", "desc": ""})
    age_range = info["range"].replace("+", "")
    parts = [p.strip() for p in age_range.split("-") if p.strip().isdigit()]
    if len(parts) == 2:
        low, high = int(parts[0]), int(parts[1])
        range_mid = (low + high) / 2
        range_span = (high - low) / 2
        variation = (confidence - 0.5) * range_span
        est_age = int(range_mid + variation)
        est_age = max(low, min(high, est_age))
    elif len(parts) == 1:
        base_age = int(parts[0])
        est_age = base_age + int(confidence * 20)
    else:
        est_age = 30
    return {"range": info["range"], "estimated_age": est_age, "description": info.get("desc", "")}

def log_prediction(filename, label, conf, bbox, age_info, all_probs):
    if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
        bbox = [int(x) for x in bbox]
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "filename": filename,
        "prediction": label,
        "confidence": f"{conf:.4f}",
        "bbox": bbox,
        "estimated_age_range": age_info["range"],
        "estimated_age": age_info["estimated_age"],
        "condition_description": age_info["description"],
        "all_probabilities": all_probs
    }
    log_file = os.path.join(LOG_DIR, f"log_{datetime.now().strftime('%Y%m%d')}.json")
    try:
        with open(log_file, "a") as f:
            json.dump(log_entry, f)
            f.write("\n")
    except Exception:
        pass
    return log_entry

# ---------------- Model & Detector loading (cached) ----------------
@st.cache_resource(show_spinner=False)
def load_face_detector():
    # Slightly more sensitive settings will be used at detection time
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return detector

@st.cache_resource(show_spinner=False)
def load_tflite_interpreter_from_keras(model_path):
    """
    Load Keras .h5 and convert to TFLite (in-memory) if possible, return (interpreter, input_details, output_details)
    If conversion fails, return None.
    """
    try:
        tf.config.optimizer.set_jit(True)
    except Exception:
        pass
    try:
        keras_model = tf.keras.models.load_model(model_path, compile=False)
    except Exception:
        return None
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        # warmup
        try:
            inp = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
            input_details = interpreter.get_input_details()
            interpreter.set_tensor(input_details[0]['index'], inp.astype(input_details[0]['dtype']))
            interpreter.invoke()
        except Exception:
            pass
        return interpreter, interpreter.get_input_details(), interpreter.get_output_details()
    except Exception:
        return None

def try_load_model_candidates():
    """
    Try candidate files and return a dict containing either a tflite interpreter or keras model.
    """
    for candidate in MODEL_CANDIDATES:
        if os.path.exists(candidate):
            try:
                interpreter_data = load_tflite_interpreter_from_keras(candidate)
                if interpreter_data is not None:
                    return {"type": "tflite", "data": interpreter_data, "path": candidate}
                else:
                    keras = tf.keras.models.load_model(candidate, compile=False)
                    # warmup keras model
                    try:
                        _ = keras.predict(np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32))
                    except Exception:
                        pass
                    return {"type": "keras", "data": keras, "path": candidate}
            except Exception:
                continue
    return None

model_info = try_load_model_candidates()
if model_info is None:
    st.error("❌ No model file found. Please place your model (.h5) in this directory (best_model.h5 or model.h5 etc.).")
    st.stop()

st.success(f"✅ Model ready: {model_info['path']} (mode: {model_info['type']})")

detector = load_face_detector()

# ---------------- Prediction function (supports TFLite & Keras fallback) ----------------
def predict_with_model(img_arr, model_info):
    """
    img_arr: preprocessed numpy array shape (1, IMG_SIZE, IMG_SIZE, 3), dtype float32
    model_info: dict returned by try_load_model_candidates()
    returns: (label, confidence_float_0_1, age_info, all_probs_dict)
    """
    preds = None
    if model_info["type"] == "tflite":
        interpreter, input_details, output_details = model_info["data"]
        input_index = input_details[0]['index']
        try:
            interpreter.set_tensor(input_index, img_arr.astype(input_details[0]['dtype']))
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]['index'])[0]
        except Exception:
            # fallback: cast to float32
            try:
                interpreter.set_tensor(input_index, img_arr.astype(np.float32))
                interpreter.invoke()
                preds = interpreter.get_tensor(output_details[0]['index'])[0]
            except Exception:
                preds = np.zeros(len(CLASSES))
    else:
        keras_model = model_info["data"]
        try:
            preds = keras_model.predict(img_arr)[0]
        except Exception:
            preds = np.zeros(len(CLASSES))

    preds = np.array(preds, dtype=float).reshape(-1)

    # If preds look like probabilities (all in [0,1] and sum approx 1), use as-is.
    if np.all(preds >= 0) and np.all(preds <= 1) and np.isclose(np.sum(preds), 1.0, atol=1e-3):
        probs = preds / np.sum(preds)
    else:
        # apply stable softmax to logits
        ex = np.exp(preds - np.max(preds)) if preds.size else preds
        probs = ex / np.sum(ex) if ex.size else np.zeros(len(CLASSES))

    # ensure length matches classes
    if probs.size < len(CLASSES):
        probs = np.pad(probs, (0, len(CLASSES) - probs.size), constant_values=0.0)

    idx = int(np.argmax(probs))
    label = CLASSES[idx]
    confidence = float(probs[idx])
    age_info = estimate_age(label, confidence)
    all_probs = {CLASSES[i]: float(probs[i] * 100.0) for i in range(len(CLASSES))}
    return label, confidence, age_info, all_probs

# ---------------- Annotation (PIL) - improved placement and ID colors ----------------
def annotate_image_pil(img_bgr, bbox, label, conf_pct, age_info, face_id=None):
    """
    Improved annotation:
    - semi-transparent label background (RGBA overlay)
    - automatic contrast for text (white on dark bg, dark on light bg)
    - text stroke for extra legibility
    - larger font size and safe clamping for label position
    """
    # convert to PIL RGBA for overlay support
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw_base = ImageDraw.Draw(pil_img)

    # font (scale with image size)
    base_font_size = max(14, int(min(pil_img.size) * 0.035))
    try:
        font = ImageFont.truetype("arial.ttf", base_font_size)
    except Exception:
        font = ImageFont.load_default()

    x, y, w, h = bbox
    label_txt = f"{label.replace('_',' ').title()} | {conf_pct:.1f}% | Age: {age_info['estimated_age']}y"
    if face_id is not None:
        label_txt = f"#{face_id} " + label_txt

    # color palette and outline
    palette = ["#00faff", "#7cffb2", "#ffd166", "#ff6b6b", "#c084fc"]
    if face_id is None:
        outline_col = "#00faff"
    else:
        outline_col = palette[(face_id - 1) % len(palette)]

    # compute text size
    try:
        text_bbox = font.getbbox(label_txt)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
    except Exception:
        text_w, text_h = (len(label_txt) * (base_font_size // 2), base_font_size + 6)

    # preferred label position (above bbox), else below
    margin = 8
    y_offset = y - text_h - margin
    if y_offset < 2:
        y_offset = y + 4
    x_offset = x + 4
    # clamp to image width
    if x_offset + text_w + 12 > pil_img.size[0] - 4:
        x_offset = max(4, pil_img.size[0] - text_w - 12)

    # label background coordinates (with small padding)
    bg_left = x_offset - 6
    bg_top = y_offset - 4
    bg_right = x_offset + text_w + 6
    bg_bottom = y_offset + text_h + 4

    # ensure bg box inside image
    bg_left = max(2, bg_left)
    bg_top = max(2, bg_top)
    bg_right = min(pil_img.size[0] - 2, bg_right)
    bg_bottom = min(pil_img.size[1] - 2, bg_bottom)

    # draw face bounding box on base image with thickness
    def hex_to_rgb(hexcol):
        hexcol = hexcol.lstrip('#')
        return tuple(int(hexcol[i:i+2], 16) for i in (0, 2, 4))
    outline_rgb = hex_to_rgb(outline_col)
    rect_thickness = max(2, int(min(pil_img.size) * 0.0035))
    for t in range(rect_thickness):
        draw_base.rectangle([x - t, y - t, x + w + t, y + h + t], outline=outline_rgb + (255,))

    # draw semi-transparent bg on overlay (dark translucent)
    bg_fill = (6, 35, 55, 200)
    draw_overlay.rectangle([bg_left, bg_top, bg_right, bg_bottom], fill=bg_fill)

    # composite overlay onto image
    combined = Image.alpha_composite(pil_img, overlay)
    draw_combined = ImageDraw.Draw(combined)

    # pick text color with contrast check vs bg_fill
    def luminance(rgb):
        r, g, b = rgb
        return 0.299*r + 0.587*g + 0.114*b
    sample_rgb = (bg_fill[0], bg_fill[1], bg_fill[2])
    text_color = (255, 255, 255) if luminance(sample_rgb) < 140 else (10, 10, 10)

    # draw text with simple stroke (offsets)
    stroke_color = (0, 0, 0) if text_color == (255,255,255) else (255,255,255)
    offsets = [(-1,0),(1,0),(0,-1),(0,1)]
    for dx, dy in offsets:
        draw_combined.text((x_offset+dx, y_offset+dy), label_txt, font=font, fill=stroke_color)
    draw_combined.text((x_offset, y_offset), label_txt, font=font, fill=text_color)

    # convert back to BGR numpy for returning consistent type
    out_bgr = cv2.cvtColor(np.array(combined.convert("RGB")), cv2.COLOR_RGB2BGR)
    return out_bgr

# ---------------- Session-state for logs ----------------
if "session_log" not in st.session_state:
    st.session_state["session_log"] = []

# ---------------- Sidebar (info) ----------------
with st.sidebar:
    st.header("🛈 About")
    st.write("**Model:** Loaded from local .h5 (converted to TFLite for fast inference when possible)")
    st.write(f"**Mode:** {model_info['type']}")
    st.write("**Classes:**")
    for cls in CLASSES:
        st.write(f"- {cls.replace('_',' ').title()}")
    st.markdown("---")
    st.write("⚡ **Target:** < 2s processing (depends on image & model)")
    st.write(f"⚖️ Confidence Threshold: {int(CONFIDENCE_THRESHOLD*100)}%")
    st.write(f"🖼️ Max display image size: {MAX_IMAGE_SIZE}px")
    st.markdown("---")
    st.write("**Age Ranges (heuristic):**")
    for k, v in AGE_MAP.items():
        st.write(f"- {k.replace('_',' ').title()}: {v['range']} — {v['desc']}")
    st.markdown("---")
    st.write("Developed by **Mohammed Mujahid Ahmed**")
    st.write("© 2025")

# ---------------- Upload UI ----------------
col_main, col_right = st.columns([2, 1])
with col_main:
    st.markdown("<div class='glass-card'><b style='font-size:1.15em;'>Upload a face image to start futuristic analysis</b></div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility='collapsed')

if not uploaded:
    st.stop()

# ---------------- Begin processing ----------------
start_all = time.time()
file_bytes = np.frombuffer(uploaded.read(), np.uint8)
original_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

if original_bgr is None:
    st.error("⚠️ Invalid image file.")
    st.stop()

# Resize for display & speed
resized_img, scale = resize_image_cv(original_bgr.copy(), max_size=MAX_IMAGE_SIZE)

# Display original
with col_main:
    st.image(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB), caption="🧠 Uploaded (resized for speed)", use_column_width=True)

# Face detection (tuned)
gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
faces = detector.detectMultiScale(
    gray,
    scaleFactor=1.1,   # slightly smaller step for better detection
    minNeighbors=5,    # a bit stricter to reduce false positives
    minSize=(40, 40),
    flags=cv2.CASCADE_SCALE_IMAGE
)

annotated_img = resized_img.copy()
results = []

# If multiple faces, process all (left-to-right order)
if isinstance(faces, (list, np.ndarray)) and len(faces) > 0:
    # sort faces by x coordinate so labeling is left-to-right
    faces = sorted(faces, key=lambda r: r[0])
    face_id = 1
    for (x, y, w, h) in faces:
        # expand bbox a little (to include forehead/chin) and clamp to image
        pad_x = int(0.12 * w)
        pad_y = int(0.12 * h)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(annotated_img.shape[1], x + w + pad_x)
        y2 = min(annotated_img.shape[0], y + h + pad_y)
        face_roi = resized_img[y1:y2, x1:x2]
        # pad to square before resizing to avoid distortion
        face_roi_sq = pad_to_square(face_roi, fill=(128, 128, 128))
        input_arr = preprocess_for_model(face_roi_sq, model_info=model_info)
        # Run prediction
        t0 = time.time()
        label, conf, age_info, all_probs = predict_with_model(input_arr, model_info)
        latency = round(time.time() - t0, 3)
        conf_pct = conf * 100.0

        # Annotate always, but visually emphasize high-confidence
        annotated_img = annotate_image_pil(annotated_img, (x1, y1, x2 - x1, y2 - y1), label, conf_pct, age_info, face_id=face_id)

        results.append({
            "Face": face_id,
            "Condition": label.replace('_', ' ').title(),
            "Confidence (%)": f"{conf_pct:.2f}",
            "Age Range": age_info["range"],
            "Est. Age": age_info["estimated_age"],
            "Description": age_info["description"],
            "BBox": f"({x1},{y1},{x2-x1},{y2-y1})",
            "Latency (s)": latency
        })
        log_prediction(uploaded.name, label, conf, (x1, y1, x2 - x1, y2 - y1), age_info, all_probs)
        face_id += 1

else:
    # No faces: run on full image (pad -> preprocess)
    full_sq = pad_to_square(resized_img, fill=(128, 128, 128))
    input_arr = preprocess_for_model(full_sq, model_info=model_info)
    t0 = time.time()
    label, conf, age_info, all_probs = predict_with_model(input_arr, model_info)
    latency = round(time.time() - t0, 3)
    conf_pct = conf * 100.0
    h_img, w_img = resized_img.shape[:2]
    annotated_img = annotate_image_pil(annotated_img, (0, 0, w_img, h_img), label, conf_pct, age_info, face_id=None)
    results.append({
        "Face": "Full Image",
        "Condition": label.replace('_', ' ').title(),
        "Confidence (%)": f"{conf_pct:.2f}",
        "Age Range": age_info["range"],
        "Est. Age": age_info["estimated_age"],
        "Description": age_info["description"],
        "BBox": f"(0,0,{w_img},{h_img})",
        "Latency (s)": latency
    })
    log_prediction(uploaded.name, label, conf, (0, 0, w_img, h_img), age_info, all_probs)

end_all = time.time()
total_elapsed = round(end_all - start_all, 3)

# ---------------- Metrics & Table ----------------
metrics_df = pd.DataFrame({
    "Metric": ["Skin Condition", "Predicted Age", "Confidence", "Processing Time (inference)", "Total Processing Time", "Bounding Box"],
    "Value": [
        results[0]["Condition"] if results else "",
        f"{results[0]['Est. Age']} yrs (range {results[0]['Age Range']})" if results else "",
        f"{results[0]['Confidence (%)']}" if results else "",
        f"{results[0]['Latency (s)']} s" if results and "Latency (s)" in results[0] else "N/A",
        f"{total_elapsed} s",
        results[0]["BBox"] if results else ""
    ]
})

with col_right:
    st.markdown("### 📊 Analysis Report")
    st.markdown(metrics_df.to_html(index=False, escape=False), unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🧾 Session Log (recent)")
    # append session log rows (show last 20)
    st.session_state["session_log"].extend(results)
    try:
        st.dataframe(pd.DataFrame(st.session_state["session_log"]).tail(20), use_container_width=True)
    except Exception:
        st.write(st.session_state["session_log"][-10:])

# ---------------- Show annotated & downloads ----------------
col_left_img, col_right_controls = st.columns([2,1])
with col_left_img:
    st.markdown("### 🟩 AI Annotated Result")
    st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_column_width=True)
    _, buffer = cv2.imencode(".png", annotated_img)
    btn_buf = io.BytesIO(buffer.tobytes())
    st.download_button("📥 Download Annotated Image", btn_buf, file_name=f"{results[0]['Condition'] if results else 'annotated'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", mime="image/png")

with col_right_controls:
    st.markdown("### 📁 Results & Export")
    df_results = pd.DataFrame(results)
    st.dataframe(df_results, use_container_width=True)
    csv = df_results.to_csv(index=False)
    st.download_button("📥 Download CSV Report", csv, file_name=f"report_{uploaded.name.split('.')[0]}.csv", mime="text/csv")
    st.markdown("---")
    # show processing timing badge
    if total_elapsed <= 2.0:
        st.success(f"⚡ Total processed in {total_elapsed:.2f}s (EXCELLENT)")
    elif total_elapsed <= 5.0:
        st.info(f"⏱️ Processed in {total_elapsed:.2f}s (ACCEPTABLE)")
    else:
        st.warning(f"⏳ Processing took {total_elapsed:.2f}s (SLOW)")
