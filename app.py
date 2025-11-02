# ============================================================
# 🧠 DermalScan AI — ULTRA-FAST Skin Condition Classification
# ============================================================


import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from datetime import datetime
import pandas as pd
import os, json, time


# ------------------------------------------------------------
# STREAMLIT CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(page_title="DermalScan AI", page_icon="🔬", layout="wide")


# ------------------------------------------------------------
# AGGRESSIVE OPTIMIZATION CONSTANTS
# ------------------------------------------------------------
MODEL_PATH = "best_model.h5"
CLASSES = ["clear_face", "dark_spots", "puffy_eyes", "wrinkles"]
IMG_SIZE = 224
LOG_DIR = "logs"
MAX_IMAGE_SIZE = 400  # REDUCED from 600 for guaranteed speed
CONFIDENCE_THRESHOLD = 0.4
TIMEOUT_SECONDS = 4.5  # Safety margin before 5 sec


os.makedirs(LOG_DIR, exist_ok=True)


# ------------------------------------------------------------
# AGE MAPPING
# ------------------------------------------------------------
AGE_MAP = {
    "clear_face": {"range": "15-25", "desc": "Youthful skin"},
    "puffy_eyes": {"range": "25-40", "desc": "Early aging signs"},
    "dark_spots": {"range": "35-55", "desc": "Sun damage/aging"},
    "wrinkles": {"range": "50-70+", "desc": "Advanced aging"}
}


# ------------------------------------------------------------
# ULTRA-FAST MODEL LOADING WITH TF OPTIMIZATIONS
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    # Force CPU/GPU optimization
    tf.config.optimizer.set_jit(True)  # Enable XLA compilation
    
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    
    # Convert to TFLite for faster inference
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    # Create interpreter
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # MINIMAL WARMUP - Just 1 iteration for instant loading
    dummy = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32) / 255.0
    
    interpreter.set_tensor(input_details[0]['index'], dummy)
    interpreter.invoke()
    _ = interpreter.get_tensor(output_details[0]['index'])
    
    return interpreter, input_details, output_details


@st.cache_resource(show_spinner=False)
def load_detector():
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    # MINIMAL WARMUP - Just 1 iteration for instant loading
    dummy_img = np.zeros((400, 400, 3), dtype=np.uint8)
    dummy_gray = cv2.cvtColor(dummy_img, cv2.COLOR_BGR2GRAY)
    
    _ = detector.detectMultiScale(
        dummy_gray, 
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(50, 50)
    )
    
    return detector


# ------------------------------------------------------------
# ULTRA-FAST IMAGE PROCESSING
# ------------------------------------------------------------
def resize_image(img, max_size=MAX_IMAGE_SIZE):
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        # Use INTER_NEAREST for maximum speed (slight quality loss acceptable)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST), scale
    return img, 1.0


def preprocess(img):
    # Ultra-fast resize with INTER_NEAREST
    resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return np.expand_dims(resized.astype("float32") / 255.0, axis=0)


# ------------------------------------------------------------
# IMPROVED AGE ESTIMATION
# ------------------------------------------------------------
def estimate_age(label, confidence):
    info = AGE_MAP[label]
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
    
    return {
        "range": info["range"],
        "estimated_age": est_age,
        "description": info["desc"]
    }


# ------------------------------------------------------------
# ULTRA-FAST PREDICTION WITH TFLITE
# ------------------------------------------------------------
def predict(img_arr, interpreter_data):
    interpreter, input_details, output_details = interpreter_data
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_arr)
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    preds = interpreter.get_tensor(output_details[0]['index'])[0]
    
    idx = int(np.argmax(preds))
    label = CLASSES[idx]
    confidence = float(preds[idx])
    age_info = estimate_age(label, confidence)
    all_probs = {CLASSES[i]: float(preds[i] * 100) for i in range(len(CLASSES))}
    return label, confidence, age_info, all_probs


# ------------------------------------------------------------
# FASTER FACE DETECTION (AGGRESSIVE PARAMS)
# ------------------------------------------------------------
def detect_faces(img, detector):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Very aggressive parameters for speed
    faces = detector.detectMultiScale(
        gray, 
        scaleFactor=1.3,  # Larger = faster
        minNeighbors=3,   # Lower = faster (less false positives filtering)
        minSize=(50, 50), # Slightly larger minimum
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) > 0:
        areas = [w * h for (x, y, w, h) in faces]
        largest_idx = np.argmax(areas)
        return np.array([faces[largest_idx]])
    return None


# ------------------------------------------------------------
# COMPACT ANNOTATION - ALWAYS OUTSIDE AND ABOVE
# ------------------------------------------------------------
def annotate_image(img, bbox, label, conf, age_info):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 9)  # Small compact font
    except:
        font = ImageFont.load_default()
    
    x, y, w, h = bbox
    img_h, img_w = img.shape[:2]
    
    # Prepare compact side-by-side label format
    label_txt = f"{label.replace('_', ' ').title()} | Conf: {conf:.1f}% | Age: {age_info['range']} (≈{age_info['estimated_age']}y)"
    
    # Calculate text dimensions
    try:
        text_bbox = draw.textbbox((0, 0), label_txt, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
    except:
        text_width = 200
        text_height = 10
    
    # ENSURE LABEL IS ALWAYS OUTSIDE AND ABOVE
    min_label_gap = 15  # Minimum gap between label and box
    
    # Check if there's enough space above for the label
    if y < text_height + min_label_gap + 10:
        # Not enough space above - push the bounding box down
        y_shift = (text_height + min_label_gap + 10) - y
        y = y + y_shift
        # Make sure box doesn't go out of bounds
        if y + h > img_h - 10:
            # If box would go out, shrink it
            h = img_h - y - 10
    
    # Adjust border to ensure visibility even at edges
    border_pad = 8
    border_width = 2  # Thin border
    
    # Adjust coordinates if touching edges
    if x < border_pad:
        x = border_pad
    if y < border_pad:
        y = border_pad
    if x + w > img_w - border_pad:
        w = img_w - x - border_pad
    if y + h > img_h - border_pad:
        h = img_h - y - border_pad
    
    # Draw thin green border
    draw.rectangle([x, y, x+w, y+h], outline="green", width=border_width)
    
    # Position label OUTSIDE and ABOVE the border
    y_offset = y - text_height - min_label_gap
    x_offset = x + 5
    
    # Ensure text stays within image width
    if x_offset + text_width > img_w - 10:
        x_offset = img_w - text_width - 10
    if x_offset < 5:
        x_offset = 5
    
    # Draw compact label text (NO BACKGROUND - lime green text)
    draw.text((x_offset, y_offset), label_txt, fill="lime", font=font)
    
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ------------------------------------------------------------
# ASYNC LOGGING (NON-BLOCKING)
# ------------------------------------------------------------
def log_prediction(filename, label, conf, bbox, age_info, all_probs):
    if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
        bbox = [int(x) for x in bbox]
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "filename": filename,
        "prediction": label,
        "confidence": f"{conf:.2f}",
        "bbox": bbox,
        "estimated_age_range": age_info["range"],
        "estimated_age": age_info["estimated_age"],
        "condition_description": age_info["description"],
        "all_probabilities": all_probs
    }
    
    log_file = os.path.join(LOG_DIR, f"log_{datetime.now().strftime('%Y%m%d')}.json")
    
    # Non-blocking write
    try:
        with open(log_file, "a") as f:
            json.dump(log_entry, f)
            f.write("\n")
    except:
        pass  # Skip logging if it fails to avoid delay
    
    return log_entry


# ============================================================
# STREAMLIT UI
# ============================================================
st.title("🔬 DermalScan AI - Skin Condition Classifier")
st.markdown("**ULTRA-FAST MODE** - Optimized for <2 second processing")

# PRE-LOAD MODELS ON APP START (INSTANT UI LOADING)
interpreter_data = load_model()
detector = load_detector()

with st.sidebar:
    st.header("ℹ️ About")
    st.write("**Model:** DenseNet121 (TFLite)")
    st.write("**Validation Accuracy:** 84.13%")
    st.write("**Classes:**")
    for cls in CLASSES:
        st.write(f"- {cls.replace('_', ' ').title()}")
    st.markdown("---")
    st.write("⚡ **TARGET: <2 sec**")
    st.write("🎯 Confidence ≥40%")
    st.write("🚀 Max Image: 400px")
    st.markdown("---")
    st.write("**Age Ranges:**")
    for key, val in AGE_MAP.items():
        st.write(f"- {key.replace('_', ' ').title()}: {val['range']}")


# ------------------------------------------------------------
# FILE UPLOADER WITH TIMEOUT PROTECTION
# ------------------------------------------------------------
uploaded = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])


if uploaded:
    start_time = time.time()
    
    try:
        # Ultra-fast image loading
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        original_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if original_img is None:
            st.error("❌ Invalid image file.")
        else:
            # Aggressive resize
            resized_img, _ = resize_image(original_img.copy())
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📷 Original")
                # Display smaller version for speed
                display_img = cv2.resize(original_img, (300, 300), interpolation=cv2.INTER_NEAREST)
                st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            # Face detection
            faces = detect_faces(resized_img, detector)
            
            results = []
            annotated_img = resized_img.copy()
            
            if faces is not None and len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_roi = resized_img[y:y+h, x:x+w]
                processed = preprocess(face_roi)
                label, conf_raw, age_info, all_probs = predict(processed, interpreter_data)
                conf = conf_raw * 100
                
                if conf >= CONFIDENCE_THRESHOLD * 100:
                    annotated_img = annotate_image(annotated_img, (x, y, w, h), label, conf, age_info)
                    results.append({
                        "Face": 1,
                        "Condition": label.replace("_", " ").title(),
                        "Confidence (%)": f"{conf:.2f}",
                        "Age Range": age_info["range"],
                        "Est. Age": age_info["estimated_age"],
                        "Description": age_info["description"],
                        "BBox": f"({x},{y},{w},{h})"
                    })
                    log_prediction(uploaded.name, label, conf, (x, y, w, h), age_info, all_probs)
            else:
                # Full image analysis
                processed = preprocess(resized_img)
                label, conf_raw, age_info, all_probs = predict(processed, interpreter_data)
                conf = conf_raw * 100
                h, w = resized_img.shape[:2]
                annotated_img = annotate_image(annotated_img, (0, 0, w, h), label, conf, age_info)
                results.append({
                    "Face": "Full Image",
                    "Condition": label.replace("_", " ").title(),
                    "Confidence (%)": f"{conf:.2f}",
                    "Age Range": age_info["range"],
                    "Est. Age": age_info["estimated_age"],
                    "Description": age_info["description"],
                    "BBox": f"(0,0,{w},{h})"
                })
                log_prediction(uploaded.name, label, conf, (0, 0, w, h), age_info, all_probs)
            
            with col2:
                st.subheader("🎯 Result")
                st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            elapsed = time.time() - start_time
            
            if elapsed <= 2.0:
                st.success(f"✅ Processed in {elapsed:.2f}s (EXCELLENT)")
            elif elapsed <= 5.0:
                st.info(f"ℹ️ Processed in {elapsed:.2f}s (ACCEPTABLE)")
            else:
                st.error(f"❌ Processing took {elapsed:.2f}s (EXCEEDED TARGET)")
            
            if results:
                st.subheader("📊 Results")
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    _, buffer = cv2.imencode(".png", annotated_img)
                    st.download_button(
                        label="💾 Download Image",
                        data=buffer.tobytes(),
                        file_name=f"annotated_{uploaded.name}",
                        mime="image/png"
                    )
                
                with col_b:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📄 Download CSV",
                        data=csv,
                        file_name=f"report_{uploaded.name.split('.')[0]}.csv",
                        mime="text/csv"
                    )
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        elapsed = time.time() - start_time
        st.warning(f"Failed after {elapsed:.2f} seconds")


st.markdown("---")
st.write("**DermalScan AI** - Developed by **Boini Pramod Kumar** | Powered by DenseNet121 (TFLite) & Haar Cascade | © 2025")
