import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from datetime import datetime
import pandas as pd
import os
import json

st.set_page_config(page_title="DermalScan AI", page_icon="🔬", layout="wide")

MODEL_PATH = 'output/best_model.h5'
CLASSES = ["clear_face", "dark_spots", "puffy_eyes", "wrinkles"]
IMG_SIZE = 224
LOG_DIR = 'logs'
MAX_IMAGE_SIZE = 1024
CONFIDENCE_THRESHOLD = 0.4

os.makedirs(LOG_DIR, exist_ok=True)

AGE_MAP = {
    'clear_face': {'range': '10-29', 'age': 25},
    'dark_spots': {'range': '30-59', 'age': 45},
    'puffy_eyes': {'range': '30-59', 'age': 40},
    'wrinkles': {'range': '60+', 'age': 65}
}

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_detector():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def resize_image(img, max_size=MAX_IMAGE_SIZE):
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA), scale
    return img, 1.0

def preprocess(img):
    resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return resized.astype('float32') / 255.0

def predict(img_arr, model):
    preds = model(np.expand_dims(img_arr, axis=0), training=False).numpy()[0]
    idx = int(np.argmax(preds))
    return CLASSES[idx], float(preds[idx]), AGE_MAP[CLASSES[idx]], {CLASSES[i]: float(preds[i]*100) for i in range(len(CLASSES))}

def detect_faces(img, detector):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    if len(faces) > 0:
        return faces
    faces = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))
    if len(faces) > 0:
        return faces
    faces = detector.detectMultiScale(gray, scaleFactor=1.03, minNeighbors=2, minSize=(15, 15))
    return faces

def is_skin_related(img, model, threshold=CONFIDENCE_THRESHOLD):
    cls, conf, age_info, all_probs = predict(preprocess(img), model)
    if conf >= threshold:
        return True, cls, conf, age_info, all_probs
    return False, None, None, None, None

def draw_annotation(img, x, y, w, h, pred):
    """Draw GREEN box with ALL info in ONE LINE above the border"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font = ImageFont.truetype("arial.ttf", 11)
    except:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 11)
        except:
            font = ImageFont.load_default()
    
    # GREEN BOX
    draw.rectangle([x, y, x+w, y+h], outline=(0, 255, 0), width=5)
    
    # ALL INFO IN ONE LINE
    label_text = f"{pred['class'].replace('_', ' ').title()} | Conf: {pred['conf']*100:.1f}% | Age: ~{pred['age']} yrs ({pred['range']})"
    
    # Position ABOVE box
    label_x = x
    label_y = y - 25
    
    if label_y < 5:
        label_y = y + h + 10
    
    # Black outline
    outline_offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for dx, dy in outline_offsets:
        draw.text((label_x+dx, label_y+dy), label_text, fill=(0,0,0), font=font)
    
    # WHITE text
    draw.text((label_x, label_y), label_text, fill=(255,255,255), font=font)
    
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def log_pred(filename, data):
    try:
        log_file = os.path.join(LOG_DIR, 'predictions.json')
        logs = json.load(open(log_file)) if os.path.exists(log_file) else []
        logs.append({'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'file': filename, 'data': data})
        json.dump(logs, open(log_file, 'w'), indent=2)
    except:
        pass

def create_csv(filename, preds):
    data = []
    for i, p in enumerate(preds):
        data.append({
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Image': filename,
            'Face_Number': i+1,
            'Detected_Class': p['class'],
            'Confidence_%': f"{p['conf']*100:.2f}",
            'Age_Estimate': p['age'],
            'Age_Range': p['range'],
            'BBox_X': p['bbox'][0],
            'BBox_Y': p['bbox'][1],
            'BBox_Width': p['bbox'][2],
            'BBox_Height': p['bbox'][3]
        })
    return pd.DataFrame(data)

st.title("🔬 DermalScan AI")
st.write("Facial Skin Aging Detection System")

with st.sidebar:
    st.info("**AI-powered facial aging detection**")
    st.write("### Detection Classes")
    for cls in CLASSES:
        st.write(f"• {cls.replace('_', ' ').title()}")
    st.write("---")
    st.write("**Model:** MobileNetV2")
    st.write("**Validation Accuracy:** 78.12%")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload Image")
    uploaded = st.file_uploader("Choose a facial image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Original Image", use_container_width=True)

with col2:
    if uploaded:
        st.subheader("🔍 Analysis Results")
        model = load_model()
        detector = load_detector()
        start = datetime.now()
        
        with st.spinner("🔄 Analyzing..."):
            arr = np.array(img)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            bgr_resized, scale = resize_image(bgr)
            faces = detect_faces(bgr_resized, detector)
            preds = []
            
            if len(faces) > 0:
                st.success(f"✅ Detected {len(faces)} face(s)")
                (x, y, w, h) = faces[0]
                face_region = bgr_resized[y:y+h, x:x+w]
                if face_region.size > 0:
                    cls, conf, age_info, all_probs = predict(preprocess(face_region), model)
                    orig_x, orig_y, orig_w, orig_h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
                    preds.append({'class': cls, 'conf': conf, 'age': age_info['age'], 'range': age_info['range'], 'bbox': [orig_x, orig_y, orig_w, orig_h], 'probs': all_probs})
                    annotated_img = draw_annotation(bgr, orig_x, orig_y, orig_w, orig_h, preds[0])
                    can_analyze = True
            else:
                st.info("ℹ️ No face detected - Checking for skin content...")
                is_skin, cls, conf, age_info, all_probs = is_skin_related(bgr_resized, model)
                if is_skin:
                    st.success(f"✅ Skin content detected: {cls.replace('_', ' ').title()}")
                    h, w = bgr.shape[:2]
                    margin = int(min(h, w) * 0.05)
                    box_x, box_y, box_w, box_h = margin, margin, w - 2*margin, h - 2*margin
                    preds.append({'class': cls, 'conf': conf, 'age': age_info['age'], 'range': age_info['range'], 'bbox': [box_x, box_y, box_w, box_h], 'probs': all_probs})
                    annotated_img = draw_annotation(bgr, box_x, box_y, box_w, box_h, preds[0])
                    can_analyze = True
                else:
                    st.error("❌ No facial skin content detected")
                    st.warning("**Please upload an image containing:**\n- Clear facial skin\n- Skin aging features")
                    can_analyze = False
            
            if can_analyze:
                st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="Annotated Result", use_container_width=True)
                elapsed = (datetime.now() - start).total_seconds()
                log_pred(uploaded.name, preds)
                p = preds[0]
                st.write(f"### {p['class'].replace('_', ' ').title()}")
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Confidence", f"{p['conf']*100:.1f}%")
                with m2:
                    st.metric("Age Estimate", f"~{p['age']} yrs")
                with m3:
                    st.metric("Age Range", p['range'])
                if elapsed <= 2:
                    st.success(f"⚡ Processing Time: {elapsed:.2f}s")
                elif elapsed <= 5:
                    st.info(f"⏱️ Processing Time: {elapsed:.2f}s")
                st.write("---")
                st.write("### 💾 Export & Download")
                e1, e2 = st.columns(2)
                with e1:
                    _, buf = cv2.imencode('.png', annotated_img)
                    st.download_button("📥 Download Annotated Image", buf.tobytes(), f"dermalscan_{uploaded.name}", "image/png", use_container_width=True)
                with e2:
                    csv = create_csv(uploaded.name, preds)
                    st.download_button("📊 Download CSV Report", csv.to_csv(index=False), f"report_{uploaded.name.split('.')[0]}.csv", "text/csv", use_container_width=True)
                with st.expander("📋 View Detailed Report"):
                    st.dataframe(csv, use_container_width=True)

st.write("---")
st.write("**DermalScan AI** - Developed by **Boini Pramod Kumar** | Powered by MobileNetV2 & Haar Cascade | © 2025")
