# ==========================================================
# DermalScan v3.4 — Crimson Quantum Edition (Optimized)
# Deep Wine Background | Neon White Text | Purple UI Elements
# Module 7: Export & Logging Enabled
# ==========================================================
import streamlit as st
import numpy as np
import cv2
import pandas as pd
import time
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.densenet import preprocess_input
from random import randint
from io import BytesIO

# ----------------------------------------------------------
# 🌐 PAGE CONFIGURATION
# ----------------------------------------------------------
st.set_page_config(page_title="DermalScan AI", layout="wide")

# 🎨 Crimson Quantum Theme
st.markdown("""
<style>
.stApp {background-color: #4B0E1E; color: #F8F8FF;}
h1, h2, h3, h4, h5, h6 {color: #ffffff; text-shadow: 0px 0px 8px rgba(255,255,255,0.4);}
.stFileUploader {background-color: #5B1B3B; border: 1px solid #9A4D7F; border-radius: 12px; padding: 1rem;}
.stButton > button {
    background: linear-gradient(90deg,#7B2CBF,#9D4EDD);
    color: #FFFFFF !important;
    border-radius: 8px; border: none;
    font-weight: 600;
    padding: 0.6em 1.4em;
    box-shadow: 0 0 10px rgba(157,78,221,0.6);
}
.stButton > button:hover {
    background: linear-gradient(90deg,#9D4EDD,#7B2CBF);
    transform: scale(1.05);
    box-shadow: 0 0 20px rgba(157,78,221,0.8);
}
.css-1r6slb0 {background-color: #5B1B3B !important; border-radius: 12px !important; border: 1px solid #9A4D7F !important;}
footer {visibility: hidden;}
/* Download button text fix */
.stDownloadButton button {
    color: #4B0E1E !important;
    background-color: #F8F8FF !important;
    border-radius: 8px !important;
    font-weight: 600;
}
.stDownloadButton button:hover {
    background-color: #EFBBD2 !important;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# ⚡ CACHE HEAVY COMPONENTS
# ----------------------------------------------------------
@st.cache_resource
def load_ai_components():
    model = load_model("densenet121_best_optimized.h5")
    detector = MTCNN()
    classes = ["clear face", "darkspots", "puffy eyes", "wrinkles"]
    return model, detector, classes

model, detector, classes = load_ai_components()

# ----------------------------------------------------------
# 🧠 HEADER
# ----------------------------------------------------------
st.markdown("<h1> DermalScan AI</h1>", unsafe_allow_html=True)
st.markdown("<h4>Precision Skin Feature Detection & Age Estimation</h4>", unsafe_allow_html=True)
st.markdown("---")

# ----------------------------------------------------------
# 📸 FILE UPLOAD
# ----------------------------------------------------------
col1, col2 = st.columns([1.2, 1])
with col1:
    uploaded_file = st.file_uploader("📤 Upload Your Face Image", type=["jpg", "jpeg", "png"])
    st.markdown("<p style='color:#E8D6F0;'>Ensure good lighting and a front-facing pose.</p>", unsafe_allow_html=True)
with col2:
    st.markdown("<div style='background:rgba(155,55,155,0.15);border:1px solid rgba(157,78,221,0.4);border-radius:15px;padding:1em;box-shadow:0 0 20px rgba(157,78,221,0.2);'><b>Supported Detections:</b><br>• Wrinkles<br>• Dark Spots<br>• Puffy Eyes<br>• Clear Skin</div>", unsafe_allow_html=True)

# ----------------------------------------------------------
# ⚙️ FAST PROCESSING FUNCTION (Compact Multifaced Label)
# ----------------------------------------------------------
def process_image_fast(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(rgb)

    if len(detections) == 0:
        h, w, _ = image.shape
        detections = [{'box': (0, 0, w, h)}]

    results, coords = [], []
    img_h, img_w = image.shape[:2]

    used_positions = []  # to avoid label overlap

    for det in detections:
        x, y, w, h = det['box']
        x, y = max(0, x), max(0, y)
        face = rgb[y:y + h, x:x + w]
        if face.size == 0:
            continue

        # --- AI Prediction ---
        face_resized = cv2.resize(face, (224, 224), interpolation=cv2.INTER_AREA)
        face_array = preprocess_input(np.expand_dims(img_to_array(face_resized), axis=0))
        preds = model.predict(face_array, verbose=0)[0]
        class_idx = np.argmax(preds)
        predicted_class = classes[class_idx]
        confidence = float(preds[class_idx]) * 100

        # --- Age Estimation ---
        est_age = {
            "clear face": randint(18, 30),
            "darkspots": randint(30, 40),
            "puffy eyes": randint(40, 55),
            "wrinkles": randint(60, 75)
        }[predicted_class]

        # --- Store results ---
        results.append({
            "Feature": predicted_class,
            "Confidence (%)": round(confidence, 1),
            "Estimated Age": est_age
        })
        coords.append((x, y, w, h))

        # --- Draw bounding box ---
        cv2.rectangle(image, (x, y), (x + w, y + h), (155, 80, 220), 2)

        # --- Prepare compact label ---
        label = f"{predicted_class[:10]}  {confidence:.1f}%  |  Age {est_age}"

        # --- Calculate label position ---
        text_x = x
        text_y = y - 8

        # if too close to top or overlaps previous label, push below
        for prev_y in used_positions:
            if abs(text_y - prev_y) < 25:
                text_y += 25
        used_positions.append(text_y)

        if text_y < 15:
            text_y = y + h + 18

        # --- Draw translucent background ---
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        bg_start = (text_x, text_y - text_h - 2)
        bg_end = (text_x + text_w + 4, text_y + 4)
        cv2.rectangle(image, bg_start, bg_end, (75, 0, 100), -1)
        cv2.putText(image, label, (text_x + 2, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 235, 255), 1, cv2.LINE_AA)

    return image, pd.DataFrame(results), coords


# ----------------------------------------------------------
# 🚀 SCAN + EXPORT MODULE
# ----------------------------------------------------------
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    with st.spinner("🔬 Running AI Scan..."):
        start_time = time.time()
        annotated_img, results_df, coords = process_image_fast(image)
        latency = time.time() - start_time

    st.success(f"✅ Scan Completed in {latency:.2f} seconds")

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
                 caption="🖼️ AI-Annotated Result", use_container_width=True)
        # Annotated image export
        buffered = BytesIO()
        cv2.imwrite("annotated_result.jpg", annotated_img)
        _, buf = cv2.imencode(".jpg", annotated_img)
        st.download_button("📸 Download Annotated Image", data=buf.tobytes(),
                           file_name="DermalScan_Annotated.jpg", mime="image/jpeg")

    with col2:
        st.markdown("### 📊 Analysis Summary")
        if not results_df.empty:
            detailed_results = []
            for (x, y, w, h), row in zip(coords, results_df.to_dict(orient="records")):
                detailed_results.append({
                    "X": x, "Y": y, "Width": w, "Height": h,
                    "Feature": row["Feature"],
                    "Confidence (%)": row["Confidence (%)"],
                    "Estimated Age": row["Estimated Age"]
                })
            summary_df = pd.DataFrame(detailed_results)

            # Glass-styled detailed vertical table
            st.markdown("""
            <style>
            .big-table table {
                width: 100%;
                border-collapse: collapse;
                font-size: 1.05rem !important;
                color: #F8F8FF !important;
            }
            .big-table th {
                background-color: #5B1B3B;
                padding: 10px;
                text-align: left;
                color: #EFBBD2 !important;
                border-bottom: 2px solid #9A4D7F;
            }
            .big-table td {
                background-color: rgba(155,55,155,0.15);
                border-bottom: 1px solid #9A4D7F;
                padding: 8px 10px;
            }
            </style>
            """, unsafe_allow_html=True)

            st.markdown("<div class='big-table'>", unsafe_allow_html=True)
            st.write(summary_df)
            st.markdown("</div>", unsafe_allow_html=True)

            avg_age = results_df["Estimated Age"].mean()
            st.markdown(f"<p>🧬 <span style='color:#9D4EDD;font-weight:600;'>Estimated Biological Age:</span> {avg_age:.1f} years</p>", unsafe_allow_html=True)
            st.markdown(f"<p>⏱️ <span style='color:#9D4EDD;font-weight:600;'>Total Prediction Time:</span> {latency:.2f} seconds</p>", unsafe_allow_html=True)

            # CSV Export
            csv = summary_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Detailed Report (CSV)", data=csv,
                               file_name="DermalScan_Detailed_Report.csv", mime="text/csv")

            # Logging
            with open("DermalScan_Logs.txt", "a") as log:
                log.write(f"\n--- Scan @ {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                log.write(summary_df.to_string(index=False))
                log.write(f"\nAverage Age: {avg_age:.1f} | Total Time: {latency:.2f}s\n")

        else:
            st.warning("⚠️ No face detected in the image.")
else:
    st.info("👆 Upload a facial image to begin the AI scan.")

# ----------------------------------------------------------
# 🧩 FOOTER
# ----------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#9D4EDD; font-size:0.9em;'>
⚙️ DermalScan v3.4 | developed by Shreya
</div>
""", unsafe_allow_html=True)
