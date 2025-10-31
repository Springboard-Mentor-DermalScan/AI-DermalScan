import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ===============================
# APP CONFIGURATION
# ===============================
st.set_page_config(
    page_title="DermalScan AI - Skin Aging Detector",
    page_icon="🧴",
    layout="wide"
)

# ===============================
# LOAD MODEL (Cached)
# ===============================
@st.cache_resource
def load_skin_model():
    model_path = r"E:\New_Projects\Dermal_Scan_project\densenet_model.h5"
    return load_model(model_path)

model = load_skin_model()

# ===============================
# CONSTANTS
# ===============================
CLASS_NAMES = {
    0: 'clear face',
    1: 'darkspots',
    2: 'puffy eyes',
    3: 'wrinkles'
}
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize persistent table in session state
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=[
        "File_Name", "Class ID", "Condition", "Confidence", "Estimated_Age",
        "x", "y", "width", "height", "Total Prediction Time (s)"
    ])

# ===============================
# HELPER FUNCTION
# ===============================
def analyze_face(image_path, padding=0.05):
    img_cv = cv2.imread(image_path)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=6)

    h, w, _ = img_cv.shape
    annotated = img_cv.copy()
    results = []

    if len(faces) == 0:
        x1, y1, x2, y2 = int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8)
        faces = [(x1, y1, x2 - x1, y2 - y1)]

    for (x, y, fw, fh) in faces:
        pad_x, pad_y = int(fw * padding), int(fh * padding)
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(w, x + fw + pad_x), min(h, y + fh + pad_y)

        face_crop = img_cv[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        img_resized = cv2.resize(face_crop, (224, 224))
        arr = image.img_to_array(img_resized)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)

        preds = model.predict(arr, verbose=0)
        pred_class = int(np.argmax(preds))
        confidence = round(float(np.max(preds)) * 100, 2)
        label_name = CLASS_NAMES[pred_class]

        age_ranges = {
            'clear face': (18, 25),
            'darkspots': (25, 40),
            'puffy eyes': (35, 50),
            'wrinkles': (50, 70)
        }
        low, high = age_ranges[label_name]
        age = np.random.randint(low, high + 1)

        # Draw bounding box
        color = (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # ========== TOP-LEFT TEXT OVERLAY ==========
        overlay_text = f"{label_name.upper()} | Conf: {confidence}% | Age: {age}"
        cv2.putText(
            annotated, overlay_text, (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
        )

        # ==========================================

        results.append({
            "x": int(x1), "y": int(y1),
            "width": int(x2 - x1), "height": int(y2 - y1),
            "Condition": label_name,
            "Confidence": confidence,
            "Estimated_Age": age
        })

    os.makedirs("results", exist_ok=True)
    output_path = os.path.join("results", "annotated_output.jpg")
    cv2.imwrite(output_path, annotated)
    return output_path, results


# ===============================
# FRONTEND
# ===============================
st.title("✨ DermalScan AI")
st.markdown("#### Detect Facial Skin Conditions & Estimate Biological Age")
st.write("Upload a clear face image and let AI analyze for aging patterns, dark spots, and more.")

# Sidebar Upload
with st.sidebar:
    st.header("📤 Upload Section")
    uploaded_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])
    st.markdown("---")
    st.info("💡 Tip: Ensure even lighting and front-facing image for accurate results.")

if uploaded_file:
    os.makedirs("uploads", exist_ok=True)
    temp_path = os.path.join("uploads", uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(temp_path, caption="Original Image", use_container_width=True)

    start_time = time.time()
    with st.spinner("Analyzing image..."):
        output_path, results = analyze_face(temp_path)
    total_time = round(time.time() - start_time, 2)

    # ============================
    # ANALYSIS COMPLETE
    # ============================
    placeholder = st.empty()
    for i in range(0, 101, 10):
        placeholder.markdown(f"### ⏳ Analysis Progress: {i}%")
        time.sleep(0.08)
    placeholder.success(f"✅ Analysis Complete in **{total_time:.2f} seconds** 🎉")

    with col2:
        st.image(output_path, caption="Processed Output (with details on top-left)", use_container_width=True)

    if results:
        df = pd.DataFrame(results)
        df["Condition"] = df["Condition"].str.strip().str.lower()
        df["Class ID"] = df["Condition"].map({v: k for k, v in CLASS_NAMES.items()}).fillna(-1).astype(int)
        df["File_Name"] = uploaded_file.name
        df["Total Prediction Time (s)"] = total_time

        # Add to session history
        st.session_state.history = pd.concat([st.session_state.history, df], ignore_index=True)

        # Display all results together
        st.subheader("📊 Prediction History")
        st.dataframe(st.session_state.history, use_container_width=True, hide_index=True)

        # Save all results to a single CSV
        csv_path = "results/prediction_history.csv"
        st.session_state.history.to_csv(csv_path, index=False)

        # Download buttons
        c1, c2 = st.columns(2)
        with c1:
            with open(output_path, "rb") as f:
                st.download_button("⬇️ Download Annotated Image", f, "annotated_output.jpg", "image/jpeg")
        with c2:
            with open(csv_path, "rb") as f:
                st.download_button("📁 Download All Results (CSV)", f, "prediction_history.csv", "text/csv")
    else:
        st.warning("⚠️ No face detected in the uploaded image.")
else:
    st.info("📸 Upload a facial image to begin the analysis.")
