import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ===============================
# PAGE CONFIGURATION
# ===============================
st.set_page_config(page_title="💆‍♀️ Dermal Scan", layout="centered")

# ===============================
# FULL PAGE BACKGROUND COLOR (Pastel Lavender)
# ===============================
st.markdown(
    """
    <style>
    body {
        background-color: #F3E8FF; /* pastel lavender */
        color: black;
    }
    [data-testid="stAppViewContainer"] {
        background-color: #F3E8FF;
        color: black;
    }
    [data-testid="stSidebar"] {
        background-color: #F5EFFF;
        color: black;
    }
    [data-testid="stHeader"] {
        background-color: #F3E8FF;
    }
    h1, h2, h3, h4, h5, h6, p, span, div {
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===============================
# HEADER SECTION
# ===============================
st.title("💆‍♀️ Dermal Scan")
st.markdown("### AI-Powered Skin Condition & Age Prediction 🧴")
st.write("Upload a face image and let our AI analyze your **skin condition** and **predict your age**.")

# ===============================
# LOAD MODELS AND HAAR CASCADE
# ===============================
with st.expander("🧠 Model Initialization Status"):
    try:
        with st.spinner("Loading models and face detector..."):
            skin_model = load_model(
                r"C:\Users\vaish\OneDrive\Desktop\infosys virtual\models\best_densenet_model.h5"
            )
            age_model = load_model(
                r"C:\Users\vaish\OneDrive\Desktop\infosys virtual\models\age_model.h5"
            )
            face_cascade = cv2.CascadeClassifier(
                r"C:\Users\vaish\OneDrive\Desktop\infosys virtual\haarcascade_frontalface_default.xml"
            )
            time.sleep(1)
            if face_cascade.empty():
                st.error("❌ Haar Cascade not loaded. Please check the file path.")
            else:
                st.success("✅ Models and Haar Cascade loaded successfully!")
    except Exception as e:
        st.error(f"⚠️ Error loading models: {e}")
        face_cascade = None

# ===============================
# CLASS LABELS
# ===============================
class_labels = ["Clear Face", "Dark Spots", "Puffy Eyes", "Wrinkles"]
img_size = (224, 224)

# ===============================
# IMAGE UPLOAD
# ===============================
st.header("📤 Upload Your Image")
uploaded_file = st.file_uploader(
    "Select an image file (JPG or PNG):",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear face image for better analysis."
)

# ===============================
# PROCESS IMAGE
# ===============================
if uploaded_file is not None:
    if face_cascade is None:
        st.error("⚠️ Face detector not initialized properly. Please reload the app.")
    else:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            caption="🖼️ Uploaded Image",
            use_container_width=True
        )

        analyze = st.button("🔍 Analyze Image")
        if analyze:
            with st.spinner("Analyzing your image... please wait ⏳"):
                time.sleep(1)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60)
                )

                if len(faces) == 0:
                    st.warning("⚠️ No face detected in the image. Please try another image.")
                else:
                    st.success(f"✅ Detected {len(faces)} face(s). Running predictions...")

                results = []
                for (x, y, w, h) in faces:
                    face_roi = image[y:y+h, x:x+w]
                    face_resized = cv2.resize(face_roi, img_size)
                    face_array = img_to_array(face_resized) / 255.0
                    face_array = np.expand_dims(face_array, axis=0)

                    # Skin prediction
                    skin_pred = skin_model.predict(face_array)[0]
                    skin_label = class_labels[np.argmax(skin_pred)]
                    skin_confidence = np.max(skin_pred) * 100

                    # Age prediction
                    predicted_age = age_model.predict(face_array)[0][0]

                    # Draw on image
                    cv2.rectangle(image, (x, y), (x + w, y + h), (140, 82, 255), 2)  # lavender tone
                    text = f"{skin_label} ({skin_confidence:.1f}%) | Age: {predicted_age:.1f}"
                    cv2.putText(image, text, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 82, 255), 2)

                    results.append((skin_label, skin_confidence, predicted_age))

                # Display results
                st.subheader("🩺 Analysis Results")
                st.image(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    caption="Processed Image with Predictions",
                    use_container_width=True
                )

                # Display each face result
                for i, (skin_label, skin_confidence, predicted_age) in enumerate(results):
                    st.markdown(f"### 👤 Face {i+1} Results")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🧴 Skin Condition", skin_label)
                    with col2:
                        st.metric("💯 Confidence", f"{skin_confidence:.2f}%")
                    with col3:
                        st.metric("👶 Estimated Age", f"{predicted_age:.1f} years")

                # ===============================
                # DOWNLOAD BUTTON SECTION
                # ===============================
                result_data = []
                for i, (skin_label, skin_confidence, predicted_age) in enumerate(results):
                    result_data.append({
                        "Face #": i + 1,
                        "Skin Condition": skin_label,
                        "Confidence (%)": f"{skin_confidence:.2f}",
                        "Estimated Age (years)": f"{predicted_age:.1f}"
                    })

                if len(result_data) > 0:
                    df = pd.DataFrame(result_data)
                    csv = df.to_csv(index=False).encode('utf-8')

                    st.download_button(
                        label="⬇️ Download Results as CSV",
                        data=csv,
                        file_name=f"dermal_scan_results_{int(time.time())}.csv",
                        mime="text/csv"
                    )

# ===============================
# SIDEBAR INFO
# ===============================
with st.sidebar:
    st.header("ℹ️ About Dermal Scan")
    st.write("""
    **Dermal Scan** uses deep learning to analyze facial skin and predict:
    - Skin conditions like wrinkles or dark spots  
    - Estimated biological age
    """)
    st.write("Developed using **TensorFlow, Keras, and OpenCV** 🧠")
    st.info("For best results, use a well-lit image with your face clearly visible.")
