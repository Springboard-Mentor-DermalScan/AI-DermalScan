import streamlit as st
import tempfile
import io
import pandas as pd
from PIL import Image
from backend import analyze_face

# -------------------- Page Configuration --------------------
st.set_page_config(page_title="DermalScan: AI Facial Skin Aging Detection", layout="wide")

# -------------------- Header --------------------
st.title(" DermalScan: AI Facial Skin Aging Detection App")

st.markdown("""
Welcome to **DermalScan**, an AI-powered facial skin assessment platform built using 
deep learning and computer vision.  
This intelligent system detects facial regions, classifies **skin types** such as *clear face*, 
*dark spots*, *puffy eyes*, and *wrinkles*, and estimates an **approximate age range** 
based on detected features.

> 💡 Powered by DenseNet deep learning model & OpenCV face detection for high-accuracy skin analysis.
""")

st.divider()

# -------------------- Upload Section --------------------
st.header("📤 Upload a Facial Image")
uploaded_file = st.file_uploader("Select an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    # Run backend analysis
    with st.spinner("🔍 Analyzing facial features... (Approx. 5 seconds)"):
        annotated_image, results_df, latency, face_count = analyze_face(temp_path)

    # -------------------- Display Results --------------------
    st.divider()
    st.header("🧠 Facial Analysis Result")

    col1, col2 = st.columns(2)

    with col1:
        st.image(uploaded_file, caption="Original Uploaded Image", use_container_width=True)

    with col2:
        st.image(annotated_image, caption="Predicted (Annotated) Output", use_container_width=True)

    # Summary Section
    st.subheader("📊 Analysis Summary")
    st.write(f"**Detected Faces:** {face_count}")
    st.write(f"**Processing Time:** {latency} seconds per image")

    if results_df is not None and not results_df.empty:
        st.dataframe(results_df, use_container_width=True)

        # -------------------- Download Buttons --------------------
        csv = results_df.to_csv(index=False).encode("utf-8")
        img_bytes = io.BytesIO()
        annotated_img = Image.fromarray(annotated_image)
        annotated_img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        col3, col4 = st.columns(2)
        with col3:
            st.download_button(
                label="📄 Download Analysis Results (CSV)",
                data=csv,
                file_name="dermalscan_analysis_results.csv",
                mime="text/csv"
            )

        with col4:
            st.download_button(
                label="🖼️ Download Annotated Image (PNG)",
                data=img_bytes,
                file_name="dermalscan_predicted_output.png",
                mime="image/png"
            )

    else:
        st.warning("⚠️ No faces detected. Please upload a clear image of a face.")

else:
    st.info("📁 Upload an image above to begin your AI-powered facial skin assessment.")

st.divider()

# -------------------- Footer --------------------
st.markdown("""
### 💬 About DermalScan  
**DermalScan** is a deep learning–based application that integrates:
- 🧠 **DenseNet Model** for skin condition classification  
- 👁️ **OpenCV Haar Cascade** for face detection  
- ⏱️ Real-time inference (<5 seconds per image)  
- 📊 Exportable insights (CSV + Annotated Image)

This project aims to assist dermatological research and skincare diagnostics by automating 
the detection of skin conditions and early aging indicators using AI.
""")
