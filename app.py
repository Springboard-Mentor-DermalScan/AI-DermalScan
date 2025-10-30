# ==========================================================
# app.py — DermalScan Frontend (Tech-Lab Themed + Scan Animation)
# ==========================================================

import streamlit as st
import numpy as np
import cv2
import pandas as pd
import time
from DermalScan import process_image

# ----------------------------------------------------------
# 🌐 PAGE CONFIGURATION — must be first Streamlit command
# ----------------------------------------------------------


# ----------------------------------------------------------
# 🧠 CUSTOM STYLING (Dark Futuristic Tech-Lab Theme)
# ----------------------------------------------------------
st.markdown("""
    <style>
    body {
        background-color: #0E1117;
        color: #E0E0E0;
    }
    .main {
        background-color: #0E1117;
        color: #E0E0E0;
    }
    h1, h2, h3, h4 {
        color: #00FFFF;
        text-align: center;
        text-shadow: 0px 0px 10px #00FFFF;
    }
    .stButton button {
        background-color: #0E76A8;
        color: white;
        border-radius: 10px;
        border: none;
        font-weight: 600;
        box-shadow: 0 0 10px #00FFFF;
        transition: 0.3s;
    }
    .stButton button:hover {
        background-color: #00FFFF;
        color: black;
        box-shadow: 0 0 20px #00FFFF;
    }
    .stProgress > div > div > div > div {
        background-color: #00FFFF;
    }
    footer {visibility: hidden;}
    /* 🔬 Scanning animation overlay */
    .scan-container {
        position: relative;
        display: inline-block;
    }
    .scan-line {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00FFFF, transparent);
        animation: scanMove 2s linear infinite;
    }
    @keyframes scanMove {
        0% {top: 0;}
        100% {top: 100%;}
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# 🧠 HEADER SECTION
# ----------------------------------------------------------
st.markdown("<h1>🧠 DermalScan AI Lab</h1>", unsafe_allow_html=True)
st.markdown("<h4>AI-Powered Facial Skin & Age Analysis</h4>", unsafe_allow_html=True)
st.markdown("---")

with st.expander("ℹ️ About DermalScan", expanded=True):
    st.markdown("""
    **DermalScan** uses AI & Deep Learning (DenseNet121 + MTCNN)  
    to detect skin features like *wrinkles*, *dark spots*, and *puffy eyes*,  
    while estimating your **biological age**.

    ⚙️ Built with TensorFlow and OpenCV  
    ⚠️ For research use only — not a medical diagnostic tool.
    """, unsafe_allow_html=True)

st.markdown("---")

# ----------------------------------------------------------
# 📸 FILE UPLOADER
# ----------------------------------------------------------
st.markdown("<h3>📡 Upload Image for AI Scan</h3>", unsafe_allow_html=True)
st.markdown("<p style='color:#00FFFF;'>🛰️ Upload a clear, front-facing image to start scanning...</p>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

# ----------------------------------------------------------
# ⚙️ PROCESSING SECTION
# ----------------------------------------------------------
if uploaded_file:
    try:
        # Read and decode the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        filename = uploaded_file.name

        # Display uploaded image with scanning overlay
        st.markdown("<h4>📷 Input Image</h4>", unsafe_allow_html=True)
        st.markdown("""
        <div class='scan-container'>
            <div class='scan-line'></div>
        </div>
        """, unsafe_allow_html=True)
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.markdown("---")

        # Animated scanning progress
        st.markdown("<h4>🚀 Initiating AI Scan Sequence...</h4>", unsafe_allow_html=True)
        progress_text = "Scanning facial regions and analyzing dermal features..."
        progress_bar = st.progress(0, text=progress_text)
        for percent in range(100):
            time.sleep(0.02)
            progress_bar.progress(percent + 1, text=progress_text)
        progress_bar.empty()

        # Process with backend
        start = time.time()
        annotated, results_df = process_image(image, filename)
        latency = time.time() - start
        st.success(f"✅ Scan Complete — Processing Time: {latency:.2f} seconds")

        # ----------------------------------------------------------
        # 🧬 DISPLAY RESULTS
        # ----------------------------------------------------------
        st.markdown("---")
        st.markdown("<h3>🧬 AI Insights</h3>", unsafe_allow_html=True)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                 caption="🔬 AI Annotated Image",
                 use_container_width=True)

        if results_df is not None and not results_df.empty:
            st.subheader("📊 Prediction Results")
            st.dataframe(results_df, use_container_width=True)

            if "Estimated Age" in results_df.columns:
                avg_age = results_df["Estimated Age"].mean()
                st.info(f"🧠 **Estimated Biological Age:** {avg_age:.1f} years")

            # Download buttons
            csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv,
                file_name=f"results_{filename}.csv",
                mime="text/csv",
            )
            retval, buffer = cv2.imencode(".jpg", annotated)
            st.download_button(
                label="📷 Download Annotated Image",
                data=buffer.tobytes(),
                file_name=f"annotated_{filename}",
                mime="image/jpeg",
            )
        else:
            st.warning("⚠️ No detectable features found. Try a clearer image.")

    except Exception as e:
        st.error(f"❌ Error during AI processing: {str(e)}")

else:
    st.info("👆 Upload a face image to start the AI scan.")

# ----------------------------------------------------------
# 🧩 FOOTER
# ----------------------------------------------------------
st.markdown("---")
st.markdown("""
    <div style='text-align:center; color:#888;'>
    <p>⚙️ DermalScan v2.0 | Powered by DenseNet121 + MTCNN | © 2025 Shreya</p>
    </div>
""", unsafe_allow_html=True)
