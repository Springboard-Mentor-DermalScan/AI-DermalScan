import streamlit as st
from PIL import Image
import io
from backend import predict_aging_signs

st.set_page_config(page_title="DermalScan – Facial Aging Detection", layout="wide")

st.title("🌿 DermalScan – Facial Aging Detection System")
st.markdown("""
This AI-powered system detects **wrinkles, dark spots, puffy eyes,** and **clear skin**  
using a deep learning model (InceptionV3). Upload your image to visualize detected aging signs.
""")

uploaded_file = st.file_uploader("📤 Upload a facial image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    with st.spinner("⏳ Analyzing image... Please wait"):
        image_bytes = uploaded_file.read()
        annotated_path, csv_path, results_df = predict_aging_signs(image_bytes)

    with col2:
        st.image(annotated_path, caption="🔍 Annotated Results", use_container_width=True)
        st.dataframe(results_df, use_container_width=True)

        st.download_button("⬇️ Download Annotated Image", open(annotated_path, "rb"), file_name="annotated_result.jpg")
        st.download_button("⬇️ Download Predictions CSV", open(csv_path, "rb"), file_name="predictions_log.csv")

else:
    st.info("Please upload an image to start the analysis.")

st.markdown("---")
st.caption("© 2025 DermalScan | Developed for detecting and analyzing facial aging patterns.")
