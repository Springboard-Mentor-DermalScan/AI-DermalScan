import streamlit as st
import os
import time
import pandas as pd
import cv2
from module6_backend_inference import predict_skin_condition
st.set_page_config(page_title="DermalSkin Analyzer", page_icon=" ", layout="wide")
st.markdown("""
<style>
div[data-testid="metric-container"] {
    font-size: 0.85rem !important;
}
div[data-testid="stMetricLabel"] p {
    font-size: 0.75rem !important;
    color: #666666 !important;
}
div[data-testid="stMetricValue"] {
    font-size: 0.95rem !important;
}
h3, h4 {
    font-size: 1rem !important;
}
</style>
""", unsafe_allow_html=True)
st.title(" DermalSkin Analyzer")
st.markdown("### Upload a clear face image to detect **face type(s)** and **estimated age(s)**.")
uploaded_file = st.file_uploader("📷 Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img_path = "uploaded_image.jpg"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_path, caption="🖼 Uploaded Image", use_container_width=True)
    with st.spinner(" Analyzing image... please wait..."):
        start_time = time.time()
        result_condition = predict_skin_condition(img_path)
        total_time = round(time.time() - start_time, 2)
    if not result_condition.get("results"):
        st.error(" No face detected. Please upload a clearer image.")
        st.stop()
    annotated_path = result_condition["annotated_image"]
    estimation = result_condition["estimation"]
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    dimensions = f"{width} × {height} px"
    with col2:
        st.success(" Prediction Complete!")
        st.image(annotated_path, caption="🩺 Final Prediction Result", use_container_width=True)
    st.markdown("---")
    st.markdown("###  **Prediction Summary**")
    rows = []
    for idx, face_result in enumerate(result_condition["results"], start=1):
        rows.append({
            "Face #": idx,
            "Face Type": face_result["Predicted Label"],
            "Confidence": f"{face_result['Confidence (%)']:.1f}%",
            "Estimated Age": f"{face_result['Estimated Age']} years",
            "Estimation Speed": estimation,
            "⏱ Total Time": f"{total_time} sec",
            "Image Dimensions": dimensions
        })
    summary_df = pd.DataFrame(rows)
    st.dataframe(summary_df, hide_index=True, use_container_width=True)
    csv_path = "multi_face_results.csv"
    summary_df.to_csv(csv_path, index=False)
    st.markdown("---")
    st.markdown("###  **Download Results**")
    col_down1, col_down2 = st.columns(2)
    with col_down1:
        st.download_button(
            label="⬇ Download Annotated Image",
            data=open(annotated_path, "rb").read(),
            file_name="DermalSkin_Result.jpg",
            mime="image/jpeg"
        )
    with col_down2:
        st.download_button(
            label="⬇ Download Prediction CSV",
            data=open(csv_path, "rb").read(),
            file_name="DermalSkin_Predictions.csv",
            mime="text/csv"
        )

else:
    st.info("ℹ Please upload a face image to start analysis.")
