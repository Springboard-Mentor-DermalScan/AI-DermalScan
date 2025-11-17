import os
import time
import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from backend import detect_predict_image, export_results_zip, OUTPUTS_DIR

# Streamlit page config
st.set_page_config(page_title="💆 AI Dermal Skin Analyzer - Rasool Baig", layout="wide")

# Basic styling
st.markdown("""
<style>
body {background:#0f1724; color:#e6eef3;}
h1 {color:#64d8cb;}
.sidebar .sidebar-content {background:linear-gradient(180deg,#0b1220,#111827);}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center'>💆 AI Dermal Skin Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center'>Upload a clear face photo → get condition, confidence & estimated age</p>", unsafe_allow_html=True)

# Ensure outputs
os.makedirs("uploads", exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Layout
col_left, col_right = st.columns([1, 1])

with st.sidebar:
    st.write("## Controls")
    max_detect_dim = st.slider("Max detection image size (pixels)", min_value=400, max_value=1200, value=800, step=50)
    conf_th = st.slider("Face detection confidence threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
    st.write("---")
    st.info("Outputs saved into `outputs/` (annotated images, predictions_log.csv).")

uploaded = st.file_uploader("📤 Upload image (jpg/png)", type=["jpg","jpeg","png"])
if uploaded is None:
    st.info("Upload an image to analyze. Best results: frontal, clear faces.")
    st.stop()

# Save uploaded file
save_path = os.path.join("uploads", uploaded.name)
with open(save_path, "wb") as f:
    f.write(uploaded.getbuffer())

st.sidebar.write(f"Saved: {save_path}")
st.info("Processing (first run may load model and take ~2-4s)...")

# Run backend
try:
    t0 = time.time()
    annotated_path, df_results, total_time = detect_predict_image(
        save_path,
        target_size=(224,224),
        conf_threshold=conf_th,
        max_detect_dim=max_detect_dim
    )
    t1 = time.time()
    runtime = round(t1-t0,3)
except Exception as e:
    st.error(f"Processing failed: {e}")
    st.stop()

# Display images
col_left.subheader("Original")
col_left.image(Image.open(save_path), use_container_width=True)
col_right.subheader("Annotated / Predicted")
col_right.image(Image.open(annotated_path), use_container_width=True)

# Show predictions table
df_display = df_results.copy()
st.markdown("### 🧑 Detected Faces & Predictions")
st.dataframe(df_display, use_container_width=True)

# Charts
st.markdown("### Visual Summary")
fig1, ax1 = plt.subplots(figsize=(6,2.5))
ax1.bar(df_display["Face #"].astype(str), df_display["Confidence (%)"].astype(float))
ax1.set_ylabel("Confidence (%)")
ax1.set_xlabel("Face #")
ax1.set_title("Confidence per detected face")
st.pyplot(fig1)

fig2, ax2 = plt.subplots(figsize=(4,3))
class_counts = df_display["Predicted Class"].value_counts()
ax2.pie(class_counts, labels=class_counts.index, autopct="%1.1f%%", startangle=140)
ax2.set_title("Class distribution")
st.pyplot(fig2)

fig3, ax3 = plt.subplots(figsize=(6,2.5))
ax3.hist(df_display["Estimated Age"].astype(float), bins=5)
ax3.set_title("Age distribution (estimated)")
st.pyplot(fig3)

# Downloads
st.markdown("### Downloads & Export")
cols = st.columns(3)
with cols[0]:
    with open(annotated_path, "rb") as f:
        st.download_button("📸 Download annotated image", data=f, file_name=os.path.basename(annotated_path), mime="image/jpeg")
with cols[1]:
    csv_path = os.path.join(OUTPUTS_DIR, "predictions_log.csv")
    if os.path.exists(csv_path):
        with open(csv_path, "rb") as f:
            st.download_button("📊 Download full predictions CSV", data=f, file_name="predictions_log.csv", mime="text/csv")
with cols[2]:
    if st.button("🗜️ Export ZIP (annotated + summary CSV + JSON)"):
        try:
            zip_path = export_results_zip(save_path, annotated_path, df_display)
            with open(zip_path, "rb") as zf:
                st.download_button("⬇ Download result ZIP", data=zf, file_name=os.path.basename(zip_path), mime="application/zip")
        except Exception as e:
            st.error(f"Export failed: {e}")

st.success(f"Completed in {total_time} s (backend runtime: {runtime}s)")
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<small>Developed by <b>Rasool Baig</b></small>", unsafe_allow_html=True)
