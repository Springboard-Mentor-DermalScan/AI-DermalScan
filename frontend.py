import streamlit as st
from PIL import Image
import tempfile
import os
import pandas as pd
import time
import numpy as np
from backend import detect_predict_face
import traceback
from io import BytesIO

# -------------------------------
# Streamlit Configuration
# -------------------------------
st.set_page_config(layout="wide", page_title="AI Dermal Scan")

st.write("## AI Dermal Scan")
st.markdown("*From pixels to precision — intelligent skin analysis reimagined*")
st.write(
    ":microscope: Upload a facial image to analyze skin conditions and predict age. "
    "This app uses AI for intelligent skin analysis. Full results can be viewed below."
)
st.sidebar.write("## Upload and download :gear:")

# -------------------------------
# Settings
# -------------------------------
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_IMAGE_SIZE = 2000  # Max image dimension
OUTPUT_FILE = "session_logs.csv"

# -------------------------------
# Session Log Setup
# -------------------------------
if not os.path.exists(OUTPUT_FILE):
    df = pd.DataFrame(columns=[
        "Image Name", "X", "Y", "Width", "Height",
        "Predicted Condition", "Confidence", "Predicted Age", "Processing Time (s)"
    ])
    df.to_csv(OUTPUT_FILE, index=False)

if "session_log" not in st.session_state:
    st.session_state.session_log = pd.DataFrame(columns=[
        "Image Name", "X", "Y", "Width", "Height",
        "Predicted Condition", "Confidence", "Predicted Age", "Processing Time (s)"
    ])

# -------------------------------
# Helper Functions
# -------------------------------
def convert_image(img):
    """Convert image (NumPy array or PIL) to downloadable PNG bytes."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype('uint8'))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def resize_image(image, max_size):
    """Resize image while maintaining aspect ratio."""
    width, height = image.size
    if width <= max_size and height <= max_size:
        return image

    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))

    return image.resize((new_width, new_height), Image.LANCZOS)


@st.cache_data
def process_image(image_bytes):
    """Process image with caching to avoid redundant processing."""
    try:
        image = Image.open(BytesIO(image_bytes))
        resized = resize_image(image, MAX_IMAGE_SIZE)

        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, "temp_image.png")
        resized.save(temp_path)

        start_time = time.time()
        result = detect_predict_face(temp_path)
        end_time = time.time()
        processing_time = round(end_time - start_time, 2)

        if len(result) == 5:
            result_img, condition, age, confidence, bbox = result
            x, y, w, h = bbox
        else:
            result_img, condition, age, confidence = result
            x, y, w, h = "-", "-", "-", "-"

        os.remove(temp_path)
        os.rmdir(temp_dir)

        return image, result_img, condition, age, confidence, x, y, w, h, processing_time

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None, None, None, None, None, None, None, None, None


def analyze_image(upload):
    """Handles full flow of analyzing and displaying results for an image."""
    try:
        start_time = time.time()
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()

        status_text.text("Loading image...")
        progress_bar.progress(10)

        if isinstance(upload, str):
            with open(upload, "rb") as f:
                image_bytes = f.read()
        else:
            image_bytes = upload.getvalue()

        status_text.text("Analyzing skin...")
        progress_bar.progress(30)

        image, result_img, condition, age, confidence, x, y, w, h, processing_time = process_image(image_bytes)
        if image is None:
            return

        progress_bar.progress(80)
        status_text.text("Displaying results...")

        col1.write("Original Image :camera:")
        col1.image(image)

        col2.write("Analysis Output :microscope:")
        col2.image(result_img)

        with col2:
            st.markdown("### Prediction Details")
            details_df = pd.DataFrame({
                "Metric": ["Skin Condition", "Predicted Age", "Confidence", "Processing Time", "Bounding Box"],
                "Value": [
                    condition if condition else "-",
                    f"{age} yrs" if age else "-",
                    f"{confidence:.2f}%" if confidence else "-",
                    f"{processing_time} sec",
                    f"Top-left ({x}, {y}), Size: {w}×{h}" if x != "-" else "-"
                ]
            })
            st.table(details_df)

        st.sidebar.download_button(
            "Download analysis image",
            convert_image(result_img),
            "analysis.png",
            "image/png"
        )

        # Validate coordinate values for logging
        if all(isinstance(val, (int, float)) for val in [x, y, w, h]) and x != "-":
            x, y, w, h = map(int, [x, y, w, h])
        else:
            x, y, w, h = "-", "-", "-", "-"

        log_entry = pd.DataFrame([{
            "Image Name": upload.name if hasattr(upload, 'name') else upload,
            "X": x, "Y": y, "Width": w, "Height": h,
            "Predicted Condition": condition,
            "Confidence": confidence,
            "Predicted Age": age,
            "Processing Time (s)": processing_time
        }])

        st.session_state.session_log = pd.concat(
            [st.session_state.session_log, log_entry], ignore_index=True
        )
        st.session_state.session_log.to_csv(OUTPUT_FILE, index=False)

        progress_bar.progress(100)
        total_time = time.time() - start_time
        status_text.text(f"Completed in {total_time:.2f} seconds")

    except Exception:
        st.error("An error occurred during analysis.")
        print(traceback.format_exc())

# -------------------------------
# UI Layout
# -------------------------------
col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload a facial image", type=["png", "jpg", "jpeg"])

with st.sidebar.expander("ℹ️ Image Guidelines"):
    st.write("""
    - Maximum file size: 10MB  
    - Large images are automatically resized  
    - Supported formats: PNG, JPG, JPEG  
    - Best results with clear, front-facing facial images
    """)

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error(f"The uploaded file is too large. Please upload an image smaller than {MAX_FILE_SIZE/1024/1024:.1f}MB.")
    else:
        analyze_image(upload=my_upload)
        st.markdown("## 📊 Session Log")
        st.dataframe(st.session_state.session_log, use_container_width=True)
else:
    default_images = ["./default_face.jpg"]
    for img_path in default_images:
        if os.path.exists(img_path):
            analyze_image(img_path)
            st.markdown("## 📊 Session Log")
            st.dataframe(st.session_state.session_log, use_container_width=True)
            break
    else:
        st.info("Please upload a facial image to get started!")
        st.markdown("## 📊 Session Log")
        st.dataframe(st.session_state.session_log, use_container_width=True)

# -------------------------------
# Footer
# -------------------------------
st.markdown("""
<div style="text-align: center; margin-top: 60px; font-size: 14px; color: #333; padding-top: 15px; border-top: 1px solid rgba(0,0,0,0.1);">
    ✨ Developed by <b>Pakhi Sharma</b> | Powered by <i>AI Dermal Scan</i> 💆‍♀️
</div>
""", unsafe_allow_html=True)
