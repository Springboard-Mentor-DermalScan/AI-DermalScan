import streamlit as st
from PIL import Image
import tempfile
import os
import pandas as pd
import time
import numpy as np  # Added for handling numpy arrays
from backend import detect_predict_face
import traceback
from io import BytesIO

st.set_page_config(layout="wide", page_title="AI Dermal Scan")

st.write("## AI Dermal Scan")
st.markdown("*From pixels to precision — intelligent skin analysis reimagined*")
st.write(
    ":microscope: Upload a facial image to analyze skin conditions and predict age. This app uses AI for intelligent skin analysis. Full results can be viewed below."
)
st.sidebar.write("## Upload and download :gear:")

# Increased file size limit
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Max dimensions for processing
MAX_IMAGE_SIZE = 2000  # pixels

# Setup for session log CSV and session state (from original)
OUTPUT_FILE = "session_logs.csv"
if not os.path.exists(OUTPUT_FILE):
    df = pd.DataFrame(columns=["Image Name", "X", "Y", "Width", "Height",
                               "Predicted Condition", "Confidence", "Predicted Age", "Processing Time (s)"])
    df.to_csv(OUTPUT_FILE, index=False)

if "session_log" not in st.session_state:
    st.session_state.session_log = pd.DataFrame(columns=[
        "Image Name", "X", "Y", "Width", "Height",
        "Predicted Condition", "Confidence", "Predicted Age", "Processing Time (s)"
    ])

# Download the result image (updated to handle numpy arrays)
def convert_image(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype('uint8'))
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

# Resize image while maintaining aspect ratio
def resize_image(image, max_size):
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
    """Process image with caching to avoid redundant processing"""
    try:
        image = Image.open(BytesIO(image_bytes))
        # Resize large images to prevent memory issues
        resized = resize_image(image, MAX_IMAGE_SIZE)
        # Save resized image to temp file for backend processing
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, "temp_image.png")
        resized.save(temp_path)
        
        # Call backend logic
        start_time = time.time()
        result = detect_predict_face(temp_path)
        end_time = time.time()
        processing_time = round(end_time - start_time, 2)
        
        # Parse result (keeping original logic)
        if len(result) == 3:
            result_img, condition, age = result
            confidence, x, y, w, h = 0.95, 120, 180, 360, 380
        elif len(result) == 5:
            result_img, condition, age, confidence, bbox = result
            x, y, w, h = bbox
        else:
            result_img, condition, age = result
            confidence, x, y, w, h = 0.9, "-", "-", "-", "-"
        
        # Clean up temp file
        os.remove(temp_path)
        os.rmdir(temp_dir)
        
        return image, result_img, condition, age, confidence, x, y, w, h, processing_time
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None, None, None, None, None, None, None, None, None

def analyze_image(upload):
    try:
        start_time = time.time()
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        status_text.text("Loading image...")
        progress_bar.progress(10)
        
        # Read image bytes
        if isinstance(upload, str):
            # Default image path
            if not os.path.exists(upload):
                st.error(f"Default image not found at path: {upload}")
                return
            with open(upload, "rb") as f:
                image_bytes = f.read()
        else:
            # Uploaded file
            image_bytes = upload.getvalue()
        
        status_text.text("Analyzing skin...")
        progress_bar.progress(30)
        
        # Process image (using cache if available)
        image, result_img, condition, age, confidence, x, y, w, h, processing_time = process_image(image_bytes)
        if image is None:
            return
        
        progress_bar.progress(80)
        status_text.text("Displaying results...")
        
        # Display images
        col1.write("Original Image :camera:")
        col1.image(image)
        
        col2.write("Analysis Output :microscope:")
        col2.image(result_img)
        
        # Display prediction info in table format
        with col2:
            st.markdown("### Prediction Details")
            details_df = pd.DataFrame({
                "Metric": ["Skin Condition", "Predicted Age", "Confidence", "Processing Time", "Bounding Box"],
                "Value": [condition, f"{age} yrs", confidence, f"{processing_time} sec", f"X={x}, Y={y}, W={w}, H={h}" if x != "-" else "-"]
            })
            st.table(details_df)
        
        # Prepare download button
        st.sidebar.markdown("\n")
        st.sidebar.download_button(
            "Download analysis image", 
            convert_image(result_img), 
            "analysis.png", 
            "image/png"
        )
        
        # Session log (adapted from original)
        log_entry = pd.DataFrame([{
            "Image Name": upload.name if hasattr(upload, 'name') else upload,
            "X": x, "Y": y, "Width": w, "Height": h,
            "Predicted Condition": condition,
            "Confidence": confidence,
            "Predicted Age": age,
            "Processing Time (s)": processing_time
        }])
        
        st.session_state.session_log = pd.concat([st.session_state.session_log, log_entry], ignore_index=True)
        st.session_state.session_log.to_csv(OUTPUT_FILE, index=False)
        
        progress_bar.progress(100)
        total_time = time.time() - start_time
        status_text.text(f"Completed in {total_time:.2f} seconds")
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.sidebar.error("Failed to analyze image")
        # Log the full error for debugging
        print(f"Error in analyze_image: {traceback.format_exc()}")

# UI Layout
col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload a facial image", type=["png", "jpg", "jpeg"])

# Information about limitations
with st.sidebar.expander("ℹ️ Image Guidelines"):
    st.write("""
    - Maximum file size: 10MB
    - Large images will be automatically resized
    - Supported formats: PNG, JPG, JPEG
    - Processing time depends on image size
    - Best results with clear facial images
    """)

# Process the image
if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error(f"The uploaded file is too large. Please upload an image smaller than {MAX_FILE_SIZE/1024/1024:.1f}MB.")
    else:
        analyze_image(upload=my_upload)
        
        # Display log table after analysis (added from original code)
        st.markdown("## 📊 Session Log")
        st.dataframe(st.session_state.session_log, use_container_width=True)
else:
    # Try default images if available (optional, can remove if not needed)
    default_images = ["./default_face.jpg"]  # Add a default image path if desired
    for img_path in default_images:
        if os.path.exists(img_path):
            analyze_image(img_path)
            # Display log table after analysis
            st.markdown("## 📊 Session Log")
            st.dataframe(st.session_state.session_log, use_container_width=True)
            break
    else:
        st.info("Please upload a facial image to get started!")
        # Display empty log table if no processing has occurred
        st.markdown("## 📊 Session Log")
        st.dataframe(st.session_state.session_log, use_container_width=True)

# Footer (from original)
st.markdown("""
<div style="text-align: center; margin-top: 60px; font-size: 14px; color: #333; padding-top: 15px; border-top: 1px solid rgba(0,0,0,0.1);">
    ✨ Developed by <b>Pakhi Sharma</b> | Powered by <i>AI Dermal Scan</i> 💆‍♀️
</div>
""", unsafe_allow_html=True)



