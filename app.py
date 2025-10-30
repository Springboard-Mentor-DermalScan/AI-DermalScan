import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io
import pandas as pd
import base64
from backend import process_and_predict, log_to_csv  

# Streamlit Page Config 
st.set_page_config(
    page_title="AI DermalScan – Facial Aging Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling 
st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            color: #fff;
        }
        [data-testid="stSidebar"] {
            background: #1b1f24;
        }
        h1, h2, h3 {
            color: #00e6ac !important;
            font-weight: 700 !important;
        }
        .stButton>button {
            background-color: #00e6ac;
            color: #000;
            font-weight: bold;
            border-radius: 10px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #00ffcc;
            color: #000;
        }
        .stDataFrame {
            border: 1px solid #00e6ac !important;
            border-radius: 10px;
        }
        hr {
            border: 1px solid #00e6ac;
            margin: 1.5em 0;
        }
        img {
            border-radius: 10px !important;
        }
    </style>
""", unsafe_allow_html=True)

# Helper Functions for Downloads 
def get_image_download_link(img_np, filename, text):
    rgb_img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<a href="data:file/png;base64,{b64}" download="{filename}" style="color:#00ffcc;">{text}</a>'

def get_csv_download_link(df, filename, text):
    csv = df.to_csv(index=False).encode()
    b64 = base64.b64encode(csv).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="color:#00ffcc;">{text}</a>'

# Sidebar Upload 
st.sidebar.title("⚙️ Configuration Panel")
st.sidebar.markdown("### Upload an Image for Analysis")

uploaded_file = st.sidebar.file_uploader(
    "Choose a facial image (.jpg, .jpeg, .png)",
    type=["jpg", "jpeg", "png"]
)

# Main Page Content 
st.title("💎 AI DermalScan")
st.markdown("### Advanced Facial Aging Sign Detection & Age Estimation")
st.write("Upload a facial image to detect **skin aging signs** such as wrinkles, dark spots, or puffy eyes, "
         "and receive an estimated age range using the **DenseNet121** deep learning model.")
st.markdown("---")

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_np_bgr = cv2.imdecode(file_bytes, 1)

    st.subheader("📸 Uploaded Image Preview")
    st.image(img_np_bgr, channels="BGR", width=600, caption=uploaded_file.name)
    st.markdown("---")

    st.subheader("🔍 Analysis Results")

    with st.spinner("Analyzing image using AI model... ⏳"):
        try:
            annotated_img, results_df, latency = process_and_predict(img_np_bgr, uploaded_file.name)
            st.info(f"**🕒 Processing Time:** {latency:.2f} seconds")

            if not results_df.empty:
                st.success("✅ Analysis Complete — Faces Detected and Classified Successfully")

                st.image(annotated_img, channels="BGR", width=600, caption="AI Annotated Output")

                st.subheader("📊 Prediction Summary")
                st.dataframe(results_df)

                # Download Options
                st.markdown("---")
                st.subheader("📥 Export Results")

                col1, col2 = st.columns(2)
                with col1:
                    img_filename = f"AI_DermalScan_{uploaded_file.name.split('.')[0]}_annotated.png"
                    st.markdown(get_image_download_link(annotated_img, img_filename, "⬇ Download Annotated Image (PNG)"),
                                unsafe_allow_html=True)

                with col2:
                    csv_filename = f"AI_DermalScan_{uploaded_file.name.split('.')[0]}_predictions.csv"
                    st.markdown(get_csv_download_link(results_df, csv_filename, "⬇ Download Prediction Data (CSV)"),
                                unsafe_allow_html=True)

                # Logging 
                log_to_csv(results_df)
                st.sidebar.success("✅ Predictions logged to prediction_log.csv")

            else:
                st.warning("⚠️ No face was detected in the uploaded image.")

        except ConnectionError as e:
            st.error(f"❌ Model Loading Error: {e}")
        except Exception as e:
            st.error(f"⚠️ An unexpected error occurred: {e}")
else:
    st.info("👈 Please upload a facial image from the sidebar to begin analysis.")
