import streamlit as st
import pandas as pd
from io import BytesIO
from PIL import Image
from backend import load_dermalscan_models, analyze_image_with_model

# --- Application Configuration ---
# Set the page to wide layout and use a custom title
st.set_page_config(
    page_title="DermalScan - Facial Aging Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Model Loading (Cached) ---
@st.cache_resource
def load_models():
    """Loads models once and caches them to improve performance."""
    # This calls the load function from the backend file
    return load_dermalscan_models()

dermalscan_model, face_cascade = load_models()

# Global check for models
if dermalscan_model is None or face_cascade is None:
    st.error("FATAL: Deep Learning or Face Detection models failed to load. Please check `backend.py` and ensure model files are present.")
    st.stop()


# Custom CSS for the dark theme look, main title, and footer
st.markdown("""
<style>
/* Streamlit's implicit dark theme already handles most background/text colors */

/* Style for the main title bar (mimicking the original design) */
.title-container {
    display: flex;
    align-items: center;
    margin-bottom: 20px;
}
.title-container .icon {
    font-size: 2.5rem;
    color: #4CAF50; /* Green icon color */
    margin-right: 15px;
}
.title-container h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0;
}

/* Style for the footer */
.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    padding: 10px;
    background-color: #0E1117; /* Match sidebar/dark background for a seamless look */
    color: #A3A3A3;
    text-align: center;
    font-size: 0.8rem;
    border-top: 1px solid #333333;
    z-index: 100;
}
</style>
""", unsafe_allow_html=True)

# --- Metric Helper Function ---
def get_feature_mean_confidence(df, feature_name):
    """Safely retrieves the mean confidence score for a specific feature."""
    # Check if the feature exists in the DataFrame
    if feature_name in df['Skin Type/Feature'].values:
        # Calculate the mean confidence for that feature
        return df.loc[df['Skin Type/Feature'] == feature_name, 'Confidence (%)'].mean()
    return None # Returns None if feature is not found

# --- CSV Conversion Function ---
def convert_df_to_csv(df):
    """Converts a DataFrame to a CSV string."""
    return df.to_csv(index=False).encode('utf-8')

# --- Header & Description ---

# Using st.markdown with HTML for the stylized title
st.markdown("""
<div class="title-container">
    <div class="icon">🌿</div>
    <h1>DermalScan - Facial Aging Detection System</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("""
This AI-powered system detects wrinkles, dark spots, puffy eyes, and clear skin 
using a deep learning model (DenseNet121). Upload your image to visualize detected aging signs.
""")

# --- Sidebar (Modifications Added) ---
st.sidebar.title("Analysis Configuration")
detection_sensitivity = st.sidebar.slider(
    "Detection Sensitivity",
    min_value=0.0,
    max_value=1.0,
    value=0.75,
    step=0.05,
    help="Higher sensitivity detects subtle signs but may increase false positives."
)
st.sidebar.markdown("---")
st.sidebar.info("Model: DenseNet121\nLast Updated: Oct 2025")

# --- Main Uploader and Display Logic ---
uploaded_file = st.file_uploader(
    "Upload a facial image",
    type=['jpg', 'jpeg', 'png'],
    accept_multiple_files=False
)

# Initialize a container for results to keep them structured
results_container = st.container()

if uploaded_file is not None:
    # --- Processing State ---
    
    # Display the uploaded image
    st.subheader("Input Image | Predicted Output")
    
    # Use PIL to open the image and display it (safer for different formats)
    try:
        # Read image data once
        image_data = uploaded_file.read()
        image = Image.open(BytesIO(image_data))
    except Exception as e:
        st.error(f"Error loading image: {e}")
        image = None
        
    if image is not None:
        
        with st.spinner('Analyzing facial patterns with Deep Learning...'):
            # Correct function call from backend.py with ALL five required arguments 
            # and ALL four returned values.
            annotated_image_png_bytes, summary, df_results, wrinkle_index_value = analyze_image_with_model(
                image, 
                uploaded_file.name, 
                detection_sensitivity, 
                dermalscan_model, 
                face_cascade
            )
            
        st.success('Analysis Complete!')
        
        # --- Display Annotated Image ---
        col_input, col_output = st.columns(2)
        
        with col_input:
            st.image(image, caption='Original Input Image', use_column_width=True)
        
        with col_output:
            # Display the annotated image from the bytes buffer
            annotated_image = Image.open(annotated_image_png_bytes)
            st.image(annotated_image, caption='Predicted Output Image', use_column_width=True)


        # --- Display Results ---
        with results_container:
            st.markdown("## Analysis Results")
            
            # 1. Metrics for quick overview
            col1, col2, col3, col4 = st.columns(4)
            
            # Get specific feature confidences using the helper function
            clarity_score = get_feature_mean_confidence(df_results, 'Clear Face') if get_feature_mean_confidence(df_results, 'Clear Face') is not None else 50.0
            darkspots_score = get_feature_mean_confidence(df_results, 'Darkspots')
            puffyeyes_score = get_feature_mean_confidence(df_results, 'Puffy Eyes')
            
            clarity_status = "Good 🟢" if clarity_score >= 70 else "Fair 🟡"
            
            col1.metric("Overall Skin Clarity", f"{clarity_score:.1f}%", clarity_status)
            col2.metric(
                "Wrinkle Index", 
                f"{wrinkle_index_value:.1f}%", 
                help="Average confidence of all 'Wrinkle' detections."
            )
            col3.metric(
                "Dark Spots Density", 
                f"{darkspots_score:.1f}%" if darkspots_score is not None else "N/A", 
                help="Average confidence of 'Darkspots' detections."
            )
            col4.metric("Detection Sensitivity Used", f"{detection_sensitivity*100:.0f}%")

            st.markdown("---")
            
            # 2. Detailed Summary
            st.subheader("Summary Report")
            st.info(summary)

            st.markdown("---")
            
            # 3. Detailed Data Table
            st.subheader("Feature Breakdown")
            
            # Row for the table and the download buttons
            col_table, col_download_csv, col_download_png = st.columns([0.6, 0.2, 0.2])
            
            with col_table:
                st.dataframe(
                    df_results, 
                    use_container_width=True, 
                    hide_index=True
                )
            
            # --- Download Button (CSV) ---
            csv = convert_df_to_csv(df_results)
            
            with col_download_csv:
                st.download_button(
                    label="Download Analysis Results (CSV)",
                    data=csv,
                    file_name='DermalScan_Analysis_Report.csv',
                    mime='text/csv',
                    key='download-csv',
                    help="Click to download the detailed feature breakdown as a CSV file."
                )

            # --- Download Button (PNG) ---
            with col_download_png:
                st.download_button(
                    label="Download Annotated Image (PNG)",
                    data=annotated_image_png_bytes.getvalue(),
                    file_name="dermalscan_annotated_output.png",
                    mime="image/png",
                    key='download-png',
                    help="Click to download the image with detected features highlighted."
                )


else:
    # --- Initial State Warning (Mimicking the image) ---
    with results_container:
        st.info("Please upload an image to start the analysis.")

# --- Footer (Mimicking the image) ---
st.markdown("""
<div class="footer">
    © 2025 DermalScan | Developed for detecting and analyzing facial aging patterns.
</div>
""", unsafe_allow_html=True)

# Padding at the bottom for the fixed footer
st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
