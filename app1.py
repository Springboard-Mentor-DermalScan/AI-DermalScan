import streamlit as st
import numpy as np
from io import BytesIO, StringIO 
from PIL import Image
import os 
import pandas as pd
import uuid 
import backend1 as backend 
def convert_df_to_csv(df):
    """Converts a pandas DataFrame to a UTF-8 encoded CSV string."""
    return df.to_csv(index=False).encode('utf-8')
st.set_page_config(
    page_title="DermalScan AI",
    page_icon="✨",
    layout="wide"
)
st.title("✨ DermalScan AI Face Analysis")
st.markdown("Upload an image to detect faces and classify common skin conditions (clear face, darkspots, puffy eyes, wrinkles).")
st.markdown(f"**Model Status:** {'✅ Loaded' if backend.model is not None else '❌ ERROR: Model not found!'}")
st.markdown("---")
uploaded_file = st.file_uploader(
    "Choose an image (JPG, JPEG, or PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    unique_filename = f"temp_upload_{uuid.uuid4()}_{uploaded_file.name}"
    temp_file_path = os.path.join("./", unique_filename) 

    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.subheader("Uploaded Image")
    st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)

    
    if st.button("🔬 **Analyze Face**", type="primary"):
        
        if backend.model is None:
            st.error("Cannot run analysis because the AI model was not loaded in the backend.")
        else:
            
            annotated_image_path = None
            try:
                with st.spinner('Analyzing image using backend logic...'):
                   
                    analysis_output = backend.predict_skin_condition(temp_file_path)
                
                results_list = analysis_output["results"]
                annotated_image_path = analysis_output["annotated_image"]
                latency = analysis_output["total_time"]
                face_count = len(results_list)

                st.subheader("🔍 Analysis Results")
                
                if face_count == 0:
                    st.warning("No faces were detected in the image.")
                    st.image(uploaded_file, caption="No annotation available", use_container_width=True)
                else:
                    results_df = pd.DataFrame(results_list)
                    results_df['Original Image File'] = uploaded_file.name 
                    results_df['Annotated Image Path'] = annotated_image_path
                    annotated_image = Image.open(annotated_image_path)
                    st.image(annotated_image, caption="Annotated Image", use_container_width=True)
                    download_col1, download_col2 = st.columns(2)

                    csv = convert_df_to_csv(results_df)
                    download_col1.download_button(
                        label="⬇️ Download Analysis (CSV)",
                        data=csv,
                        file_name=f"{uploaded_file.name.split('.')[0]}_analysis.csv",
                        mime='text/csv',
                        key='analysis_csv_download' 
                    )
                    
                    with open(annotated_image_path, "rb") as file:
                        btn = download_col2.download_button(
                                label="⬇️ Download Annotated Image",
                                data=file.read(),
                                file_name=f"{uploaded_file.name.split('.')[0]}_annotated.jpg", # Or .png if you prefer
                                mime="image/jpeg", 
                                key="annotated_image_download"
                            )
                    st.markdown("---")
                    col1, col2 = st.columns(2) 
                    col1.metric("Faces Detected", value=face_count)
                    col2.metric("Processing Time", value=f"{latency} s")
                    st.markdown("---")
                    st.subheader("Detailed Face Data")
                    st.dataframe(results_df, use_container_width=True)
                    
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
                
            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                if annotated_image_path and os.path.exists(annotated_image_path):
                     os.remove(annotated_image_path)
                 
st.markdown("---")
st.caption("Backend logic separated into `backend1.py` for clean architecture.")