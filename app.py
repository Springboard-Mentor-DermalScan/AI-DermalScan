
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import os
import plotly.graph_objects as go
import time
import json
from datetime import datetime
import pandas as pd
import io

# Page configuration
st.set_page_config(
    page_title="DermalScan:AI_Facial Skin Aging Detection App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, beautiful UI
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header gradient */
    .header-gradient {
        background: linear-gradient(135deg, #89f7fe 0%, #66a6ff 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    
    .header-title {
        font-size: 3rem;
        font-weight: 800;
        color: white;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .header-tagline {
        font-size: 1.2rem;
        color: rgba(255,255,255,0.95);
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        font-size: 18px;
        font-weight: 600;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.6);
    }
    
    /* Card styling */
    .prediction-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin: 15px 0;
        border-left: 5px solid #667eea;
    }
    
    .upload-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin: 15px 0;
        text-align: center;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 10px 0;
        color: white;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
        font-weight: 500;
    }
    
    /* Success/Warning boxes */
    .success-box {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 15px;
        border-radius: 10px;
        color: #0d4d4d;
        font-weight: 600;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #ffa751 0%, #ffe259 100%);
        padding: 15px;
        border-radius: 10px;
        color: #663d00;
        font-weight: 600;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .error-box {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        font-weight: 600;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Instructions box */
    .instructions {
        background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        border-left: 5px solid #667eea;
    }
    
    .instructions h4 {
        color: #2d3748;
        margin-bottom: 10px;
        font-weight: 700;
    }
    
    .instructions ul {
        color: #4a5568;
        font-weight: 500;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* File uploader styling */
    .uploadedFile {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Constants
CLASS_NAMES = ['clear face', 'darkspots', 'puffy eyes', 'wrinkles']
MODEL_PATH = r"C:\Users\shanm\Downloads\dataset\clear face\resnet50_multitask_dermal_age.h5"
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

AGE_CALIBRATION_A = 0.9
AGE_CALIBRATION_B = -2

# Performance tracking
inference_times = []

@st.cache_resource
def load_models():
    """Load and cache the model and face cascade"""
    try:
        model = load_model(MODEL_PATH, compile=False)
        face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        return model, face_cascade
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def preprocess_face_for_model(face_img):
    """Preprocess face image for model prediction"""
    face_resized = cv2.resize(face_img, (224, 224))
    img_array = np.expand_dims(face_resized, axis=0).astype('float32')
    return preprocess_input(img_array)

def log_prediction(bbox, predictions, age, processing_time):
    """Log prediction to JSON file"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'bbox': bbox,
        'predictions': predictions,
        'age': age,
        'top_condition': max(predictions, key=predictions.get),
        'processing_time': round(processing_time, 3)
    }
    
    log_file = 'prediction_logs.json'
    try:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(log_entry)
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        st.warning(f"Could not save prediction log: {str(e)}")

def process_image(image, model, face_cascade):
    """Complete image processing pipeline with logging"""
    start_time = time.time()
    
    # Convert to OpenCV format
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Resize large images for performance (max 1920px)
    max_dimension = 1920
    height, width = img.shape[:2]
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    original_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Denoise
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # Multi-scale face detection
    all_faces = []
    faces1 = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(80, 80))
    all_faces.extend(faces1)
    
    faces2 = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=12, minSize=(100, 100))
    all_faces.extend(faces2)
    
    faces3 = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(90, 90))
    all_faces.extend(faces3)
    
    # Group duplicate detections
    def group_faces(faces, threshold=50):
        if len(faces) == 0:
            return []
        grouped = []
        faces_list = list(faces)
        while faces_list:
            current = faces_list.pop(0)
            x, y, w, h = current
            group = [current]
            remaining = []
            for face in faces_list:
                fx, fy, fw, fh = face
                dist = np.sqrt((x + w/2 - fx - fw/2)**2 + (y + h/2 - fy - fh/2)**2)
                if dist < threshold:
                    group.append(face)
                else:
                    remaining.append(face)
            faces_list = remaining
            avg_x = int(np.mean([f[0] for f in group]))
            avg_y = int(np.mean([f[1] for f in group]))
            avg_w = int(np.mean([f[2] for f in group]))
            avg_h = int(np.mean([f[3] for f in group]))
            grouped.append((avg_x, avg_y, avg_w, avg_h))
        return grouped
    
    faces = group_faces(all_faces)
    
    # Validate faces
    valid_faces = []
    img_height, img_width = img.shape[:2]
    
    for (x, y, w, h) in faces:
        if y + h/2 < img_height * 0.80:
            aspect_ratio = w / h
            if 0.7 <= aspect_ratio <= 1.4:
                if w > 60 and h > 60:
                    if x > img_width * 0.05 and x + w < img_width * 0.95:
                        valid_faces.append((x, y, w, h))
    
    faces = np.array(valid_faces) if valid_faces else np.array([])
    
    # Keep only largest face
    if len(faces) > 1:
        faces = sorted(faces, key=lambda face: face[2] * face[3], reverse=True)
        faces = [faces[0]]
    
    if len(faces) == 0:
        return {
            'status': 'error',
            'message': 'No faces detected in the image',
            'processing_time': time.time() - start_time
        }
    
    # Process the detected face
    x, y, w, h = faces[0]
    
    # Add padding
    padding = int(0.1 * max(w, h))
    x_pad = max(0, x - padding)
    y_pad = max(0, y - padding)
    w_pad = min(img.shape[1] - x_pad, w + 2 * padding)
    h_pad = min(img.shape[0] - y_pad, h + 2 * padding)
    
    face_crop = original_img[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
    
    # Ensemble predictions
    predictions_ensemble = []
    
    preds1 = model.predict(preprocess_face_for_model(face_crop), verbose=0)
    predictions_ensemble.append(preds1)
    
    brightened = cv2.convertScaleAbs(face_crop, alpha=1.15, beta=5)
    preds2 = model.predict(preprocess_face_for_model(brightened), verbose=0)
    predictions_ensemble.append(preds2)
    
    lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe_face = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe_face.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    preds3 = model.predict(preprocess_face_for_model(enhanced), verbose=0)
    predictions_ensemble.append(preds3)
    
    darkened = cv2.convertScaleAbs(face_crop, alpha=0.90, beta=-5)
    preds4 = model.predict(preprocess_face_for_model(darkened), verbose=0)
    predictions_ensemble.append(preds4)
    
    denoised = cv2.fastNlMeansDenoisingColored(face_crop, None, 10, 10, 7, 21)
    preds5 = model.predict(preprocess_face_for_model(denoised), verbose=0)
    predictions_ensemble.append(preds5)
    
    # Weighted average ensemble
    weights = [2.0, 1.5, 1.5, 1.0, 1.0]
    
    if isinstance(predictions_ensemble[0], list) and len(predictions_ensemble[0]) == 2:
        skin_preds = [p[0][0] for p in predictions_ensemble]
        skin_pred = np.average(skin_preds, axis=0, weights=weights)
        
        age_preds = [p[1][0][0] for p in predictions_ensemble]
        age_pred = np.average(age_preds, weights=weights)
        
        calibrated_age = int(round(AGE_CALIBRATION_A * age_pred + AGE_CALIBRATION_B))
    else:
        skin_preds = [p[0] if isinstance(p, list) else p[0] for p in predictions_ensemble]
        skin_pred = np.average(skin_preds, axis=0, weights=weights)
        calibrated_age = None
    
    predictions = {CLASS_NAMES[i]: float(skin_pred[i] * 100) for i in range(len(CLASS_NAMES))}
    
    # Draw annotations
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
    
    if calibrated_age:
        label = f"Age: {calibrated_age}"
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2)
    
    top_condition = max(predictions, key=predictions.get)
    top_conf = predictions[top_condition]
    condition_label = f"{top_condition}: {top_conf:.1f}%"
    cv2.putText(img, condition_label, (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 
               0.6, (0, 255, 0), 2)
    
    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    processing_time = time.time() - start_time
    
    # Log prediction
    bbox = {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
    log_prediction(bbox, predictions, calibrated_age, processing_time)
    
    # Track performance
    inference_times.append(processing_time)
    
    return {
        'status': 'success',
        'annotated_image': img_rgb,
        'bbox': bbox,
        'predictions': predictions,
        'age': calibrated_age,
        'top_condition': top_condition,
        'top_confidence': float(top_conf),
        'processing_time': processing_time,
        'num_faces': 1
    }

def main():
    st.markdown("""
        <div class="header-gradient">
            <h1 class="header-title">DermalScan:AI_Facial Skin Aging Detection App</h1>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## About the Project")
        st.markdown("""
        **AI_DERMAL** is an advanced deep learning system for:
        - **Age Estimation** (Accurate prediction)
        - **Skin Condition Analysis** (4 conditions)
        - **Confidence Scores** (Multi-class predictions)
        """)
        
        st.markdown("---")
        
        st.markdown("##How It Works")
        with st.expander("Click to learn more"):
            st.markdown("""
            ### Detection Pipeline:
            1. **Face Detection**: Multi-scale Haar Cascade detection
            2. **Preprocessing**: CLAHE enhancement + denoising
            3. **Feature Extraction**: ResNet50 deep features
            4. **Prediction**: Ensemble of 5 augmented predictions
            5. **Calibration**: Weighted averaging for accuracy
            
            ### Model Architecture:
            - **Base**: ResNet50 (ImageNet pre-trained)
            - **Output**: Multi-task (Skin + Age)
            - **Classes**: 4 skin conditions
            - **Accuracy**: ~85-90% on test set
            """)
        
        st.markdown("---")
        
        show_all_predictions = st.checkbox(
            "Show All Predictions", 
            value=True,
            help="Display all condition probabilities"
        )
        
        st.markdown("---")
        st.caption("**Model:** ResNet50 Multi-Task")
        st.caption("**Framework:** TensorFlow + Streamlit")
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("## Upload Your Image")
        
        st.markdown("""
        <div class="instructions">
            <h4> Instructions for Best Results:</h4>
            <ul>
                <li> Upload a <strong>clear, front-facing image</strong></li>
                <li> Use <strong>good lighting</strong> (natural light preferred)</li>
                <li> Ensure <strong>only one face</strong> is visible</li>
                <li> Avoid <strong>filters, makeup</strong>, or heavy editing</li>
                <li> No <strong>sunglasses, masks</strong>, or face coverings</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an image file (JPG, JPEG, PNG)",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear face image for analysis"
        )
        
        if uploaded_file is not None:
            st.markdown('<div class="upload-card">', unsafe_allow_html=True)
            image = Image.open(uploaded_file)
            st.image(image, caption="Your Uploaded Image", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Run Analysis", key="analyze", use_container_width=True):
                with st.spinner("Analyzing image... Please wait..."):
                    try:
                        model, face_cascade = load_models()
                        if model is None or face_cascade is None:
                            st.markdown('<div class="error-box">Failed to load models</div>', unsafe_allow_html=True)
                            return
                        
                        result = process_image(image, model, face_cascade)
                        
                        if result['status'] == 'error':
                            st.markdown(f'<div class="error-box">{result["message"]}</div>', unsafe_allow_html=True)
                            return
                        
                        # Store results in session state
                        st.session_state['processed_img'] = result['annotated_image']
                        st.session_state['result'] = result
                        st.session_state['image_analyzed'] = True
                        
                        # Display processing time
                        st.success(f"Analysis completed in {result['processing_time']:.2f}s")
                        
                    except Exception as e:
                        st.markdown(f'<div class="error-box">Error: {str(e)}</div>', unsafe_allow_html=True)
        else:
            st.info(" Please upload an image to begin analysis")
    
    with col2:
        st.markdown("## Analysis Results")
        
        if 'image_analyzed' in st.session_state and st.session_state['image_analyzed']:
            processed_img = st.session_state['processed_img']
            result = st.session_state['result']
            
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.image(processed_img, caption="Detected Face with Predictions", 
                    use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display processing time badge
            time_color = "#84fab0" if result['processing_time'] <= 5.0 else "#ffa751"
            st.markdown(f"""
            <div style='background: {time_color}; padding: 10px; border-radius: 8px; text-align: center; margin: 10px 0;'>
                <strong>⚡ Processing Time: {result['processing_time']:.3f}s</strong>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="success-box">Face detected successfully! Analysis complete.</div>', unsafe_allow_html=True)
            
            # Download buttons
            st.markdown("<br>", unsafe_allow_html=True)
            download_col1, download_col2 = st.columns(2)
            
            with download_col1:
                # Convert annotated image to bytes for download
                img_pil = Image.fromarray(processed_img)
                buf = io.BytesIO()
                img_pil.save(buf, format='PNG')
                byte_im = buf.getvalue()
                
                st.download_button(
                    label=" Download Annotated Image",
                    data=byte_im,
                    file_name=f"dermal_scan_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            with download_col2:
                # Create CSV with predictions and bounding box data
                csv_data = {
                    'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                    'Bounding_Box_X': [result['bbox']['x']],
                    'Bounding_Box_Y': [result['bbox']['y']],
                    'Bounding_Box_Width': [result['bbox']['w']],
                    'Bounding_Box_Height': [result['bbox']['h']],
                    'Predicted_Age': [result['age'] if result['age'] else 'N/A'],
                    'Top_Condition': [result['top_condition']],
                    'Top_Confidence': [f"{result['top_confidence']:.2f}%"],
                    'Clear_Face_Confidence': [f"{result['predictions']['clear face']:.2f}%"],
                    'Darkspots_Confidence': [f"{result['predictions']['darkspots']:.2f}%"],
                    'Puffy_Eyes_Confidence': [f"{result['predictions']['puffy eyes']:.2f}%"],
                    'Wrinkles_Confidence': [f"{result['predictions']['wrinkles']:.2f}%"],
                    'Processing_Time': [f"{result['processing_time']:.3f}s"]
                }
                df = pd.DataFrame(csv_data)
                csv_string = df.to_csv(index=False)
                
                st.download_button(
                    label="Download Predictions CSV",
                    data=csv_string,
                    file_name=f"dermal_scan_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # Display detailed results table
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### Detailed Analysis Results")
            
            # Create comprehensive results table
            table_data = {
                'Property': [
                    'Timestamp',
                    'Bounding Box (x, y, w, h)',
                    'Predicted Age',
                    'Top Condition',
                    'Top Confidence',
                    'Clear Face',
                    'Darkspots',
                    'Puffy Eyes',
                    'Wrinkles',
                    'Processing Time'
                ],
                'Value': [
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    f"({result['bbox']['x']}, {result['bbox']['y']}, {result['bbox']['w']}, {result['bbox']['h']})",
                    f"{result['age']} years" if result['age'] else 'N/A',
                    result['top_condition'].title(),
                    f"{result['top_confidence']:.2f}%",
                    f"{result['predictions']['clear face']:.2f}%",
                    f"{result['predictions']['darkspots']:.2f}%",
                    f"{result['predictions']['puffy eyes']:.2f}%",
                    f"{result['predictions']['wrinkles']:.2f}%",
                    f"{result['processing_time']:.3f}s"
                ]
            }
            
            results_df = pd.DataFrame(table_data)
            
            # Style the dataframe with custom CSS
            st.markdown("""
            <style>
            .dataframe {
                font-size: 14px;
                width: 100%;
            }
            .dataframe th {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                font-weight: 600;
                padding: 12px;
                text-align: left;
            }
            .dataframe td {
                padding: 10px;
                border-bottom: 1px solid #e0e0e0;
            }
            .dataframe tr:hover {
                background-color: #f5f7fa;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Property": st.column_config.TextColumn(
                        "Property",
                        width="medium",
                    ),
                    "Value": st.column_config.TextColumn(
                        "Value",
                        width="large",
                    )
                }
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            metric_col1, metric_col2 = st.columns(2)
            
            with metric_col1:
                if result['age']:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">PREDICTED AGE</div>
                        <div class="metric-value">{result['age']}</div>
                        <div class="metric-label">years old</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with metric_col2:
                condition_colors = {
                    'clear face': '#84fab0',
                    'darkspots': '#ffa751',
                    'puffy eyes': '#f5576c',
                    'wrinkles': '#f093fb'
                }
                condition_color = condition_colors.get(result['top_condition'].lower(), '#667eea')
                
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, {condition_color} 0%, #8ec5fc 100%);">
                    <div class="metric-label">SKIN CONDITION</div>
                    <div class="metric-value" style="font-size: 1.5rem;">{result['top_condition'].upper()}</div>
                    <div class="metric-label">{result['top_confidence']:.1f}% confidence</div>
                </div>
                """, unsafe_allow_html=True)
            
            if show_all_predictions:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### Detailed Confidence Scores")
                
                sorted_preds = sorted(result['predictions'].items(), 
                                    key=lambda x: x[1], reverse=True)
                
                conditions = [pred[0].title() for pred in sorted_preds]
                confidences = [pred[1] for pred in sorted_preds]
                
                colors_map = {
                    'Clear Face': '#84fab0',
                    'Darkspots': '#ffa751',
                    'Puffy Eyes': '#f5576c',
                    'Wrinkles': '#f093fb'
                }
                bar_colors = [colors_map.get(cond, '#667eea') for cond in conditions]
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=confidences,
                        y=conditions,
                        orientation='h',
                        marker=dict(
                            color=bar_colors,
                            line=dict(color='rgba(0,0,0,0.3)', width=1)
                        ),
                        text=[f'{conf:.1f}%' for conf in confidences],
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    title="Skin Condition Probability Distribution",
                    xaxis_title="Confidence (%)",
                    yaxis_title="Condition",
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### All Predictions:")
                pred_cols = st.columns(2)
                for idx, (condition, confidence) in enumerate(sorted_preds):
                    col = pred_cols[idx % 2]
                    
                    with col:
                        if confidence >= 70:
                            indicator = "🟢"
                        elif confidence >= 40:
                            indicator = "🟡"
                        else:
                            indicator = "🔴"
                        
                        st.metric(
                            label=f"{indicator} {condition.title()}",
                            value=f"{confidence:.1f}%",
                            delta=None
                        )
        else:
            st.markdown("""
            <div class="prediction-card" style="text-align: center; padding: 50px;">
                <h3 style="color: #666;"> Upload an image and click "Run Analysis"</h3>
                <p style="color: #999; margin-top: 20px;">
                    Your results will appear here with detailed predictions,
                    confidence scores, and visual analysis.
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; color: #888; margin-top: 50px; padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 10px; color: white;'>
        <p style='font-weight: 600; font-size: 1.1rem;'>DermalScan:AI_Facial Skin Aging Detection App</p>
        <p style='font-size: 0.9rem; opacity: 0.9;'>Powered by ResNet50, TensorFlow & Streamlit</p>
        <p style='font-size: 0.8rem; opacity: 0.8;'>© 2025 | For Educational & Research Purposes Only</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
