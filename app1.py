import streamlit as st
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input
from deepface import DeepFace
from PIL import Image
import io
import time

st.set_page_config(layout="wide", page_title="DermalScan")

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {background-color: #FCF9EA; color: #C95B5B;}
    [data-testid="stSidebar"] {background-color: #BADFDB; border-right: 2px solid #FFA4A4;}
    h1, h2, h3, h4, h5, h6, div, p, span, li, label {color: #C95B5B;}
    [data-testid="stButton"] button {
        background-color: #FFA4A4; color: #FCF9EA; font-weight: bold; border: 1px solid #FFA4A4;
        border-radius: 5px; padding: 8px 16px;
    }
    [data-testid="stButton"] button:hover {background-color: #FFBDBD; color: #C95B5B; border: 1px solid #FFBDBD;}
    [data-testid="stDownloadButton"] button {
        background-color: #FFBDBD; color: #C95B5B; border: 1px solid #FFA4A4;
    }
    [data-testid="stDownloadButton"] button:hover {
        background-color: #FFA4A4; color: #FCF9EA;
    }
    [data-testid="stFileUploader"] {
        background-color: #BADFDB; border-radius: 10px; padding: 15px;
    }
    [data-testid="stFileUploader"] label {color: #C95B5B;}
    [data-testid="stMetric"] {
        background-color: #FFFFFF; border: 1px solid #BADFDB; border-radius: 10px; padding: 10px;
    }
    [data-testid="stMetric"] label {color: #FFA4A4;}
    [data-testid="stMetric"] div {color: #C95B5B;}
    [data-testid="stDataFrame"] {
        background-color: #FFFFFF; border: 1px solid #BADFDB;
    }
</style>
""", unsafe_allow_html=True)

SKIN_MODEL_IMG_SIZE = 299
skin_model_path = 'best_inceptionv3_model.h5'
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# <<< CHANGED: YOU MUST UPDATE THIS LIST to match your model's output classes in order
SKIN_CLASSES = ["Clear", "Acne", "Pigmentation", "Wrinkles", "Eczema"] 

@st.cache_resource
def load_all_models():
    try:
        skin_model = load_model(skin_model_path)
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        _ = DeepFace.analyze(
            img_path=np.zeros((100, 100, 3), dtype=np.uint8),
            actions=['age'],
            enforce_detection=False
        )
        return skin_model, face_cascade
    except:
        return None, None

skin_model, face_cascade = load_all_models()

def predict_and_annotate(image_bytes):
    if skin_model is None or face_cascade is None:
        return None, []
    
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None, []
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    predictions_list = []
    
    for i, (x, y, w, h) in enumerate(faces):
        face_roi = img[y:y+h, x:x+w]
        face_id = i + 1
        
        # Skin condition prediction # <<< CHANGED
        try:
            resized_skin = cv2.resize(face_roi, (SKIN_MODEL_IMG_SIZE, SKIN_MODEL_IMG_SIZE))
            input_skin = np.expand_dims(resized_skin, axis=0)
            input_skin = preprocess_input(input_skin)
            
            skin_predictions = skin_model.predict(input_skin)[0] # e.g., [0.1, 0.7, 0.05, 0.1, 0.05]
            
            # Get the highest probability and its index
            confidence_score = float(np.max(skin_predictions))
            predicted_index = int(np.argmax(skin_predictions))
            
            # Get the corresponding condition name
            predicted_condition = SKIN_CLASSES[predicted_index] # e.g., "Acne"
            
            # Format for display
            confidence_percent = round(confidence_score * 100, 1) # e.g., 70.0

        except:
            predicted_condition = "N/A" # <<< CHANGED
            confidence_score = 0.0      # <<< CHANGED
            confidence_percent = 0.0    # <<< CHANGED
        
        # Age prediction
        try:
            analysis = DeepFace.analyze(
                img_path=face_roi,
                actions=['age'],
                enforce_detection=False,
                detector_backend='skip'
            )
            predicted_age = analysis[0]['age']
        except:
            predicted_age = "N/A"
        
        #skin_clarity = round(clarity_score * 100, 1) # <<< REMOVED
        
        face_data = {
            'face_id': face_id,
            'estimated_age': predicted_age,
            'condition': predicted_condition,      # <<< CHANGED
            'confidence': round(confidence_score, 3), # <<< CHANGED
            'bbox_x': x,
            'bbox_y': y,
            'bbox_w': w,
            'bbox_h': h
        }
        predictions_list.append(face_data)
        
        # Label and background rectangle
        # <<< CHANGED: Updated label format
        label = f"Age: {predicted_age} | {predicted_condition}: {confidence_percent}%" 
        (t_w, t_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 6, 3) # <<< CHANGED: Font size
        cv2.rectangle(img, (x, y - t_h - 10), (x + t_w + 5, y), (164, 164, 255), -1) # <<< CHANGED: Box size
        cv2.rectangle(img, (x, y), (x + w, y + h), (164, 164, 255), 2)
        cv2.putText(img, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 0), 3) # <<< CHANGED: Font size/position

    annotated_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return annotated_img_rgb, predictions_list

st.title(" DermalScan: AI Facial Skin Aging Detection App 🧖‍♀️✨")
st.write("Upload a facial image, then click 'Run Analysis' to see the results.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

if uploaded_file:
    button_placeholder = st.empty()
    if button_placeholder.button("Run Analysis"):
        button_placeholder.empty()
        st.header(f"Results for: {uploaded_file.name}")
        start_time = time.time()
        image_bytes = uploaded_file.getvalue()
        
        with st.spinner(f"Analyzing {uploaded_file.name}... 🤔"):
            annotated_img, prediction_data = predict_and_annotate(image_bytes)
        
        end_time = time.time()
        
        if annotated_img is None:
            st.error(f"Could not process the image: {uploaded_file.name}")
        else:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(annotated_img, caption='Analysis Result.', use_container_width=True)
            with col2:
                st.metric(label="Total Prediction Time", value=f"{end_time - start_time:.2f} s")
                
                if prediction_data:
                    df = pd.DataFrame(prediction_data)
                    # <<< CHANGED: Updated columns for CSV and display
                    df = df[['face_id', 'estimated_age', 'condition', 'confidence', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']]
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No prediction data generated (e.g., no faces found).")
                
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    im = Image.fromarray(annotated_img)
                    buf = io.BytesIO()
                    im.save(buf, format="PNG")
                    st.download_button("Image (PNG)", buf.getvalue(), f"annotated_{uploaded_file.name}.png", "image/png")
                with btn_col2:
                    if prediction_data:
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button("Data (CSV)", csv, f"predictions_{uploaded_file.name}.csv", "text/csv")
else:
    st.info("Please upload an image file using the button above to begin analysis.")

st.sidebar.header("About DermalScan")
st.sidebar.info(
    "This application uses a deep learning model (InceptionV3) "
    "to identify facial skin conditions and predict age. " # <<< CHANGED
    "It also utilizes the DeepFace library for age estimation.\n\n"
    "**Disclaimer:** The results are indicative and generated by AI. "
    "They are not a substitute for professional medical diagnosis."
)
st.sidebar.markdown("---")
st.sidebar.write("Developed as of the VIP_25 project.")