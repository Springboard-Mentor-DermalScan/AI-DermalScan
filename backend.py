import cv2
import numpy as np
import tensorflow as tf
from random import randint
import pandas as pd
import datetime
import os
from io import BytesIO
from PIL import Image


HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
DERMALSCAN_MODEL_PATH = 'modelfile.h5' 
DERMALSCAN_CLASS_NAMES = ['clear face', 'darkspots', 'puffy eyes', 'wrinkles']
IMAGE_SIZE = (224, 224)


@tf.keras.utils.register_keras_serializable()
def load_dermalscan_models():
    """Loads the Deep Learning model and Haar Cascade classifier."""
    try:
        # Suppress TensorFlow warnings/logs during load
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
        
        # Use fully qualified path for load_model
        dermalscan_model = tf.keras.models.load_model(DERMALSCAN_MODEL_PATH)
        face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        
        if face_cascade.empty():
             print(f"WARNING: Haar Cascade file not found at {HAAR_CASCADE_PATH}")

        return dermalscan_model, face_cascade
    except Exception as e:
        print(f"FATAL: Error loading models. Check file paths ({DERMALSCAN_MODEL_PATH}, {HAAR_CASCADE_PATH}): {e}")
        return None, None

# --- Main Analysis Function ---
def analyze_image_with_model(image_pil, filename, detection_sensitivity, dermalscan_model, face_cascade):
    """
    Processes a PIL image, detects faces, predicts signs of aging, 
    and prepares the results for the Streamlit frontend.
    """
    if dermalscan_model is None or face_cascade is None:
        # Return empty data structures if models failed to load
        return BytesIO(), "Analysis models failed to load.", pd.DataFrame(), 0.0

    start_time = datetime.datetime.now()
    
    # Convert PIL Image to OpenCV NumPy array (BGR format for OpenCV)
    image_np = np.array(image_pil.convert('RGB'))
    image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Face Detection
    min_neighbors = max(3, int(5 + (1 - detection_sensitivity) * 5))
    faces = face_cascade.detectMultiScale(
        gray_image, 
        scaleFactor=1.1, 
        minNeighbors=min_neighbors,
        minSize=(30, 30)
    )
    
    results_list = []
    
    # If no faces found
    if len(faces) == 0:
        latency = (datetime.datetime.now() - start_time).total_seconds()
        annotated_image_png = BytesIO()
        image_pil.save(annotated_image_png, format="PNG")
        annotated_image_png.seek(0)
        summary = f"Selected Faces: 0 | No faces detected. Processing Time: {latency:.2f} seconds."
        return annotated_image_png, summary, pd.DataFrame(columns=['Skin Type/Feature', 'Confidence (%)', 'Estimated Age', 'Coordinates (X1, Y1, W, H)']), 0.0

    total_wrinkle_confidence = 0
    wrinkle_count = 0
    
    # Process each detected face
    for (x, y, width, height) in faces:
        # Use RGB version for Keras input
        face_crop = image_np[y:y + height, x:x + width] 
        
        face_resized = cv2.resize(face_crop, IMAGE_SIZE)
        face_normalized = face_resized / 255.0

        # Deep Learning Prediction
        dermalscan_prediction = dermalscan_model.predict(np.expand_dims(face_normalized, axis=0), verbose=0)
        predicted_index = np.argmax(dermalscan_prediction)
        predicted_class = DERMALSCAN_CLASS_NAMES[predicted_index]
        confidence_score = dermalscan_prediction[0][predicted_index] * 100

        # Estimated Age Logic (kept for mock realism)
        if predicted_class == "clear face":
            est_age = randint(25, 30)
        elif predicted_class == "darkspots":
            est_age = randint(30, 40)
        elif predicted_class == "puffy eyes":
            est_age = randint(40, 55)
        else: # wrinkles
            est_age = randint(55, 75)

        # Annotation (on the BGR image)
        label_text = f"{predicted_class.title()} ({confidence_score:.1f}%)"
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2) # BGR: Green box
        
        FONT_SCALE = 0.55
        THICKNESS = 1
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, THICKNESS)
        
        # Draw background rectangle for text
        cv2.rectangle(image, (x, y - text_h - 15), 
                      (x + text_w + 10, y - 5), 
                      (0, 255, 0), -1) 
                      
        # Put text
        cv2.putText(image, label_text, (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), THICKNESS) # Black text

        # Prepare data for DataFrame (matching Streamlit column names)
        results_list.append({
            "Skin Type/Feature": predicted_class.title(),
            "Confidence (%)": round(confidence_score, 1),
            "Estimated Age": est_age,
            "Coordinates (X1, Y1, W, H)": f"({x}, {y}, {width}, {height})",
        })
        
        if predicted_class == 'wrinkles':
            total_wrinkle_confidence += confidence_score
            wrinkle_count += 1

    latency = (datetime.datetime.now() - start_time).total_seconds()
    
    # Finalize DataFrame (Matching Frontend Column Names)
    results_df = pd.DataFrame(results_list)
    
    # Calculate Wrinkle Index
    if wrinkle_count > 0:
        # Average confidence of wrinkle detections
        wrinkle_index_value = total_wrinkle_confidence / wrinkle_count
    else:
        # Fallback index if no wrinkles were detected. Base it on clarity if possible.
        clear_face_conf = results_df[results_df['Skin Type/Feature'] == 'Clear Face']['Confidence (%)'].mean()
        wrinkle_index_value = 100.0 - clear_face_conf if not pd.isna(clear_face_conf) else 5.0
    
    # Convert OpenCV BGR Image back to PNG Bytes for Streamlit
    annotated_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    annotated_image_pil = Image.fromarray(annotated_image_rgb)
    annotated_image_png = BytesIO()
    annotated_image_pil.save(annotated_image_png, format="PNG")
    annotated_image_png.seek(0)
    
    summary = (
        f"Selected Faces: {len(faces)} | Processing Time: {latency:.2f} seconds."
    )
    
    return annotated_image_png, summary, results_df, wrinkle_index_value
