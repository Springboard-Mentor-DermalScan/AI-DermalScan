import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from random import randint
import pandas as pd
import datetime
import os

# Configuration
HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
DERMALSCAN_MODEL_PATH = 'DenseNet121_best_model.h5'

DERMALSCAN_CLASS_NAMES = ['clear face', 'darkspots', 'puffy eyes', 'wrinkles']
IMAGE_SIZE = (224, 224)

# CSV Logging
def log_to_csv(df):
    """Append predictions to a persistent CSV log file."""
    log_file = 'prediction_log.csv'
    if not df.empty:
        file_exists = os.path.exists(log_file)
        df.to_csv(log_file, mode='a', header=not file_exists, index=False)
        print(f"[LOG] Results appended to {log_file}")

# Load Models 
@tf.keras.utils.register_keras_serializable()
def load_all_models():
    try:
        dermalscan_model = load_model(DERMALSCAN_MODEL_PATH)
        face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        return dermalscan_model, face_cascade
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None

DERMALSCAN_MODEL, FACE_CASCADE = load_all_models()

# Prediction Function 
def process_and_predict(image_np, filename="uploaded_image"):
    if DERMALSCAN_MODEL is None or FACE_CASCADE is None:
        raise ConnectionError("Models failed to load. Check file paths and ensure the model is correctly saved.")

    start_time = datetime.datetime.now()
    image = image_np.copy()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    results_list = []

    if len(faces) == 0:
        return image, pd.DataFrame(), (datetime.datetime.now() - start_time).total_seconds()

    for (x, y, width, height) in faces:
        face_crop = image[y:y + height, x:x + width]
        face_resized = cv2.resize(face_crop, IMAGE_SIZE)
        face_normalized = face_resized / 255.0

        dermalscan_prediction = DERMALSCAN_MODEL.predict(np.expand_dims(face_normalized, axis=0), verbose=0)
        predicted_index = np.argmax(dermalscan_prediction)
        predicted_class = DERMALSCAN_CLASS_NAMES[predicted_index]
        confidence_score = dermalscan_prediction[0][predicted_index] * 100

        # Estimated Age Logic
        if predicted_class == "clear face":
            est_age = randint(25, 30)
        elif predicted_class == "darkspots":
            est_age = randint(30, 40)
        elif predicted_class == "puffy eyes":
            est_age = randint(40, 55)
        else:
            est_age = randint(70, 95)

        # Annotation
        label_text = f"{predicted_class} ({confidence_score:.1f}%), Age: {est_age}"
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)

        FONT_SCALE = 0.55
        THICKNESS = 2
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, THICKNESS)
        label_x = max(x - 5, 0)
        label_y = max(y - 10, text_h + 10)
        label_x = min(label_x, image.shape[1] - text_w - 5)

        cv2.rectangle(image, (label_x, label_y - text_h - 6),
                      (label_x + text_w + 6, label_y),
                      (0, 255, 0), -1)
        cv2.putText(image, label_text, (label_x + 3, label_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), THICKNESS)

        results_list.append({
            "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Filename": filename,
            "Bounding_Box": f"({x}, {y}, {width}, {height})",
            "Predicted_Sign": predicted_class,
            "Confidence": f"{confidence_score:.1f}%",
            "Estimated_Age": est_age
        })

    latency = (datetime.datetime.now() - start_time).total_seconds()
    results_df = pd.DataFrame(results_list)
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Processed {filename}. "
          f"Faces: {len(faces)} | Time: {latency:.2f}s")

    return image, results_df, latency
