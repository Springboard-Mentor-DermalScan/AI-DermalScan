# backend.py
import os
import cv2
import numpy as np
import pandas as pd
from random import randint
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, BatchNormalization
from tensorflow.keras.models import Model

# Paths
MODEL_PATH = "best_inceptionv3_model2.h5"
HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Cache model and cascade
_model, _face_cascade = None, None

def load_inception_model():
    global _model
    if _model is not None:
        return _model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    try:
        _model = load_model(MODEL_PATH)
    except:
        base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        predictions = Dense(4, activation='softmax')(x)
        _model = Model(inputs=base_model.input, outputs=predictions)
        _model.load_weights(MODEL_PATH)
    return _model

def load_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(HAAR_PATH)
    return _face_cascade

def predict_aging_signs(image_bytes):
    model = load_inception_model()
    face_cascade = load_face_cascade()

    # Read image
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3)
    if len(faces) == 0:
        h, w = img.shape[:2]
        faces = [(0, 0, w, h)]

    results = []
    class_names = ['clear face', 'darkspots', 'puffy eyes', 'wrinkles']

    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (224, 224))
        face_array = img_to_array(face_resized)
        face_array = np.expand_dims(face_array, axis=0)
        face_array = preprocess_input(face_array)

        preds = model.predict(face_array, verbose=0)[0]
        pred_idx = np.argmax(preds)
        conf = float(preds[pred_idx]) * 100
        condition = class_names[pred_idx]

        # Predict age range
        if condition == 'clear face': est_age = randint(18, 30)
        elif condition == 'darkspots': est_age = randint(30, 40)
        elif condition == 'puffy eyes': est_age = randint(40, 55)
        else: est_age = randint(56, 75)

        label = f"{condition} ({conf:.2f}%) | Age: {est_age}"

        # Draw bounding box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Determine label size and adjust position
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_y = y - 10 if y - 10 > label_height else y + h + label_height + 10
        
        # Make sure text doesn’t go outside image
        if text_y + label_height > img.shape[0]:
            text_y = y - 10
        
        # Add background rectangle for better readability
        cv2.rectangle(
            img,
            (x, text_y - label_height - baseline),
            (x + label_width, text_y + baseline),
            (0, 255, 0),
            cv2.FILLED
        )
        
        # Put label text on the rectangle
        cv2.putText(img, label, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


        results.append({
            "x": x, "y": y, "width": w, "height": h,
            "Condition": condition,
            "Confidence (%)": round(conf, 2),
            "Estimated Age": est_age
        })

    annotated_path = "annotated_result.jpg"
    cv2.imwrite(annotated_path, img)
    results_df = pd.DataFrame(results)
    csv_path = "predictions_log.csv"
    results_df.to_csv(csv_path, index=False)

    return annotated_path, csv_path, results_df
