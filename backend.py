import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from random import randint
import os

# -------------------------------
# Load your trained model
# -------------------------------
MODEL_PATH = "MobileNetV2_best_model.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please check the path.")

model = load_model(MODEL_PATH)

# -------------------------------
# Define class labels (must match your dataset)
# -------------------------------
class_labels = ['puffy_eyes', 'darkspots', 'clear_face', 'wrinkles']

# -------------------------------
# Load Haar Cascade for face detection
# -------------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# -------------------------------
# Main function: Detect & Predict
# -------------------------------
def detect_predict_face(image_path, model=model, class_labels=class_labels, target_size=(224, 224)):
    """
    Detect faces in an image, predict dermal condition, and estimate age.

    Returns:
        annotated_image (np.ndarray): Image with bounding boxes and labels.
        condition (str): Predicted facial skin condition.
        age (int): Estimated age.
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read image at {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    results = []
    if len(faces) == 0:
        print("⚠️ No faces detected!")
        return img_rgb, None, None

    for (x, y, w, h) in faces:
        # Preprocess the face ROI
        face_roi = img_rgb[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, target_size)
        face_array = np.expand_dims(face_resized / 255.0, axis=0)

        # Predict skin condition
        preds = model.predict(face_array, verbose=0)[0]
        class_idx = np.argmax(preds)
        confidence = preds[class_idx] * 100
        predicted_class = class_labels[class_idx]

        # Assign estimated age range
        if predicted_class.lower() == "clear_face":
            est_age = randint(18, 30)
        elif predicted_class.lower() == "darkspots":
            est_age = randint(30, 40)
        elif predicted_class.lower() == "puffy_eyes":
            est_age = randint(40, 55)
        else:
            est_age = randint(55, 70)

        label = f"{predicted_class}: {confidence:.2f}%, Age: {est_age} yrs"
        label_y = y - 10 if y - 10 > 20 else y + 25

        # Draw bounding box and label
        cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img_rgb, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        results.append({
            "predicted_class": predicted_class,
            "confidence": round(confidence, 2),
            "age": est_age
        })

    # For frontend display — take first detected face
    condition = results[0]['predicted_class']
    age = results[0]['age']

    return img_rgb, condition, age


