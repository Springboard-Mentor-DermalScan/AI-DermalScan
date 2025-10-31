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
# Load OpenCV DNN Face Detector
# -------------------------------
FACE_PROTO = "models/deploy.prototxt"
FACE_MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"

if not os.path.exists(FACE_PROTO) or not os.path.exists(FACE_MODEL):
    raise FileNotFoundError("Face detector model files not found. Check paths in 'models/' folder.")

face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)

# -------------------------------
# Main Function: Detect & Predict
# -------------------------------
def detect_predict_face(image_path, model=model, class_labels=class_labels, target_size=(224, 224)):
    """
    Detect faces using OpenCV DNN, predict skin condition & estimate age.

    Returns:
        annotated_image (np.ndarray): Image with boxes and labels
        results (list): List of dicts for each face
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read image at {image_path}")

    (h, w) = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create blob and perform detection
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    results = []

    # Loop over detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            # Extract and preprocess face ROI
            face_roi = img_rgb[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue
            face_resized = cv2.resize(face_roi, target_size)
            face_array = np.expand_dims(face_resized / 255.0, axis=0)

            # Predict skin condition
            preds = model.predict(face_array, verbose=0)[0]
            class_idx = np.argmax(preds)
            pred_conf = preds[class_idx] * 100
            predicted_class = class_labels[class_idx]

            # Estimate age range (randomized for now)
            if predicted_class.lower() == "clear_face":
                est_age = randint(18, 30)
            elif predicted_class.lower() == "darkspots":
                est_age = randint(30, 40)
            elif predicted_class.lower() == "puffy_eyes":
                est_age = randint(40, 55)
            else:
                est_age = randint(55, 70)

            # Label text
            label = f"{predicted_class}: {pred_conf:.2f}%, Age: {est_age}"
            label_y = y1 - 8 if y1 - 8 > 15 else y1 + 15

            # Draw bounding box + smaller label text
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_rgb, label, (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

            results.append({
                "predicted_class": predicted_class,
                "confidence": round(pred_conf, 2),
                "age": est_age,
                "bbox": (x1, y1, x2 - x1, y2 - y1)
            })

    # If no faces found
    if len(results) == 0:
        print("⚠️ No faces detected!")
        return img_rgb, []

    return img_rgb, results
