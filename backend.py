import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from random import randint
import pandas as pd

# ---------------- Load DenseNet Model ----------------
skin_model_path = r"C:\ARPEETA MOHANTY\final_densenet_model.h5"
skin_model = load_model(skin_model_path)
skin_classes = ['clear face', 'darkspots', 'puffy eyes', 'wrinkles']

# ---------------- Load Haar Cascade ----------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def analyze_face(image_path):
    """Main backend function: detects faces, classifies skin, estimates age."""
    start_time = time.time()

    image = cv2.imread(image_path)
    if image is None:
        return None, None, None, "Error: Unable to load image"

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (h, w) = image.shape[:2]
    padding = 20

    results = []
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))

    for (x, y, fw, fh) in faces:
        x1, y1 = max(0, x - padding), max(0, y - padding)
        x2, y2 = min(w - 1, x + fw + padding), min(h - 1, y + fh + padding)
        face_crop = image[y1:y2, x1:x2]

        if face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
            continue

        # Preprocess for model
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (224, 224))
        face_array = np.expand_dims(img_to_array(face_resized) / 255.0, axis=0)

        # Model prediction
        pred = skin_model.predict(face_array, verbose=0)[0]
        skin_index = np.argmax(pred)
        skin_type = skin_classes[skin_index]
        confidence = pred[skin_index] * 100

        # Estimated age (randomized logic)
        if skin_type == "clear face":
            est_age = randint(18, 30)
        elif skin_type == "darkspots":
            est_age = randint(30, 40)
        elif skin_type == "puffy eyes":
            est_age = randint(40, 55)
        else:
            est_age = randint(56, 70)

        # Draw bounding box + label
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{skin_type} ({confidence:.2f}%), Age: {est_age}"
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        results.append({
            "Skin Type": skin_type,
            "Confidence (%)": round(confidence, 2),
            "Estimated Age": est_age,
            "Coordinates (x,y,w,h)": f"{x1},{y1},{x2},{y2}",
            "Processing Time (s)": round(time.time() - start_time, 2),
            "Image Width": w,
            "Image Height": h
        })

    latency = round(time.time() - start_time, 2)
    annotated_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    df = pd.DataFrame(results)

    return annotated_image, df, latency, len(faces)
