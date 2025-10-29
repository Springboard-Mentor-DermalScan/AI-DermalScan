import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ====== Load Model ======
MODEL_PATH = r"C:\Users\Shiva\OneDrive\Desktop\infosys\best_model.h5"
model = load_model(MODEL_PATH)

# ====== Class Names ======
CLASS_NAMES = {
    0: 'clear face',
    1: 'darkspots',
    2: 'puffy eyes',
    3: 'wrinkles'
}

# ====== Haarcascade for Multi-face Detection ======
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ====== Prediction Function ======
def predict_image(image_path, padding=0.05):
    img_cv = cv2.imread(image_path)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=6)

    h, w, _ = img_cv.shape
    annotated = img_cv.copy()
    results = []

    # If no faces detected — use center crop fallback
    if len(faces) == 0:
        x1, y1, x2, y2 = int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8)
        faces = [(x1, y1, x2 - x1, y2 - y1)]

    for (x, y, fw, fh) in faces:
        # Add padding but ensure stays inside image bounds
        pad_x, pad_y = int(fw * padding), int(fh * padding)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + fw + pad_x)
        y2 = min(h, y + fh + pad_y)

        # Crop face region
        face_crop = img_cv[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        # Preprocess input (MobileNetV2 style)
        img_resized = cv2.resize(face_crop, (224, 224))
        arr = image.img_to_array(img_resized)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)

        preds = model.predict(arr, verbose=0)
        pred_class = int(np.argmax(preds))
        confidence = round(float(np.max(preds)) * 100, 2)
        label_name = CLASS_NAMES[pred_class]

        # Estimated Age by Class
        age_ranges = {
            'clear face': (18, 25),
            'darkspots': (25, 40),
            'puffy eyes': (35, 50),
            'wrinkles': (50, 70)
        }
        low, high = age_ranges[label_name]
        age = np.random.randint(low, high + 1)

        # Draw Bounding Box
        color = (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label Texts
        text_lines = [
            f"{label_name}",
            f"Conf: {confidence}%",
            f"Age: {age}"
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        line_height = 20

        # Draw text above box if space available
        text_y = y1 - 10
        if text_y < 20:
            text_y = y2 + 25  # place below box

        for i, txt in enumerate(text_lines):
            y_pos = text_y + i * line_height
            cv2.putText(
                annotated, txt, (x1 + 5, y_pos),
                font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA
            )

        results.append({
            "x": int(x1), "y": int(y1),
            "width": int(x2 - x1), "height": int(y2 - y1),
            "Condition": label_name,
            "Confidence": confidence,
            "Estimated_Age": age
        })

    # Save Annotated Output
    os.makedirs("results", exist_ok=True)
    output_path = os.path.join("results", "annotated_output.jpg")
    cv2.imwrite(output_path, annotated)

    return output_path, results
