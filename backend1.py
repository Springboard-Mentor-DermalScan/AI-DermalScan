import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
import cv2
import time
from mtcnn import MTCNN
from random import randint
MODEL_PATH = os.environ.get("MODEL_PATH", r"DermalSkin_MobileNetV2_Finetuned (1).h5")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")
print(" Loading MobileNetV2 model...")
@tf.keras.utils.register_keras_serializable()
class FixedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)
@tf.keras.utils.register_keras_serializable()
def load_model_once():
    try:
        return load_model(MODEL_PATH, compile=False)
    except TypeError:
        print(" Compatibility fix applied...")
        custom_objects = {"DepthwiseConv2D": FixedDepthwiseConv2D}
        return load_model(MODEL_PATH, compile=False, custom_objects=custom_objects)

model = load_model_once()
print(" Model loaded successfully!")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
detector = MTCNN()
def enhance_image(img):
    """
    Gentle, natural enhancement for skin — avoids oily/glossy look.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    enhanced_img = cv2.bilateralFilter(enhanced_img, d=5, sigmaColor=50, sigmaSpace=50)
    enhanced_img = cv2.convertScaleAbs(enhanced_img, alpha=1.0, beta=-10)
    return enhanced_img
def predict_skin_condition(img_path):
    start_time = time.time()
    if not os.path.exists(img_path):
        raise FileNotFoundError(f" Image not found: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(" Unable to read image file.")
    img = enhance_image(img)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mtcnn_results = detector.detect_faces(rgb_img)
    faces = [(r['box'][0], r['box'][1], r['box'][2], r['box'][3]) for r in mtcnn_results]
    if not faces:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if not faces:
        return {
            "results": [],
            "annotated_image": img_path,
            "total_time": 0,
            "estimation": " No face detected"
        }
    results = []
    class_labels = ["clear face", "dark spots", "puffy eyes", "wrinkles"]
    for (x, y, w, h) in faces:
        x, y = max(0, x), max(0, y)
        pad = int(0.1 * w)
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
        face_crop = img[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue
        face_resized = cv2.resize(face_crop, (224, 224), interpolation=cv2.INTER_AREA)
        face_array = np.expand_dims(face_resized / 255.0, axis=0)
        preds = model.predict(face_array, verbose=0)
        predicted_class_index = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]) * 100)
        predicted_label = class_labels[predicted_class_index]
        if predicted_label == "oily face":
            predicted_label = "normal face"
        if predicted_label in ["clear face", "normal face"]:
            age_estimated = randint(18, 30)
        elif predicted_label == "dark spots":
            age_estimated = randint(30, 40)
        elif predicted_label == "puffy eyes":
            age_estimated = randint(40, 55)
        else:
            age_estimated = randint(55, 70)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label_text = f"{predicted_label} ({confidence:.1f}%)"
        age_text = f"Age: {age_estimated} years"
        text_y = max(y1 - 10, 30)
        overlay = img.copy()
        text_bg_width = max(220, len(label_text) * 12)
        cv2.rectangle(overlay, (x1, text_y - 40), (x1 + text_bg_width, text_y - 5), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
        cv2.putText(img, label_text, (x1 + 10, text_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, age_text, (x1 + 10, text_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1, cv2.LINE_AA)
        results.append({
            "x": int(x1),
            "y": int(y1),
            "w": int(x2 - x1),
            "h": int(y2 - y1),
            "Predicted Label": predicted_label,
            "Confidence (%)": round(confidence, 2),
            "Estimated Age": age_estimated
        })
    annotated_path = "annotated_output_clear.jpg"
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(annotated_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    total_time = round(time.time() - start_time, 2)
    estimation = " Fast (≤5s)" if total_time <= 5 else " Slow (>5s)"
    return {
        "results": results,
        "annotated_image": annotated_path,
        "total_time": total_time,
        "estimation": estimation
    }