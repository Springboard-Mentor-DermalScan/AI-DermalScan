# backend_inference.py
import io
import json
import time
import sqlite3
import logging
from datetime import datetime
from typing import List, Tuple, Optional, Dict
from pathlib import Path

from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess

# -----------------------
# Configuration
# -----------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_densenet_model.h5"
HAAR_CASCADE_PATH = BASE_DIR / "haarcascade_frontalface_default.xml"
DB_PATH = BASE_DIR / "logs" / "predictions.db"

INPUT_SIZE = (224, 224)
DEBUG = True

# -----------------------
# Logging setup
# -----------------------
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# -----------------------
# Database helpers
# -----------------------
def init_db(db_path: Path = DB_PATH):
    db_path.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            filename TEXT,
            bbox TEXT,
            predictions TEXT,
            latency_ms INTEGER
        )
    """)
    conn.commit()
    conn.close()
    logging.info(f"✅ Database initialized at {db_path}")

def log_to_db(filename: str, bbox: Optional[Tuple[int,int,int,int]],
              predictions: Dict[str, float], latency_ms: int,
              db_path: Path = DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        "INSERT INTO predictions (timestamp, filename, bbox, predictions, latency_ms) VALUES (?,?,?,?,?)",
        (datetime.utcnow().isoformat(), filename, json.dumps(bbox) if bbox else None,
         json.dumps(predictions), int(latency_ms))
    )
    conn.commit()
    conn.close()

# -----------------------
# Model loading
# -----------------------
_model = None
_labels = None

def load_densenet_model(model_path: Path = MODEL_PATH):
    global _model, _labels
    if _model is not None:
        return _model
    if not model_path.exists():
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")
    logging.info(f"Loading DenseNet model from {model_path}")
    _model = load_model(model_path, compile=False)

    labels_path = model_path.parent / "labels.json"
    if labels_path.exists():
        with open(labels_path, "r") as f:
            _labels = json.load(f)
        logging.info("✅ Labels loaded successfully.")
    else:
        _labels = None
        logging.warning("⚠️ No labels.json found. Using index numbers instead.")

    return _model

# -----------------------
# Face detection
# -----------------------
_face_cascade = None  # <-- Add this line here (outside the function)

def maybe_init_face_cascade(path=HAAR_CASCADE_PATH):
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(path)
        if _face_cascade.empty():
            logging.error(f"❌ Failed to load cascade from {path}. File not found or invalid.")
            _face_cascade = None
        else:
            logging.info(f"✅ Face detector loaded successfully from {path}")
    return _face_cascade


# -----------------------
# Preprocessing
# -----------------------
def crop_and_preprocess(img: Image.Image, bbox: Optional[Tuple[int,int,int,int]] = None,
                        target_size: Tuple[int,int] = INPUT_SIZE) -> np.ndarray:
    if bbox:
        x, y, w, h = bbox
        img = img.crop((x, y, x+w, y+h))
    img = img.convert("RGB").resize(target_size, Image.BILINEAR)
    arr = np.array(img).astype(np.float32)
    arr = densenet_preprocess(arr)
    return np.expand_dims(arr, axis=0)

# -----------------------
# Inference
# -----------------------
def infer_image(img: Image.Image, bbox: Optional[Tuple[int,int,int,int]] = None) -> Dict[str, float]:
    model = load_densenet_model()
    batch = crop_and_preprocess(img, bbox=bbox)
    preds = model.predict(batch, verbose=0)[0]

    if _labels:
        result = {str(_labels[i]): float(preds[i]) for i in range(len(preds))}
    else:
        result = {str(i): float(preds[i]) for i in range(len(preds))}
    return result

# -----------------------
# Flask API
# -----------------------
app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": _model is not None}), 200

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    start_total = time.time()
    if 'file' not in request.files:
        return jsonify({"error": "no file provided"}), 400

    file = request.files['file']
    filename = file.filename or "upload"
    try:
        img = Image.open(io.BytesIO(file.read()))
    except Exception as e:
        return jsonify({"error": "invalid image file", "details": str(e)}), 400

    detect_faces_flag = request.form.get("detect_faces", "true").lower() in ("1", "true", "yes")
    bbox = None
    if all(k in request.form for k in ("x","y","w","h")):
        try:
            bbox = tuple(map(int, (request.form['x'], request.form['y'], request.form['w'], request.form['h'])))
        except Exception:
            bbox = None

    detections = []
    np_img = np.array(img.convert("RGB"))
    bboxes_list = [bbox] if bbox else (detect_faces_np(np_img) if detect_faces_flag else [])

    if not bboxes_list:
        width, height = img.size
        bboxes_list = [(0, 0, width, height)]

    for bb in bboxes_list:
        t0 = time.time()
        preds = infer_image(img, bbox=bb)
        latency_ms = int((time.time() - t0) * 1000)
        log_to_db(filename, bb, preds, latency_ms)
        detections.append({"bbox": list(bb), "predictions": preds, "latency_ms": latency_ms})

    total_latency_ms = int((time.time() - start_total) * 1000)
    return jsonify({
        "filename": filename,
        "detections": detections,
        "total_latency_ms": total_latency_ms
    }), 200

if __name__ == "__main__":
    init_db()
    maybe_init_face_cascade()
    load_densenet_model()

    # Warm-up for better first-call latency
    dummy = np.zeros((1, INPUT_SIZE[0], INPUT_SIZE[1], 3), dtype=np.float32)
    _model.predict(dummy, verbose=0)
    logging.info("🔥 DenseNet model warm-up complete.")

    app.run(host="0.0.0.0", port=8000, debug=DEBUG, threaded=True)
