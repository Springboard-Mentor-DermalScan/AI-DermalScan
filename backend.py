import os
import time
import json
import shutil
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# ------------ CONFIG ------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

H5_NAME = "DermalSkin_MobileNetV2_Finetuned.h5"
MODEL_PATH = os.path.join(MODELS_DIR, H5_NAME)

FACE_PROTO = os.path.join(MODELS_DIR, "deploy.prototxt")
FACE_MODEL = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

LOG_CSV = os.path.join(OUTPUTS_DIR, "predictions_log.csv")

CLASS_LABELS = ['clear_face', 'dark_spots', 'puffy_eyes', 'wrinkles']  

# ------------ UTILITIES ------------
def _load_keras_model_safe(path):
    """Load Keras .h5 with compatibility for older DepthwiseConv2D that had 'groups' kwarg."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}. Place the .h5 in models/ folder.")
    try:
        m = load_model(path, compile=False)
        return m
    except TypeError as e:
        msg = str(e)
        if "DepthwiseConv2D" in msg and "groups" in msg:
            from tensorflow.keras.layers import DepthwiseConv2D as KDepthwise
            class DepthwiseConv2DFix(KDepthwise):
                def __init__(self, *args, **kwargs):
                    kwargs.pop("groups", None)
                    super().__init__(*args, **kwargs)
            from tensorflow.keras import utils
            utils.get_custom_objects()["DepthwiseConv2D"] = DepthwiseConv2DFix
            m = load_model(path, compile=False)
            return m
        else:
            raise

MODEL = _load_keras_model_safe(MODEL_PATH)

# Face detector
if not os.path.exists(FACE_PROTO) or not os.path.exists(FACE_MODEL):
    raise FileNotFoundError("Face detector files missing in models/. Add deploy.prototxt and res10_300x300_ssd_iter_140000.caffemodel")

FACE_NET = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)

# ------------ Small helpers ------------
def _resize_for_detection(img_bgr, max_dim=800):
    h, w = img_bgr.shape[:2]
    if max(h, w) <= max_dim:
        return img_bgr, 1.0
    scale = max_dim / float(max(h, w))
    return cv2.resize(img_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA), scale

def _variance_of_laplacian(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def _assess_blur(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    var = _variance_of_laplacian(gray)
    if var < 100:
        label = "very_blurry"
    elif var < 180:
        label = "slightly_blurry"
    else:
        label = "sharp"
    return label, round(float(var), 2)

def _estimate_skin_tone(face_rgb):
    ycrcb = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2YCrCb)
    meanY = float(np.mean(ycrcb[:, :, 0]))
    if meanY >= 140:
        return "fair"
    elif meanY >= 100:
        return "medium"
    else:
        return "dark"

def _color_for_label(label):
    mapping = {
        "clear_face": (50, 205, 50),
        "dark_spots": (0, 140, 255),
        "puffy_eyes": (255, 0, 255),
        "wrinkles": (200, 200, 0)
    }
    return mapping.get(label, (0, 200, 0))

# ------------ MAIN API ------------
def detect_predict_image(input_path, target_size=(224,224), conf_threshold=0.5, max_detect_dim=800):
    start_all = time.time()
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Image not found: {input_path}")

    img_bgr = cv2.imread(input_path)
    if img_bgr is None:
        raise ValueError("Unable to read image (cv2.imread returned None)")

    blur_label, blur_score = _assess_blur(img_bgr)
    img_small, scale = _resize_for_detection(img_bgr, max_dim=max_detect_dim)
    h_small, w_small = img_small.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(img_small, (300,300)), 1.0, (300,300), (104.0,177.0,123.0))
    FACE_NET.setInput(blob)
    detections = FACE_NET.forward()

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = []
    face_idx = 0

    for i in range(detections.shape[2]):
        conf = float(detections[0,0,i,2])
        if conf < conf_threshold:
            continue

        rx1 = int(detections[0,0,i,3] * w_small)
        ry1 = int(detections[0,0,i,4] * h_small)
        rx2 = int(detections[0,0,i,5] * w_small)
        ry2 = int(detections[0,0,i,6] * h_small)

        x1, y1 = max(0,int(rx1/scale)), max(0,int(ry1/scale))
        x2, y2 = min(img_bgr.shape[1]-1,int(rx2/scale)), min(img_bgr.shape[0]-1,int(ry2/scale))
        if x2 <= x1 or y2 <= y1:
            continue

        face_idx += 1
        face_crop_bgr = img_bgr[y1:y2, x1:x2]
        if face_crop_bgr.size == 0:
            continue
        face_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        skin_tone = _estimate_skin_tone(face_rgb)

        # preprocess for model
        face_resized = cv2.resize(face_rgb, target_size)
        face_input = np.expand_dims(face_resized.astype("float32")/255.0, axis=0)
        preds = MODEL.predict(face_input, verbose=0)[0]
        idx = int(np.argmax(preds))
        label = CLASS_LABELS[idx] if idx < len(CLASS_LABELS) else str(idx)
        conf_pct = float(preds[idx]*100.0)

        # deterministic-ish age estimate
        if label == "clear_face":
            est_age = int(18 + (1 - conf_pct/100.0)*7)
        elif label == "dark_spots":
            est_age = int(30 + (1 - conf_pct/100.0)*7)
        elif label == "puffy_eyes":
            est_age = int(40 + (1 - conf_pct/100.0)*10)
        else:
            est_age = int(55 + (1 - conf_pct/100.0)*10)

        color = _color_for_label(label)
        thickness = max(1, int(img_bgr.shape[1]/400))
        cv2.rectangle(img_rgb, (x1,y1),(x2,y2), color, thickness)

        # ✅ fix putText error: force strings
        label_lines = [
            str(label).replace("_"," ").title(),
            f"{conf_pct:.1f}% Age:{int(est_age)} Tone:{str(skin_tone)}"
        ]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.45, img_bgr.shape[1]/1600.0)
        t = max(1,int(img_bgr.shape[1]/800))
        heights = [cv2.getTextSize(str(ln), font, font_scale, t)[0][1] for ln in label_lines]
        widths  = [cv2.getTextSize(str(ln), font, font_scale, t)[0][0] for ln in label_lines]
        box_w = max(widths)+8
        box_h = sum(heights)+6
        tx = x1
        ty = max(0, y1-box_h-6)
        cv2.rectangle(img_rgb, (tx,ty),(tx+box_w, ty+box_h), (255,255,224), -1)
        y_text = ty+heights[0]
        for ln in label_lines:
            cv2.putText(img_rgb, str(ln), (tx+4, y_text), font, font_scale, (0,0,0), t, lineType=cv2.LINE_AA)
            y_text += int(heights[0]*0.9)+2

        results.append({
            "Face #": face_idx,
            "Predicted Class": label,
            "Confidence (%)": round(conf_pct,2),
            "Estimated Age": int(est_age),
            "Skin Tone": skin_tone,
            "Blur Score": blur_score,
            "X": x1, "Y": y1, "W": x2-x1, "H": y2-y1
        })

    # Save annotated
    annotated_path = os.path.join(OUTPUTS_DIR, f"annotated_{os.path.basename(input_path)}")
    cv2.imwrite(annotated_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    # DataFrame
    if results:
        df = pd.DataFrame(results)
    else:
        df = pd.DataFrame([{
            "Face #": 0, "Predicted Class":"No face", "Confidence (%)":0.0,
            "Estimated Age":0,"Skin Tone":"-","Blur Score":blur_score,
            "X":"-","Y":"-","W":"-","H":"-"
        }])

    df.insert(0, "Timestamp", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    df.insert(1, "Image", os.path.basename(input_path))
    total_time = round(time.time()-start_all,3)
    df["Total Time (s)"] = total_time

    # CSV log
    if not os.path.exists(LOG_CSV):
        df.to_csv(LOG_CSV,index=False)
    else:
        df.to_csv(LOG_CSV, mode="a", header=False, index=False)

    return annotated_path, df, total_time

# ------------ Export helper (zip) ------------
def export_results_zip(input_image_path, annotated_path, df_row):
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(OUTPUTS_DIR, f"summary_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(df_row.to_dict(orient="records"), jf, indent=2)
    csv_path = os.path.join(OUTPUTS_DIR, f"row_{timestamp}.csv")
    df_row.to_csv(csv_path,index=False)
    import zipfile
    precise_zip = os.path.join(OUTPUTS_DIR, f"result_precise_{timestamp}.zip")
    with zipfile.ZipFile(precise_zip, 'w') as zf:
        zf.write(annotated_path, arcname=os.path.basename(annotated_path))
        zf.write(json_path, arcname=os.path.basename(json_path))
        zf.write(csv_path, arcname=os.path.basename(csv_path))
    return precise_zip
