# 💎 AI DermalScan – Facial Aging Sign Detection

**AI DermalScan** is a deep learning–based web application that detects **facial aging signs** such as *wrinkles*, *dark spots*, and *puffy eyes*, while also estimating the **approximate age** of the person using a fine-tuned **DenseNet121** model.  
The app is built with **Streamlit** for an intuitive and modern user interface.

---

## 🚀 Features

✅ Real-time **Face Detection** using OpenCV Haar Cascade  
✅ Accurate **Aging Sign Classification** (`clear face`, `darkspots`, `puffy eyes`, `wrinkles`)  
✅ Logical **Age Estimation** based on detected facial condition  
✅ **Streamlit-based UI** with dark gradient theme  
✅ **Downloadable Results** – annotated image + prediction CSV  
✅ **Automatic CSV Logging** of all predictions  
✅ Average **processing time under 5 seconds**

---

## ⚙️ Project Structure

```bash
AI_DermalScan/
│
├── Documentation/
│   └── Naman Kapoor(AI_DermalScan) Documentation.pdf
├── Naman Kapoor(AI_DermalScan).ipynb    # Model training & experimentation notebook
├── test_images/                         # Sample test images                 
├── app.py                               # Streamlit frontend (UI + interaction)
├── backend.py                           # Model loading & prediction logic
├── haarcascade_frontalface_default.xml  # Face detector
├── requirements.txt                     # Dependencies
├── prediction_log.csv                   # Auto-generated prediction records
└── README.md                            # You are here
```
---

## 🧠 Model Overview

| Parameter | Details |
|------------|----------|
| **Base Architecture** | DenseNet121 (Transfer Learning) |
| **Input Size** | 224 × 224 pixels |
| **Optimizer** | Adam |
| **Loss Function** | Categorical Crossentropy |
| **Framework** | TensorFlow / Keras |
| **Augmentation** | Rotation, Zoom, Flip, Shift |

---

## 🧩 Tech Stack

| Layer | Technology |
|--------|-------------|
| **Frontend** | Streamlit |
| **Backend** | TensorFlow / Keras |
| **Detection** | OpenCV Haar Cascade |
| **Data Handling** | Pandas, NumPy |
| **Visualization** | Matplotlib, Streamlit |
| **Logging** | CSV via Pandas |

---

## ⚙️ Setup & Installation 

### 1️⃣ Clone the Repository 
```bash
git clone -b Naman https://github.com/Springboard-Mentor-DermalScan/AI-DermalScan.git
cd AI-DermalScan
```
### 2️⃣ Create and Activate Virtual Environment(Recommended)
```bash
python -m venv dermalscan_env

# 👉 For Windows
dermalscan_env\Scripts\activate

# 👉 For macOS/Linux
source dermalscan_env/bin/activate
```
### 3️⃣ Install All Required Dependencies
```
pip install -r requirements.txt
```
### 4️⃣ Generate Trained Model File 
```
The pretrained model file DenseNet121_best_model.h5
is not included directly in the repository.
You can easily generate it by running the following Jupyter Notebook: 
Naman Kapoor(AI_DermalScan).ipynb available in repo.
Once executed successfully, it will save the file:
DenseNet121_best_model.h5  ✅
```
### 5️⃣ Ensure Haar Cascade File Exists for Face Detection
```
# This file is used by OpenCV to detect faces before classification.
# The required file 'haarcascade_frontalface_default.xml' is already included.
```
### 6️⃣ Run the Streamlit Application
```
streamlit run app.py

The application will automatically open in your browser:
http://localhost:8501

You can now upload an image → get real-time predictions → download results.
```
### 7️⃣ Test Images(Optional) 
```
After the Streamlit app is running, you can test with sample images provided in:
AI_DermalScan/test_images/
Files include:
  test1.jpg
  test2.jpg
  test3.jpg
  test4.jpg

Upload these from the Streamlit sidebar to validate the model output.
```
### 8️⃣ View Prediction Logs(Optional)
```
Every prediction is automatically saved to:
prediction_log.csv
You can open this file in Excel or any CSV viewer to see:
 Timestamp, Filename, Bounding_Box, Predicted_Sign, Confidence, Estimated_Age

✅ Setup Complete!
You are now ready to explore AI DermalScan’s facial aging sign detection.
```

---

## 📊 Output Example

**Uploaded Image → Annotated Result**

✅ **Detected Sign:** Wrinkles  
📊 **Confidence:** 92.5%  
🎯 **Estimated Age:** 68 years  
⚡ **Processing Time:** 3.42 seconds  

The model successfully identifies visible facial aging signs and overlays bounding boxes with predicted class, confidence score, and estimated age.

---

## 👥 Contributors

- **Intern:** Naman Kapoor  
- **Mentor:** Praveen (Infosys Springboard)

