#  AI DermalScan: Facial Skin Disease Detection using Deep Learning

**DermalScan** is an AI-powered dermatological assistant that analyzes facial skin images to detect and classify skin diseases and early aging indicators using **DenseNet121**.  
The system provides **real-time inference**, **annotated predictions**, and **exportable reports** to assist in dermatological research and skincare analysis.

---

##  Project Summary

The **DermalScan** project integrates **Deep Learning**, **Computer Vision**, and **Web-based AI deployment** to automate the detection of facial skin conditions.

###  Core Objectives
- Automate skin type and aging sign detection from facial images.  
- Provide real-time prediction (<5 seconds per image).  
- Offer easy-to-use visualization through a web interface.  
- Allow export of analyzed results for research or clinical use.

###  How It Works
1. **Face Detection:** Performed using OpenCV Haar Cascade.  
2. **Feature Extraction:** Handled by the pre-trained **DenseNet121** model fine-tuned on dermatological datasets.  
3. **Prediction:** The model classifies the detected skin region into predefined skin type/aging categories.  
4. **Visualization:** The results are displayed with **confidence scores** and **bounding box annotations**.  
5. **Export:** Users can download a **CSV** or **annotated image** for record-keeping.

---

##  Web UI Overview

An intuitive **Streamlit-based web interface** that enables:
-  Image upload (single/multiple)
-  Real-time predictions with bounding boxes
-  Confidence scores and class labels
-  Export options: CSV & annotated images

---

##  Tech Stack

| Area | Tools / Libraries |
|------|--------------------|
| **Image Operations** | OpenCV, NumPy, Haarcascade |
| **Model** | TensorFlow / Keras, **DenseNet121** |
| **Frontend** | Streamlit |
| **Backend** | Python |
| **Evaluation** | Accuracy, Loss, Confusion Matrix |
| **Export Options** | CSV, Annotated Image |

---

##  Installation

### 1. Clone this repository

git clone https://github.com/Springboard-Mentor-DermalScan/AI-DermalScan.git
cd AI-DermalScan
###  2️. Install Dependencies
pip install -r requirement.txt
### 3️. Run the Application
streamlit run app.py

### 4️. Upload & Analyze

- Upload a facial image (.jpg / .png).

- The app will detect facial regions using Haar Cascade.

- It will classify skin aging signs (wrinkles, dark spots, puffy eyes, clear skin) using DenseNet121.

- View annotated predictions and download results (CSV + image).


 **Intern:** Arpeeta Mohanty  
  *C.V. Raman Global University*  

 **Mentor:** Praveen (Infosys Springboard)



