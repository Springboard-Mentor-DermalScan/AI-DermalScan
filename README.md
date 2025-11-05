# DermalScan: AI Facial Skin Aging Detection App
### (A Machine Learning Model to Predict AI Facial Skin Aging)

**Infosys SpringBoard Virtual Internship Program**

**Submitted by:**  
Jatin Agrawal  

**Under the Guidance of Mentor:**  
Praveen  

---

## 📘 Project Overview

DermalScan is a hybrid **machine learning** and **deep learning** system designed to automatically detect and classify key facial aging signs such as wrinkles, dark spots, puffy eyes, and clear skin.  
It combines image preprocessing, classical machine learning benchmarking, transfer learning with CNNs, and a **Streamlit-based web interface** for real-time analysis and result visualization.

---

## 🧾 Table of Contents
1. [Data Preparation & Cleaning](#1-data-preparation--cleaning)  
2. [Preprocessing and Encoding](#2-preprocessing-and-encoding)  
3. [Train-Test Split](#3-train-test-split)  
4. [Classical Machine Learning Benchmarking](#4-classical-machine-learning-benchmarking)  
5. [Deep Learning Pipeline (Transfer Learning)](#5-deep-learning-pipeline-transfer-learning)  
6. [Face Detection and Annotation](#6-face-detection-and-annotation)  
7. [Batch Inference](#7-batch-inference)  
8. [Streamlit Web Application Features](#8-streamlit-web-application-features)  
9. [Technology Stack](#9-technology-stack)  
10. [Workflow Summary](#10-workflow-summary)  
11. [Example Outputs and Reports](#11-example-outputs-and-reports)  
12. [Learning Reflections](#12-learning-reflections)  
13. [Disclaimer](#13-disclaimer)

---

## 1. Data Preparation & Cleaning

- Images are stored in subfolders representing each skin condition class.  
- Corrupt or unreadable images are removed automatically.  
- Duplicate images are identified and removed using the **dHash algorithm**.  
- Cleaned data is structured as a Pandas DataFrame.
- Visualization through pie charts, histograms, and scatter plots ensures balanced and high-quality data.

---

## 2. Preprocessing and Encoding

- Images are resized to **224x224** pixels and converted to RGB.  
- Pixel values are normalized for model compatibility.  
- For ML models, images are flattened into 1D vectors.  
- Labels are encoded numerically using **LabelEncoder**.

---

## 3. Train-Test Split

- The dataset is divided into **80% training** and **20% testing** using **stratified sampling** to preserve class ratios.

---

## 4. Classical Machine Learning Benchmarking

Trained and compared the following models:
- Decision Tree  
- Random Forest  
- Support Vector Machines (SVM)  
- AdaBoost Ensemble  

**Evaluation Metrics:** Accuracy, Precision, Recall, and F1-score.  
**Optimization:** Hyperparameter tuning with **GridSearchCV** for Decision Tree and Random Forest to maximize performance.

---

## 5. Deep Learning Pipeline (Transfer Learning)

- Base model: **MobileNetV2** pretrained on ImageNet.  
- Added custom layers: GlobalAveragePooling, Dense (ReLU), BatchNormalization, Dropout, and Softmax output.  
- Optimizer: **Adam (lr=0.001)** with **Categorical Crossentropy** loss.  
- Implemented callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau.  
- Achieved **Test Accuracy: 87%** and **Test Loss: 0.3088**.

---

## 6. Face Detection and Annotation

- Face detection via **OpenCV Haar Cascade**.  
- Supports multiple face detection per image.  
- Cropped, resized, and normalized face regions pass through the trained model.  
- Output includes **label**, **confidence score**, and **age group** annotation displayed on the image.

---

## 7. Batch Inference

- Batch processing for multiple test images.  
- Automated detection, prediction, and annotation pipeline.  
- Saves annotated images and CSV summaries for each run.

---

## 8. Streamlit Web Application Features

### Pages
- **Home:** Overview and app introduction.  
- **Upload & Analyze:** Upload images, detect faces, and view predicted skin signs with confidence scores.  
- **Reports:** Access previous analyses, view interactive plots, and metrics in real time.  
- **About:** Covers methodology, technologies used, and disclaimers.

### Highlights
- AI animation during loading.  
- Modern gradient background for an engaging interface.  
- Export results as **CSV** and **annotated images**.

---

## 9. Technology Stack

**Backend:** Python, OpenCV, NumPy, Pandas, Scikit-learn, TensorFlow, Keras, Matplotlib, Seaborn  
**Frontend:** Streamlit, Plotly, Pillow (PIL)  
**Tools:** Jupyter Notebook, VS Code  
**Face Detection:** OpenCV Haar Cascade  
**Hardware Requirements:** 4GB RAM (minimum), GPU support recommended

---

## 10. Workflow Summary

| Step | Description |
|------|--------------|
| Upload Data | Images organized by class folders |
| Clean Data | Remove corrupt/duplicate images |
| Visualize | Generate class distribution plots |
| Preprocess | Resize, normalize, encode labels |
| Classical ML | Train/test split, baseline models |
| Tune ML | Optimize with GridSearchCV |
| Deep Learning | Train MobileNetV2 model |
| Detect Face | Use Haar Cascade for detection |
| Batch Infer | Predict and annotate images |
| Web UI | Streamlit-based live analysis and export |

---

## 11. Example Outputs and Reports

- Annotated facial images with bounding boxes and class labels.  
- Detailed accuracy, confusion matrices, and F1 metrics.  
- JSON logs and graphical summaries of model predictions.  
- Model metadata like file size, date, and training metrics retained.

---

## 12. Learning Reflections

- Comprehensive dataset cleaning and balancing improved model robustness.  
- Comparison of classical and deep models highlighted transfer learning benefits.  
- Successfully integrated ML analysis into a web-based interactive tool.  
- Enhanced understanding of AI deployment through Streamlit.

---

## 13. Disclaimer

DermalScan AI is strictly for **educational and informational purposes**.  
It is **not a medical diagnostic or treatment tool** and should not be used in clinical decision-making.

---

## 📎 Repository Details

- **Project Type:** Deep Learning + Streamlit Web Application  
- **Language:** Python (3.9+)  
- **Frameworks:** TensorFlow, Keras, Scikit-learn  
- **Interface:** Streamlit  
- **File Outputs:** Annotated Images, CSV Reports

---

## 🚀 Run Instructions

### Prerequisites
- Python 3.9+
- pip (Python package manager)

### Setup

git clone https://github.com/YourUsername/DermalScan-AI.git
cd DermalScan-AI
python -m venv venv
source venv/bin/activate # or .\venv\Scripts\activate (Windows)
pip install -r requirements.txt

### Run Application

streamlit run app.py


Then open the displayed local URL in your web browser to interact with the DermalScan interface.

---

## 🧑‍💻 Contributors

- **Developer:** Jatin Agrawal  
- **Mentor:** Praveen  
- **Program:** Infosys SpringBoard Virtual Internship Program  

---

## 📄 License
This project is released for academic and non-commercial use, subject to appropriate citation of the author and mentorship program.

