DermalScan: AI Facial Skin Aging Detection App
🧠 Overview

DermalScan is an AI-powered application that analyzes facial images to predict skin type and age indicators using a deep learning model trained with MobileNetV2 and image processing through OpenCV.
It helps users visualize aging signs and understand their skin health using annotated predictions and class probabilities.

🚀 Features

📸 Upload single or multiple facial images

🧩 Detect and visualize skin type, age, and facial features

🧠 Pre-trained deep learning model MobileNetV2 for classification

🎯 Annotated visualization with bounding boxes and class confidence

💾 Export results as annotated images and CSV files

🧍‍♀️ Web UI built with Streamlit for smooth interaction

🧰 Tech Stack
Area	Tools / Libraries
Image Ops	OpenCV, NumPy, Haarcascade
Model	TensorFlow/Keras, MobileNetV2
Frontend	Streamlit
Backend	Python, Modularized Inference
Evaluation	Accuracy, Loss, Confusion Matrix
Exporting	CSV, Annotated Image, PDF (optional)
📂 Repository Structure
AI-DermalScan/
│
├── app.py                    # Streamlit Frontend (UI)
├── backend.py                # Backend Logic & Model Integration
├── Shivani_AI_DermalScan.ipynb  # Model Training / Testing Notebook
│
├── Documents/
│   ├── AI_DermalScan.pdf
│   ├── DermalScan_Documentation.docx
│__ Requirement.txt  
│
├── uploads/                  # Sample Test Images
├── results/                  # Annotated Images & CSV Exports
│
├── LICENSE                   # License Information
└── README.md                 # Project Overview

⚙️ Installation

1️⃣ Clone this repository

git clone https://github.com/Springboard-Mentor-DermalScan/AI-DermalScan.git
cd AI-DermalScan


2️⃣ Install dependencies

pip install -r Requirement.txt


3️⃣ Run the application

streamlit run app.py

📈 Project Summary


DermalScan: AI Facial Skin Aging Detection App

Developed a deep learning–based system to detect and classify facial aging signs such as wrinkles, dark spots, puffy eyes, and clear skin. The model was trained using a pretrained MobileNetV2 network with data preprocessing, augmentation, and one-hot encoding for accuracy improvement.

Implemented face detection using Haar Cascades and built an end-to-end prediction pipeline capable of identifying multiple faces in an image, estimating age, and labeling each face with bounding boxes and confidence percentages.

Integrated a Streamlit-based frontend that enables users to upload images, view real-time annotated predictions, and download results as CSV files. The backend was modularized in Python to ensure smooth and efficient inference with a prediction time of less than 5 seconds per image.

All project files, including datasets, model scripts, and results, are documented and maintained on GitHub for transparency and reproducibility.
👩‍💻 Intern
Akkala Shivani Reddy
🎓 Malla Reddy Engineering College
🤝 Infosys Springboard Mentor – Praveen
