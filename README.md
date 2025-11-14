📸 DermalScan: AI Facial Skin Analysis

🌟 Project At a Glance
DermalScan is an advanced, AI-powered web application designed to analyze facial images for signs of aging and common skin conditions. It leverages a fine-tuned deep learning model (MobileNetV2) and robust image processing (OpenCV) to offer users detailed, annotated insights into their skin health and estimated age indicators.


The application is deployed via Streamlit, providing a fast, interactive experience for real-time analysis.


💡 Core Functionality
1. Deep Learning Classification
Model: A pre-trained MobileNetV2 model, specialized through rigorous training, serves as the core classification engine.

Goal: To accurately classify the dominant skin condition in the analyzed facial region.

2. Condition Detection
The model is trained to identify and categorize the following key indicators:


✅ Clear Skin

👵 Wrinkles

⚫ Dark Spots

👁️ Puffy Eyes

3. End-to-End Analysis Pipeline
Face Detection: Uses OpenCV's Haar Cascade classifier for precise, real-time localization of faces, even when multiple faces are present.

Prediction & Annotation: After processing the cropped face, the application displays results directly on the image, including:

A bounding box around the detected face.

The predicted skin condition and its confidence score.

An estimate of the associated biological age range.
Category,Key Tools / Libraries,Purpose in DermalScan
User Interface (UI),Streamlit,"Interactive, smooth, and deployment-ready web frontend."
Deep Learning,"TensorFlow/Keras, MobileNetV2","Training, inference, and core model architecture."
Computer Vision,"OpenCV, Haar Cascade","Face detection, image preprocessing, and result drawing."
Data Handling,"Pandas, NumPy","Handling data augmentations, result logging, and CSV export."
Backend,Python (Modularized),Ensures robust and quick (under 5 seconds) inference execution.

    Installation and Run

Clone the Repository:
git clone https://github.com/Springboard-Mentor-DermalScan/AI-DermalScan.git
cd AI-DermalScan

Install Dependencies:
pip install -r requirement.txt

 👩Intern: Vennu Bhavana🎓 Vijaya Institute of technology for women 🤝 Infosys Springboard Mentor – Praveen
