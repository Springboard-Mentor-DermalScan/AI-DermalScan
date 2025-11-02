🧠 DermalScan: AI Facial Skin Aging Detection App

This project is an interactive AI-powered web application that uses deep learning to analyze facial images for common signs of aging — such as wrinkles, dark spots, and puffy eyes — and estimate the user's age.

The application is built in Python using Streamlit for the web interface and a fine-tuned ResNet50 model (trained with TensorFlow/Keras) for facial skin condition classification. The app automatically detects faces, classifies visible skin signs, estimates approximate age, and provides an intuitive visual and analytical summary.

🌟 Features
🖼️ File Upload

Simple drag-and-drop or file selection for .jpg, .jpeg, and .png images.

👁️ Face Detection

Automatically detects faces in uploaded images using OpenCV’s Haar Cascade Classifier.

🔍 Skin Sign Classification

Analyzes detected faces and provides a percentage breakdown for four categories:

Wrinkles

Dark Spots

Puffy Eyes

Clear Skin

🎚️ Age Estimation

Provides an approximate predicted age for each detected face using model-calibrated regression.

📊 Visual Feedback

Displays both the original image and the annotated image with:

Bounding boxes

Predicted age

Confidence percentages for each skin sign

💾 Export Results

Download Annotated Image — Save a .png of the processed image with visual labels.

Download CSV — Export a .csv file containing detailed prediction data (confidence scores, age, bounding box values, and processing time).

🚀 Tech Stack
Category	Technologies Used
Backend & ML	Python, TensorFlow, Keras
Web Framework	Streamlit
Image Processing	OpenCV, Pillow
Data Handling	NumPy, Pandas
Visualization	Plotly, Matplotlib
Development Tools	VS Code, Jupyter Notebook
📁 Project Structure
DermalScan/
│
├── 📁 test/
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── 3.jpg
│   ├── 4.jpg
│   └── 5.jpg
│
├── 📁 Documentation/
│   └── ProjectDocumnetation1.pdf
│
├── 📄 app.py                     # Main Streamlit web application
├── 📄 AI_DERMALSCAN.py           # Model training and backend pipeline
├── 📄 AI_DERMAL.ipynb              # Optional Jupyter Notebook for retraining
├── 📄 README.md                  # Project documentation (this file)
└── 📄 requirements.txt           # Python dependencies

🔧 Setup & Installation

Follow these steps to set up and run the project on your local system.

1️⃣ Prerequisites

Python 3.9 or higher

pip (Python package manager)

A GPU is recommended for model training (optional)

2️⃣ Clone the Repository
git clone https://github.com/Springboard-Mentor-DermalScan/AI-DermalScan.git
cd DermalScan

3️⃣ Create a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

# Create a virtual environment
python -m venv venv

# Activate the environment (Windows)
.\venv\Scripts\activate

# Activate the environment (macOS/Linux)
source venv/bin/activate

4️⃣ Install Dependencies

Install all required libraries from the requirements.txt file.

pip install -r requirements.txt

5️⃣ Download / Generate Model Files

The application requires a trained ResNet50 model file in the project directory.

To generate this model, open and run all cells in AI_DERMALSCAN.py or Project.ipynb.

The script will output a trained model file named:

resnet50_multitask_dermal_age.h5


Place this file in the main DermalScan/ folder.

⚙️ Note: The Haar Cascade file for face detection (haarcascade_frontalface_default.xml) is automatically provided by OpenCV.

🏃‍♂️ How to Use

There are two main modes to use this project:

1️⃣ Running the Pre-trained Streamlit Application

Once the model and dependencies are ready, launch the web interface:

streamlit run app.py


Then:

A browser tab will open automatically.

Upload a clear, front-facing image under good lighting.

Click Run Analysis to begin processing.

Wait a few seconds for detection and analysis.

View:

Annotated image with bounding boxes.

Predicted age and confidence scores.

Option to download image and results in CSV format.

2️⃣ (Optional) Re-training the Model

If you want to train the model from scratch:

Organize your dataset in the following format:

dataset/
├── clear face/
├── darkspots/
├── puffy eyes/
└── wrinkles/


Ensure the following libraries are installed:

pip install jupyter matplotlib scikit-learn


Run the training pipeline:

python AI_DERMALSCAN.py


or open and execute Project.ipynb to train interactively in Jupyter Notebook.

📊 Model Performance

The fine-tuned ResNet50 model achieved the following results on a four-class dataset:

Metric	Accuracy
Final Training Accuracy	98.75%
Final Validation Accuracy	89.74% (peaked at 91.03%)
Test Accuracy	88.9%
Best Validation Epoch	Epoch 34 (Fine-tuning phase)

This performance demonstrates the model’s robustness and reliability in detecting visible facial skin conditions.

⚠️ Limitations

Not for Medical Diagnosis
This system is for educational and informational use only. It does not provide medical or dermatological advice.

Image Quality Sensitivity
Predictions may vary depending on lighting, angle, facial expressions, and image clarity.

Dataset Bias
The model performance depends on the diversity of training data. It may underperform on underrepresented skin tones or rare conditions.

Age Estimation Accuracy
Predicted age is approximate and calibrated for general adult facial features; it may not be accurate for very young or elderly subjects.

🧠 Behind the Scenes

Face Detection: Multi-scale Haar Cascade detection (OpenCV)

Preprocessing: CLAHE (contrast enhancement), denoising, and resizing

Model: Multi-task ResNet50 — classifies skin signs and predicts age

Ensemble Prediction: Uses five augmented face versions for stable averaging

Visualization: Annotated bounding boxes and confidence bars (Plotly)

Export: Results saved in .json, .csv, and .png formats

🧩 Sample Output

Input: User uploads a clear face image
Output:

Predicted Age: 28 years

Dominant Condition: Wrinkles (83.2% confidence)

Annotated image with green bounding box

CSV summary of all detected conditions

💡 Future Enhancements

Integration with YOLOv8 or EfficientNet for faster real-time detection.

Incorporate gender classification and skin tone balancing.

Expand dataset for more diverse demographics.

Add live webcam detection mode via Streamlit.

🧾 Credits

Developed by Team DermalScan
Department of Computer Science and Engineering (CSE)
Under the guidance of project mentors.

© 2025 DermalScan | For Educational & Research Purposes Only
