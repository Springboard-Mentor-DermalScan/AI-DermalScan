DermalScan: AI Facial Skin Aging Detection App

This project is an interactive web application that uses deep learning to analyze facial images for common signs of aging (wrinkles, dark spots, puffy eyes) and estimate the user's age and gender.

The application is built in Python using Streamlit for the frontend and a fine-tuned InceptionV3 model (trained with TensorFlow/Keras) for skin sign classification. Age and gender estimation are performed using the deepface library.

🌟 Features

File Upload: Simple drag-and-drop or file selection for jpg, jpeg, and png images.

Face Detection: Automatically locates faces in the uploaded image using OpenCV's Haar Cascades.

Skin Sign Classification: Analyzes each detected face and provides a percentage breakdown for four categories:

Wrinkles

Dark Spots

Puffy Eyes

Clear Skin

Age Estimation: Provides an estimated age for each detected face.

Visual Feedback: Displays the original image alongside the annotated image, showing bounding boxes and prediction results.

Export Results:

Download Annotated Image: Save a PNG of the resulting image with annotations.

Download CSV: Save a .csv file with the detailed prediction percentages for each detected face.

🚀 Tech Stack

Backend & ML: Python, TensorFlow, Keras

Web Framework: Streamlit

Image Processing: OpenCV, Pillow

Facial Analysis: DeepFace

Data Handling: Pandas, NumPy

Development: Jupyter Notebook (for model training), VS Code

📁 Project Structure

DermalScan/
│├── 📁 test/

│   └── 📄 1.jpg 

│   └── 📄 2.jpg 

│   └── 📄 3.jpg 

│   └── 📄 4.jpg 

│   └── 📄 5.jpg 

│├── 📁 Documentation/

│   └── 📄 DermalScan(Deepika).pdf 

├── 📄 app1.py                     # The main Streamlit application
├── 📄 Project.ipynb               # Jupyter Notebook used for training
├── 📄 README.md                   # This file
└── 📄 requirements.txt            # Python dependencies


🔧 Setup & Installation

Follow these steps to set up and run the project on your local machine.

1. Prerequisites

Python 3.9+

pip (Python package installer)

2. Clone Repository

Clone this project to your local machine (or download the source files).

git clone https://github.com/Springboard-Mentor-DermalScan/AI-DermalScan.git

cd DermalScan


3. Create a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

# Create a virtual environment
python -m venv venv

# Activate the environment (Windows)
.\venv\Scripts\activate

# Activate the environment (macOS/Linux)
source venv/bin/activate


4. Install Dependencies

Install all required libraries from the requirements.txt file.

pip install -r requirements.txt


5. Download Model Files

This project requires one pre-trained model file that must be placed in the root directory:

best_inceptionv3_model.h5: This is the skin sign classifier trained in Module 3. Place this file in the main DermalScan/ folder.

(Note: The age/gender models are downloaded automatically by the deepface library, and the Haar Cascade file is loaded automatically from opencv-python)

🏃‍♂️ How to Use

There are two main parts to this project: running the pre-trained application and (optionally) re-training the model.

1. Running the Streamlit Application (Main)

After completing the setup, run the following command in your terminal from the project's root directory:

streamlit run app1.py


Streamlit will automatically open a new tab in your web browser.

Drag and drop an image file (or click "Browse files") to upload a photo.

Wait for the analysis to complete (a spinner will appear).

View the "Analysis Result" on the right, showing the image with annotations.

Below the result, use the buttons to download the annotated image or a CSV file of the predictions.

2. (Optional) Re-training the Model

If you want to re-train the skin sign classification model (Module 3):

Ensure your dataset is organized correctly in the dataset/ folder, with subfolders for each class.

Make sure you have jupyter notebook, matplotlib, and scikit-learn installed (pip install jupyter matplotlib scikit-learn).

Run Jupyter Notebook:

jupyter notebook


Open the DermalScan_Training.ipynb file and run the cells in order. The notebook will generate a new best_inceptionv3_model.h5 file.

📊 Model Performance

The skin sign classification model (InceptionV3) was fine-tuned on a 4-class dataset. The final model achieved:

Final Training Accuracy: 98.75%

Final Validation Accuracy: 89.74% (peaked at 91.03%)

This high accuracy demonstrates the model's effectiveness in distinguishing between the defined skin signs on the provided dataset.

⚠️ Limitations

Not a Medical Diagnosis: This tool is for informational and educational purposes only. The predictions are not medical advice and should not be used to diagnose or treat any health condition.

Image Quality: Prediction accuracy depends heavily on the input image quality (lighting, pose, resolution, occlusions).

Dataset Bias: The model's performance reflects the dataset it was trained on. It may perform differently on skin tones or conditions not well-represented in the data.

Age Estimation Inaccuracy: The deepface library, while robust, is known to be inaccurate for certain demographics, especially infants and very young children (e.g., it may predict 17 for a baby).
