🧠 AI DermalScan

From pixels to precision — Smart AI that understands your skin.

🌟 Overview

AI DermalScan is an advanced AI-based dermatological analysis project that can analyze a facial image to predict:

Your skin condition (such as wrinkles, dark spots, puffy eyes, or clear skin), and

Your estimated biological age.

It uses MTCNN for accurate face detection and a fine-tuned DenseNet121 deep learning model for skin feature analysis and age estimation.
The app runs seamlessly through Streamlit, offering an elegant Tech-Lab themed interface that anyone can use right in the browser.

🚀 Key Features

✅ AI Skin Detection – Detects facial skin issues using a fine-tuned deep learning model.
✅ Face Detection (MTCNN) – Robust multi-face localization with high accuracy.
✅ Age Prediction – Estimates biological age based on facial features.
✅ Confidence Score – Displays AI confidence in each detection.
✅ Multiple Face Support – Detects and labels more than one face in an image.
✅ Fast & Optimized – Lightweight model with near real-time inference.
✅ Modern UI – Streamlit-powered interface with scan animations and futuristic visuals.

⚙️ Tools and Technologies
Category	Tools Used
Programming Language	Python 3.12
Deep Learning	TensorFlow, Keras
Face Detection	MTCNN
Image Processing	OpenCV, Pillow
Frontend	Streamlit
Data Handling	NumPy, Pandas
Visualization	Matplotlib
Model Architecture	DenseNet121 (Transfer Learning)
🧩 Model & Resource Setup

The project requires a trained model file:

densenet121_best_optimized.h5

Place it inside your project folder (same directory as backend.py and app.py).
The Streamlit app will automatically load this file during startup.

📁 Project Structure
AI_DermalScan/
│
├── app.py                        # Streamlit frontend (Tech-Lab themed)
├── backend.py                    # Image processing + AI inference logic
├── densenet121_best_optimized.h5 # Trained model weights
│
├── dataset_extraction.ipynb      # Dataset cleanup and preprocessing
├── documentation/                # Project documentation and reports
├── test_images/                  # Sample images for testing
│
├── requirements.txt              # Dependencies
└── README.md                     # Project guide (this file)

💻 How to Run the Project
Step 1️⃣: Clone the Repository
git clone https://github.com/yourusername/AI-DermalScan.git
cd AI-DermalScan

Step 2️⃣: Create and Activate a Virtual Environment
python -m venv dermalscan_env
dermalscan_env\Scripts\activate     # (Windows)
# OR
source dermalscan_env/bin/activate  # (Mac/Linux)

Step 3️⃣: Install Required Libraries
pip install -r requirements.txt

Step 4️⃣: Run the Streamlit App
streamlit run app.py

Step 5️⃣: Upload an Image

Upload a clear, front-facing image (JPG/PNG ≤ 10MB).

Wait for the scan animation to finish.

The app will display:

Predicted skin condition

Confidence score

Estimated biological age

Annotated image preview

Downloadable CSV of results

🧠 Model Details
Parameter	Description
Model Used	DenseNet121
Face Detector	MTCNN
Input Image Size	224 × 224 × 3
Optimizer	AdamW (Cosine LR Scheduler)
Loss Function	Categorical Crossentropy (Label Smoothing = 0.1)
Data Augmentation	Flip, rotation, brightness, contrast
Validation Accuracy	~95%
Classes	wrinkles, darkspots, puffy_eyes, clear_skin
⚡ Performance Tips

Use @st.cache_resource to cache model and detector.

Resize images ≤ 1024px before inference.

For GPU acceleration:

pip install tensorflow[and-cuda]


Convert model to TFLite for faster loading and deployment.

📜 License

Released under the MIT License.
You are free to use, modify, and distribute this project with proper attribution.

⚠️ Disclaimer

DermalScan is not a medical diagnostic tool.
It is developed for research and educational purposes only.
Always consult a dermatologist for professional medical advice.

👩‍💻 Author: Shreya Bhat
🚀 Building AI that cares for your skin.
