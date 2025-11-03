🌸 **AI DermalScan – Intelligent Facial Skin & Age Detection System**

From pixels to precision — Smart AI that understands your skin.  

---

## 🌟 Overview
**AI DermalScan** is an advanced AI-based dermatological analysis system that can analyze a facial image to predict:

- Your **skin condition** (such as *wrinkles, dark spots, puffy eyes,* or *clear skin*), and  
- Your **estimated biological age**.  

It uses **MTCNN** for robust face detection and a fine-tuned **DenseNet121** deep learning model for skin feature analysis and age estimation.  
The system runs seamlessly through **Streamlit**, offering a **Tech-Lab themed futuristic UI** that anyone can use directly from their browser.

---

## 🧠 Key Features
✅ **AI Skin Detection** – Detects facial skin conditions using a fine-tuned deep learning model.  
✅ **Face Detection (MTCNN)** – High-accuracy, multi-face detection.  
✅ **Age Prediction** – Estimates biological age based on facial features.  
✅ **Confidence Score** – Displays AI confidence for each detection.  
✅ **Multiple Face Support** – Analyzes and labels multiple faces per image.  
✅ **Fast & Optimized** – Lightweight model with near real-time inference.  
✅ **Modern UI** – Streamlit-powered futuristic interface with scan animations.

---

## ⚙️ Tools and Technologies

| **Category** | **Tools Used** |
|---------------|----------------|
| **Programming Language** | Python 3.12 |
| **Deep Learning** | TensorFlow, Keras |
| **Face Detection** | MTCNN |
| **Image Processing** | OpenCV, Pillow |
| **Frontend Framework** | Streamlit |
| **Data Handling** | NumPy, Pandas |
| **Visualization** | Matplotlib |
| **Model Architecture** | DenseNet121 *(Transfer Learning)* |

---

## 🧩 Model & Resource Setup
The project requires a pre-trained model file:  
**`densenet121_best_optimized.h5`**

Place it in your **project root directory** (same location as `backend.py` and `app.py`).  
The Streamlit frontend will automatically load this file at startup.

---

## 📁 Project Structure

AI_DermalScan/
│
├── app.py # Streamlit frontend (Tech-Lab themed)
├── backend.py # Image processing & AI inference logic
├── densenet121_best_optimized.h5 # Trained DenseNet121 model
│
├── dataset_extraction.ipynb # Dataset cleaning & preprocessing notebook
├── documentation/ # Reports, results, and documentation
├── test_images/ # Sample input images for testing
│
├── requirements.txt # Dependencies
└── README.md # Project guide (this file)


---

## 💻 How to Run the Project

### Step 1️⃣: Clone the Repository
```bash
git clone https://github.com/yourusername/AI-DermalScan.git
cd AI-DermalScan

Step 2️⃣: Create and Activate Virtual Environment
# Windows
python -m venv dermalscan_env
dermalscan_env\Scripts\activate

# Mac/Linux
python3 -m venv dermalscan_env
source dermalscan_env/bin/activate

Step 3️⃣: Install Dependencies
pip install -r requirements.txt

Step 4️⃣: Run the Streamlit App
streamlit run app.py

Then open the local URL shown in the terminal (usually http://localhost:8501
).

Step 5️⃣: Upload an Image

Upload a clear, front-facing JPG/PNG image (≤ 10MB).

Wait for the scan animation to finish.

The app will display:

Predicted skin condition

Confidence score

Estimated biological age

Annotated image preview

Downloadable CSV of results

| **Parameter**           | **Description**                                    |
| ----------------------- | -------------------------------------------------- |
| **Model Used**          | DenseNet121 (Fine-tuned)                           |
| **Face Detector**       | MTCNN                                              |
| **Input Image Size**    | 224 × 224 × 3                                      |
| **Optimizer**           | AdamW (Cosine LR Scheduler)                        |
| **Loss Function**       | Categorical Crossentropy *(Label Smoothing = 0.1)* |
| **Data Augmentation**   | Flip, Rotation, Brightness, Contrast               |
| **Validation Accuracy** | ~95%                                               |
| **Classes**             | wrinkles, darkspots, puffy_eyes, clear_skin        |

⚡ Performance Tips

Use @st.cache_resource to cache model & detector for faster inference.

Resize images ≤ 1024px before prediction.

For GPU acceleration:
pip install tensorflow[and-cuda]
Convert the model to TFLite for faster web & mobile deployment.

🚀 Future Scope

Real-time camera capture integration

Skin tone & texture analysis

Cloud-based API deployment for scalability

Model optimization for web/mobile inference

Dermatological recommendation system integration

📜 License

Released under the MIT License — you are free to use, modify, and distribute this project with proper attribution.

⚠️ Disclaimer

AI DermalScan is not a medical diagnostic tool.
It is designed purely for research and educational purposes.
For medical concerns, always consult a certified dermatologist.

👩‍💻 Developer

Shreya Bhat
B.E. in Computer Science and Engineering
RNS Institute of Technology, Bengaluru
