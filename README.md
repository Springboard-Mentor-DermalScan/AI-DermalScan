💆‍♀️ Dermal Scan – AI-Powered Skin Analysis App

Dermal Scan is an AI-driven web application that analyzes facial skin images to detect and classify common skin conditions like wrinkles, dark spots, puffy eyes, and clear skin.
Developed using TensorFlow, OpenCV, and Streamlit, it offers an elegant and interactive interface for real-time dermatological insights.

🚀 Key Features

🧠 AI-Powered Detection – Deep learning–based model for precise skin feature classification.

📸 Real-Time Image Upload – Upload a facial or skin image for instant analysis.

🎨 Elegant UI – Pastel lavender theme with soft visuals and readable black text.

⚡ Fast & Accurate – Optimized backend inference with TensorFlow and OpenCV.

📥 Download Option – Export analysis results as a downloadable report.

🧩 Tech Stack
Layer	Technology
Frontend	Streamlit, HTML/CSS
Backend	Python, TensorFlow, OpenCV, NumPy
Model	Convolutional Neural Network (CNN)
Libraries	Keras, Pillow, Time, OS
Deployment	Streamlit Web App
🧠 Model Overview

The Convolutional Neural Network (CNN) model was trained on facial skin images to detect:

🕓 Wrinkles – Fine lines indicating aging.

🌑 Dark Spots – Hyperpigmentation or blemishes.

👁️ Puffy Eyes – Under-eye puffiness or swelling.

🌸 Clear Skin – Smooth, healthy, and even-toned skin.

Each prediction includes a confidence percentage indicating model certainty.

🖥️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/dermal-scan.git
cd dermal-scan

2️⃣ Create and Activate Virtual Environment
python -m venv venv
venv\Scripts\activate       # Windows
# or
source venv/bin/activate    # macOS/Linux

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Application
streamlit run app.py


Then open your browser and go to 👉 http://localhost:8501



🌈 User Interface

Home Page:

Upload a facial or skin image (.jpg, .jpeg, .png).

Click Analyze to view prediction results.

Displays detected condition and model confidence.

Option to Download Report for reference.

🧾 Example Output
Image	Predicted Condition	Confidence
face1.jpg	Wrinkles	93.6%

face2.jpg	Clear Skin	97.1%

face3.jpg	Puffy Eyes	89.4%

face4.jpg	Dark Spots	95.2%

🌟 Future Enhancements

📊 Severity-level classification (mild/moderate/severe).

🩺 Integration with dermatologist feedback API.

📱 Responsive mobile interface.

☁️ Cloud-based model hosting for faster processing.

💡 Personalized skincare recommendations.
