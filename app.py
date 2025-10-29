import streamlit as st
import os
import pandas as pd
import time
from backend import predict_image

# Streamlit Page Configuration
st.set_page_config(page_title="Facial Skin Analyzer", layout="wide")
st.title("🪞DermalScan:AI_Facial Skin Aging Detection App ")
st.write("Upload an image to detect facial skin conditions, predicted age, and total processing time.")

# Upload Image
uploaded_file = st.file_uploader("📤 Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    temp_dir = "uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    col1, col2 = st.columns(2)
    with col1:
        st.image(temp_path, caption="📸 Input Image", use_container_width=True)

    # Prediction Timer
    start_time = time.time()
    with st.spinner("🔍 Analyzing Image... Please wait"):
        output_path, results = predict_image(temp_path)
    total_time = time.time() - start_time

    with col2:
        st.image(output_path, caption="🧠 Annotated Output", use_container_width=True)

    st.markdown("### 📋 Prediction Results")

    if len(results) > 0:
        # Convert results to DataFrame
        df = pd.DataFrame(results)

        # Normalize labels (remove spaces and lowercase)
        df["Condition"] = df["Condition"].str.strip().str.lower()

        # Mapping class names to IDs
        class_map = {
            "clear face": 0,
            "darkspots": 1,
            "puffy eyes": 2,
            "wrinkles": 3
        }

        # Add Class ID column safely
        df["Class ID"] = df["Condition"].map(class_map).fillna(-1).astype(int)

        # Add total prediction time column
        df["Total Prediction Time (s)"] = round(total_time, 3)

        # Reorder columns cleanly
        df = df[[
            "Class ID",
            "Condition",
            "Confidence",
            "Estimated_Age",
            "x",
            "y",
            "width",
            "height",
            "Total Prediction Time (s)"
        ]]

        # Reset index to avoid showing index 0, 1, 2...
        df.reset_index(drop=True, inplace=True)

        # Display clean table
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Save results to CSV
        os.makedirs("results", exist_ok=True)
        csv_path = f"results/{os.path.splitext(uploaded_file.name)[0]}_results.csv"
        df.to_csv(csv_path, index=False)

        # Show total prediction time summary
        st.success(f"✅ Total Prediction Time: **{total_time:.3f} seconds**")

        # Download buttons
        col3, col4 = st.columns(2)
        with col3:
            with open(output_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Annotated Image",
                    data=f,
                    file_name="annotated_output.jpg",
                    mime="image/jpeg"
                )
        with col4:
            with open(csv_path, "rb") as f:
                st.download_button(
                    label="📊 Download Results (CSV)",
                    data=f,
                    file_name="prediction_results.csv",
                    mime="text/csv"
                )

    else:
        st.warning("⚠️ No faces detected. Try another image with a clearer view.")
else:
    st.info("📁 Please upload a facial image to start analysis.")
