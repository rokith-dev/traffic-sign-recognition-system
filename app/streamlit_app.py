import os
import sys
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

# ======================================================
# Project Path
# ======================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.preprocessing.preprocess import preprocess_image
from src.utils.class_names import CLASS_NAMES

# ======================================================
# Page Configuration
# ======================================================

st.set_page_config(
    page_title="AI Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide"
)

# ======================================================
# Sidebar
# ======================================================

st.sidebar.title("🚦 Project Information")

st.sidebar.success("Model Loaded Successfully")

st.sidebar.markdown("### Model Details")

st.sidebar.write("**Model:** Custom CNN")

st.sidebar.write("**Dataset:** 4145 Images")

st.sidebar.write("**Classes:** 58")

st.sidebar.write("**Accuracy:** 91.56%")

st.sidebar.markdown("---")

st.sidebar.write("Developed by")

st.sidebar.write("**Rokith**")

# ======================================================
# Load Model
# ======================================================

model = load_model("models/traffic_sign_cnn.keras")

# ======================================================
# Main Title
# ======================================================

st.title("🚦 AI-Powered Traffic Sign Recognition System")

st.write(
    "Upload a traffic sign image and let the AI predict the traffic sign."
)

st.divider()

# ======================================================
# Upload Image
# ======================================================

uploaded_file = st.file_uploader(
    "Upload Traffic Sign Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🚀 Predict"):

        processed = preprocess_image(image)

        processed = np.expand_dims(processed, axis=0)

        prediction = model.predict(processed)

        predicted_class = np.argmax(prediction)

        confidence = float(np.max(prediction) * 100)

        sign_name = CLASS_NAMES[predicted_class]

        st.divider()

        st.subheader("Prediction Result")

        st.success(f"🚸 Traffic Sign : {sign_name}")

        st.metric(
            label="Prediction Confidence",
            value=f"{confidence:.2f}%"
        )

        st.progress(confidence / 100)

        st.info(
            "The prediction is generated using the trained CNN model."
        )