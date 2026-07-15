import os
import sys
import tempfile

# =====================================================
# Add Project Root to Python Path
# =====================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# =====================================================
# Imports
# =====================================================

import streamlit as st
from PIL import Image

from src.prediction.predictor import TrafficSignPredictor

# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="AI Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide"
)

# =====================================================
# Title
# =====================================================

st.title("🚦 AI-Powered Traffic Sign Recognition System")

st.write(
    "Upload a traffic sign image and let MobileNetV2 classify it."
)

st.divider()

# =====================================================
# Load Predictor
# =====================================================

predictor = TrafficSignPredictor()

# =====================================================
# Upload Image
# =====================================================

uploaded_file = st.file_uploader(
    "Choose a Traffic Sign Image",
    type=["png", "jpg", "jpeg"]
)

# =====================================================
# Prediction
# =====================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".png"
    ) as temp:

        image.save(temp.name)

        result = predictor.predict(temp.name)

    with col2:

        st.subheader("Prediction")

        st.success(result["class_name"])

        st.metric(
            "Confidence",
            f"{result['confidence']:.2f}%"
        )

        st.metric(
            "Inference Time",
            f"{result['inference_time']:.3f} sec"
        )

        st.metric(
            "Class ID",
            result["class_id"]
        )