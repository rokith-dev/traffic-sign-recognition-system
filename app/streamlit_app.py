import os
import sys
import tempfile
from contextlib import suppress

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
from PIL import Image, UnidentifiedImageError

from src.prediction.predictor import TrafficSignPredictor

# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="AI Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide"
)


def inject_custom_styles() -> None:
    """Apply a compact visual system for the dashboard."""

    st.markdown(
        """
        <style>
            .stApp {
                background: radial-gradient(circle at top, #f7fbff 0%, #eef4fb 55%, #e7edf5 100%);
            }

            .hero-card {
                padding: 1.5rem 1.75rem;
                border-radius: 1.25rem;
                background: linear-gradient(135deg, rgba(12, 19, 31, 0.96), rgba(17, 55, 100, 0.92));
                color: white;
                border: 1px solid rgba(255, 255, 255, 0.08);
                box-shadow: 0 18px 50px rgba(12, 19, 31, 0.16);
            }

            .hero-card h1 {
                margin: 0;
                font-size: 2.35rem;
                line-height: 1.1;
            }

            .hero-card p {
                margin: 0.7rem 0 0;
                max-width: 62rem;
                color: rgba(255, 255, 255, 0.82);
                font-size: 1rem;
            }

            .metric-card {
                padding: 1rem 1.1rem;
                border-radius: 1rem;
                background: rgba(255, 255, 255, 0.82);
                border: 1px solid rgba(10, 24, 43, 0.08);
                box-shadow: 0 10px 30px rgba(16, 24, 40, 0.06);
            }

            .prediction-pill {
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.55rem 0.85rem;
                border-radius: 999px;
                background: linear-gradient(135deg, #0f7b6c, #12a480);
                color: white;
                font-weight: 600;
                margin-bottom: 0.5rem;
            }

            .top5-item {
                padding: 0.7rem 0.8rem;
                border-radius: 0.85rem;
                background: rgba(248, 250, 252, 0.95);
                border: 1px solid rgba(15, 23, 42, 0.06);
                margin-bottom: 0.6rem;
            }

            .sidebar-note {
                padding: 0.95rem;
                border-radius: 0.9rem;
                background: rgba(15, 23, 42, 0.05);
                border: 1px solid rgba(15, 23, 42, 0.08);
            }
        </style>
        """,
        unsafe_allow_html=True
    )


@st.cache_resource(show_spinner=False)
def load_predictor() -> TrafficSignPredictor:
    """Load the model once and reuse it across Streamlit reruns."""

    return TrafficSignPredictor()


def render_sidebar() -> None:
    """Show app guidance and model context."""

    with st.sidebar:
        st.markdown("## 🚦 Dashboard")
        st.markdown(
            """
            <div class="sidebar-note">
                Upload a traffic sign image, review the predicted class, and inspect the top-5 alternatives.
                The model is preloaded with caching so the page stays responsive during reruns.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### What this version includes")
        st.write("- Clean hero layout")
        st.write("- Faster cached model loading")
        st.write("- Better prediction summary")
        st.write("- Safer temporary-file cleanup")

        st.markdown("### Supported formats")
        st.write("PNG, JPG, JPEG")


def save_image_to_tempfile(image: Image.Image) -> str:
    """Persist the uploaded image to a temporary file for inference."""

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        image.save(temp_file.name)
        return temp_file.name


def render_metric_card(label: str, value: str) -> None:
    """Render a small metric container."""

    st.markdown(
        f"""
        <div class="metric-card">
            <div style="font-size: 0.85rem; color: #475569; margin-bottom: 0.35rem;">{label}</div>
            <div style="font-size: 1.45rem; font-weight: 700; color: #0f172a;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_top_predictions(top5: list[dict]) -> None:
    """Show the five strongest classes with progress bars."""

    st.subheader("Top 5 Predictions")

    for index, item in enumerate(top5, start=1):
        st.markdown(
            f"""
            <div class="top5-item">
                <strong>{index}. {item['class_name']}</strong><br />
                <span style="color: #475569;">Class ID: {item['class_id']}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.progress(item["confidence"] / 100)
        st.caption(f"{item['confidence']:.2f}% confidence")


def render_header() -> None:
    """Render the main page hero block."""

    st.markdown(
        """
        <div class="hero-card">
            <h1>AI-Powered Traffic Sign Recognition System</h1>
            <p>
                Upload a road sign image to get an immediate MobileNetV2 prediction with confidence,
                inference time, and the top competing classes. This page is designed as the first
                Phase 2 UI improvement without changing the trained model pipeline.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def main() -> None:
    """Run the Streamlit interface."""

    inject_custom_styles()
    render_sidebar()
    render_header()

    st.write("")

    uploaded_file = st.file_uploader(
        "Upload a traffic sign image",
        type=["png", "jpg", "jpeg"],
        help="Choose a clear image of a traffic sign for classification."
    )

    if uploaded_file is None:
        st.info("Upload an image to see the predicted sign class and confidence breakdown.")
        return

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except UnidentifiedImageError:
        st.error("The uploaded file is not a valid image.")
        return

    st.caption(
        f"File: {uploaded_file.name} | Size: {uploaded_file.size / 1024:.1f} KB | "
        f"Mode: {image.mode} | Dimensions: {image.width} x {image.height}"
    )

    col1, col2 = st.columns([1.15, 0.95], gap="large")

    temp_path = save_image_to_tempfile(image)

    try:
        result = load_predictor().predict(temp_path)
    finally:
        with suppress(OSError):
            os.unlink(temp_path)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Prediction Summary")
        st.markdown(
            f"<div class='prediction-pill'>Predicted: {result['class_name']}</div>",
            unsafe_allow_html=True
        )

        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            render_metric_card("Confidence", f"{result['confidence']:.2f}%")
        with metric_col2:
            render_metric_card("Inference Time", f"{result['inference_time']:.3f} sec")

        st.write("")
        render_metric_card("Class ID", str(result["class_id"]))

        st.write("")
        render_top_predictions(result["top5"])


if __name__ == "__main__":
    main()