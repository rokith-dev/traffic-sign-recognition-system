"""Grad-CAM explainability helpers for traffic sign predictions.

This module provides reusable utilities for generating Grad-CAM heatmaps and
overlay visualizations from any compatible Keras classification model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from configs.config import IMAGE_SIZE


@dataclass(frozen=True)
class GradCAMResult:
    """Container for Grad-CAM outputs."""

    heatmap: np.ndarray
    overlay_image: np.ndarray


class GradCAMExplainer:
    """Generate Grad-CAM visualizations for image classification models."""

    def __init__(self, model: tf.keras.Model, last_conv_layer_name: str | None = None):
        """Initialize the explainer.

        Args:
            model: A loaded Keras classification model.
            last_conv_layer_name: Optional explicit convolution layer name.
        """

        self.model = model
        self.last_conv_layer_name = last_conv_layer_name or self._find_last_conv_layer_name()

    def _find_last_conv_layer_name(self) -> str:
        """Find the last convolutional layer in the model.

        The MobileNetV2-based architecture used in this project ends with a
        convolutional feature map layer named ``Conv_1``. The fallback search
        keeps this module reusable if the head changes later.
        """

        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name

            if isinstance(layer, tf.keras.Model):
                for nested_layer in reversed(layer.layers):
                    if isinstance(nested_layer, tf.keras.layers.Conv2D):
                        return nested_layer.name

        raise ValueError("No convolutional layer found in the model.")

    def preprocess_image(self, image: Image.Image | np.ndarray) -> np.ndarray:
        """Convert an image into the model input tensor format."""

        if isinstance(image, Image.Image):
            image_array = np.array(image.convert("RGB"))
        else:
            image_array = np.asarray(image)

        if image_array.ndim != 3 or image_array.shape[-1] not in (3, 4):
            raise ValueError("Grad-CAM expects an RGB image with 3 channels.")

        if image_array.shape[-1] == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)

        resized_image = cv2.resize(image_array, IMAGE_SIZE)
        model_input = resized_image.astype(np.float32) / 255.0
        return np.expand_dims(model_input, axis=0)

    def generate_heatmap(
        self,
        image: Image.Image | np.ndarray,
        class_index: int | None = None,
    ) -> np.ndarray:
        """Generate a normalized Grad-CAM heatmap for the selected class."""

        model_input = self.preprocess_image(image)

        grad_model = tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=[
                self.model.get_layer(self.last_conv_layer_name).output,
                self.model.output,
            ],
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(model_input)

            if class_index is None:
                class_index = int(tf.argmax(predictions[0]))

            class_channel = predictions[:, class_index]

        gradients = tape.gradient(class_channel, conv_outputs)
        pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_gradients, axis=-1)
        heatmap = tf.maximum(heatmap, 0)

        max_value = tf.reduce_max(heatmap)
        if tf.equal(max_value, 0):
            return np.zeros(IMAGE_SIZE, dtype=np.float32)

        heatmap = heatmap / max_value
        heatmap = cv2.resize(heatmap.numpy(), IMAGE_SIZE)
        return heatmap

    def overlay_heatmap(
        self,
        image: Image.Image | np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.45,
    ) -> np.ndarray:
        """Overlay a Grad-CAM heatmap on top of the original image."""

        if isinstance(image, Image.Image):
            base_image = np.array(image.convert("RGB"))
        else:
            base_image = np.asarray(image)

        if base_image.shape[-1] == 4:
            base_image = cv2.cvtColor(base_image, cv2.COLOR_RGBA2RGB)

        base_image = cv2.resize(base_image, IMAGE_SIZE)
        base_image = base_image.astype(np.uint8)

        heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(base_image, 1 - alpha, heatmap_color, alpha, 0)
        return overlay

    def explain(
        self,
        image: Image.Image | np.ndarray,
        class_index: int | None = None,
    ) -> GradCAMResult:
        """Generate both the heatmap and the overlay visualization."""

        heatmap = self.generate_heatmap(image, class_index=class_index)
        overlay_image = self.overlay_heatmap(image, heatmap)
        return GradCAMResult(heatmap=heatmap, overlay_image=overlay_image)