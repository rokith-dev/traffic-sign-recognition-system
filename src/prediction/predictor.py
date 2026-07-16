import time
import cv2
import numpy as np

from tensorflow.keras.models import load_model

from configs.config import IMAGE_SIZE
from src.utils.class_names import CLASS_NAMES


class TrafficSignPredictor:

    def __init__(self):

        print("Loading MobileNetV2 Model...")

        self.model = load_model(
            "models/traffic_sign_mobilenetv2.keras"
        )

        print("Model Loaded Successfully!")

    def preprocess(self, image_path):

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError("Image not found!")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, IMAGE_SIZE)

        image = image.astype("float32") / 255.0

        image = np.expand_dims(image, axis=0)

        return image

    def predict(self, image_path):

        image = self.preprocess(image_path)

        start_time = time.time()

        prediction = self.model.predict(image, verbose=0)

        end_time = time.time()

        prediction = prediction[0]

        predicted_class = np.argmax(prediction)

        confidence = prediction[predicted_class] * 100

        # Top 5 Predictions
        top5_indices = np.argsort(prediction)[::-1][:5]

        top5 = []

        for idx in top5_indices:

            top5.append({
                "class_id": int(idx),
                "class_name": CLASS_NAMES.get(idx, "Unknown"),
                "confidence": float(prediction[idx] * 100)
            })

        inference_time = end_time - start_time

        return {
            "class_id": predicted_class,
            "class_name": CLASS_NAMES.get(predicted_class, "Unknown"),
            "confidence": confidence,
            "top5": top5,
            "inference_time": inference_time
        }