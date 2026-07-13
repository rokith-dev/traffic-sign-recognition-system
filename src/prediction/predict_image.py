import os
import sys
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Add project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

from src.preprocessing.preprocess import preprocess_image

# ==========================================
# Load Trained Model
# ==========================================

model = load_model("models/traffic_sign_cnn.keras")

print("Model Loaded Successfully!")

# ==========================================
# Image Path
# ==========================================

IMAGE_PATH = "data/raw/traffic_Data/DATA/0/000_1_0001.png"

# ==========================================
# Read Image
# ==========================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Image not found!")
    exit()

# ==========================================
# Preprocess Image
# ==========================================

processed_image = preprocess_image(image)

processed_image = np.expand_dims(processed_image, axis=0)

# ==========================================
# Prediction
# ==========================================

prediction = model.predict(processed_image)

predicted_class = np.argmax(prediction)

print(f"\nPredicted Class : {predicted_class}")

# ==========================================
# Show Image
# ==========================================

cv2.imshow("Traffic Sign", image)
cv2.waitKey(0)
cv2.destroyAllWindows()