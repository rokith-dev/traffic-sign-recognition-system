import os
import sys
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

from src.preprocessing.prepare_data import get_data

# ==========================================
# Load Test Data
# ==========================================

_, X_test, _, y_test = get_data()

# ==========================================
# Load Trained Model
# ==========================================

model = load_model("models/traffic_sign_cnn.keras")

print("=" * 50)
print("Model Loaded Successfully")
print("=" * 50)

# ==========================================
# Predict
# ==========================================

predictions = model.predict(X_test)

y_pred = np.argmax(predictions, axis=1)

# ==========================================
# Accuracy
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy : {accuracy:.4f}")

# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix Shape :", cm.shape)

# ==========================================
# Classification Report
# ==========================================

print("\nClassification Report")
print(classification_report(y_test, y_pred))