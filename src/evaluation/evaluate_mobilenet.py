import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from tensorflow.keras.models import load_model

from src.data.dataset import load_dataset

# ==========================================
# Load Dataset
# ==========================================

print("=" * 50)
print("Loading Dataset...")
print("=" * 50)

X_train, X_test, y_train, y_test = load_dataset()

# ==========================================
# Load Trained Model
# ==========================================

print("\nLoading MobileNetV2 Model...")

model = load_model("models/traffic_sign_mobilenetv2.keras")

print("Model Loaded Successfully!")

# ==========================================
# Predict
# ==========================================

print("\nPredicting Test Images...")

predictions = model.predict(X_test)

y_pred = np.argmax(predictions, axis=1)

# ==========================================
# Accuracy
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 50)
print("Evaluation Results")
print("=" * 50)

print(f"Accuracy : {accuracy * 100:.2f}%")

# ==========================================
# Classification Report
# ==========================================

print("\nClassification Report\n")

print(classification_report(y_test, y_pred))

# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix Shape :", cm.shape)