import os
import sys

from tensorflow.keras.optimizers import Adam

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

from src.preprocessing.prepare_data import get_data
from src.training.cnn_model import build_cnn_model

# ==========================================
# Load Dataset
# ==========================================

X_train, X_test, y_train, y_test = get_data()

# ==========================================
# Build Model
# ==========================================

NUM_CLASSES = 58

model = build_cnn_model(NUM_CLASSES)

# ==========================================
# Compile Model
# ==========================================

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("=" * 50)
print("CNN Model Summary")
print("=" * 50)

model.summary()

# ==========================================
# Train Model
# ==========================================

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=32,
    verbose=1
)

# ==========================================
# Create Models Folder
# ==========================================

os.makedirs("models", exist_ok=True)

# ==========================================
# Save Model
# ==========================================

model.save("models/traffic_sign_cnn.keras")

print("\n==========================================")
print("Model Trained Successfully!")
print("Model Saved: models/traffic_sign_cnn.keras")
print("==========================================")