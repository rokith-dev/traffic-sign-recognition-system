from tensorflow.keras.optimizers import Adam

from cnn_model import build_cnn_model
from src.preprocessing.prepare_data import get_data

# ==========================================
# Load Dataset
# ==========================================

X_train, X_test, y_train, y_test = get_data()

# ==========================================
# Build CNN
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
# Save Model
# ==========================================

model.save("models/traffic_sign_cnn.keras")

print("\n✅ Model saved successfully!")