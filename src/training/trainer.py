import os

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from src.data.dataset import load_dataset
from src.models.mobilenetv2 import build_mobilenetv2

from configs.config import *

# ==========================================
# Load Dataset
# ==========================================

print("=" * 50)
print("Loading Dataset")
print("=" * 50)

X_train, X_test, y_train, y_test = load_dataset()

# ==========================================
# Build Model
# ==========================================

print("\nBuilding MobileNetV2 Model...")

model = build_mobilenetv2()

model.summary()

# ==========================================
# Callbacks
# ==========================================

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

checkpoint = ModelCheckpoint(
    filepath=os.path.join(
        MODEL_SAVE_PATH,
        MODEL_FILE_NAME
    ),
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=3,
    verbose=1
)

# ==========================================
# Train Model
# ==========================================

print("\nTraining Started...\n")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr
    ],
    verbose=1
)

print("\nTraining Completed Successfully!")