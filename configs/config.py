# ==========================================================
# DATASET CONFIGURATION
# ==========================================================

DATASET_PATH = "data/raw/traffic_Data/DATA"

IMAGE_SIZE = (32, 32)

NUM_CHANNELS = 3

NUM_CLASSES = 58

# ==========================================================
# TRAINING CONFIGURATION
# ==========================================================

BATCH_SIZE = 32

EPOCHS = 20

LEARNING_RATE = 0.001

TEST_SIZE = 0.20

RANDOM_STATE = 42

# ==========================================================
# MODEL CONFIGURATION
# ==========================================================

MODEL_NAME = "MobileNetV2"

MODEL_SAVE_PATH = "models"

MODEL_FILE_NAME = "traffic_sign_mobilenetv2.keras"

# ==========================================================
# PROJECT CONFIGURATION
# ==========================================================

PROJECT_NAME = "AI-Powered Traffic Sign Recognition System"

AUTHOR = "Rokith"

VERSION = "2.0"