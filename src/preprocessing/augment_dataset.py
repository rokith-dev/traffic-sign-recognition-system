from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =====================================================
# Data Augmentation Configuration
# =====================================================

train_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.20,
    width_shift_range=0.20,
    height_shift_range=0.20,
    brightness_range=[0.8, 1.2],
    shear_range=0.15,
    fill_mode="nearest"
)

print("=" * 50)
print("Data Augmentation Configuration")
print("=" * 50)

print(train_datagen)