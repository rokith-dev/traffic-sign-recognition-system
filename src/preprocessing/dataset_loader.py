import os
import cv2
import numpy as np
from src.preprocessing.preprocess import preprocess_image

# ==========================================================
# Dataset Path
# ==========================================================

DATASET_PATH = "data/raw/traffic_Data/DATA"

# ==========================================================
# Lists to Store Images and Labels
# ==========================================================

images = []
labels = []

# ==========================================================
# Get All Class Folders
# ==========================================================

class_folders = sorted(os.listdir(DATASET_PATH))

print("=" * 50)
print("Traffic Sign Dataset Loader")
print("=" * 50)

print(f"Total Classes Found : {len(class_folders)}")
print()

# ==========================================================
# Read Every Class Folder
# ==========================================================

for class_name in class_folders:

    class_path = os.path.join(DATASET_PATH, class_name)

    # Skip if it is not a folder
    if not os.path.isdir(class_path):
        continue

    image_files = os.listdir(class_path)

    print(f"Reading Class {class_name} --> {len(image_files)} Images")

    # Read Every Image
    for image_name in image_files:

        image_path = os.path.join(class_path, image_name)

        image = cv2.imread(image_path)

        # Skip corrupted images
        if image is None:
            continue

        # Preprocess image
        image = preprocess_image(image)

        # Store image and label
        images.append(image)
        labels.append(int(class_name))

# ==========================================================
# Convert Lists to NumPy Arrays
# ==========================================================

images = np.array(images)
labels = np.array(labels)

# ==========================================================
# Dataset Summary
# ==========================================================

print("\n" + "=" * 50)
print("Dataset Loaded Successfully")
print("=" * 50)

print("Total Images :", len(images))
print("Total Labels :", len(labels))
print("Total Classes:", len(class_folders))

# ==========================================================
# NumPy Information
# ==========================================================

print("\n" + "=" * 50)
print("NumPy Information")
print("=" * 50)

print("Images Shape :", images.shape)
print("Labels Shape :", labels.shape)

print("Images Data Type :", images.dtype)
print("Labels Data Type :", labels.dtype)

# ==========================================================
# First Image Information
# ==========================================================

print("\n" + "=" * 50)
print("First Image Information")
print("=" * 50)

print("Height   :", images[0].shape[0])
print("Width    :", images[0].shape[1])
print("Channels :", images[0].shape[2])

# ==========================================================
# Display First Image
# ==========================================================

if __name__ == "__main__":
    cv2.imshow(
        "First Traffic Sign",
        cv2.cvtColor((images[0] * 255).astype("uint8"), cv2.COLOR_RGB2BGR)
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()