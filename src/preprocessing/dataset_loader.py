import os
import cv2
from preprocess import preprocess_image
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
# Read Every Class
# ==========================================================

for class_name in class_folders:

    class_path = os.path.join(DATASET_PATH, class_name)

    # Skip if not a folder
    if not os.path.isdir(class_path):
        continue

    image_files = os.listdir(class_path)

    print(f"Reading Class {class_name} -> {len(image_files)} Images")

    # Read Every Image
    for image_name in image_files:

        image_path = os.path.join(class_path, image_name)

        image = cv2.imread(image_path)

        if image is None:
                continue

# Preprocess the image
        image = preprocess_image(image)

        images.append(image)
        labels.append(int(class_name))

# ==========================================================
# Dataset Summary
# ==========================================================

print("\n" + "=" * 50)
print("Dataset Loaded Successfully")
print("=" * 50)

print(f"Total Images Loaded : {len(images)}")
print(f"Total Labels Loaded : {len(labels)}")
print(f"Number of Classes   : {len(class_folders)}")

# ==========================================================
# Display First Image Information
# ==========================================================

print("\nFirst Image Details")
print("-" * 30)

print("Shape    :", images[0].shape)
print("Height   :", images[0].shape[0])
print("Width    :", images[0].shape[1])
print("Channels :", images[0].shape[2])
print("Datatype :", images[0].dtype)

# ==========================================================
# Show First Image
# ==========================================================

cv2.imshow("First Traffic Sign", images[0])

cv2.waitKey(0)

cv2.destroyAllWindows()