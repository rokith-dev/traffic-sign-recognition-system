import os
import cv2

# Dataset path
DATASET_PATH = "data/raw/traffic_Data/DATA"

# Get all class folders
class_folders = sorted(os.listdir(DATASET_PATH))

print(f"Total Classes Found: {len(class_folders)}")

# Select the first class
first_class = class_folders[0]

# Path of the first class
first_class_path = os.path.join(DATASET_PATH, first_class)

# Get first image
first_image = os.listdir(first_class_path)[0]

# Complete image path
image_path = os.path.join(first_class_path, first_image)

print("Image Path:", image_path)


# Read image
image = cv2.imread(image_path)

# Print image information
print("Image Shape:", image.shape)

# Display image
cv2.imshow("Traffic Sign", image)

cv2.waitKey(0)

cv2.destroyAllWindows()