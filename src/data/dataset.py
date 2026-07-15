import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split

from configs.config import *


def load_dataset():

    images = []
    labels = []

    class_folders = sorted(os.listdir(DATASET_PATH))

    print("=" * 50)
    print("Loading Dataset")
    print("=" * 50)

    for class_name in class_folders:

        class_path = os.path.join(DATASET_PATH, class_name)

        if not os.path.isdir(class_path):
            continue

        image_files = [
            file for file in os.listdir(class_path)
            if file.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        print(f"Class {class_name}: {len(image_files)} images")

        for image_name in image_files:

            image_path = os.path.join(class_path, image_name)

            image = cv2.imread(image_path)

            if image is None:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = cv2.resize(image, IMAGE_SIZE)

            image = image.astype("float32") / 255.0

            images.append(image)

            labels.append(int(class_name))

    images = np.array(images)

    labels = np.array(labels)

    print("\nDataset Loaded Successfully")
    print("Images :", images.shape)
    print("Labels :", labels.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        images,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    return X_train, X_test, y_train, y_test