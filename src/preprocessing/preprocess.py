import cv2

# Target image size
IMAGE_SIZE = (32, 32)


def preprocess_image(image):
    """
    Preprocess a traffic sign image.
    """

    # Resize image
    image = cv2.resize(image, IMAGE_SIZE)

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize pixel values
    image = image.astype("float32") / 255.0

    return image