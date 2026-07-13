from sklearn.model_selection import train_test_split

# Import processed dataset
from dataset_loader import images, labels

print("=" * 50)
print("Preparing Dataset")
print("=" * 50)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    images,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

print("Training Images :", X_train.shape)
print("Testing Images  :", X_test.shape)

print("Training Labels :", y_train.shape)
print("Testing Labels  :", y_test.shape)


def get_data():
    """
    Returns the prepared dataset.
    """
    return X_train, X_test, y_train, y_test