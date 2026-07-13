from sklearn.model_selection import train_test_split

from dataset_loader import images, labels

print("=" * 50)
print("Train Test Split")
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    images,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

print("Training Images :", X_train.shape)
print("Testing Images  :", X_test.shape)

print("Training Labels :", y_train.shape)
print("Testing Labels  :", y_test.shape)