from src.data.dataset import load_dataset

X_train, X_test, y_train, y_test = load_dataset()

print()

print("Training :", X_train.shape)

print("Testing  :", X_test.shape)