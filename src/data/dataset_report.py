import os
import pandas as pd

from configs.config import DATASET_PATH

class_counts = {}

for class_name in sorted(os.listdir(DATASET_PATH)):

    class_path = os.path.join(DATASET_PATH, class_name)

    if not os.path.isdir(class_path):
        continue

    image_count = len([
        file for file in os.listdir(class_path)
        if file.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    class_counts[int(class_name)] = image_count

df = pd.DataFrame({
    "Class ID": class_counts.keys(),
    "Images": class_counts.values()
})

print("=" * 50)
print(df)
print("=" * 50)

print("\nSmallest Classes")
print(df.nsmallest(10, "Images"))

print("\nLargest Classes")
print(df.nlargest(10, "Images"))