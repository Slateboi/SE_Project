import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

DATASET_DIR = "dataset/asl_alphabet_train"
OUTPUT_DIR = "dataset"
IMG_SIZE = 64

X = []
y = []

classes = sorted(os.listdir(DATASET_DIR))
print("Found classes:", classes)

for idx, cls in enumerate(classes):
    folder = os.path.join(DATASET_DIR, cls)
    print(f"\nProcessing class '{cls}' ({idx+1}/{len(classes)})...")

    count = 0
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = cv2.imread(path)
        if img is None:
            print(f"⚠️ Skipping unreadable file: {file}")
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(img)
        y.append(idx)
        count += 1

        if count % 500 == 0:
            print(f"  Processed {count} images...")

print("\n✅ Image loading complete. Converting to arrays...")

X = np.array(X, dtype=np.float32) / 255.0
y = np.array(y)

print(f"Total samples: {len(X)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

print(f"\n✅ Dataset preprocessed and saved to '{OUTPUT_DIR}'")
