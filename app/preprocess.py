"""
app/preprocess.py

- Walks each top-level class folder in dataset_path (e.g. dataset/leapgestrecog/00, 01, ...)
- Recursively finds image files in nested subfolders
- Loads, resizes, converts to grayscale, normalizes
- Builds X, y arrays and splits into train/test
- Saves X_train.npy, X_test.npy, y_train.npy, y_test.npy, label_dict.npy
"""
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

DATASET_PATH = "dataset/asl_alphabet/asl_alphabet_train/asl_alphabet_train"
IMG_SIZE = (64, 64)
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def find_image_files_in_class(class_folder: Path):
    files = []
    for root, _, filenames in os.walk(class_folder):
        for fname in filenames:
            if Path(fname).suffix.lower() in VALID_EXTS:
                files.append(os.path.join(root, fname))
    return files

def load_dataset(dataset_path, img_size=(64,64)):
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"{dataset_path} does not exist")

    labels = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])
    if not labels:
        raise ValueError(f"No class folders found in {dataset_path}. Check path/structure.")

    print("Found classes:", labels)
    label_dict = {label: idx for idx, label in enumerate(labels)}

    X = []
    y = []
    counts = {}

    for label in labels:
        class_folder = dataset_path / label
        image_paths = find_image_files_in_class(class_folder)
        counts[label] = len(image_paths)
        if len(image_paths) == 0:
            print(f"Warning: no images found for class '{label}' (folder: {class_folder})")
            continue

        for img_path in image_paths:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                # corrupt / unreadable file: skip
                continue
            img = cv2.resize(img, img_size)
            X.append(img)
            y.append(label_dict[label])

    X = np.array(X, dtype=np.float32)
    if X.size == 0:
        raise ValueError("No images loaded. Check dataset path and image files.")

    X = X.reshape(-1, img_size[0], img_size[1], 1) / 255.0
    y = np.array(y, dtype=np.int32)

    print("Images per class:", counts)
    print("Total images loaded:", len(y))
    return train_test_split(X, y, test_size=0.2, random_state=42), label_dict

if __name__ == "__main__":
    (X_train, X_test, y_train, y_test), label_dict = load_dataset(DATASET_PATH, IMG_SIZE)
    out_dir = Path("dataset")
    out_dir.mkdir(exist_ok=True)
    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "X_test.npy", X_test)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "y_test.npy", y_test)
    np.save(out_dir / "label_dict.npy", label_dict)
    print("✅ Dataset preprocessed and saved to dataset/*.npy")
