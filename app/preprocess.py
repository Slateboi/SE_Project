"""
app/preprocess.py

- Walks each top-level class folder in dataset_path
- Recursively finds image files in nested subfolders
- Loads, resizes, converts to grayscale, normalizes
- Builds X, y arrays and splits into train/test
- Saves x_train.npy, x_test.npy, y_train.npy, y_test.npy, label_dict.npy

Supports two dataset structures:
1. ASL Alphabet (Kaggle): dataset/asl_alphabet/asl_alphabet_train/asl_alphabet_train/[A-Z, del, nothing, space]
2. LeapGestRecog: dataset/leapGestRecog/[00-09]/[01_palm, 02_l, ...]
"""
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

# Config - Change this to switch between datasets
USE_ASL_ALPHABET = True  # Set to False to use LeapGestRecog

if USE_ASL_ALPHABET:
    DATASET_PATH = "dataset/asl_alphabet/asl_alphabet_train/asl_alphabet_train"
else:
    DATASET_PATH = "dataset/leapGestRecog"

IMG_SIZE = (64, 64)
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def find_image_files_in_class(class_folder: Path):
    files = []
    for root, _, filenames in os.walk(class_folder):
        for fname in filenames:
            if Path(fname).suffix.lower() in VALID_EXTS:
                files.append(os.path.join(root, fname))
    return files

def load_dataset(dataset_path, img_size=(64,64), use_asl_alphabet=USE_ASL_ALPHABET):
    """
    Load gesture dataset.
    
    For ASL Alphabet: folders are A-Z, del, nothing, space
    For LeapGestRecog: top folders are people, subfolders are gestures
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"{dataset_path} does not exist")

    if use_asl_alphabet:
        # ASL Alphabet structure: direct class folders
        labels = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])
        if not labels:
            raise ValueError(f"No class folders found in {dataset_path}")
        
        print("Found ASL alphabet classes:", labels)
        label_dict = {label: idx for idx, label in enumerate(labels)}
        
        X = []
        y = []
        counts = {}
        
        for label in labels:
            class_folder = dataset_path / label
            image_paths = find_image_files_in_class(class_folder)
            counts[label] = len(image_paths)
            
            if len(image_paths) == 0:
                print(f"Warning: no images found for class '{label}'")
                continue

            for img_path in image_paths:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, img_size)
                X.append(img)
                y.append(label_dict[label])
    
    else:
        # LeapGestRecog structure: people folders contain gesture folders
        first_person = sorted([d for d in dataset_path.iterdir() if d.is_dir()])[0]
        gesture_types = sorted([d.name for d in first_person.iterdir() if d.is_dir()])
        
        if not gesture_types:
            raise ValueError(f"No gesture folders found in {first_person}")

        print("Found gesture types:", gesture_types)
        label_dict = {label: idx for idx, label in enumerate(gesture_types)}

        
        X = []
        y = []
        counts = {}

        # Iterate through all people folders
        person_folders = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
        
        for person_folder in person_folders:
            print(f"Processing person folder: {person_folder.name}")
            
            # Iterate through gesture type folders for this person
            for gesture_folder in sorted(person_folder.iterdir()):
                if not gesture_folder.is_dir():
                    continue
                    
                gesture_name = gesture_folder.name
                if gesture_name not in label_dict:
                    continue
                    
                image_paths = find_image_files_in_class(gesture_folder)
                if gesture_name not in counts:
                    counts[gesture_name] = 0
                counts[gesture_name] += len(image_paths)
                
                if len(image_paths) == 0:
                    print(f"Warning: no images found for {person_folder.name}/{gesture_name}")
                    continue

                for img_path in image_paths:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        # corrupt / unreadable file: skip
                        continue
                    img = cv2.resize(img, img_size)
                    X.append(img)
                    y.append(label_dict[gesture_name])

    X = np.array(X, dtype=np.float32)
    if X.size == 0:
        raise ValueError("No images loaded. Check dataset path and image files.")

    X = X.reshape(-1, img_size[0], img_size[1], 1) / 255.0
    y = np.array(y, dtype=np.int32)

    print("Images per class:", counts)
    print("Total images loaded:", len(y))
    return train_test_split(X, y, test_size=0.2, random_state=42), label_dict

if __name__ == "__main__":
    print(f"Using dataset: {DATASET_PATH}")
    print(f"Dataset type: {'ASL Alphabet' if USE_ASL_ALPHABET else 'LeapGestRecog'}")
    
    (X_train, X_test, y_train, y_test), label_dict = load_dataset(DATASET_PATH, IMG_SIZE)
    out_dir = Path("dataset")
    out_dir.mkdir(exist_ok=True)
    np.save(out_dir / "x_train.npy", X_train)
    np.save(out_dir / "x_test.npy", X_test)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "y_test.npy", y_test)
    np.save(out_dir / "label_dict.npy", label_dict)
    print("✅ Dataset preprocessed and saved to dataset/*.npy")
    print(f"   Total images: {len(y_train) + len(y_test)}")
    print(f"   Classes: {len(label_dict)}")
    print(f"   Class names: {list(label_dict.keys())}")
