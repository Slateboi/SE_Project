# app/train.py
"""
Train a CNN on the preprocessed dataset.

Usage:
  conda activate gesture
  python app/train.py
"""

import json
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,
                                     Dropout, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------- CONFIG ----------------
DATA_DIR = "dataset"
MODEL_OUT = "app/gesture_model.h5"
LABEL_MAP_OUT = "app/label_map.json"

IMG_H, IMG_W = 64, 64        # must match preprocess resize
BATCH = 64
EPOCHS = 25
LR = 1e-3
AUGMENT = True               # set False to disable data augmentation
# ----------------------------------------

# ---------------- helper ----------------
def load_data():
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    label_map = np.load(os.path.join(DATA_DIR, "label_dict.npy"), allow_pickle=True).item()
    # ensure shapes & types
    X_train = X_train.astype("float32")
    X_test  = X_test.astype("float32")
    return X_train, X_test, y_train, y_test, label_map

def get_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation="relu", padding="same", input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])
    return model

# ---------------- training ----------------
def main():
    print("Loading data...")
    X_train, X_test, y_train, y_test, label_map = load_data()
    num_classes = len(label_map)
    print(f"Shapes: X_train={X_train.shape}, X_test={X_test.shape}")
    print("Num classes:", num_classes)

    # Convert labels to categorical
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat  = to_categorical(y_test, num_classes)

    inp_shape = X_train.shape[1:]  # (H, W, C)
    model = get_model(inp_shape, num_classes)
    model.compile(optimizer=Adam(learning_rate=LR),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    # Callbacks
    ckpt_cb = ModelCheckpoint(MODEL_OUT, monitor="val_accuracy", save_best_only=True, verbose=1)
    es_cb = EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1)
    rl_cb = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)

    # Data augmentation
    if AUGMENT:
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True
        )
        datagen.fit(X_train)
        train_gen = datagen.flow(X_train, y_train_cat, batch_size=BATCH)
        steps_per_epoch = max(1, len(X_train) // BATCH)
        history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=EPOCHS,
            validation_data=(X_test, y_test_cat),
            callbacks=[ckpt_cb, es_cb, rl_cb]
        )
    else:
        history = model.fit(
            X_train, y_train_cat,
            batch_size=BATCH,
            epochs=EPOCHS,
            validation_data=(X_test, y_test_cat),
            callbacks=[ckpt_cb, es_cb, rl_cb]
        )

    # Save final model (best already saved by checkpoint)
    if not os.path.exists(MODEL_OUT):
        model.save(MODEL_OUT)
    print("Model training completed. Model saved to:", MODEL_OUT)

    # Save label map (index -> label) for inference
    # label_map currently maps folder_name->index; we'll invert to idx->folder
    inv_label_map = {str(idx): label for label, idx in label_map.items()}
    with open(LABEL_MAP_OUT, "w") as f:
        json.dump(inv_label_map, f, indent=2)
    print("Saved label map to:", LABEL_MAP_OUT)

if __name__ == "__main__":
    main()
