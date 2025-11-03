"""
Train a CNN (MobileNetV2 transfer learning) on the preprocessed ASL dataset.

Usage:
  conda activate gesture
  python app/train.py
"""

import json
import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------- CONFIG ----------------
DATA_DIR = "dataset"
MODEL_OUT = "app/gesture_model.h5"
LABEL_MAP_OUT = "app/label_map.json"

IMG_H, IMG_W = 64, 64
BATCH = 64
EPOCHS = 15           # can increase to 25+ later
LR = 1e-4
AUGMENT = True
# ----------------------------------------

# ---------------- helper ----------------
def load_data():
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    label_map = np.load(os.path.join(DATA_DIR, "label_dict.npy"), allow_pickle=True).item()

    X_train = X_train.astype("float32")
    X_test  = X_test.astype("float32")
    return X_train, X_test, y_train, y_test, label_map

def get_model(input_shape, num_classes):
    base = MobileNetV2(include_top=False, weights="imagenet", input_shape=input_shape)
    base.trainable = False  # freeze pretrained layers for faster training

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=outputs)
    return model

# ---------------- training ----------------
def main():
    print("📦 Loading data...")
    X_train, X_test, y_train, y_test, label_map = load_data()
    num_classes = len(label_map)
    print(f"✅ Data shapes: X_train={X_train.shape}, X_test={X_test.shape}")
    print("🧩 Classes:", num_classes)

    # Convert grayscale → 3-channel RGB (MobileNetV2 expects 3 channels)
    if X_train.shape[-1] == 1:
        X_train = np.repeat(X_train, 3, axis=-1)
        X_test  = np.repeat(X_test, 3, axis=-1)

    # Convert labels to one-hot
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat  = to_categorical(y_test, num_classes)

    # Model setup
    model = get_model(X_train.shape[1:], num_classes)
    model.compile(optimizer=Adam(learning_rate=LR),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    # Callbacks
    ckpt_cb = ModelCheckpoint(MODEL_OUT, monitor="val_accuracy", save_best_only=True, verbose=1)
    es_cb = EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1)
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
            callbacks=[ckpt_cb, es_cb, rl_cb],
            verbose=1
        )
    else:
        history = model.fit(
            X_train, y_train_cat,
            batch_size=BATCH,
            epochs=EPOCHS,
            validation_data=(X_test, y_test_cat),
            callbacks=[ckpt_cb, es_cb, rl_cb],
            verbose=1
        )

    # Save model & label map
    if not os.path.exists(MODEL_OUT):
        model.save(MODEL_OUT)
    print("✅ Model training complete! Saved to:", MODEL_OUT)

    inv_label_map = {str(idx): label for label, idx in label_map.items()}
    with open(LABEL_MAP_OUT, "w") as f:
        json.dump(inv_label_map, f, indent=2)
    print("📁 Saved label map to:", LABEL_MAP_OUT)

if __name__ == "__main__":
    main()
