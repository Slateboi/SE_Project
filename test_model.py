"""
Quick test script to verify model predictions across all classes
"""
import numpy as np
import tensorflow as tf
import json

# Load model
model = tf.keras.models.load_model("app/gesture_model.h5")
print(f"Model loaded. Output shape: {model.output_shape}")
print(f"Number of classes in model: {model.output_shape[-1]}")

# Load actual label dict from dataset
label_dict = np.load("dataset/label_dict.npy", allow_pickle=True).item()
print(f"\nActual labels in dataset: {label_dict}")
print(f"Number of classes in dataset: {len(label_dict)}")

# Load test data
X_test = np.load("dataset/x_test.npy")
y_test = np.load("dataset/y_test.npy")
print(f"\nTest data shape: {X_test.shape}")
print(f"Test labels shape: {y_test.shape}")
print(f"Unique labels in test set: {np.unique(y_test)}")

# Convert to 3-channel if needed (for MobileNetV2)
if X_test.shape[-1] == 1 and model.input_shape[-1] == 3:
    X_test = np.repeat(X_test, 3, axis=-1)
    print(f"Converted to 3-channel: {X_test.shape}")

# Test predictions on a few samples from each class
print("\n=== Testing predictions ===")
idx_to_label = {idx: label for label, idx in label_dict.items()}

for class_idx in range(len(label_dict)):
    # Find samples of this class
    class_samples = np.where(y_test == class_idx)[0]
    if len(class_samples) > 0:
        # Take first sample
        sample_idx = class_samples[0]
        sample = X_test[sample_idx:sample_idx+1]
        
        pred = model.predict(sample, verbose=0)[0]
        pred_class = int(np.argmax(pred))
        confidence = float(pred[pred_class])
        
        true_label = idx_to_label.get(class_idx, f"Class_{class_idx}")
        pred_label = idx_to_label.get(pred_class, f"Class_{pred_class}")
        
        status = "✓" if pred_class == class_idx else "✗"
        print(f"{status} True: {true_label} ({class_idx}) | Predicted: {pred_label} ({pred_class}) | Conf: {confidence:.3f}")

# Overall accuracy
print("\n=== Overall Test Accuracy ===")
predictions = model.predict(X_test, verbose=0)
pred_classes = np.argmax(predictions, axis=1)
accuracy = np.mean(pred_classes == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Check label_map.json
print("\n=== Checking label_map.json ===")
try:
    with open("app/label_map.json", "r") as f:
        json_map = json.load(f)
    print(f"label_map.json contents: {json_map}")
    print(f"Number of classes in label_map.json: {len(json_map)}")
    
    if len(json_map) != len(label_dict):
        print(f"⚠️  WARNING: Mismatch! label_map.json has {len(json_map)} classes but model was trained on {len(label_dict)} classes")
except Exception as e:
    print(f"Error loading label_map.json: {e}")
