import cv2
import numpy as np
import tensorflow as tf
import time
import pyttsx3

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "model/gesture_model.h5"
IMG_SIZE = 64  # must match training size
LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
USE_TTS = True  # set False if you don’t want speech output
CONF_THRESHOLD = 0.7  # minimum confidence for detection
# ===============================

# Load trained model
print("🔹 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot access webcam!")

# Initialize text-to-speech
tts_engine = None
if USE_TTS:
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 160)

# To prevent repeated announcements
last_prediction = None
last_spoken_time = 0
speak_interval = 2  # seconds

def preprocess_frame(frame):
    """Crop center and resize to match model input."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    return normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

def speak(text):
    """Speak text asynchronously."""
    global last_spoken_time
    if USE_TTS and time.time() - last_spoken_time > speak_interval:
        tts_engine.say(text)
        tts_engine.runAndWait()
        last_spoken_time = time.time()

# Main loop
print("🎥 Starting webcam inference. Press 'q' to quit.")
prev_time = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not captured. Skipping...")
        continue

    # Preprocess and predict
    input_data = preprocess_frame(frame)
    preds = model.predict(input_data, verbose=0)[0]
    confidence = np.max(preds)
    label_idx = np.argmax(preds)
    label = LABELS[label_idx]

    # Update FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Overlay
    display_text = f"{label} ({confidence:.2f})" if confidence > CONF_THRESHOLD else "?"
    color = (0, 255, 0) if confidence > CONF_THRESHOLD else (0, 0, 255)
    cv2.putText(frame, display_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow("Gesture Recognition (Press 'q' to Quit)", frame)

    # Speak new predictions
    if label != last_prediction and confidence > CONF_THRESHOLD:
        speak(label)
        last_prediction = label

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Inference stopped.")
