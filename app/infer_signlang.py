import time
import json
import cv2
import numpy as np
import tensorflow as tf

# ---------- CONFIG ----------
MODEL_PATH = "app/gesture_model.h5"
LABEL_MAP_PATH = "app/label_map.json"
IMG_SIZE = (64, 64)         # must match training preprocess
ROI = (200, 80, 520, 400)   # (x1, y1, x2, y2) - adjust for your camera/frame
CONF_THRESHOLD = 0.75       # min probability to consider prediction
STABLE_FRAMES = 5           # same label required for this many frames to accept
TEXT_COOLDOWN = 1.2         # seconds before adding another token
USE_TTS = True              # set False to disable speech
# ----------------------------

# Optional TTS
try:
    if USE_TTS:
        import pyttsx3
        tts_engine = pyttsx3.init()
        tts_engine.setProperty("rate", 160)
    else:
        tts_engine = None
except Exception:
    tts_engine = None

def safe_speak(text: str):
    if tts_engine:
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception:
            pass

# Load model
print("🔹 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded.")

# Load label map and build idx->label robustly
with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)

def build_idx_to_label(mapping):
    if all(isinstance(v, int) for v in mapping.values()):
        return {v: k for k, v in mapping.items()}
    try:
        if all(k.isdigit() for k in mapping.keys()):
            return {int(k): v for k, v in mapping.items()}
    except Exception:
        pass
    try:
        return {int(v): k for k, v in mapping.items()}
    except Exception:
        return {int(k): v for k, v in mapping.items()}

idx_to_label = build_idx_to_label(label_map)
print("Labels loaded. Classes:", len(idx_to_label))

# Model input info
model_input_shape = model.input_shape
expected_channels = model_input_shape[-1] if model_input_shape else 1

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

# State for smoothing + sentence building
frame_count_same = 0
last_frame_label = None
last_accepted_label = None
last_accept_time = 0
sentence = ""
registered_words = []   # <-- stores completed words
prev_time = time.time()
fps = 0.0

print("🎥 Inference started — press 'q' to quit, 'c' to clear, Enter to register word.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not read from camera — exiting.")
        break

    frame = cv2.flip(frame, 1)
    now = time.time()
    dt = now - prev_time
    prev_time = now
    if dt > 0:
        fps = 0.9 * fps + 0.1 * (1.0 / dt)

    # Draw ROI box
    x1, y1, x2, y2 = ROI
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)

    # Crop ROI and preprocess
    roi = frame[y1:y2, x1:x2]
    roi_resized = cv2.resize(roi, IMG_SIZE)

    if expected_channels == 1:
        proc = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
        proc = proc.reshape(IMG_SIZE[1], IMG_SIZE[0], 1)
    else:
        proc = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
        proc = proc.reshape(IMG_SIZE[1], IMG_SIZE[0], 3)

    proc = proc.astype("float32") / 255.0
    input_tensor = np.expand_dims(proc, axis=0)

    preds = model.predict(input_tensor, verbose=0)[0]
    class_idx = int(np.argmax(preds))
    conf = float(preds[class_idx])
    label = idx_to_label.get(class_idx, str(class_idx))

    # Temporal smoothing
    if conf >= CONF_THRESHOLD:
        if label == last_frame_label:
            frame_count_same += 1
        else:
            frame_count_same = 1
            last_frame_label = label
    else:
        frame_count_same = 0
        last_frame_label = None

    # Accept label if stable enough
    if frame_count_same >= STABLE_FRAMES:
        if (label != last_accepted_label) or (now - last_accept_time > TEXT_COOLDOWN):
            if label.lower() == "space":
                sentence += " "
                safe_speak("space")
            elif label.lower() in ["del", "delete"]:
                sentence = sentence[:-1]
                safe_speak("deleted")
            elif label.lower() == "nothing":
                pass
            else:
                sentence += label
                safe_speak(label)

            last_accepted_label = label
            last_accept_time = now
            frame_count_same = 0
            last_frame_label = None

    # Display overlays
    cv2.putText(frame, f"Pred: {label} ({conf:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

    # Sentence box
    box_y = y2 + 30
    cv2.rectangle(frame, (10, box_y - 30), (frame.shape[1] - 10, box_y + 80), (0, 0, 0), -1)
    cv2.putText(frame, "Sentence: " + sentence, (20, box_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Words: " + ", ".join(registered_words[-3:]), (20, box_y + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2, cv2.LINE_AA)

    cv2.imshow("Sign Language -> Text", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        sentence = ""
        registered_words.clear()
        safe_speak("cleared")
    elif key == ord("p"):
        print("Current sentence:", sentence)
    elif key == 13:  # Enter key pressed
        if sentence.strip():
            registered_words.append(sentence.strip())
            safe_speak("word registered")
            print("✅ Registered word:", sentence.strip())
            sentence = ""  # reset for next word

cap.release()
cv2.destroyAllWindows()
print("\nFinal sentence:", " ".join(registered_words))
