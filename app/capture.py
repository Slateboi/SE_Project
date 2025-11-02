"""
capture.py

Usage:
  1. Activate your conda env: conda activate gesture
  2. Run: python app/capture.py
  3. Controls:
     - 'n' : start a new labeled recording (you'll be prompted in terminal)
     - 'r' : stop current recording early
     - 'q' : quit program
Saved files:
  dataset/raw/<label>/sample_<timestamp>.npy  (shape: (T,126))
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time
from pathlib import Path

# ---------------- CONFIG ----------------
DATA_DIR = Path("dataset/raw")
CAMERA_ID = 0                # change if your webcam is on another index
DEFAULT_SECONDS = 2.5
RECORD_FPS = 15              # approx frames per second captured during recording
MAX_NUM_HANDS = 2
MIN_DET_CONF = 0.6
MIN_TRACK_CONF = 0.6
# ----------------------------------------

DATA_DIR.mkdir(parents=True, exist_ok=True)

class LandmarkRecorder:
    def __init__(self, max_num_hands=2, min_det_conf=0.6, min_track_conf=0.6):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_track_conf
        )

    def frame_to_flat(self, frame_bgr):
        """Return np.array shape (126,) or None if no hands detected."""
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        if not results.multi_hand_landmarks:
            return None

        hands_list = []
        for hand_landmarks in results.multi_hand_landmarks:
            lm = [(l.x, l.y, l.z) for l in hand_landmarks.landmark]
            hands_list.append(lm)

        flat = []
        # up to 2 hands; pad zeros for missing hand
        for i in range(2):
            if i < len(hands_list):
                lms = hands_list[i]
                wrist = lms[0]
                rel = [(p[0]-wrist[0], p[1]-wrist[1], p[2]-wrist[2]) for p in lms]
                for pt in rel:
                    flat.extend([pt[0], pt[1], pt[2]])
            else:
                flat.extend([0.0]*63)
        return np.array(flat, dtype=np.float32)

    def close(self):
        self.hands.close()

def ensure_label_dir(label: str):
    p = DATA_DIR / label
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_sample(label: str, frames):
    if len(frames) == 0:
        print("No frames to save.")
        return None
    arr = np.stack(frames, axis=0)
    ts = int(time.time())
    filename = ensure_label_dir(label) / f"sample_{ts}.npy"
    np.save(filename, arr)
    print(f"Saved {filename}  shape={arr.shape}")
    return filename

def record_sequence(cap, recorder, duration=DEFAULT_SECONDS, target_fps=RECORD_FPS):
    print(f"Recording {duration}s at ~{target_fps} FPS. Press 'r' to stop early.")
    frames = []
    start = time.time()
    frame_interval = 1.0 / target_fps
    next_capture = start

    while True:
        now = time.time()
        if now - start >= duration:
            break
        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed.")
            break

        vis = frame.copy()
        cv2.putText(vis, "RECORDING...", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        cv2.imshow("capture", vis)

        if now >= next_capture:
            lm = recorder.frame_to_flat(frame)
            if lm is not None:
                frames.append(lm)
            next_capture += frame_interval

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            print("Early stop requested.")
            break

    print(f"Recorded {len(frames)} landmark frames.")
    return frames

def main():
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)  # CAP_DSHOW helps on Windows
    if not cap.isOpened():
        print(f"ERROR: cannot open camera index {CAMERA_ID}")
        return

    recorder = LandmarkRecorder(MAX_NUM_HANDS, MIN_DET_CONF, MIN_TRACK_CONF)
    print("Camera opened. Press 'n' to record, 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame.")
                break
            frame = cv2.flip(frame, 1)  # mirror for natural webcam view
            cv2.putText(frame, "Press 'n' to record | 'q' to quit", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow("capture", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                label = input("Enter label for this sample (no spaces): ").strip()
                if label == "":
                    print("Empty label — skipping.")
                    continue
                ds = input(f"Enter duration in seconds (default {DEFAULT_SECONDS}): ").strip()
                try:
                    duration = float(ds) if ds != "" else DEFAULT_SECONDS
                except:
                    duration = DEFAULT_SECONDS
                frames = record_sequence(cap, recorder, duration=duration, target_fps=RECORD_FPS)
                if frames:
                    save_sample(label, frames)
                else:
                    print("No frames captured; nothing saved.")
    finally:
        recorder.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
