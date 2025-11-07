"""
FastAPI Backend for ASL Sign Language Recognition
Integrates with your existing ML model and provides REST + WebSocket APIs

File: backend/main.py
Run: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from passlib.context import CryptContext
from jose import JWTError, jwt
import json
import base64
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import asyncio
from collections import deque
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


# ============= Configuration =============
SECRET_KEY = "your-secret-key-change-in-production-use-openssl-rand-hex-32"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

MODEL_PATH = "../app/gesture_model.h5"
LABEL_MAP_PATH = "../app/label_map.json"
IMG_SIZE = (64, 64)
# ROI coordinates - will be adjusted based on actual video dimensions
# Format: (x1, y1, x2, y2) - top-left and bottom-right corners
# These are relative percentages that will be scaled to actual video size
ROI = None  # Will be calculated dynamically based on video dimensions
CONF_THRESHOLD = 0.5  # Lowered to detect more letters (was 0.75)
STABLE_FRAMES = 5
TEXT_COOLDOWN = 1.2

DATABASE_URL = "sqlite:///./asl_recognition.db"

# ============= Database Setup =============
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class DBUser(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class DBSession(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    total_letters = Column(Integer, default=0)
    full_text = Column(Text)

class DBDetection(Base):
    __tablename__ = "detections"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, index=True)
    letter = Column(String)
    confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# ============= FastAPI App =============
app = FastAPI(
    title="ASL Sign Language Recognition API",
    description="Real-time American Sign Language fingerspelling recognition",
    version="1.0.0"
)

# Serve frontend
from pathlib import Path
frontend_path = Path(__file__).parent.parent / "frontend"

@app.get("/")
async def serve_frontend():
    return FileResponse(frontend_path / "index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= Security =============
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user = db.query(DBUser).filter(DBUser.username == username).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# ============= Pydantic Models =============
class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime

class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

class SessionResponse(BaseModel):
    id: int
    start_time: datetime
    end_time: Optional[datetime]
    total_letters: int
    full_text: Optional[str]

class DetectionResponse(BaseModel):
    letter: str
    confidence: float
    timestamp: datetime

# ============= ML Model Loader =============
class ASLRecognizer:
    def __init__(self):
        print("🔹 Loading ASL model...")
        self.model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully")
        
        with open(LABEL_MAP_PATH, "r") as f:
            label_map = json.load(f)
        
        # Build index to label mapping
        self.idx_to_label = self._build_idx_to_label(label_map)
        self.expected_channels = self.model.input_shape[-1] if self.model.input_shape else 3
        print(f"📋 Loaded {len(self.idx_to_label)} classes")
    
    def _build_idx_to_label(self, mapping):
        if all(isinstance(v, int) for v in mapping.values()):
            return {v: k for k, v in mapping.items()}
        try:
            if all(k.isdigit() for k in mapping.keys()):
                return {int(k): v for k, v in mapping.items()}
        except:
            pass
        try:
            return {int(v): k for k, v in mapping.items()}
        except:
            return {int(k): v for k, v in mapping.items()}
    
    def predict(self, frame, roi=ROI):
        """Predict ASL letter from frame"""
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # If ROI is provided, use it; otherwise calculate a centered ROI
        if roi:
            x1, y1, x2, y2 = roi
        else:
            # Default: center region (30% margin on all sides)
            margin_w = int(w * 0.3)
            margin_h = int(h * 0.3)
            x1, y1 = margin_w, margin_h
            x2, y2 = w - margin_w, h - margin_h
        
        # Validate and clamp ROI to frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Check if ROI is valid
        if x2 <= x1 or y2 <= y1:
            print(f"⚠️ Invalid ROI: ({x1}, {y1}, {x2}, {y2}) for frame {w}x{h}")
            return "nothing", 0.0
        
        # Note: Frontend video is mirrored for display, but canvas sends original frame
        # So ROI coordinates should match the original (non-mirrored) frame
        
        roi_crop = frame[y1:y2, x1:x2]
        
        if roi_crop.size == 0:
            print(f"⚠️ Empty ROI crop from ({x1}, {y1}, {x2}, {y2})")
            return "nothing", 0.0
        
        roi_resized = cv2.resize(roi_crop, IMG_SIZE)
        
        # Convert to grayscale (model expects 1 channel, not 3)
        if len(roi_resized.shape) == 3:
            gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi_resized
        
        # Normalize to [0, 1] range (matching training)
        proc = gray.astype("float32") / 255.0
        
        # Reshape to (64, 64, 1) - model expects 1 channel
        proc = proc.reshape(IMG_SIZE[1], IMG_SIZE[0], 1)
        
        input_tensor = np.expand_dims(proc, axis=0)
        
        # Get model predictions
        preds = self.model.predict(input_tensor, verbose=0)[0]
        
        # Use temperature scaling to reduce overconfidence
        # This helps when model is overconfident (like 100% confidence)
        # Temperature scaling: divide logits by temperature, then softmax
        temperature = 3.0  # Higher temperature = softer predictions (reduces overconfidence more)
        
        # Apply temperature scaling (preds are already softmax outputs, so we need to convert back to logits)
        # Since preds are probabilities, we convert to logits: log(p) / temperature
        epsilon = 1e-10  # Avoid log(0)
        logits = np.log(preds + epsilon) / temperature
        preds_scaled = np.exp(logits - np.max(logits))  # Numerical stability
        preds_scaled = preds_scaled / np.sum(preds_scaled)  # Renormalize to probabilities
        
        # SPECIAL HANDLING: If B has very high original confidence, penalize it
        b_idx = None
        for idx, lbl in self.idx_to_label.items():
            if lbl == "B":
                b_idx = idx
                break
        
        if b_idx is not None and preds[b_idx] > 0.8:
            # Penalize B predictions by reducing their confidence
            preds_scaled[b_idx] = preds_scaled[b_idx] * 0.5  # Reduce B confidence by 50%
            # Renormalize
            preds_scaled = preds_scaled / np.sum(preds_scaled)
        
        class_idx = int(np.argmax(preds_scaled))
        conf = float(preds_scaled[class_idx])
        label = self.idx_to_label.get(class_idx, str(class_idx))
        
        # Also get original confidence for comparison
        original_conf = float(preds[class_idx])
        
        # Debug: Print top 3 predictions occasionally (every 30 frames to avoid spam)
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1
        
        if self._debug_counter % 30 == 0:  # Print every 30 predictions
            top_3_indices = np.argsort(preds_scaled)[-3:][::-1]
            top_3_labels = [self.idx_to_label.get(int(idx), str(idx)) for idx in top_3_indices]
            top_3_confs = [float(preds_scaled[int(idx)]) for idx in top_3_indices]
            print(f"🔍 Top 3 (scaled): {list(zip(top_3_labels, [f'{c:.2f}' for c in top_3_confs]))} | Selected: {label} (scaled={conf:.2f}, orig={original_conf:.2f})")
        
        # Check if prediction is suspiciously confident (might indicate model bias)
        if original_conf > 0.95:
            # Get all predictions to see distribution
            all_preds = [(self.idx_to_label.get(int(i), str(i)), float(preds[i])) for i in range(len(preds))]
            all_preds.sort(key=lambda x: x[1], reverse=True)
            top_5 = all_preds[:5]
            if self._debug_counter % 10 == 0:  # Print more frequently for high confidence
                print(f"⚠️ High orig conf ({original_conf:.3f}) for '{label}'. Top 5: {top_5}")
        
        return label, conf

recognizer = ASLRecognizer()

# ============= Session State Manager =============
class RecognitionSession:
    def __init__(self, user_id: int, session_id: int):
        self.user_id = user_id
        self.session_id = session_id
        self.sentence = ""
        self.registered_words = []
        self.frame_count_same = 0
        self.last_frame_label = None
        self.last_accepted_label = None
        self.last_accept_time = 0
        self.pending_letter = None  # Store pending letter waiting for confirmation
        self.pending_confidence = 0.0
        self.prediction_history = []  # Track recent predictions to detect stuck model
        self.max_history = 50  # Keep last 50 predictions

active_sessions: Dict[str, RecognitionSession] = {}

# ============= API Routes =============

@app.get("/")
async def root():
    return {
        "message": "ASL Sign Language Recognition API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs"
    }

@app.post("/api/auth/register", response_model=UserResponse)
async def register(user: UserRegister, db: Session = Depends(get_db)):
    """Register a new user"""
    # Check if username exists
    if db.query(DBUser).filter(DBUser.username == user.username).first():
        raise HTTPException(status_code=400, detail="Username already registered")
    
    # Check if email exists
    if db.query(DBUser).filter(DBUser.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user
    hashed_password = get_password_hash(user.password)
    db_user = DBUser(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return db_user

@app.post("/api/auth/login", response_model=Token)
async def login(user: UserLogin, db: Session = Depends(get_db)):
    """Login and get access token"""
    db_user = db.query(DBUser).filter(DBUser.username == user.username).first()
    
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": db_user
    }

@app.get("/api/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: DBUser = Depends(get_current_user)):
    """Get current user information"""
    return current_user

@app.get("/api/sessions", response_model=List[SessionResponse])
async def get_user_sessions(
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all sessions for current user"""
    sessions = db.query(DBSession).filter(
        DBSession.user_id == current_user.id
    ).order_by(DBSession.start_time.desc()).limit(20).all()
    
    return sessions

@app.get("/api/sessions/{session_id}")
async def get_session_detail(
    session_id: int,
    current_user: DBUser = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get detailed session information"""
    session = db.query(DBSession).filter(
        DBSession.id == session_id,
        DBSession.user_id == current_user.id
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    detections = db.query(DBDetection).filter(
        DBDetection.session_id == session_id
    ).order_by(DBDetection.timestamp).all()
    
    return {
        "session": session,
        "detections": detections
    }

@app.get("/api/alphabet")
async def get_alphabet():
    """Get list of recognizable letters"""
    return {
        "alphabet": list(recognizer.idx_to_label.values()),
        "total_classes": len(recognizer.idx_to_label)
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": recognizer.model is not None,
        "total_classes": len(recognizer.idx_to_label),
        "timestamp": datetime.utcnow().isoformat()
    }

# ============= WebSocket for Real-time Recognition =============

@app.websocket("/ws/recognition")
async def websocket_recognition(websocket: WebSocket):
    """Real-time ASL recognition via WebSocket"""
    await websocket.accept()
    
    session_state = None
    db = SessionLocal()
    
    try:
        # Wait for authentication
        auth_data = await websocket.receive_json()
        
        if auth_data.get("type") != "authenticate":
            await websocket.send_json({"type": "error", "message": "Authentication required"})
            return
        
        token = auth_data.get("token")
        
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            user = db.query(DBUser).filter(DBUser.username == username).first()
            
            if not user:
                await websocket.send_json({"type": "error", "message": "Invalid token"})
                return
        except JWTError:
            await websocket.send_json({"type": "error", "message": "Invalid token"})
            return
        
        # Create new session
        db_session = DBSession(user_id=user.id)
        db.add(db_session)
        db.commit()
        db.refresh(db_session)
        
        session_state = RecognitionSession(user.id, db_session.id)
        
        await websocket.send_json({
            "type": "session_started",
            "session_id": db_session.id,
            "message": "Ready to recognize ASL signs!",
            "alphabet": list(recognizer.idx_to_label.values())
        })
        
        # Recognition loop
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "frame":
                # Decode base64 frame
                frame_data = data.get("frame")
                if not frame_data or "," not in frame_data:
                    continue
                
                frame_data = frame_data.split(",")[1]
                img_bytes = base64.b64decode(frame_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue
                
                # Calculate dynamic ROI based on frame size
                h, w = frame.shape[:2]
                
                # ROI: Left-center region (where hand typically appears)
                # Frontend shows mirrored video, but backend gets original frame
                # So left side of original = right side of mirrored display
                margin_w = int(w * 0.25)  # 25% margin
                margin_h = int(h * 0.2)   # 20% margin top/bottom
                
                # Use left-center region (this appears as right-center in mirrored view)
                # This avoids capturing the face which is typically center-right
                roi_x1 = margin_w
                roi_y1 = margin_h
                roi_x2 = int(w * 0.65)  # End at 65% of width (left-center region)
                roi_y2 = h - margin_h
                
                # Validate ROI crop before prediction
                roi_crop = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                
                # Check if ROI is valid (not empty, has sufficient variance)
                if roi_crop.size == 0:
                    continue
                
                # Calculate image statistics to detect if it's too dark/empty
                gray_roi = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2GRAY) if len(roi_crop.shape) == 3 else roi_crop
                mean_brightness = np.mean(gray_roi)
                std_brightness = np.std(gray_roi)
                
                # Only skip if image is VERY dark (< 15) or has VERY little variance (< 5)
                # This filters out completely black screens, covered cameras, etc.
                # But allows normal hand signs which might be in shadow
                if mean_brightness < 15 or std_brightness < 5:
                    if hasattr(recognizer, '_skip_counter'):
                        recognizer._skip_counter += 1
                    else:
                        recognizer._skip_counter = 1
                    
                    # Only print every 30 skips to avoid spam
                    if recognizer._skip_counter % 30 == 0:
                        print(f"⚠️ Skipping: ROI too dark/low variance (mean={mean_brightness:.1f}, std={std_brightness:.1f})")
                    continue
                
                # Predict with dynamic ROI
                label, conf = recognizer.predict(frame, roi=(roi_x1, roi_y1, roi_x2, roi_y2))
                
                # Track prediction history to detect if model is stuck on one letter
                session_state.prediction_history.append((label, conf))
                if len(session_state.prediction_history) > session_state.max_history:
                    session_state.prediction_history.pop(0)
                
                # Check if model is stuck predicting the same letter (especially B)
                if len(session_state.prediction_history) >= 10:
                    recent_labels = [p[0] for p in session_state.prediction_history[-10:]]
                    most_common = max(set(recent_labels), key=recent_labels.count)
                    count_most_common = recent_labels.count(most_common)
                    
                    # If B appears in 70%+ of recent predictions, block ALL B predictions
                    if count_most_common >= 7 and most_common == "B":
                        recent_confs = [p[1] for p in session_state.prediction_history[-10:] if p[0] == "B"]
                        avg_conf = np.mean(recent_confs) if recent_confs else 0
                        if avg_conf > 0.85:
                            print(f"🚫 Model stuck on 'B' ({count_most_common}/10 frames, avg conf={avg_conf:.2f}) - blocking all B predictions")
                            # Block this prediction if it's B
                            if label == "B":
                                continue
                
                # Additional validation: if confidence is suspiciously high, require more stability
                # But don't block completely - just require more frames
                if conf >= 0.99:
                    # Very high confidence - require more stability but still allow it
                    required_stable_frames = STABLE_FRAMES * 3  # 15 frames instead of 5
                    if not hasattr(recognizer, '_high_conf_counter'):
                        recognizer._high_conf_counter = 0
                    recognizer._high_conf_counter += 1
                    if recognizer._high_conf_counter % 10 == 0:
                        print(f"⚠️ Very high confidence ({conf:.3f}) for '{label}' - requiring extra stability (15 frames)")
                else:
                    required_stable_frames = STABLE_FRAMES * 2 if conf > 0.95 else STABLE_FRAMES
                
                # Filter out "nothing" predictions
                if label.lower() == "nothing":
                    continue
                
                # AGGRESSIVE FILTER: Block B predictions if model appears stuck
                # If B is being predicted too frequently, it's likely a model bias issue
                if label == "B":
                    # Check recent history for B predictions
                    recent_b_count = sum(1 for p in session_state.prediction_history[-10:] if p[0] == "B")
                    
                    # If B appears in 70%+ of recent predictions, block it
                    if recent_b_count >= 7:
                        print(f"🚫 Blocking B prediction - appears in {recent_b_count}/10 recent predictions (model stuck)")
                        continue
                    
                    # If B has very high confidence (>0.9), require even more stability
                    if conf > 0.9:
                        required_stable_frames = max(required_stable_frames, STABLE_FRAMES * 4)  # 20 frames
                        print(f"🔍 B detected with {conf:.3f} confidence - requiring {required_stable_frames} stable frames")
                    
                    # If B has extremely high confidence (>0.95), block it entirely
                    if conf > 0.95:
                        print(f"🚫 Blocking B prediction with suspiciously high confidence ({conf:.3f})")
                        continue
                
                # Temporal smoothing
                now = datetime.utcnow().timestamp()
                
                # required_stable_frames is set above based on confidence level
                
                if conf >= CONF_THRESHOLD:
                    if label == session_state.last_frame_label:
                        session_state.frame_count_same += 1
                    else:
                        session_state.frame_count_same = 1
                        session_state.last_frame_label = label
                else:
                    session_state.frame_count_same = 0
                    session_state.last_frame_label = None
                
                # Send current prediction
                await websocket.send_json({
                    "type": "prediction",
                    "label": label,
                    "confidence": conf,
                    "stable": session_state.frame_count_same >= required_stable_frames
                })
                
                # When stable prediction is detected, send it as pending (waiting for confirmation)
                if session_state.frame_count_same >= required_stable_frames:
                    if (label != session_state.last_accepted_label) or \
                       (now - session_state.last_accept_time > TEXT_COOLDOWN):
                        
                        # Don't auto-add, instead send as pending for user confirmation
                        if label.lower() == "nothing":
                            continue
                        
                        # Store as pending letter
                        session_state.pending_letter = label
                        session_state.pending_confidence = conf
                        
                        # Send pending letter notification (waiting for user confirmation)
                        await websocket.send_json({
                            "type": "letter_pending",
                            "letter": label,
                            "confidence": conf,
                            "message": "Letter detected - waiting for confirmation"
                        })
                        
                        # Reset frame count to avoid spamming
                        session_state.frame_count_same = 0
                        session_state.last_frame_label = None
            
            elif data.get("type") == "accept_letter":
                # User confirmed they want to add the pending letter
                if session_state.pending_letter:
                    label = session_state.pending_letter
                    conf = session_state.pending_confidence
                        
                    if label.lower() == "space":
                        session_state.sentence += " "
                        final_letter = "SPACE"
                    elif label.lower() in ["del", "delete"]:
                        if session_state.sentence:
                            session_state.sentence = session_state.sentence[:-1]
                        final_letter = "DELETE"
                    else:
                        session_state.sentence += label
                        final_letter = label
                        
                        # Save detection
                        detection = DBDetection(
                            session_id=session_state.session_id,
                            letter=final_letter,
                            confidence=conf
                        )
                        db.add(detection)
                        
                        # Update session
                        db_session = db.query(DBSession).filter(
                            DBSession.id == session_state.session_id
                        ).first()
                        db_session.total_letters += 1
                        db_session.full_text = session_state.sentence
                        db.commit()
                        
                        session_state.last_accepted_label = label
                    session_state.last_accept_time = datetime.utcnow().timestamp()
                    session_state.pending_letter = None
                    session_state.pending_confidence = 0.0
                        
                    await websocket.send_json({
                        "type": "letter_detected",
                        "letter": final_letter,
                        "confidence": conf,
                        "sentence": session_state.sentence,
                        "word_count": len(session_state.registered_words)
                    })
            
            elif data.get("type") == "reject_letter":
                # User rejected the pending letter - just clear it
                session_state.pending_letter = None
                session_state.pending_confidence = 0.0
                await websocket.send_json({
                    "type": "letter_rejected",
                    "message": "Letter rejected"
                        })
            
            elif data.get("type") == "register_word":
                if session_state.sentence.strip():
                    session_state.registered_words.append(session_state.sentence.strip())
                    session_state.sentence = ""
                    
                    await websocket.send_json({
                        "type": "word_registered",
                        "words": session_state.registered_words,
                        "message": "Word registered!"
                    })
            
            elif data.get("type") == "clear":
                session_state.sentence = ""
                session_state.registered_words = []
                session_state.pending_letter = None
                session_state.pending_confidence = 0.0
                
                await websocket.send_json({
                    "type": "cleared",
                    "message": "Text cleared"
                })
            
            elif data.get("type") == "stop":
                break
    
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        # End session
        if session_state:
            db_session = db.query(DBSession).filter(
                DBSession.id == session_state.session_id
            ).first()
            if db_session:
                db_session.end_time = datetime.utcnow()
                db.commit()
        
        db.close()
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)