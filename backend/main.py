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
from fastapi.middleware.cors import CORSMiddleware



# ============= Configuration =============
SECRET_KEY = "your-secret-key-change-in-production-use-openssl-rand-hex-32"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

MODEL_PATH = "../app/gesture_model.h5"
LABEL_MAP_PATH = "../app/label_map.json"
IMG_SIZE = (64, 64)
ROI = (200, 80, 520, 400)
CONF_THRESHOLD = 0.75
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
        x1, y1, x2, y2 = roi
        roi_crop = frame[y1:y2, x1:x2]
        roi_resized = cv2.resize(roi_crop, IMG_SIZE)
        
        if self.expected_channels == 1:
            proc = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
            proc = proc.reshape(IMG_SIZE[1], IMG_SIZE[0], 1)
        else:
            proc = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
            proc = proc.reshape(IMG_SIZE[1], IMG_SIZE[0], 3)
        
        proc = proc.astype("float32") / 255.0
        input_tensor = np.expand_dims(proc, axis=0)
        
        preds = self.model.predict(input_tensor, verbose=0)[0]
        class_idx = int(np.argmax(preds))
        conf = float(preds[class_idx])
        label = self.idx_to_label.get(class_idx, str(class_idx))
        
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
                
                # Predict
                label, conf = recognizer.predict(frame)
                
                # Temporal smoothing
                now = datetime.utcnow().timestamp()
                
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
                    "stable": session_state.frame_count_same >= STABLE_FRAMES
                })
                
                # Accept stable prediction
                if session_state.frame_count_same >= STABLE_FRAMES:
                    if (label != session_state.last_accepted_label) or \
                       (now - session_state.last_accept_time > TEXT_COOLDOWN):
                        
                        if label.lower() == "space":
                            session_state.sentence += " "
                            final_letter = "SPACE"
                        elif label.lower() in ["del", "delete"]:
                            if session_state.sentence:
                                session_state.sentence = session_state.sentence[:-1]
                            final_letter = "DELETE"
                        elif label.lower() == "nothing":
                            continue
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
                        session_state.last_accept_time = now
                        session_state.frame_count_same = 0
                        
                        await websocket.send_json({
                            "type": "letter_detected",
                            "letter": final_letter,
                            "confidence": conf,
                            "sentence": session_state.sentence,
                            "word_count": len(session_state.registered_words)
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