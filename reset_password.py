#!/usr/bin/env python3
"""Reset user password or create new user"""

from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from passlib.context import CryptContext
from datetime import datetime
import sys

Base = declarative_base()

class DBUser(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

# Connect to database
engine = create_engine('sqlite:///./backend/asl_recognition.db')
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

print("\n=== ASL User Management ===\n")
print("What would you like to do?")
print("1. Create a new test user")
print("2. Reset password for existing user")
print("\nChoice (1 or 2): ", end='')

choice = input().strip()

if choice == '1':
    print("\n--- Create New User ---")
    username = "testuser"
    email = "test@test.com"
    password = "test123"
    
    # Check if user already exists
    existing = db.query(DBUser).filter(DBUser.username == username).first()
    if existing:
        print(f"⚠️  User '{username}' already exists!")
        print("Use option 2 to reset the password")
    else:
        new_user = DBUser(
            username=username,
            email=email,
            hashed_password=pwd_context.hash(password)
        )
        db.add(new_user)
        db.commit()
        print(f"✅ User created successfully!")
        print(f"   Username: {username}")
        print(f"   Password: {password}")

elif choice == '2':
    print("\n--- Reset User Password ---")
    print("Available users:")
    users = db.query(DBUser).all()
    for i, user in enumerate(users, 1):
        print(f"  {i}. {user.username} ({user.email})")
    
    print("\nEnter username to reset: ", end='')
    username = input().strip()
    
    user = db.query(DBUser).filter(DBUser.username == username).first()
    if user:
        new_password = "newpass123"
        user.hashed_password = pwd_context.hash(new_password)
        db.commit()
        print(f"\n✅ Password reset successfully!")
        print(f"   Username: {username}")
        print(f"   New Password: {new_password}")
    else:
        print(f"❌ User '{username}' not found!")
else:
    print("Invalid choice!")

db.close()
