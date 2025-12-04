#!/usr/bin/env python3
"""Check users in the database"""

from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from passlib.context import CryptContext
from datetime import datetime

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

# Get all users
users = db.query(DBUser).all()
print(f'\n=== Database Users ({len(users)} total) ===\n')
for user in users:
    print(f'Username: {user.username}')
    print(f'Email: {user.email}')
    print(f'Created: {user.created_at}')
    print(f'Password Hash: {user.hashed_password[:50]}...')
    print('-' * 50)

# Offer to create a test user
if len(users) == 0:
    print('\n⚠️  No users found! Creating a test user...')
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    test_user = DBUser(
        username="testuser",
        email="test@example.com",
        hashed_password=pwd_context.hash("password123")
    )
    db.add(test_user)
    db.commit()
    print('✅ Test user created!')
    print('   Username: testuser')
    print('   Password: password123')
else:
    print('\n💡 You can try logging in with one of the usernames above')
    print('   If you forgot passwords, you can register a new user')
    print('\n❓ Want to add a new test user? (y/n): ', end='')
    response = input().strip().lower()
    if response == 'y':
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        test_user = DBUser(
            username="admin",
            email="admin@example.com",
            hashed_password=pwd_context.hash("admin123")
        )
        try:
            db.add(test_user)
            db.commit()
            print('✅ Admin user created!')
            print('   Username: admin')
            print('   Password: admin123')
        except Exception as e:
            print(f'❌ Error: {e}')
            print('   User may already exist')

db.close()
