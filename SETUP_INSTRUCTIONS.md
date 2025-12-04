# ASL Sign Language Recognition - Setup Instructions

## Problem Identified
Your model was trained on the wrong dataset (leapGestRecog with only 10 gesture types like "palm", "fist", "L", etc.) but you need to detect A-Z alphabet letters for sign language.

## Solution: Download and Train on ASL Alphabet Dataset

### Step 1: Install Kaggle API
```bash
pip install kaggle
```

### Step 2: Set up Kaggle API credentials
1. Go to https://www.kaggle.com/account
2. Scroll down to "API" section
3. Click "Create New API Token"
4. This downloads `kaggle.json`
5. Move it to the right location:
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Step 3: Download ASL Alphabet Dataset
```bash
cd /home/slateboi/Documents/SE_Project/Codes/Backend/SE_Project
python3 download_asl_dataset.py
```

This will download ~1.1GB of data with images for:
- A-Z (26 letters)
- del (delete)
- nothing (no gesture)
- space

Total: 29 classes, ~87,000 images

### Step 4: Preprocess the Dataset
```bash
python3 app/preprocess.py
```

This creates:
- `dataset/x_train.npy` - Training images
- `dataset/x_test.npy` - Test images  
- `dataset/y_train.npy` - Training labels
- `dataset/y_test.npy` - Test labels
- `dataset/label_dict.npy` - Label mapping

### Step 5: Train the Model
```bash
python3 app/train.py
```

This will:
- Train a CNN model (MobileNetV2 transfer learning)
- Take 15-30 minutes depending on your hardware
- Save to `app/gesture_model.h5`
- Create `app/label_map.json` with A-Z mappings

### Step 6: Run the Web App
```bash
cd backend
python3 -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Then open in browser: **http://localhost:8000**

(Must use `localhost` not `127.0.0.1` for camera to work!)

## Alternative: Use LeapGestRecog Dataset

If you want to use the existing leapGestRecog dataset (10 gestures only):

1. Edit `app/preprocess.py`:
   ```python
   USE_ASL_ALPHABET = False  # Change to False
   ```

2. Run preprocessing and training as above

## Files Modified

✅ `download_asl_dataset.py` - Downloads Kaggle ASL dataset
✅ `app/preprocess.py` - Supports both ASL alphabet and LeapGestRecog
✅ `app/train.py` - Already correct
✅ `app/infer_signlang.py` - Fixed label loading
✅ `backend/main.py` - Fixed label loading

## Current Status

- ❌ Model trained on wrong dataset (only 10 gestures, not A-Z)
- ✅ Code fixed to support ASL alphabet
- ⏳ Need to download dataset and retrain

## Next Steps

1. Run `python3 download_asl_dataset.py`
2. Run `python3 app/preprocess.py`
3. Run `python3 app/train.py` (takes 15-30 min)
4. Restart the backend server
5. Test in browser!
