"""
Download ASL Alphabet dataset from Kaggle
You need to have kaggle API set up: pip install kaggle
And place your kaggle.json in ~/.kaggle/

Dataset: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
"""
import os
import subprocess
import sys

def download_asl_dataset():
    dataset_dir = "dataset/asl_alphabet"
    
    # Check if kaggle is installed
    try:
        import kaggle
    except ImportError:
        print("❌ Kaggle not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"], check=True)
    
    # Check if kaggle.json exists
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json):
        print("❌ Kaggle API key not found!")
        print("Please follow these steps:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Download kaggle.json")
        print("4. Place it in ~/.kaggle/kaggle.json")
        print("5. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    # Create dataset directory
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Download dataset
    print("📥 Downloading ASL Alphabet dataset from Kaggle...")
    print("   This may take a few minutes (1.1 GB)...")
    
    try:
        subprocess.run([
            "kaggle", "datasets", "download", "-d", "grassknoted/asl-alphabet",
            "-p", dataset_dir, "--unzip"
        ], check=True)
        print("✅ Dataset downloaded successfully!")
        print(f"   Location: {dataset_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading dataset: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("ASL Alphabet Dataset Downloader")
    print("=" * 60)
    success = download_asl_dataset()
    if success:
        print("\n✅ Next steps:")
        print("1. Run: python3 app/preprocess.py")
        print("2. Run: python3 app/train.py")
        print("3. Start the web app!")
    else:
        print("\n❌ Download failed. Please check the error messages above.")
