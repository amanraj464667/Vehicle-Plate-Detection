#!/usr/bin/env python3
"""
Dataset Setup Script for Vehicle Number Plate Detection Project

This script helps you download and setup test images for the project.
It provides multiple options for getting test data.
"""

import os
import requests
import zipfile
from pathlib import Path

# Create necessary directories
data_dir = Path("data/raw")
data_dir.mkdir(parents=True, exist_ok=True)

print("🚗 Vehicle Number Plate Detection - Dataset Setup")
print("=" * 50)

# Sample image URLs that should work
sample_urls = [
    "https://images.unsplash.com/photo-1519641471654-76ce0107ad1b?w=800",  # Car image
    "https://images.unsplash.com/photo-1621007947382-bb3c3994e3fb?w=800",  # Another car
]

def download_sample_images():
    """Download sample images from reliable sources"""
    print("\n📥 Downloading sample images...")
    
    for i, url in enumerate(sample_urls, 1):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            filename = f"test{i}.jpg"
            filepath = data_dir / filename
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✅ Downloaded: {filename}")
            
        except Exception as e:
            print(f"❌ Failed to download image {i}: {e}")

def setup_kaggle_instructions():
    """Provide instructions for Kaggle setup"""
    print("\n📚 Kaggle Dataset Setup Instructions:")
    print("-" * 40)
    print("1. Go to https://www.kaggle.com and create an account")
    print("2. Go to Account > API > Create New API Token")
    print("3. Download kaggle.json and place it in:")
    print(f"   {Path.home() / '.kaggle' / 'kaggle.json'}")
    print("4. Run: kaggle datasets download -d andrewmvd/car-plate-detection")
    print("5. Extract the downloaded ZIP to data/raw/")
    
def manual_setup_guide():
    """Provide manual setup instructions"""
    print("\n🔧 Manual Setup Options:")
    print("-" * 30)
    print("Option 1: Use your own images")
    print("- Take photos of vehicles with number plates")
    print("- Save them as JPG files in data/raw/")
    print("- Rename one to 'test1.jpg' for testing")
    print()
    print("Option 2: Download from internet")
    print("- Search for 'vehicle license plate images'")
    print("- Download and save to data/raw/test1.jpg")
    print()
    print("Option 3: Popular datasets to search on Kaggle:")
    print("- 'Indian Vehicle Dataset'")
    print("- 'License Plate Detection Dataset'")
    print("- 'Car Plate Detection'")
    print("- 'ANPR Dataset'")

def test_setup():
    """Test if the setup is working"""
    test_files = list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png"))
    
    if test_files:
        print(f"\n✅ Found {len(test_files)} test images:")
        for f in test_files[:5]:  # Show first 5
            print(f"   - {f.name}")
        
        print("\n🧪 You can now test with:")
        print(f"   python src/main.py --input {test_files[0]}")
    else:
        print("\n❌ No test images found in data/raw/")
        print("Please follow one of the setup options above.")

if __name__ == "__main__":
    # Try to download sample images
    try:
        download_sample_images()
    except Exception as e:
        print(f"❌ Download failed: {e}")
    
    # Show setup instructions
    setup_kaggle_instructions()
    manual_setup_guide()
    
    # Test current setup
    test_setup()
    
    print("\n" + "=" * 50)
    print("Setup complete! Follow the instructions above to get test data.")
