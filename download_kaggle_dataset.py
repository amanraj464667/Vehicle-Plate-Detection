#!/usr/bin/env python3
"""
Kaggle Dataset Downloader for License Plate Detection

This script downloads popular license plate datasets from Kaggle.
"""

import os
import subprocess
import zipfile
from pathlib import Path

# Popular license plate datasets on Kaggle
DATASETS = {
    "1": {
        "name": "Car Plate Detection",
        "kaggle_id": "andrewmvd/car-plate-detection",
        "description": "Car plate detection dataset with annotations"
    },
    "2": {
        "name": "License Plate Recognition",
        "kaggle_id": "aslanahmedov/license-plate-recognition",
        "description": "License plate recognition dataset"
    },
    "3": {
        "name": "Indian Vehicle Dataset", 
        "kaggle_id": "dataturks/indian-vehicle-dataset",
        "description": "Indian vehicle dataset with plates"
    },
    "4": {
        "name": "Vehicle License Plates",
        "kaggle_id": "andrewmvd/car-plate-detection",
        "description": "Vehicle license plate detection dataset"
    }
}

def check_kaggle_setup():
    """Check if Kaggle API is properly configured"""
    try:
        result = subprocess.run(["kaggle", "--version"], 
                              capture_output=True, text=True, check=True)
        print("✅ Kaggle CLI is installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Kaggle CLI not configured properly")
        print("\nTo setup Kaggle API:")
        print("1. Go to https://www.kaggle.com/settings/account")
        print("2. Click 'Create New API Token'")
        print("3. Place kaggle.json in:")
        kaggle_dir = Path.home() / ".kaggle"
        print(f"   {kaggle_dir}")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json (on Linux/Mac)")
        return False

def download_dataset(dataset_id, extract_to="data/kaggle_raw"):
    """Download and extract a dataset from Kaggle"""
    extract_path = Path(extract_to)
    extract_path.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"📥 Downloading dataset: {dataset_id}")
        
        # Download the dataset
        cmd = ["kaggle", "datasets", "download", "-d", dataset_id, "-p", str(extract_path)]
        subprocess.run(cmd, check=True)
        
        # Find and extract the ZIP file
        zip_files = list(extract_path.glob("*.zip"))
        if zip_files:
            zip_file = zip_files[0]
            print(f"📦 Extracting {zip_file.name}...")
            
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            
            # Remove the ZIP file
            zip_file.unlink()
            
            print(f"✅ Dataset extracted to {extract_path}")
            return True
        else:
            print("❌ No ZIP file found after download")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download dataset: {e}")
        return False
    except Exception as e:
        print(f"❌ Error extracting dataset: {e}")
        return False

def organize_images():
    """Organize downloaded images into the project structure"""
    kaggle_raw = Path("data/kaggle_raw")
    project_raw = Path("data/raw")
    
    if not kaggle_raw.exists():
        print("❌ No Kaggle data found")
        return
    
    # Find all image files
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(kaggle_raw.glob(f"**/{ext}"))
    
    if not image_files:
        print("❌ No image files found in downloaded dataset")
        return
    
    print(f"📁 Found {len(image_files)} images, organizing...")
    
    # Copy some images to the main data/raw directory
    project_raw.mkdir(parents=True, exist_ok=True)
    
    for i, img_file in enumerate(image_files[:10]):  # Take first 10 images
        dest_name = f"sample_{i+1}{img_file.suffix}"
        dest_path = project_raw / dest_name
        
        try:
            # Copy file
            with open(img_file, 'rb') as src, open(dest_path, 'wb') as dst:
                dst.write(src.read())
            print(f"   📋 Copied: {dest_name}")
        except Exception as e:
            print(f"   ❌ Failed to copy {img_file.name}: {e}")
    
    # Create a test1.jpg if it doesn't exist
    if not (project_raw / "test1.jpg").exists() and image_files:
        test_file = project_raw / "test1.jpg"
        try:
            with open(image_files[0], 'rb') as src, open(test_file, 'wb') as dst:
                dst.write(src.read())
            print("✅ Created test1.jpg for testing")
        except Exception as e:
            print(f"❌ Failed to create test1.jpg: {e}")

def main():
    """Main function"""
    print("🚗 Kaggle Dataset Downloader for License Plate Detection")
    print("=" * 60)
    
    if not check_kaggle_setup():
        return
    
    print("\nAvailable datasets:")
    for key, dataset in DATASETS.items():
        print(f"{key}. {dataset['name']}")
        print(f"   ID: {dataset['kaggle_id']}")
        print(f"   {dataset['description']}")
        print()
    
    try:
        choice = input("Choose a dataset (1-4) or 'q' to quit: ").strip()
        
        if choice.lower() == 'q':
            return
        
        if choice not in DATASETS:
            print("❌ Invalid choice")
            return
        
        dataset = DATASETS[choice]
        print(f"\n📦 Selected: {dataset['name']}")
        
        # Download the dataset
        if download_dataset(dataset['kaggle_id']):
            organize_images()
            
            print("\n" + "=" * 60)
            print("✅ Dataset setup complete!")
            print("\nYou can now test with:")
            print("   python src/main.py --input data/raw/test1.jpg")
        
    except KeyboardInterrupt:
        print("\n👋 Cancelled by user")

if __name__ == "__main__":
    main()
