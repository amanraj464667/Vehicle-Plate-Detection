
# Vehicle Number Plate Detection System for Indian Vehicles

This repository contains an end-to-end scaffold for a Vehicle Number Plate Detection and Recognition system
(Indian license plates). The scaffold includes:
- Plate detection (OpenCV contour baseline + easyOCR example)
- Character segmentation stub
- OCR recognition using EasyOCR or Tesseract (configurable)
- Simple speed estimation stub
- Simple SQLite-based storage
- Flask webapp skeleton to view results

**Important:** This is a starter project scaffold with working example code. You must install dependencies from `requirements.txt`.

## Quick start
1. Create a virtual environment: `python -m venv venv && source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
2. Install dependencies: `pip install -r requirements.txt`
3. Run a simple test:
   - Place a test image at `data/raw/test1.jpg`
   - Run: `python src/main.py --input data/raw/test1.jpg`
4. Start webapp: `python webapp/app.py` then open http://127.0.0.1:5000

## Project structure
See the repo layout in the ZIP. Extend YOLO or CNN-based detector by adding model weights into `models/` and updating `src/plate_detection.py`.

## Notes
- For production-grade detection use YOLOv8/YOLOv5 for localization and a fine-tuned CNN for character recognition.
- This scaffold uses **EasyOCR** as a ready-made OCR option; Tesseract config and a CNN option are included as stubs.


## YOLOv8 Integration (recommended first step)

This scaffold now includes a YOLOv8 integration example. YOLO improves plate localization significantly and is **recommended** before training OCR models or building tracking systems.

### Install Ultralytics YOLOv8
```bash
pip install ultralytics
```

### Download weights (example: yolov8n pretrained)
```bash
# official ultralytics weights (small model)
# this downloads ~14MB - ensure you have internet
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
# or download from https://github.com/ultralytics/assets/releases/ if you prefer
```

### How to run the YOLO inference example
```bash
# Run on a single image
python examples/yolo_infer.py --input data/raw/test1.jpg --weights models/yolov8n.pt --save-crops results/detected_images

# Run main pipeline using YOLO detector
python src/main.py --input data/raw/test1.jpg --detector yolo --weights models/yolov8n.pt
```

Notes:
- Pretrained weights included with Ultralytics are trained on COCO (no specific "license plate" class).
  For best performance, fine-tune YOLO on an annotated Indian license plate dataset (YOLO format).
- Place your custom-trained weights under `models/` and pass the path via `--weights` argument.


## OCR: Character-level CNN (optional, more accurate than Tesseract/EasyOCR for plate fonts)

This repo now includes a simple CNN classifier for individual characters (0-9, A-Z).
You can train the CNN on a character image dataset arranged in the ImageFolder format:
```
data/char_dataset/
   0/
   1/
   ...
   9/
   A/
   B/
   ...
   Z/
```
Each class folder should contain grayscale/colored images of characters.

### Train the CNN
```bash
python scripts/train_char_cnn.py --data-dir data/char_dataset --epochs 30 --batch-size 64 --save-path models/char_cnn.h5
```

### Use the CNN at inference
- Put the trained model at `models/char_cnn.h5` and run the main pipeline with `--ocr cnn`:
```bash
python src/main.py --input data/raw/test1.jpg --detector yolo --weights models/yolov8n.pt --ocr cnn --cnn-model models/char_cnn.h5
```
The pipeline will attempt to segment characters and use the CNN to predict each character. If segmentation fails, it will fall back to EasyOCR/Tesseract where configured.


## Speed Estimation & Tracking (priority #3)

This repository includes a simple tracker-based speed estimation demo.
It uses YOLO detections (plates) per frame and a centroid-based tracker to assign IDs and estimate speed.

### How it works (simple calibration)
- The tracker matches detected bounding boxes across consecutive frames via centroid distance.
- Speed = (pixel_distance * meters_per_pixel) / time_delta * 3.6 -> (km/h).
- `meters_per_pixel` must be calibrated for your camera and scene (provide approximate value).

### Run tracking on a video
```bash
python examples/track_speed.py --input data/raw/test_video.mp4 --weights models/yolov8n.pt --output results/track_output.mp4 --meters-per-pixel 0.02
```

Notes:
- This is a demo-grade estimator. For production-grade speed estimation you'll need camera calibration, homography to map image pixels to ground-plane meters, and a robust multi-object tracker like SORT/DeepSORT.
- The script will save cropped plates with detected speeds into `results/detected_images` and log them into the SQLite DB.


## Web App Enhancements (Dashboard & Challan)
- Dashboard at `/` shows detected plates and allows generating challans (simple HTML challan saved under `results/challans/`).
- API endpoint `/api/plates` returns recent plates as JSON.
- Endpoint `/challan/create` accepts POST form data to create a challan and returns its URL.

### Run webapp
```bash
python webapp/app.py
# open http://127.0.0.1:5000 in your browser
```


## Final: Homography + SORT integration complete

You can now calibrate homography and run the SORT-based tracker with homography-based speed estimation.
Use `examples/calibrate_homography.py` to compute `models/homography.npz` from correspondences, then run `examples/track_sort.py` with `--homography` to get accurate speeds in km/h (assuming good correspondences).
