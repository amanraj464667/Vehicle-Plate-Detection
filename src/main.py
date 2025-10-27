
import argparse, os, cv2, datetime
from plate_detection import detect_plate_contour, crop_plate
try:
    from yolo_detector import YOLODetector
    _HAS_YOLO = True
except Exception:
    _HAS_YOLO = False

from ocr_recognition import OCRReader
from database import init_db, insert_plate
from robust_plate_pipeline import RobustPlatePipeline

def run_single_image(image_path, ocr_backend='easyocr', detector='yolo', weights=None):
    image = cv2.imread(image_path)
    if image is None:
        print('Failed to read', image_path)
        return

    # YOLO path first
    if detector == 'yolo':
        if not _HAS_YOLO:
            print('YOLO detector not available. Install ultralytics or choose --detector robust.')
        else:
            if weights is None:
                weights = 'models/yolov8n-license-plate.pt'
            try:
                yd = YOLODetector(weights_path=weights)
                dets = yd.detect(image)
                if dets:
                    best = max(dets, key=lambda d: d['conf'])
                    x1,y1,x2,y2 = best['xyxy']
                    # padding
                    h,w = image.shape[:2]
                    pad = int(0.12*max(h,w))
                    x1 = max(0,x1-pad); y1=max(0,y1-pad); x2=min(w-1,x2+pad); y2=min(h-1,y2+pad)
                    plate_img = image[y1:y2, x1:x2]
                    reader = OCRReader(backend=ocr_backend)
                    text = reader.read_plate(plate_img)
                    print(f"Detected (yolo): {text}")
                    os.makedirs('results/detected_images', exist_ok=True)
                    out_path = os.path.join('results/detected_images', os.path.basename(image_path))
                    cv2.imwrite(out_path, plate_img)
                    init_db()
                    insert_plate(text, datetime.datetime.now().isoformat(), speed=None, image_path=out_path)
                    return
            except Exception as e:
                print('YOLO inference failed:', e)
    
    # Robust pipeline path
    if detector == 'robust':
        pipeline = RobustPlatePipeline(languages=['en'])
        text, box, plate_img, conf, info = pipeline.process_image(image)
        if not text or box is None or plate_img is None:
            print('No plate found for', image_path)
            return
        print(f"Detected (robust): {text} (conf: {conf:.1f}%)")
        # Save result image
        os.makedirs('results/detected_images', exist_ok=True)
        out_path = os.path.join('results/detected_images', os.path.basename(image_path))
        cv2.imwrite(out_path, plate_img)
        # Save to DB
        init_db()
        insert_plate(text, datetime.datetime.now().isoformat(), speed=None, image_path=out_path)
        return

    # contour fallback
    box = None
    if detector == 'yolo':
        if not _HAS_YOLO:
            print('YOLO detector not available (install ultralytics and ensure src/yolo_detector.py is present). Falling back to contour.')
            box = detect_plate_contour(image)
        else:
            if weights is None:
                print('YOLO weights not provided. Falling back to contour detector.')
                box = detect_plate_contour(image)
            else:
                yd = YOLODetector(weights_path=weights)
                dets = yd.detect(image)
                if dets:
                    # choose highest confidence box
                    best = max(dets, key=lambda d: d['conf'])
                    x1,y1,x2,y2 = best['xyxy']
                    box = (x1, y1, x2-x1, y2-y1)
                else:
                    box = None
    else:
        box = detect_plate_contour(image)

    if box is None:
        print('No plate found for', image_path)
        return
    plate_img = crop_plate(image, box)
    # Initialize OCR reader
    if ocr_backend == 'cnn':
        reader = OCRReader(backend='cnn', cnn_model_path=cnn_model)
        # attempt segmentation and cnn-based recognition
        try:
            from character_segmentation import segment_characters
            segments = segment_characters(plate_img)
        except Exception:
            segments = None
        text = reader.read_plate(plate_img, segments=segments)
    else:
        reader = OCRReader(backend=ocr_backend)
        text = reader.read_plate(plate_img)
    print('Detected:', text)
    # Save result image
    os.makedirs('results/detected_images', exist_ok=True)
    out_path = os.path.join('results/detected_images', os.path.basename(image_path))
    cv2.imwrite(out_path, plate_img)
    # Save to DB
    init_db()
    insert_plate(text, datetime.datetime.now().isoformat(), speed=None, image_path=out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to input image')
    parser.add_argument('--ocr', default='easyocr', help='OCR backend: easyocr or tesseract')
    parser.add_argument('--detector', default='yolo', help='Detection method: yolo, robust or contour')
    parser.add_argument('--weights', default='models/yolov8n-license-plate.pt', help='Path to YOLO weights file')
    parser.add_argument('--cnn-model', default=None, help='Path to CNN model for character recognition')
    args = parser.parse_args()
    
    # Set global cnn_model variable for the cnn ocr option
    global cnn_model
    cnn_model = args.cnn_model
    
    run_single_image(args.input, args.ocr, args.detector, args.weights)
