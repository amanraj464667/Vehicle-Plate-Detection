
import argparse, os, cv2
from src.yolo_detector import YOLODetector, draw_detections

def run_image(input_path, weights, save_crops=None):
    img = cv2.imread(input_path)
    detector = YOLODetector(weights_path=weights)
    dets = detector.detect(img)
    out = draw_detections(img, dets)
    os.makedirs('results/detected_images', exist_ok=True)
    out_path = os.path.join('results/detected_images', os.path.basename(input_path))
    cv2.imwrite(out_path, out)
    print(f"Saved visualization to {out_path}")
    # Save crops if requested
    if save_crops:
        for i, d in enumerate(dets):
            x1,y1,x2,y2 = d['xyxy']
            crop = img[y1:y2, x1:x2].copy()
            crop_path = os.path.join(save_crops, f"crop_{i}_" + os.path.basename(input_path))
            cv2.imwrite(crop_path, crop)
            print('Saved crop:', crop_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to input image')
    parser.add_argument('--weights', default='models/yolov8n.pt', help='Path to YOLO weights (pt)')
    parser.add_argument('--save-crops', default=None, help='Directory to save plate crops (optional)')
    args = parser.parse_args()
    run_image(args.input, args.weights, args.save_crops)
