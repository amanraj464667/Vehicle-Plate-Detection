
import argparse, cv2, os, time, numpy as np
from src.yolo_detector import YOLODetector
from src.sort_tracker import SORTSimple
from src.database import init_db, insert_plate
from src.homography import load_homography

def run_video_sort(input_video, weights, output_video=None, meters_per_pixel=0.02, conf=0.25, homography_path=None):
    if not os.path.exists(input_video):
        print('Input video not found:', input_video)
        return
    yd = YOLODetector(weights_path=weights, conf=conf)
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if output_video:
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    else:
        out = None

    tracker = SORTSimple(iou_threshold=0.3, max_missed=8)
    frame_idx = 0
    init_db()

    H = None
    if homography_path is not None:
        if os.path.exists(homography_path):
            try:
                H = load_homography(homography_path)
                print('Loaded homography from', homography_path)
            except Exception as e:
                print('Failed to load homography:', e)
        else:
            print('Homography file not found:', homography_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = frame_idx / fps
        results = yd.detect(frame)  # list of {'xyxy':(x1,y1,x2,y2), 'conf':, 'cls':}
        bboxes = [r['xyxy'] for r in results]
        tracked = tracker.update(bboxes, timestamp)
        vis = frame.copy()
        for t in tracked:
            tid = t['id']
            x1,y1,x2,y2 = t['bbox']
            vx = t.get('vx', 0.0)
            vy = t.get('vy', 0.0)
            speed = None
            # homography-based conversion if available
            if H is not None and vx is not None:
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                src = np.array([[[cx, cy]], [[cx + vx, cy + vy]]], dtype='float32')
                try:
                    dst = cv2.perspectiveTransform(src, H)
                    dx = dst[1,0,0] - dst[0,0,0]
                    dy = dst[1,0,1] - dst[0,0,1]
                    dist_m = (dx**2 + dy**2) ** 0.5
                    # vx is in pixels per second approx, so dt=1s => dist_m per second
                    speed_m_s = dist_m
                    speed = speed_m_s * 3.6
                except Exception as e:
                    # fallback to pixel-based
                    speed_m_s = ((vx**2 + vy**2) ** 0.5) * meters_per_pixel
                    speed = speed_m_s * 3.6
            else:
                if vx is not None:
                    speed_m_s = ((vx**2 + vy**2) ** 0.5) * meters_per_pixel
                    speed = speed_m_s * 3.6
            label = f"ID:{tid}" + (f" | {speed:.1f} km/h" if speed is not None else "")
            cv2.rectangle(vis, (x1,y1), (x2,y2), (255,0,0), 2)
            cv2.putText(vis, label, (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            if speed is not None and speed > 0:
                crop = frame[y1:y2, x1:x2].copy()
                os.makedirs('results/detected_images', exist_ok=True)
                fname = f"plate_sort_{tid}_{frame_idx}.jpg"
                path = os.path.join('results/detected_images', fname)
                cv2.imwrite(path, crop)
                insert_plate(str(tid), time.strftime('%Y-%m-%dT%H:%M:%S'), float(speed), path)
        if out is not None:
            out.write(vis)
        frame_idx += 1

    cap.release()
    if out is not None:
        out.release()
    print('Processing finished. Output saved to', output_video if output_video else 'no output file provided (check results)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input video path')
    parser.add_argument('--weights', default='models/yolov8n.pt', help='YOLO weights (pt)')
    parser.add_argument('--output', default='results/track_sort_output.mp4', help='Output annotated video')
    parser.add_argument('--meters-per-pixel', type=float, default=0.02, help='Calibration meters per pixel')
    parser.add_argument('--conf', type=float, default=0.25, help='YOLO confidence threshold')
    parser.add_argument('--homography', default=None, help='Path to homography npz file to convert image->world meters')
    args = parser.parse_args()
    run_video_sort(args.input, args.weights, args.output, meters_per_pixel=args.meters_per_pixel, conf=args.conf, homography_path=args.homography)
