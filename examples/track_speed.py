
import argparse, cv2, os, time, numpy as np
from src.yolo_detector import YOLODetector, draw_detections
from src.tracker import PlateTracker
from src.database import init_db, insert_plate
from src.homography import load_homography

def run_video(input_video, weights, output_video=None, meters_per_pixel=0.02, conf=0.25, homography_path=None):
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

    tracker = PlateTracker(meters_per_pixel=meters_per_pixel)
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
        tracked = tracker.update(bboxes, frame_idx, timestamp)
        # If homography provided, compute speed per tracked object using world coords
        if H is not None:
            for obj in tracked:
                oid = obj['id']
                try:
                    info = tracker.objects.get(int(oid), None)
                except Exception:
                    info = None
                if info is not None and len(info.get('trace', [])) >= 2:
                    (t_prev, c_prev), (t_curr, c_curr) = info['trace'][-2], info['trace'][-1]
                    src = np.array([[[c_prev[0], c_prev[1]], [c_curr[0], c_curr[1]]]], dtype='float32')
                    try:
                        dst = cv2.perspectiveTransform(src, H)
                        dx = dst[0,1,0] - dst[0,0,0]
                        dy = dst[0,1,1] - dst[0,0,1]
                        dist_m = (dx**2 + dy**2) ** 0.5
                        dt = t_curr - t_prev if (t_curr - t_prev) > 0 else 1e-6
                        speed_m_s = dist_m / dt
                        obj['speed'] = speed_m_s * 3.6
                    except Exception as e:
                        pass
        # Draw detections + IDs + speed
        vis = frame.copy()
        for t in tracked:
            oid = t['id']
            x1,y1,x2,y2 = t['bbox']
            speed = t.get('speed', None)
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
            label = f"ID:{oid}" + (f" | {speed:.1f} km/h" if speed is not None else "")
            cv2.putText(vis, label, (x1, max(0,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            if speed is not None:
                crop = frame[y1:y2, x1:x2].copy()
                os.makedirs('results/detected_images', exist_ok=True)
                fname = f"plate_{oid}_{frame_idx}.jpg"
                path = os.path.join('results/detected_images', fname)
                cv2.imwrite(path, crop)
                insert_plate(str(oid), time.strftime('%Y-%m-%dT%H:%M:%S'), speed, path)

        if out is not None:
            out.write(vis)
        frame_idx += 1

    cap.release()
    if out is not None:
        out.release()
    print('Processing finished. Output saved to', output_video if output_video else 'no output file provided (check live display)')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input video path')
    parser.add_argument('--weights', default='models/yolov8n.pt', help='YOLO weights (pt)')
    parser.add_argument('--output', default='results/track_output.mp4', help='Output annotated video')
    parser.add_argument('--meters-per-pixel', type=float, default=0.02, help='Calibration meters per pixel')
    parser.add_argument('--conf', type=float, default=0.25, help='YOLO confidence threshold')
    parser.add_argument('--homography', default=None, help='Path to homography npz file to convert image->world meters')
    args = parser.parse_args()
    run_video(args.input, args.weights, args.output, meters_per_pixel=args.meters_per_pixel, conf=args.conf, homography_path=args.homography)
