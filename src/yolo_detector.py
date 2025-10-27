
import os
import cv2
import numpy as np

# YOLOv8 integration (uses ultralytics package)
# Note: ultralytics must be installed in the user's environment:
# pip install ultralytics
try:
    from ultralytics import YOLO
    _HAS_ULTRALYTICS = True
except Exception:
    _HAS_ULTRALYTICS = False

class YOLODetector:
    def __init__(self, weights_path='models/yolov8n-license-plate.pt', device='cpu', conf=0.25, iou=0.5):
        if not _HAS_ULTRALYTICS:
            raise ImportError('ultralytics YOLO is not installed. pip install ultralytics')
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f'Weights not found at: {weights_path}. Download or place weights there.')
        self.model = YOLO(weights_path)
        self.conf = conf
        self.iou = iou
        # Try to find license-plate class id if available
        try:
            self.names = self.model.model.names  # type: ignore[attr-defined]
            self.plate_class_ids = {i for i,n in self.names.items() if 'plate' in str(n).lower()}
        except Exception:
            self.names = None
            self.plate_class_ids = set()

    def detect(self, image):
        """Run YOLO inference on an image (numpy BGR image).
        Returns a list of detections: [{'xyxy':(x1,y1,x2,y2), 'conf':float, 'cls':int}, ...]
        """
        # ultralytics accepts either file path or ndarray in RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model.predict(source=img_rgb, conf=self.conf, iou=self.iou, verbose=False)
        detections = []
        if len(results) == 0:
            return detections
        res = results[0]
        # res.boxes contains boxes if any
        try:
            boxes = res.boxes.xyxy.cpu().numpy()  # shape (n,4)
            scores = res.boxes.conf.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy().astype(int)
            for b, s, c in zip(boxes, scores, classes):
                if self.plate_class_ids and int(c) not in self.plate_class_ids:
                    continue
                x1, y1, x2, y2 = map(int, b.tolist())
                detections.append({'xyxy': (x1, y1, x2, y2), 'conf': float(s), 'cls': int(c)})
        except Exception:
            # older ultralytics versions may differ
            for box in res.boxes:
                b = box.xyxy[0].tolist()
                s = float(box.conf[0])
                c = int(box.cls[0])
                if self.plate_class_ids and int(c) not in self.plate_class_ids:
                    continue
                x1, y1, x2, y2 = map(int, b)
                detections.append({'xyxy': (x1, y1, x2, y2), 'conf': s, 'cls': c})
        return detections

def draw_detections(image, detections, color=(0,255,0), thickness=2):
    out = image.copy()
    for d in detections:
        x1, y1, x2, y2 = d['xyxy']
        cv2.rectangle(out, (x1,y1), (x2,y2), color, thickness)
        label = f"{d['conf']:.2f}"
        cv2.putText(out, label, (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out
