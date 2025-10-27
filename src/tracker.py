
import time
import math

class PlateTracker:
    """Simple centroid-based tracker for detected plates across frames.
    This is NOT a full SORT implementation, but a lightweight tracker suitable for demos.
    It matches detections frame-to-frame by centroid distance.
    """
    def __init__(self, max_lost=5, dist_threshold=50, meters_per_pixel=0.02):
        # object_id -> info dict: { 'centroid':(x,y), 'last_seen':frame_idx, 'trace':[(time,centroid)], 'lost':0, 'speed':None }
        self.next_object_id = 1
        self.objects = {}
        self.max_lost = max_lost
        self.dist_threshold = dist_threshold
        self.meters_per_pixel = meters_per_pixel

    def _centroid(self, bbox):
        x1,y1,x2,y2 = bbox
        cx = int((x1+x2)/2)
        cy = int((y1+y2)/2)
        return (cx, cy)

    def update(self, detections, frame_idx, timestamp):
        """detections: list of bboxes in xyxy format [(x1,y1,x2,y2), ...]
           frame_idx: integer frame index
           timestamp: float seconds (e.g., frame_idx / fps)
           Returns list of tracked objects with id, bbox, speed_kmh (or None)
        """
        # Prepare incoming centroids
        incoming = [self._centroid(d) for d in detections]
        assigned = {}
        results = []

        # If no existing objects, register all detections
        if len(self.objects) == 0:
            for bbox, cent in zip(detections, incoming):
                oid = self.next_object_id
                self.next_object_id += 1
                self.objects[oid] = {'centroid': cent, 'last_seen': frame_idx, 'trace': [(timestamp, cent)], 'lost':0, 'speed': None}
                results.append({'id': oid, 'bbox': bbox, 'speed': None})
            return results

        # Match existing objects to incoming detections by nearest centroid
        unmatched_incoming = set(range(len(incoming)))
        unmatched_existing = set(self.objects.keys())

        # Compute distance matrix
        dist_list = []
        for oid, info in self.objects.items():
            for i, cent in enumerate(incoming):
                dx = info['centroid'][0] - cent[0]
                dy = info['centroid'][1] - cent[1]
                dist = math.hypot(dx, dy)
                dist_list.append((dist, oid, i))
        # Sort by distance ascending
        dist_list.sort(key=lambda x: x[0])

        for dist, oid, i in dist_list:
            if i not in unmatched_incoming or oid not in unmatched_existing:
                continue
            if dist > self.dist_threshold:
                continue
            # assign
            unmatched_incoming.remove(i)
            unmatched_existing.remove(oid)
            bbox = detections[i]
            cent = incoming[i]
            info = self.objects[oid]
            # compute speed if we have a previous trace
            speed = None
            if len(info['trace']) > 0:
                prev_time, prev_cent = info['trace'][-1]
                dt = timestamp - prev_time if timestamp - prev_time > 0 else 1e-6
                dx = cent[0] - prev_cent[0]
                dy = cent[1] - prev_cent[1]
                dist_pixels = math.hypot(dx, dy)
                dist_meters = dist_pixels * self.meters_per_pixel
                speed_m_s = dist_meters / dt
                speed = speed_m_s * 3.6  # km/h
                info['speed'] = speed
            info['centroid'] = cent
            info['last_seen'] = frame_idx
            info['trace'].append((timestamp, cent))
            info['lost'] = 0
            results.append({'id': oid, 'bbox': bbox, 'speed': info.get('speed', None)})

        # Register unmatched incoming as new objects
        for i in list(unmatched_incoming):
            bbox = detections[i]
            cent = incoming[i]
            oid = self.next_object_id
            self.next_object_id += 1
            self.objects[oid] = {'centroid': cent, 'last_seen': frame_idx, 'trace': [(timestamp, cent)], 'lost':0, 'speed': None}
            results.append({'id': oid, 'bbox': bbox, 'speed': None})

        # Mark unmatched existing as lost
        for oid in list(unmatched_existing):
            info = self.objects[oid]
            info['lost'] += 1
            if info['lost'] > self.max_lost:
                del self.objects[oid]

        return results
