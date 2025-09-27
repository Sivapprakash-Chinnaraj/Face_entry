# modules/tracker.py
from .utils import centroid
import math

class SimpleTracker:
    """
    Simple centroid-based tracker.
    - Keeps object id -> bbox mapping
    - Registers new objects when no match is found
    - Deregisters objects after disappeared for too many frames
    """
    def __init__(self, max_disappeared=30, distance_threshold=80):
        self.next_id = 1
        self.objects = {}        # id -> bbox
        self.centroids = {}      # id -> (cx,cy)
        self.disappeared = {}    # id -> disappeared_count
        self.max_disappeared = max_disappeared
        self.distance_threshold = distance_threshold

    def register(self, bbox):
        oid = self.next_id
        self.next_id += 1
        self.objects[oid] = bbox
        self.centroids[oid] = centroid(bbox)
        self.disappeared[oid] = 0
        return oid

    def deregister(self, oid):
        if oid in self.objects:
            del self.objects[oid]
            del self.centroids[oid]
            del self.disappeared[oid]

    def update(self, detections):
        """
        detections: list of bboxes [[x1,y1,x2,y2], ...]
        Returns: dict of current objects {id: bbox}, and list of exited ids
        """
        exited = []

        if len(detections) == 0:
            # increment disappeared counters
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    exited.append(oid)
                    self.deregister(oid)
            return dict(self.objects), exited

        # compute centroids for detections
        det_centroids = [centroid(b) for b in detections]

        # if no existing objects, register all detections
        if len(self.centroids) == 0:
            for b in detections:
                self.register(b)
            return dict(self.objects), exited

        # match existing objects to nearest detection (greedy)
        used_det = set()
        # for each existing object, find nearest detection
        for oid, c in list(self.centroids.items()):
            best_det_idx = None
            best_dist = None
            for i, dc in enumerate(det_centroids):
                if i in used_det:
                    continue
                d = math.hypot(c[0] - dc[0], c[1] - dc[1])
                if best_dist is None or d < best_dist:
                    best_dist = d
                    best_det_idx = i
            if best_det_idx is not None and best_dist <= self.distance_threshold:
                # update object with matched detection
                self.objects[oid] = detections[best_det_idx]
                self.centroids[oid] = det_centroids[best_det_idx]
                self.disappeared[oid] = 0
                used_det.add(best_det_idx)
            else:
                # not matched -> increment disappeared
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    exited.append(oid)
                    self.deregister(oid)

        # any detections left unmatched -> register new objects
        for i, b in enumerate(detections):
            if i not in used_det:
                self.register(b)

        return dict(self.objects), exited
