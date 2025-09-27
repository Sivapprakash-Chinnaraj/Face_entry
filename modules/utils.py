# modules/utils.py
import os
import json
import datetime
import numpy as np

def load_config(path='config.json'):
    """Load config.json or return defaults."""
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    # defaults
    return {
        "frame_skip": 5,
        "confidence_threshold": 0.45,
        "match_threshold": 0.60,
        "db_path": "db/visitors.db",
        "logs_dir": "logs",
        "models_dir": "models",
        "log_path": "events.log",
        "track_disappeared_frames": 30,
        "distance_threshold": 80
    }

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def get_timestamp():
    """Return timestamp string (YYYY-MM-DDTHH:MM:SS)."""
    return datetime.datetime.now().replace(microsecond=0).isoformat()

def bbox_to_int(bbox):
    """Convert bbox [x1,y1,x2,y2] to ints."""
    return [int(round(float(x))) for x in bbox]

def crop_face(frame, bbox):
    """Crop bounding box from frame (safe clamping)."""
    x1, y1, x2, y2 = bbox_to_int(bbox)
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w-1))
    x2 = max(0, min(x2, w-1))
    y1 = max(0, min(y1, h-1))
    y2 = max(0, min(y2, h-1))
    if y2 <= y1 or x2 <= x1:
        # if bbox invalid, return a tiny crop to avoid errors
        return frame[0:1, 0:1].copy()
    return frame[y1:y2, x1:x2].copy()

def iou(boxA, boxB):
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    boxBArea = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    denom = boxAArea + boxBArea - interArea
    return interArea / denom if denom > 0 else 0.0

def centroid(bbox):
    x1, y1, x2, y2 = bbox_to_int(bbox)
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return (cx, cy)
