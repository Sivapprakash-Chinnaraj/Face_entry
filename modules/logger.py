# modules/logger.py
import logging
import os
import cv2
from .utils import load_config, ensure_dir, get_timestamp, crop_face
from .database import Database

cfg = load_config()
ensure_dir(cfg.get('logs_dir', 'logs'))

# Setup a simple file logger
logger = logging.getLogger('face_tracker')
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(cfg.get('log_path', 'events.log'))
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

# DB instance (shared)
db = Database(cfg.get('db_path', 'db/visitors.db'))

def log_system(message):
    """Write a line to events.log via logging module."""
    logger.info(message)

def log_face_event(face_id, event_type, bbox, frame):
    """
    Save cropped face image, insert event into DB, and log summary.
      - event_type: 'entry' or 'exit'
      - bbox: [x1,y1,x2,y2]
      - frame: full BGR image
    """
    timestamp = get_timestamp()
    date = timestamp.split('T')[0]
    base_dir = os.path.join(cfg.get('logs_dir', 'logs'), f"{event_type}s", date)
    ensure_dir(base_dir)
    filename = f"{event_type}_{face_id}_{timestamp.replace(':','-')}.jpg"
    path = os.path.join(base_dir, filename)

    # crop and save face image
    face_img = crop_face(frame, bbox)
    # OpenCV uses BGR; ensure saving works
    cv2.imwrite(path, face_img)

    # insert into DB
    try:
        db.insert_event(face_id, event_type, timestamp, path)
    except Exception as e:
        log_system(f"DB insert_event failed for face {face_id}: {e}")

    # log to events.log too
    log_system(f"{event_type.upper()} - face_id={face_id} path={path}")

def close():
    try:
        db.close()
    except Exception:
        pass
