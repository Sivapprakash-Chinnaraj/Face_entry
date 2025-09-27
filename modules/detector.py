# modules/detector.py
from ultralytics import YOLO
from .utils import load_config
import numpy as np

class Detector:
    """Simple YOLO wrapper for face detection.
       Model path can be changed in config or passed when constructing.
    """
    def __init__(self, model_path=None, device='cpu'):
        cfg = load_config()
        self.conf = cfg.get('confidence_threshold', 0.45)
        model_path = model_path or f"{cfg.get('models_dir','models')}/yolov8-face.pt"
        # create model object (will load weights)
        self.model = YOLO(model_path)

    def detect(self, frame):
        """
        Run a detection pass on `frame` (BGR numpy image).
        Returns a list of dicts: {'bbox':[x1,y1,x2,y2], 'conf':float}
        """
        # ultralytics returns a Results object; we take first result
        results = self.model(frame, conf=self.conf)
        detections = []
        if len(results) == 0:
            return detections
        res = results[0]
        if hasattr(res, 'boxes') and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()  # Nx4
            confs = res.boxes.conf.cpu().numpy()  # N
            for b, c in zip(xyxy, confs):
                detections.append({'bbox': [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                                   'conf': float(c)})
        return detections
