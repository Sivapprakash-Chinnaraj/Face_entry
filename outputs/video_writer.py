# outputs/video_writer.py
import cv2
import os
from modules.utils import ensure_dir

class VideoWriter:
    def __init__(self, output_path, frame_size, fps=20):
        ensure_dir(os.path.dirname(output_path))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()