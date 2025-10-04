# data/manager.py
import os
import numpy as np
import cv2
from modules.utils import ensure_dir

class DataManager:
    """
    Handles saving/loading of registered faces and embeddings.
    Stores:
      - cropped face images in data/registered_faces
      - embeddings in data/embeddings
    """

    def __init__(self, base_dir="data"):
        self.base_dir = base_dir
        self.face_dir = os.path.join(base_dir, "registered_faces")
        self.emb_dir = os.path.join(base_dir, "embeddings")
        ensure_dir(self.face_dir)
        ensure_dir(self.emb_dir)

    def save_face(self, face_id, face_img):
        """
        Save a cropped face image for a new registered face.
        """
        path = os.path.join(self.face_dir, f"face_{face_id}.jpg")
        cv2.imwrite(path, face_img)
        return path

    def save_embedding(self, face_id, embedding):
        """
        Save embedding as .npy file.
        """
        path = os.path.join(self.emb_dir, f"face_{face_id}.npy")
        np.save(path, np.array(embedding, dtype=float))
        return path

    def load_embedding(self, face_id):
        """
        Load embedding for a given face_id.
        """
        path = os.path.join(self.emb_dir, f"face_{face_id}.npy")
        if os.path.exists(path):
            return np.load(path)
        return None

    def list_registered_faces(self):
        """
        Return list of registered face IDs from embeddings directory.
        """
        files = [f for f in os.listdir(self.emb_dir) if f.endswith(".npy")]
        ids = [int(f.split("_")[1].split(".")[0]) for f in files]
        return sorted(ids)
