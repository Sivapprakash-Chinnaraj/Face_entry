# modules/recognizer.py
from .utils import load_config
import numpy as np

try:
    # InsightFace recommended API
    from insightface import app
except Exception as e:
    raise ImportError("insightface is required. pip install insightface") from e

class Recognizer:
    """Wrapper around InsightFace FaceAnalysis to produce normalized embeddings."""
    def __init__(self):
        self.cfg = load_config()
        self.threshold = self.cfg.get('match_threshold', 0.60)
        # FaceAnalysis will download and prepare models on first run (may take time)
        self.fa = app.FaceAnalysis()
        # use CPU by default (ctx_id=-1). If you have GPU, change to ctx_id=0
        self.fa.prepare(ctx_id=-1)

    def get_embedding(self, face_img):
        """
        Given a cropped face image (BGR/ndarray), returns a normalized 1D numpy array embedding or None.
        """
        faces = self.fa.get(face_img)
        if not faces:
            return None
        emb = np.array(faces[0].embedding, dtype=float)
        norm = np.linalg.norm(emb) + 1e-10
        return (emb / norm).astype(float)

    @staticmethod
    def cosine_similarity(a, b):
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
