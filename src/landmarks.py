"""Thin wrapper around MediaPipe Face Mesh (imported lazily)."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class FaceLandmarker:
    def __init__(self) -> None:
        import mediapipe as mp

        self._mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Return an (N, 2) array of pixel coordinates, or None if no face."""
        h, w = frame_bgr.shape[:2]
        results = self._mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        lm = results.multi_face_landmarks[0].landmark
        return np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float64)

    def close(self) -> None:
        self._mesh.close()
