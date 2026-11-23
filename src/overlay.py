"""Drawing helpers shared by the desktop app and the Streamlit dashboard."""
from __future__ import annotations

import cv2
import numpy as np

from .detector import Result

RED = (0, 0, 255)
GREEN = (0, 200, 0)
WHITE = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_calibration(frame: np.ndarray, progress: float, face_found: bool) -> None:
    h, w = frame.shape[:2]
    msg = "Calibrating: look at the road, eyes open" if face_found else "No face detected"
    cv2.putText(frame, msg, (10, 30), FONT, 0.7, WHITE, 2)
    cv2.rectangle(frame, (10, 45), (w - 10, 65), WHITE, 1)
    cv2.rectangle(frame, (10, 45), (10 + int((w - 20) * progress), 65), GREEN, -1)


def draw_status(frame: np.ndarray, res: Result, ear_threshold: float) -> None:
    lines = [
        f"EAR  {res.ear:.2f} (thr {ear_threshold:.2f})",
        f"MAR  {res.mar:.2f}",
        f"Head {res.pitch_dev:+.0f} deg",
        f"PERCLOS {res.perclos * 100:.0f}%",
    ]
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (10, 25 + 24 * i), FONT, 0.6, WHITE, 2)
    if not res.face_found:
        cv2.putText(frame, "No face detected", (10, 130), FONT, 0.7, (0, 165, 255), 2)
    if res.drowsy:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 50), (w, h), RED, -1)
        label = "DROWSINESS ALERT"
        if res.reasons:
            label += ": " + ", ".join(res.reasons)
        cv2.putText(frame, label, (10, h - 15), FONT, 0.8, WHITE, 2)
