"""Facial-feature geometry: EAR, MAR and head pitch.

Every function here works on plain NumPy arrays of *pixel* coordinates, so it can
be unit-tested without a camera or MediaPipe. ``coords`` is an (N, 2) array whose
row index is the MediaPipe Face Mesh landmark index.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# --- MediaPipe Face Mesh landmark indices -----------------------------------
# Eye order is p1..p6 as in Soukupova & Cech (2016): p1/p4 = corners,
# p2/p3 = upper lid, p6/p5 = lower lid.
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Inner-lip landmarks: three (upper, lower) pairs for mouth opening and the two
# mouth corners for mouth width.
MOUTH_VERTICAL_PAIRS = [(82, 87), (13, 14), (312, 317)]
MOUTH_CORNERS = (78, 308)

# Nose tip, chin, outer eye corners, mouth corners -> used for head pose.
HEAD_POSE_IDX = [1, 152, 33, 263, 61, 291]

# Generic 3D face model (arbitrary units) in camera-style axes: x right, y down,
# z away from the camera. A frontal face therefore gives a pose close to identity.
_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),        # nose tip
        (0.0, 330.0, 65.0),     # chin
        (-225.0, -170.0, 135.0),  # outer corner, eye on image-left
        (225.0, -170.0, 135.0),   # outer corner, eye on image-right
        (-150.0, 150.0, 125.0),   # mouth corner, image-left
        (150.0, 150.0, 125.0),    # mouth corner, image-right
    ],
    dtype=np.float64,
)


@dataclass
class FrameMetrics:
    """Raw per-frame measurements (before calibration)."""

    ear: float
    mar: float
    pitch: Optional[float]  # degrees; None if head pose could not be solved


def _dist(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def eye_aspect_ratio(eye: np.ndarray) -> float:
    """EAR = (|p2-p6| + |p3-p5|) / (2 |p1-p4|) for a (6, 2) array of eye points."""
    horizontal = _dist(eye[0], eye[3])
    if horizontal < 1e-6:
        return 0.0
    return (_dist(eye[1], eye[5]) + _dist(eye[2], eye[4])) / (2.0 * horizontal)


def mouth_aspect_ratio(coords: np.ndarray) -> float:
    """MAR = mean inner-lip opening / mouth width, from full landmark array."""
    width = _dist(coords[MOUTH_CORNERS[0]], coords[MOUTH_CORNERS[1]])
    if width < 1e-6:
        return 0.0
    openings = [_dist(coords[u], coords[l]) for u, l in MOUTH_VERTICAL_PAIRS]
    return float(np.mean(openings)) / width


def estimate_pitch(coords: np.ndarray, frame_w: int, frame_h: int) -> Optional[float]:
    """Head pitch in degrees (sign depends on convention; use deviations from a
    calibrated baseline rather than the absolute value)."""
    image_points = np.array([coords[i] for i in HEAD_POSE_IDX], dtype=np.float64)
    focal = float(frame_w)
    camera = np.array(
        [[focal, 0, frame_w / 2.0], [0, focal, frame_h / 2.0], [0, 0, 1]],
        dtype=np.float64,
    )
    ok, rvec, _ = cv2.solvePnP(
        _MODEL_POINTS, image_points, camera, np.zeros((4, 1)),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    pitch = float(cv2.RQDecomp3x3(rotation)[0][0])
    # Wrap into [-90, 90] so a frontal face is near 0 regardless of solver branch.
    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180
    return pitch


def compute_metrics(coords: np.ndarray, frame_w: int, frame_h: int) -> FrameMetrics:
    left = eye_aspect_ratio(coords[LEFT_EYE])
    right = eye_aspect_ratio(coords[RIGHT_EYE])
    return FrameMetrics(
        ear=(left + right) / 2.0,
        mar=mouth_aspect_ratio(coords),
        pitch=estimate_pitch(coords, frame_w, frame_h),
    )
