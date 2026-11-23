"""Drowsiness decision logic, independent of camera and MediaPipe.

Feed ``DrowsinessDetector.update`` one ``FrameMetrics`` (or ``None`` when no face
is found) plus a timestamp in seconds. Durations are time-based, so behaviour does
not depend on the camera frame rate.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from statistics import median
from typing import Deque, List, Optional, Tuple

from .features import FrameMetrics


@dataclass
class Config:
    # Thresholds. ear_threshold is overwritten by calibration.
    ear_threshold: float = 0.22
    mar_threshold: float = 0.60
    pitch_threshold_deg: float = 15.0
    # How long a condition must persist before it counts (seconds).
    eye_closed_seconds: float = 1.0
    yawn_seconds: float = 1.0
    nod_seconds: float = 0.7
    # Alert stays on until no condition has been seen for this long.
    recover_seconds: float = 0.5
    # PERCLOS = fraction of recent frames with eyes closed.
    perclos_window_seconds: float = 60.0
    # Calibration.
    calibration_seconds: float = 5.0
    ear_calibration_ratio: float = 0.75
    ear_threshold_min: float = 0.15
    ear_threshold_max: float = 0.30


@dataclass
class Result:
    face_found: bool
    ear: float = 0.0
    mar: float = 0.0
    pitch_dev: float = 0.0
    eyes_closed: bool = False
    yawning: bool = False
    nodding: bool = False
    drowsy: bool = False
    perclos: float = 0.0
    event: Optional[str] = None  # "DROWSY_START" | "DROWSY_END" | None
    reasons: List[str] = field(default_factory=list)


class Calibrator:
    """Collects open-eye EAR and neutral head pitch while the driver looks ahead."""

    def __init__(self, seconds: float = 5.0, min_samples: int = 10):
        self.seconds = seconds
        self.min_samples = min_samples
        self._t0: Optional[float] = None
        self._ears: List[float] = []
        self._pitches: List[float] = []
        self._now = 0.0

    def add(self, t: float, ear: float, pitch: Optional[float]) -> None:
        if self._t0 is None:
            self._t0 = t
        self._now = t
        self._ears.append(ear)
        if pitch is not None:
            self._pitches.append(pitch)

    @property
    def progress(self) -> float:
        if self._t0 is None:
            return 0.0
        return min(1.0, (self._now - self._t0) / self.seconds)

    @property
    def done(self) -> bool:
        return self.progress >= 1.0 and len(self._ears) >= self.min_samples

    def result(self) -> Tuple[float, float]:
        """(median open-eye EAR, median baseline pitch)."""
        pitch = median(self._pitches) if self._pitches else 0.0
        return median(self._ears), pitch


class DrowsinessDetector:
    def __init__(self, config: Optional[Config] = None):
        self.cfg = config or Config()
        self.baseline_pitch = 0.0
        self.calibrated = False
        self.alert_active = False
        self._since = {"eyes": None, "yawn": None, "nod": None}
        self._last_trigger: Optional[float] = None
        self._perclos: Deque[Tuple[float, bool]] = deque()

    # -- calibration ---------------------------------------------------------
    def apply_calibration(self, open_ear: float, baseline_pitch: float) -> None:
        c = self.cfg
        c.ear_threshold = min(
            c.ear_threshold_max,
            max(c.ear_threshold_min, c.ear_calibration_ratio * open_ear),
        )
        self.baseline_pitch = baseline_pitch
        self.calibrated = True

    # -- per-frame update ----------------------------------------------------
    def _persisted(self, key: str, active: bool, t: float, seconds: float) -> bool:
        if not active:
            self._since[key] = None
            return False
        if self._since[key] is None:
            self._since[key] = t
        return (t - self._since[key]) >= seconds

    def _update_perclos(self, t: float, closed: bool) -> float:
        self._perclos.append((t, closed))
        cutoff = t - self.cfg.perclos_window_seconds
        while self._perclos and self._perclos[0][0] < cutoff:
            self._perclos.popleft()
        return sum(1 for _, c in self._perclos if c) / len(self._perclos)

    def update(self, metrics: Optional[FrameMetrics], t: float) -> Result:
        c = self.cfg
        if metrics is None:
            # No face: reset persistence counters; alert may recover below.
            for k in self._since:
                self._since[k] = None
            res = Result(face_found=False)
            raw_closed = raw_yawn = raw_nod = False
        else:
            pitch_dev = 0.0 if metrics.pitch is None else metrics.pitch - self.baseline_pitch
            below = metrics.ear < c.ear_threshold
            res = Result(
                face_found=True, ear=metrics.ear, mar=metrics.mar, pitch_dev=pitch_dev,
            )
            res.eyes_closed = self._persisted("eyes", below, t, c.eye_closed_seconds)
            res.yawning = self._persisted(
                "yawn", metrics.mar > c.mar_threshold, t, c.yawn_seconds)
            res.nodding = self._persisted(
                "nod", abs(pitch_dev) > c.pitch_threshold_deg, t, c.nod_seconds)
            res.perclos = self._update_perclos(t, below)
            raw_closed, raw_yawn, raw_nod = res.eyes_closed, res.yawning, res.nodding

        triggered = raw_closed or raw_yawn or raw_nod
        if triggered:
            self._last_trigger = t
            if not self.alert_active:
                self.alert_active = True
                res.event = "DROWSY_START"
        elif self.alert_active and (
            self._last_trigger is None or t - self._last_trigger >= c.recover_seconds
        ):
            self.alert_active = False
            res.event = "DROWSY_END"

        res.drowsy = self.alert_active
        if res.eyes_closed:
            res.reasons.append("EYES_CLOSED")
        if res.yawning:
            res.reasons.append("YAWN")
        if res.nodding:
            res.reasons.append("HEAD_NOD")
        return res
