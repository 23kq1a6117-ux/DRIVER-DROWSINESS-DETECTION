"""Evaluate the detector on a recorded video, for the report's results chapter.

    python -m evaluation.evaluate_video clip.mp4 --labels clip_labels.csv

``--labels`` is a CSV with columns ``start,end`` (seconds) marking the periods in
which the driver is truly drowsy (eyes closed, yawning, nodding). Frames outside
those periods count as alert. Without labels, only the per-frame CSV is written.
The first ``--calib-seconds`` of the video are used for calibration and excluded
from the metrics, so start each clip with a few seconds of normal, alert driving.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Sequence, Tuple

import cv2

from src.detector import Calibrator, Config, DrowsinessDetector
from src.features import compute_metrics
from src.landmarks import FaceLandmarker


def load_intervals(path: str) -> List[Tuple[float, float]]:
    with open(path, newline="") as f:
        return [(float(r["start"]), float(r["end"])) for r in csv.DictReader(f)]


def in_intervals(t: float, intervals: Sequence[Tuple[float, float]]) -> bool:
    return any(s <= t <= e for s, e in intervals)


def confusion_metrics(y_true: Sequence[bool], y_pred: Sequence[bool]) -> Dict[str, float]:
    tp = sum(1 for a, b in zip(y_true, y_pred) if a and b)
    fp = sum(1 for a, b in zip(y_true, y_pred) if not a and b)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a and not b)
    tn = sum(1 for a, b in zip(y_true, y_pred) if not a and not b)
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision, "recall": recall, "f1": f1,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--labels", default=None)
    p.add_argument("--calib-seconds", type=float, default=3.0,
                   help="seconds at the start used for calibration (0 = skip)")
    p.add_argument("--out", default="evaluation/outputs")
    args = p.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    cfg = Config(calibration_seconds=max(args.calib_seconds, 0.1))
    detector = DrowsinessDetector(cfg)
    landmarker = FaceLandmarker()
    calibrator = Calibrator(cfg.calibration_seconds, min_samples=5) if args.calib_seconds > 0 else None
    intervals = load_intervals(args.labels) if args.labels else None

    rows, y_true, y_pred = [], [], []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        idx += 1
        h, w = frame.shape[:2]
        coords = landmarker.process(frame)
        metrics = compute_metrics(coords, w, h) if coords is not None else None

        if calibrator is not None:
            if metrics is not None:
                calibrator.add(t, metrics.ear, metrics.pitch)
            if calibrator.done:
                detector.apply_calibration(*calibrator.result())
                calibrator = None
            continue  # calibration frames are not scored

        res = detector.update(metrics, t)
        truth = in_intervals(t, intervals) if intervals is not None else None
        rows.append([f"{t:.3f}", f"{res.ear:.3f}", f"{res.mar:.3f}", f"{res.pitch_dev:.1f}",
                     int(res.drowsy), "" if truth is None else int(truth)])
        if truth is not None:
            y_true.append(truth)
            y_pred.append(res.drowsy)

    cap.release()
    landmarker.close()

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.video))[0]
    csv_path = os.path.join(args.out, f"{stem}_frames.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "ear", "mar", "pitch_dev", "pred_drowsy", "true_drowsy"])
        w.writerows(rows)
    print(f"Wrote {csv_path} ({len(rows)} scored frames, EAR threshold {cfg.ear_threshold:.3f})")

    if intervals is not None:
        m = confusion_metrics(y_true, y_pred)
        json_path = os.path.join(args.out, f"{stem}_metrics.json")
        with open(json_path, "w") as f:
            json.dump(m, f, indent=2)
        print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
