"""Desktop app: webcam (or video file) -> detector -> alarm / log / Arduino.

Run from the repository root:
    python -m src.main
    python -m src.main --video sample.mp4 --no-sound
Keys: q = quit, c = recalibrate.
"""
from __future__ import annotations

import argparse
import time

import cv2

from .alarm import Alarm
from .arduino import AlertOutput
from .detector import Calibrator, Config, DrowsinessDetector
from .features import compute_metrics
from .landmarks import FaceLandmarker
from .logger import EventLogger
from .overlay import draw_calibration, draw_status


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Driver drowsiness detection")
    p.add_argument("--camera", type=int, default=0, help="webcam index")
    p.add_argument("--video", type=str, default=None, help="video file instead of webcam")
    p.add_argument("--no-calibrate", action="store_true", help="skip calibration")
    p.add_argument("--no-sound", action="store_true", help="disable the audio alarm")
    p.add_argument("--arduino-port", type=str, default=None, help="e.g. COM3")
    p.add_argument("--log", type=str, default="logs/events.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()
    detector = DrowsinessDetector(cfg)
    landmarker = FaceLandmarker()
    logger = EventLogger(args.log)
    alarm = Alarm(enabled=not args.no_sound)
    output = AlertOutput(args.arduino_port)

    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera/video source.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    calibrator = None if args.no_calibrate else Calibrator(cfg.calibration_seconds)
    frame_idx = 0
    print("[INFO] Running. Press 'q' to quit, 'c' to recalibrate.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if not args.video:
                frame = cv2.flip(frame, 1)  # mirror view for webcam
            # Video files use their own clock so results are reproducible.
            t = frame_idx / fps if args.video else time.monotonic()
            frame_idx += 1

            h, w = frame.shape[:2]
            coords = landmarker.process(frame)
            metrics = compute_metrics(coords, w, h) if coords is not None else None

            if calibrator is not None:
                if metrics is not None:
                    calibrator.add(t, metrics.ear, metrics.pitch)
                draw_calibration(frame, calibrator.progress, metrics is not None)
                if calibrator.done:
                    detector.apply_calibration(*calibrator.result())
                    calibrator = None
                    print(f"[INFO] Calibrated. EAR threshold = {cfg.ear_threshold:.3f}")
            else:
                res = detector.update(metrics, t)
                if res.event == "DROWSY_START":
                    logger.log("DROWSY_START", "+".join(res.reasons), res.ear, res.mar, res.pitch_dev)
                    alarm.start()
                    output.on()
                elif res.event == "DROWSY_END":
                    logger.log("DROWSY_END", "", res.ear, res.mar, res.pitch_dev)
                    alarm.stop()
                    output.off()
                draw_status(frame, res, cfg.ear_threshold)

            cv2.imshow("Driver Drowsiness Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                calibrator = Calibrator(cfg.calibration_seconds)
                alarm.stop()
                output.off()
    finally:
        alarm.stop()
        output.close()
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
