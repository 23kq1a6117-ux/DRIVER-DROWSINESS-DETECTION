"""Streamlit dashboard. Run locally with:  streamlit run app.py

It reads the webcam of the machine running Streamlit (cv2.VideoCapture), so it
is meant for local use. Changing any sidebar setting restarts the session,
including calibration.
"""
import time

import cv2
import pandas as pd
import streamlit as st

from src.alarm import Alarm
from src.detector import Calibrator, Config, DrowsinessDetector
from src.features import compute_metrics
from src.landmarks import FaceLandmarker
from src.logger import EventLogger
from src.overlay import draw_calibration, draw_status

LOG_PATH = "logs/events.csv"

st.set_page_config(page_title="Driver Drowsiness Detection", layout="wide")
st.title("Driver Drowsiness Detection")

with st.sidebar:
    st.header("Settings")
    camera_index = st.number_input("Camera index", min_value=0, value=0, step=1)
    sound = st.checkbox("Sound alarm", value=True)
    eye_seconds = st.slider("Eyes closed for (s)", 0.5, 3.0, 1.0, 0.1)
    mar_threshold = st.slider("Yawn threshold (MAR)", 0.4, 1.0, 0.6, 0.05)
    pitch_threshold = st.slider("Head-nod threshold (deg)", 5, 40, 15, 1)
    st.caption("Changing a setting restarts the session and recalibrates.")

run = st.toggle("Start camera")
video_col, metric_col = st.columns([3, 1])
frame_slot = video_col.empty()
with metric_col:
    ear_slot = st.empty()
    mar_slot = st.empty()
    pitch_slot = st.empty()
    perclos_slot = st.empty()
alert_slot = st.empty()

if run:
    cfg = Config(
        eye_closed_seconds=eye_seconds,
        mar_threshold=mar_threshold,
        pitch_threshold_deg=float(pitch_threshold),
    )
    detector = DrowsinessDetector(cfg)
    calibrator = Calibrator(cfg.calibration_seconds)
    landmarker = FaceLandmarker()
    logger = EventLogger(LOG_PATH)
    alarm = Alarm(enabled=sound)
    cap = cv2.VideoCapture(int(camera_index))
    try:
        if not cap.isOpened():
            st.error("Could not open the camera.")
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                st.error("Camera read failed.")
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            coords = landmarker.process(frame)
            metrics = compute_metrics(coords, w, h) if coords is not None else None
            t = time.monotonic()

            if calibrator is not None:
                if metrics is not None:
                    calibrator.add(t, metrics.ear, metrics.pitch)
                draw_calibration(frame, calibrator.progress, metrics is not None)
                if calibrator.done:
                    detector.apply_calibration(*calibrator.result())
                    calibrator = None
            else:
                res = detector.update(metrics, t)
                if res.event == "DROWSY_START":
                    logger.log("DROWSY_START", "+".join(res.reasons), res.ear, res.mar, res.pitch_dev)
                    alarm.start()
                elif res.event == "DROWSY_END":
                    logger.log("DROWSY_END", "", res.ear, res.mar, res.pitch_dev)
                    alarm.stop()
                draw_status(frame, res, cfg.ear_threshold)
                ear_slot.metric("Eye Aspect Ratio", f"{res.ear:.2f}")
                mar_slot.metric("Mouth Aspect Ratio", f"{res.mar:.2f}")
                pitch_slot.metric("Head pitch deviation", f"{res.pitch_dev:+.0f}°")
                perclos_slot.metric("PERCLOS (60 s)", f"{res.perclos * 100:.0f}%")
                if res.drowsy:
                    alert_slot.error("DROWSINESS ALERT: " + ", ".join(res.reasons))
                else:
                    alert_slot.empty()

            frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
    finally:
        alarm.stop()
        landmarker.close()
        cap.release()

with st.expander("Event log"):
    try:
        st.dataframe(pd.read_csv(LOG_PATH).tail(50), use_container_width=True)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        st.write("No events logged yet.")
