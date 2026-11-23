import cv2
import numpy as np
import pytest

from evaluation.evaluate_video import confusion_metrics, in_intervals
from src.detector import Calibrator, Config, DrowsinessDetector
from src.features import (
    HEAD_POSE_IDX, LEFT_EYE, MOUTH_CORNERS, MOUTH_VERTICAL_PAIRS, _MODEL_POINTS,
    FrameMetrics, estimate_pitch, eye_aspect_ratio, mouth_aspect_ratio,
)


def make_eye(opening: float) -> np.ndarray:
    """Eye 30 px wide; `opening` is the vertical lid gap in px."""
    return np.array([
        (0, 0), (10, -opening / 2), (20, -opening / 2),
        (30, 0), (20, opening / 2), (10, opening / 2),
    ], dtype=float)


def test_ear_open_vs_closed():
    assert eye_aspect_ratio(make_eye(10)) == pytest.approx(10 / 30)
    assert eye_aspect_ratio(make_eye(1)) < 0.05


def test_ear_degenerate_eye_returns_zero():
    assert eye_aspect_ratio(np.zeros((6, 2))) == 0.0


def make_face(mouth_opening: float) -> np.ndarray:
    coords = np.zeros((478, 2))
    a, b = MOUTH_CORNERS
    coords[a], coords[b] = (0, 0), (40, 0)
    for k, (u, l) in enumerate(MOUTH_VERTICAL_PAIRS):
        x = 10 + 10 * k
        coords[u], coords[l] = (x, -mouth_opening / 2), (x, mouth_opening / 2)
    return coords


def test_mar_closed_vs_yawn():
    assert mouth_aspect_ratio(make_face(0)) == pytest.approx(0.0)
    assert mouth_aspect_ratio(make_face(30)) == pytest.approx(30 / 40)
    assert mouth_aspect_ratio(make_face(30)) > 0.6 > mouth_aspect_ratio(make_face(4))


@pytest.mark.parametrize("angle", [0, 15, 25, -20])
def test_pitch_recovered_from_synthetic_projection(angle):
    w, h = 640, 480
    camera = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=float)
    rvec = np.array([[np.radians(angle)], [0.0], [0.0]])
    tvec = np.array([[0.0], [0.0], [1500.0]])
    pts, _ = cv2.projectPoints(_MODEL_POINTS, rvec, tvec, camera, np.zeros((4, 1)))
    coords = np.zeros((478, 2))
    for i, p in zip(HEAD_POSE_IDX, pts.reshape(-1, 2)):
        coords[i] = p
    assert abs(estimate_pitch(coords, w, h)) == pytest.approx(abs(angle), abs=2.0)


# --- detector state machine --------------------------------------------------
def feed(det, seconds, ear=0.3, mar=0.1, pitch=0.0, start=0.0, fps=10):
    """Feed constant metrics; return (list of results, next timestamp)."""
    results, t = [], start
    for _ in range(int(seconds * fps)):
        results.append(det.update(FrameMetrics(ear, mar, pitch), t))
        t += 1 / fps
    return results, t


def test_normal_driving_never_alerts():
    det = DrowsinessDetector()
    res, _ = feed(det, 10)
    assert not any(r.drowsy for r in res)


def test_blink_does_not_alert_but_long_closure_does():
    det = DrowsinessDetector()
    res, t = feed(det, 0.3, ear=0.05)           # a blink
    assert not any(r.drowsy for r in res)
    res, t = feed(det, 0.5, ear=0.3, start=t)   # eyes reopen
    res, t = feed(det, 2.0, ear=0.05, start=t)  # sustained closure
    assert any(r.eyes_closed for r in res)
    assert [r.event for r in res if r.event] == ["DROWSY_START"]


def test_alert_ends_after_recovery():
    det = DrowsinessDetector()
    res, t = feed(det, 2.0, ear=0.05)
    res2, _ = feed(det, 2.0, ear=0.3, start=t)
    events = [r.event for r in res + res2 if r.event]
    assert events == ["DROWSY_START", "DROWSY_END"]
    assert not res2[-1].drowsy


def test_yawn_needs_duration():
    det = DrowsinessDetector()
    res, t = feed(det, 0.4, mar=0.9)
    assert not any(r.yawning for r in res)
    res, _ = feed(det, 1.5, mar=0.9, start=t)
    assert any(r.yawning for r in res)


def test_head_nod_uses_baseline():
    det = DrowsinessDetector()
    det.apply_calibration(open_ear=0.3, baseline_pitch=10.0)
    res, t = feed(det, 2.0, pitch=12.0)  # 2 deg from baseline: fine
    assert not any(r.nodding for r in res)
    res, _ = feed(det, 2.0, pitch=35.0, start=t)  # 25 deg from baseline
    assert any(r.nodding for r in res)


def test_no_face_resets_and_recovers():
    det = DrowsinessDetector()
    feed(det, 2.0, ear=0.05)
    assert det.alert_active
    t = 2.0
    for _ in range(10):
        res = det.update(None, t)
        t += 0.1
    assert not res.face_found and not res.drowsy


def test_calibration_sets_threshold_and_clamps():
    det = DrowsinessDetector()
    det.apply_calibration(open_ear=0.32, baseline_pitch=3.0)
    assert det.cfg.ear_threshold == pytest.approx(0.24)
    det.apply_calibration(open_ear=0.10, baseline_pitch=0.0)
    assert det.cfg.ear_threshold == det.cfg.ear_threshold_min
    det.apply_calibration(open_ear=0.60, baseline_pitch=0.0)
    assert det.cfg.ear_threshold == det.cfg.ear_threshold_max


def test_calibrator_progress_and_median():
    cal = Calibrator(seconds=2.0, min_samples=5)
    t = 0.0
    for ear in [0.3, 0.31, 0.29, 0.05, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]:  # one blink
        cal.add(t, ear, 5.0)
        t += 0.25
    assert cal.done
    open_ear, pitch = cal.result()
    assert open_ear == pytest.approx(0.3)
    assert pitch == 5.0


def test_perclos():
    det = DrowsinessDetector(Config(eye_closed_seconds=100))  # never alert here
    res, t = feed(det, 5.0, ear=0.3)
    res, _ = feed(det, 5.0, ear=0.05, start=t)
    assert res[-1].perclos == pytest.approx(0.5, abs=0.02)


# --- evaluation helpers -------------------------------------------------------
def test_confusion_metrics():
    m = confusion_metrics([1, 1, 0, 0, 1], [1, 0, 0, 1, 1])
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 1)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(3 / 5)


def test_in_intervals():
    assert in_intervals(2.0, [(1.0, 3.0)])
    assert not in_intervals(4.0, [(1.0, 3.0)])
