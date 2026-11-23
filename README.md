# Driver Drowsiness Detection

Real-time driver drowsiness detection from a webcam using MediaPipe Face Mesh.
The system watches three cues, raises an audio alert, logs every event, and can
drive an optional Arduino output (LED / buzzer / relay prototype).

| Cue | Measure | Triggers when |
|---|---|---|
| Eyes closing | Eye Aspect Ratio (EAR) | EAR stays below a calibrated threshold for ≥ 1.0 s |
| Yawning | Mouth Aspect Ratio (MAR) | MAR stays above 0.6 for ≥ 1.0 s |
| Head nodding | Head pitch (solvePnP) | Pitch deviates > 15° from the calibrated baseline for ≥ 0.7 s |

PERCLOS (share of recent frames with eyes closed, 60 s window) is also shown as a
live metric.

> **Status:** prototype for academic use. It is not a certified safety system and
> must not be relied on to control a real vehicle.

<!-- Add a demo GIF here: ![demo](docs/demo.gif) -->

## How it works

```
Webcam → MediaPipe Face Mesh (478 landmarks) → features.py (EAR, MAR, head pitch)
       → detector.py (calibration + time-based decision logic)
       → alarm.py (voice/beep) · logger.py (CSV) · arduino.py (serial) · overlay.py (UI)
```

- **EAR** = (‖p2−p6‖ + ‖p3−p5‖) / (2‖p1−p4‖), averaged over both eyes
  (Soukupová & Čech, 2016).
- **MAR** = mean of three inner-lip openings (landmark pairs 82–87, 13–14, 312–317)
  divided by mouth width (landmarks 78–308).
- **Head pitch** is solved with `cv2.solvePnP` from six landmarks (nose tip, chin,
  eye corners, mouth corners) against a generic 3D face model, and compared with a
  baseline measured during calibration.
- **Calibration** (first 5 s, driver looks ahead with eyes open): the EAR threshold
  is set to 75 % of the driver's median open-eye EAR (clamped to 0.15–0.30) and the
  neutral head pitch is recorded. Press `c` any time to recalibrate.
- All durations are **time-based**, so behaviour does not depend on camera FPS.

## Project structure

```
├── src/
│   ├── features.py     # EAR, MAR, head pitch (pure NumPy/OpenCV, unit-tested)
│   ├── detector.py     # Config, Calibrator, DrowsinessDetector state machine
│   ├── landmarks.py    # MediaPipe Face Mesh wrapper
│   ├── alarm.py        # threaded voice/beep alarm
│   ├── logger.py       # CSV event log
│   ├── arduino.py      # optional serial output
│   ├── overlay.py      # on-screen drawing helpers
│   └── main.py         # desktop OpenCV app
├── app.py              # Streamlit dashboard
├── arduino/alert.ino   # sketch for the optional hardware output
├── evaluation/         # evaluate_video.py: per-frame CSV + precision/recall/F1
├── tests/              # pytest unit tests
├── docs/               # report, screenshots, demo GIF
├── logs/               # events.csv is written here (git-ignored)
└── requirements.txt
```

## Setup (Windows / VS Code)

Requires **Python 3.10–3.12** (MediaPipe 0.10.21 does not support 3.13).

```powershell
git clone <your-repo-url>
cd driver-drowsiness-detection
python -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass     # only if activation is blocked
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Then in VS Code: **Ctrl+Shift+P → Python: Select Interpreter → `.venv`**.
Keep the folder outside OneDrive to avoid file-locking problems.
On macOS/Linux use `source .venv/bin/activate`.

> `requirements.txt` pins `mediapipe==0.10.21` on purpose: newer MediaPipe releases
> no longer include the `mp.solutions.face_mesh` API this project uses. It also
> installs `opencv-contrib-python` rather than `opencv-python`, because MediaPipe
> bundles it and having both causes `cv2` conflicts.

## Run

Desktop app (run from the repository root):

```powershell
python -m src.main                       # webcam, with calibration
python -m src.main --video clip.mp4      # video file
python -m src.main --no-sound            # silent
python -m src.main --arduino-port COM3   # with Arduino output
```

Keys: `q` quit · `c` recalibrate. Events are appended to `logs/events.csv`.

Streamlit dashboard (reads the webcam of the machine running it):

```powershell
streamlit run app.py
```

## Optional Arduino output

Upload `arduino/alert.ino`, connect an LED/buzzer/relay driver to pin 8, and start
the app with `--arduino-port <PORT>`. The app sends `1` when drowsiness starts and
`0` when it ends. If no board is found the app runs in software-only mode.

## Tests and evaluation

```powershell
python -m pip install pytest
python -m pytest
```

The unit tests cover EAR/MAR geometry, head-pitch recovery from a synthetic
projection, calibration, and the alert state machine.

To measure accuracy on recorded clips, start each clip with a few seconds of normal
alert driving (used for calibration) and list the truly drowsy periods in a CSV:

```csv
start,end
12.0,18.5
40.2,47.0
```

```powershell
python -m evaluation.evaluate_video clip.mp4 --labels clip_labels.csv
```

This writes `evaluation/outputs/<clip>_frames.csv` (per-frame EAR/MAR/pitch and
predictions) and `<clip>_metrics.json` (TP/FP/FN/TN, accuracy, precision, recall, F1).

### Results

_To be filled in after testing on your own recordings and/or a public dataset._

| Configuration | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| EAR only | | | | |
| EAR + MAR | | | | |
| EAR + MAR + head nod | | | | |

## Tuning

Defaults live in `Config` (`src/detector.py`). The MAR (0.6) and head-nod (15°)
thresholds are starting points. Check them against your own footage and adjust.
The Streamlit sidebar exposes the main ones for quick experiments.

## Limitations

- Accuracy drops with strong glare on glasses, sunglasses, very low light, or a
  face turned far from the camera.
- Talking or laughing can raise MAR; the 1 s duration requirement reduces but does
  not eliminate false yawn alerts.
- Head pose uses a generic face model and an approximate camera, so pitch is
  relative (deviation from baseline), not an absolute angle.
- A visible-light webcam cannot see eyes at night; an infrared camera would be needed.
- The Streamlit app and the voice alarm use local hardware (webcam, speakers), so
  they run locally rather than on a hosted server.

## References

- T. Soukupová and J. Čech, "Real-Time Eye Blink Detection using Facial Landmarks,"
  21st Computer Vision Winter Workshop, 2016.
- MediaPipe Face Mesh: https://developers.google.com/mediapipe

## Author

_Your name · your college · academic year_
