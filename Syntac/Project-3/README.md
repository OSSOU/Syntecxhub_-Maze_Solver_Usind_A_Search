# AXIOM — Hand Gesture Recognition
### Real-time gesture detection with MediaPipe · Python 3.8+

```
gesture_recognition/
├── gesture_engine.py      # Core classifier (no camera needed — import freely)
├── gesture_demo.py        # Webcam demo with full OpenCV UI
├── test_gesture_engine.py # Unit tests (camera-free)
└── README.md
```

---

## Quick Start

```bash
# Install dependencies
pip install mediapipe opencv-python numpy

# Run demo (default webcam)
python gesture_demo.py

# Alternate camera, save action log
python gesture_demo.py --camera 1 --save-log

# Run unit tests (no camera required)
python test_gesture_engine.py
# or with pytest:
pytest test_gesture_engine.py -v
```

---

## Gesture → Action Map

| Gesture | Hand Shape | Mapped Action |
|---|---|---|
| 👍 Thumbs Up | Thumb up, all fingers curled | ▶ **Play / Resume** |
| 👎 Thumbs Down | Thumb down, all fingers curled | ⏸ **Pause** |
| ✊ Fist | All fingers curled | 🔇 **Mute** |
| 🖐 Open Palm | All 4 fingers extended, spread | ⏹ **Stop** |
| ✌️ Peace / V-Sign | Index + middle extended | ⏭ **Next Track** |
| ☝️ Pointing | Only index finger extended | 🖱 **Cursor Click** |
| 🤟 Rock On | Index + pinky extended | 🔀 **Shuffle** |
| 🤙 Three Fingers | Index + middle + ring extended | 🔊 **Volume Up** |
| 👌 OK Sign | Thumb + index tips touching, others open | ✓ **Confirm / Select** |
| 🤏 Pinch | Thumb + index tips touching, others curled | 🔉 **Volume Down** |

---

## Architecture

### `gesture_engine.py` — Classifier Core

The engine is **camera-agnostic**: it accepts any 21-element MediaPipe landmark list and
returns a `GestureResult`. Import it into any project.

```python
from gesture_engine import classify, GestureBuffer

# Single-frame classification
result = classify(hand_landmarks, handedness="Right")
print(result.name, result.action, result.confidence)

# Temporal smoothing (recommended for real-time use)
buf = GestureBuffer(window=10, threshold=0.6)
smoothed = buf.update(result)
```

#### Classification Pipeline

```
MediaPipe 21 landmarks
        │
        ▼
  Boolean finger states
  ┌─────────────────────────────────┐
  │  tip_above_pip()  → extended?   │
  │  tip_below_mcp()  → curled?     │
  │  thumb_up/down()  → direction   │
  │  dist(tip,tip)    → pinch?      │
  └─────────────────────────────────┘
        │
  Priority rule chain (10 gestures)
        │
        ▼
  GestureResult { name, confidence, action, color }
        │
        ▼
  GestureBuffer (sliding window vote)
        │
        ▼
  Stable committed gesture + history log
```

#### Landmark Indices (MediaPipe)

```
        8   12  16  20      ← fingertips
        |    |   |   |
        7   11  15  19
        |    |   |   |
        6   10  14  18
        |    |   |   |
  4     5    9  13  17
  |     └────┴───┴───┘
  3          │
  2         WRIST (0)
  1
THUMB
```

### `GestureBuffer` — Temporal Smoothing

Raw per-frame classification is noisy. `GestureBuffer` uses a **sliding window majority vote**:
a gesture is only committed when it appears in ≥ `threshold` fraction of the last `window` frames.

```python
buf = GestureBuffer(
    window=10,      # frames to look back
    threshold=0.6   # 60% consensus required
)
```

This eliminates flickering between similar gestures (e.g., Pointing ↔ Peace).

### `gesture_demo.py` — OpenCV UI

The demo renders a full HUD over the webcam feed:

| Panel | Content |
|---|---|
| Top bar | App name · FPS counter · camera info |
| Gesture card | Current gesture · confidence bar · mapped action |
| Action flash | Centre-screen overlay when gesture fires |
| Gesture reference | All 10 gestures and their actions |
| Gesture log | Last 10 gesture events with timestamps |

---

## Keyboard Controls

| Key | Action |
|---|---|
| `Q` / `ESC` | Quit |
| `S` | Save screenshot as PNG |
| `H` | Toggle help overlay |
| `L` | Toggle gesture log panel |
| `M` | Toggle mirror mode |
| `+` / `-` | Adjust minimum detection confidence |

---

## CLI Options

```
python gesture_demo.py [OPTIONS]

Options:
  --camera INT       Camera device index (default: 0)
  --no-mirror        Disable horizontal flip
  --confidence FLOAT Min MediaPipe detection confidence 0.0–1.0 (default: 0.70)
  --save-log         Export action history as JSON when you quit
  --help             Show help
```

---

## Extending the System

### Add a new gesture

1. Add a rule in `gesture_engine.py` inside `classify()`:

```python
# Example: "Spock" — middle+ring together, index+pinky spread
if idx_ext and not mid_ext and not ring_ext and pin_ext:
    # already "Rock On" — add a spread check here
    ...
```

2. Add the gesture template to the `GESTURES` dict:

```python
"spock": GestureResult(
    name="Spock",
    confidence=0.0,
    action="LIVE LONG AND PROSPER",
    action_icon="🖖",
    description="Vulcan salute",
    color=(0, 200, 180),
),
```

3. Add a test pose in `test_gesture_engine.py`.

### Map to real system actions

Replace the print statements in `gesture_demo.py` with actual OS calls:

```python
import subprocess

ACTION_MAP = {
    "PLAY / RESUME": lambda: subprocess.run(["playerctl", "play"]),
    "PAUSE":         lambda: subprocess.run(["playerctl", "pause"]),
    "VOLUME UP":     lambda: subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", "+5%"]),
    "VOLUME DOWN":   lambda: subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", "-5%"]),
    "MUTE":          lambda: subprocess.run(["pactl", "set-sink-mute",   "@DEFAULT_SINK@", "toggle"]),
}
```

---

## Requirements

```
mediapipe>=0.10
opencv-python>=4.8
numpy>=1.24
```

Python 3.8 – 3.12 supported.

---

## How MediaPipe Works

MediaPipe Hands runs a **two-stage pipeline**:

1. **Palm detector** — fast, lightweight network that finds hand bounding boxes in the full frame.
2. **Hand landmark model** — takes each cropped hand region and regresses 21 3D landmarks (x, y, z)
   with sub-millimetre accuracy at 30+ FPS on CPU.

We use only the normalized (x, y) screen coordinates for gesture classification. The z coordinate
(depth) is available but not required for the gestures implemented here.

---

*Built with MediaPipe · OpenCV · Python*
