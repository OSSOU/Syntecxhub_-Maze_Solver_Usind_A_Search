"""
gesture_engine.py
─────────────────
Core gesture classification engine using MediaPipe hand landmarks.
Classifies 10 gestures and maps them to media-control actions.
Can be imported by the demo script or used standalone.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Optional
import mediapipe as mp

# ─── MediaPipe landmark indices ────────────────────────────────────────────────
# Wrist = 0
# Thumb:  CMC=1, MCP=2, IP=3, TIP=4
# Index:  MCP=5, PIP=6, DIP=7, TIP=8
# Middle: MCP=9, PIP=10, DIP=11, TIP=12
# Ring:   MCP=13, PIP=14, DIP=15, TIP=16
# Pinky:  MCP=17, PIP=18, DIP=19, TIP=20

WRIST         = 0
THUMB_CMC     = 1;  THUMB_MCP   = 2;  THUMB_IP    = 3;  THUMB_TIP   = 4
INDEX_MCP     = 5;  INDEX_PIP   = 6;  INDEX_DIP   = 7;  INDEX_TIP   = 8
MIDDLE_MCP    = 9;  MIDDLE_PIP  = 10; MIDDLE_DIP  = 11; MIDDLE_TIP  = 12
RING_MCP      = 13; RING_PIP    = 14; RING_DIP    = 15; RING_TIP    = 16
PINKY_MCP     = 17; PINKY_PIP   = 18; PINKY_DIP   = 19; PINKY_TIP   = 20

# ─── Gesture definitions ───────────────────────────────────────────────────────

@dataclass
class GestureResult:
    name: str
    confidence: float          # 0.0 – 1.0
    action: str
    action_icon: str
    description: str
    color: tuple               # BGR for OpenCV rendering

@dataclass
class HistoryEntry:
    gesture: str
    action: str
    timestamp: float

# ─── Helper geometry ──────────────────────────────────────────────────────────

def _lm(landmarks, idx):
    """Return (x, y, z) for a landmark index."""
    lm = landmarks[idx]
    return lm.x, lm.y, lm.z

def _dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def _tip_above_pip(landmarks, tip_idx, pip_idx):
    """True when fingertip is ABOVE its PIP joint (finger extended, y decreases upward)."""
    tip_y = landmarks[tip_idx].y
    pip_y = landmarks[pip_idx].y
    return tip_y < pip_y  # lower y-value = higher on screen

def _tip_below_mcp(landmarks, tip_idx, mcp_idx, threshold=0.0):
    """True when fingertip is BELOW its MCP (finger curled)."""
    tip_y = landmarks[tip_idx].y
    mcp_y = landmarks[mcp_idx].y
    return tip_y > mcp_y + threshold

def _finger_extended(landmarks, tip, pip, mcp):
    """Finger is extended when tip is clearly above MCP."""
    tip_y  = landmarks[tip].y
    mcp_y  = landmarks[mcp].y
    pip_y  = landmarks[pip].y
    return tip_y < pip_y and tip_y < mcp_y  # both conditions for robustness

def _thumb_extended(landmarks, handedness="Right"):
    """Thumb extended check (horizontal direction, handedness-aware)."""
    tip_x  = landmarks[THUMB_TIP].x
    ip_x   = landmarks[THUMB_IP].x
    mcp_x  = landmarks[THUMB_MCP].x
    if handedness == "Right":
        return tip_x < ip_x < mcp_x   # tip to the left for right hand
    else:
        return tip_x > ip_x > mcp_x   # tip to the right for left hand

def _thumb_up(landmarks, handedness="Right"):
    """Thumb pointing upward."""
    tip_y = landmarks[THUMB_TIP].y
    ip_y  = landmarks[THUMB_IP].y
    mcp_y = landmarks[THUMB_MCP].y
    return tip_y < ip_y < mcp_y

def _thumb_down(landmarks, handedness="Right"):
    """Thumb pointing downward."""
    tip_y = landmarks[THUMB_TIP].y
    ip_y  = landmarks[THUMB_IP].y
    mcp_y = landmarks[THUMB_MCP].y
    return tip_y > ip_y > mcp_y

def _fingers_curled(landmarks, fingers=None):
    """Check if specified fingers (list of tip/pip/mcp tuples) are curled."""
    if fingers is None:
        fingers = [
            (INDEX_TIP,  INDEX_PIP,  INDEX_MCP),
            (MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
            (RING_TIP,   RING_PIP,   RING_MCP),
            (PINKY_TIP,  PINKY_PIP,  PINKY_MCP),
        ]
    return all(_tip_below_mcp(landmarks, t, m) for t, p, m in fingers)

def _fingers_spread(landmarks):
    """Rough check: tips are far from each other (open palm)."""
    tips = [landmarks[t] for t in [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]]
    dists = []
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            a = (tips[i].x, tips[i].y)
            b = (tips[j].x, tips[j].y)
            dists.append(math.dist(a, b))
    return sum(dists) / len(dists)

def _count_extended(landmarks):
    """Count how many of the four fingers (not thumb) are extended."""
    tests = [
        _finger_extended(landmarks, INDEX_TIP,  INDEX_PIP,  INDEX_MCP),
        _finger_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
        _finger_extended(landmarks, RING_TIP,   RING_PIP,   RING_MCP),
        _finger_extended(landmarks, PINKY_TIP,  PINKY_PIP,  PINKY_MCP),
    ]
    return sum(tests)

# ─── Gesture Classifier ────────────────────────────────────────────────────────

GESTURES = {
    "thumbs_up": GestureResult(
        name="Thumbs Up",
        confidence=0.0,
        action="PLAY / RESUME",
        action_icon="▶",
        description="Play or resume media",
        color=(0, 220, 80),
    ),
    "thumbs_down": GestureResult(
        name="Thumbs Down",
        confidence=0.0,
        action="PAUSE",
        action_icon="⏸",
        description="Pause media",
        color=(0, 80, 220),
    ),
    "open_palm": GestureResult(
        name="Open Palm",
        confidence=0.0,
        action="STOP",
        action_icon="⏹",
        description="Stop playback",
        color=(0, 200, 220),
    ),
    "fist": GestureResult(
        name="Fist",
        confidence=0.0,
        action="MUTE",
        action_icon="🔇",
        description="Toggle mute",
        color=(40, 40, 180),
    ),
    "peace": GestureResult(
        name="Peace / V-Sign",
        confidence=0.0,
        action="NEXT TRACK",
        action_icon="⏭",
        description="Skip to next track",
        color=(180, 180, 0),
    ),
    "ok": GestureResult(
        name="OK Sign",
        confidence=0.0,
        action="CONFIRM / SELECT",
        action_icon="✓",
        description="Confirm selection",
        color=(0, 180, 100),
    ),
    "pointing": GestureResult(
        name="Pointing",
        confidence=0.0,
        action="CURSOR CLICK",
        action_icon="🖱",
        description="Click / select item",
        color=(200, 100, 0),
    ),
    "rock": GestureResult(
        name="Rock On  🤘",
        confidence=0.0,
        action="SHUFFLE",
        action_icon="🔀",
        description="Toggle shuffle mode",
        color=(150, 0, 180),
    ),
    "three_fingers": GestureResult(
        name="Three Fingers",
        confidence=0.0,
        action="VOLUME UP",
        action_icon="🔊",
        description="Increase volume",
        color=(0, 160, 220),
    ),
    "pinch": GestureResult(
        name="Pinch",
        confidence=0.0,
        action="VOLUME DOWN",
        action_icon="🔉",
        description="Decrease volume",
        color=(220, 120, 0),
    ),
    "unknown": GestureResult(
        name="Unknown",
        confidence=0.0,
        action="—",
        action_icon="?",
        description="No gesture recognised",
        color=(60, 60, 60),
    ),
}


def classify(landmarks, handedness: str = "Right") -> GestureResult:
    """
    Classify hand landmarks into a gesture.
    Returns a copy of the matching GestureResult with confidence filled in.
    """
    import copy

    lm = landmarks  # shorthand

    # Boolean finger states
    idx_ext  = _finger_extended(lm, INDEX_TIP,  INDEX_PIP,  INDEX_MCP)
    mid_ext  = _finger_extended(lm, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP)
    ring_ext = _finger_extended(lm, RING_TIP,   RING_PIP,   RING_MCP)
    pin_ext  = _finger_extended(lm, PINKY_TIP,  PINKY_PIP,  PINKY_MCP)
    thumb_up   = _thumb_up(lm, handedness)
    thumb_down = _thumb_down(lm, handedness)
    thumb_ext  = _thumb_extended(lm, handedness)

    n_extended = sum([idx_ext, mid_ext, ring_ext, pin_ext])

    # ── 1. FIST ───────────────────────────────────────────────────────────────
    if n_extended == 0 and not thumb_up and not thumb_down:
        # All fingers curled, thumb roughly neutral
        res = copy.copy(GESTURES["fist"])
        res.confidence = 0.92
        return res

    # ── 2. THUMBS UP ─────────────────────────────────────────────────────────
    if thumb_up and n_extended == 0:
        res = copy.copy(GESTURES["thumbs_up"])
        res.confidence = 0.95
        return res

    # ── 3. THUMBS DOWN ───────────────────────────────────────────────────────
    if thumb_down and n_extended == 0:
        res = copy.copy(GESTURES["thumbs_down"])
        res.confidence = 0.93
        return res

    # ── 4. OPEN PALM ─────────────────────────────────────────────────────────
    if n_extended == 4 and idx_ext and mid_ext and ring_ext and pin_ext:
        spread = _fingers_spread(lm)
        conf = min(0.95, 0.75 + spread * 2)
        res = copy.copy(GESTURES["open_palm"])
        res.confidence = conf
        return res

    # ── 5. PEACE / V-SIGN ────────────────────────────────────────────────────
    if idx_ext and mid_ext and not ring_ext and not pin_ext:
        # Extra: check the two tips are spread
        spread = math.dist(
            (lm[INDEX_TIP].x,  lm[INDEX_TIP].y),
            (lm[MIDDLE_TIP].x, lm[MIDDLE_TIP].y)
        )
        if spread > 0.03:
            res = copy.copy(GESTURES["peace"])
            res.confidence = 0.91
            return res

    # ── 6. POINTING ──────────────────────────────────────────────────────────
    if idx_ext and not mid_ext and not ring_ext and not pin_ext:
        res = copy.copy(GESTURES["pointing"])
        res.confidence = 0.93
        return res

    # ── 7. THREE FINGERS ─────────────────────────────────────────────────────
    if idx_ext and mid_ext and ring_ext and not pin_ext:
        res = copy.copy(GESTURES["three_fingers"])
        res.confidence = 0.90
        return res

    # ── 8. ROCK ON 🤘 ────────────────────────────────────────────────────────
    if idx_ext and not mid_ext and not ring_ext and pin_ext:
        res = copy.copy(GESTURES["rock"])
        res.confidence = 0.90
        return res

    # ── 9. OK SIGN ───────────────────────────────────────────────────────────
    # Index and thumb tips close together, other fingers extended
    thumb_tip = (lm[THUMB_TIP].x, lm[THUMB_TIP].y)
    index_tip = (lm[INDEX_TIP].x, lm[INDEX_TIP].y)
    pinch_dist = math.dist(thumb_tip, index_tip)
    if pinch_dist < 0.06 and mid_ext and ring_ext and pin_ext:
        res = copy.copy(GESTURES["ok"])
        res.confidence = 0.88
        return res

    # ── 10. PINCH ────────────────────────────────────────────────────────────
    if pinch_dist < 0.06 and not mid_ext and not ring_ext and not pin_ext:
        res = copy.copy(GESTURES["pinch"])
        res.confidence = 0.85
        return res

    # ── UNKNOWN ──────────────────────────────────────────────────────────────
    res = copy.copy(GESTURES["unknown"])
    res.confidence = 0.0
    return res


# ─── Smoothing buffer ─────────────────────────────────────────────────────────

class GestureBuffer:
    """
    Temporal smoothing: only commit a gesture if it appears consistently
    across a sliding window of frames.
    """

    def __init__(self, window: int = 8, threshold: float = 0.6):
        self.window    = window
        self.threshold = threshold
        self._buf: list[str] = []
        self.current: Optional[GestureResult] = None
        self.history: list[HistoryEntry] = []

    def update(self, result: GestureResult) -> GestureResult:
        self._buf.append(result.name)
        if len(self._buf) > self.window:
            self._buf.pop(0)

        if not self._buf:
            return result

        # Vote
        from collections import Counter
        counts = Counter(self._buf)
        best, freq = counts.most_common(1)[0]
        ratio = freq / len(self._buf)

        if ratio >= self.threshold:
            if self.current is None or self.current.name != best:
                # Gesture changed — log it
                self.current = result if result.name == best else GESTURES.get(best.lower().replace(" ", "_").replace("/", "").replace("🤘","").strip(), GESTURES["unknown"])
                entry = HistoryEntry(
                    gesture=self.current.name,
                    action=self.current.action,
                    timestamp=time.time(),
                )
                self.history.append(entry)
                if len(self.history) > 50:
                    self.history.pop(0)
            return self.current or result

        return self.current or result
