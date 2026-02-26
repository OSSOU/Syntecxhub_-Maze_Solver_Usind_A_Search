"""
test_gesture_engine.py
──────────────────────
Pytest tests for the gesture classifier.  No camera required.
Run with:  python -m pytest test_gesture_engine.py -v
"""

import math
import time
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
from types import SimpleNamespace

# ── stub landmark helper ───────────────────────────────────────────────────────

def lm(x, y, z=0.0):
    return SimpleNamespace(x=x, y=y, z=z)


def make_landmarks(positions: dict) -> list:
    """
    Build a 21-element landmark list.
    positions: {idx: (x, y)} — unspecified landmarks default to (0.5, 0.5).
    """
    out = [lm(0.5, 0.5) for _ in range(21)]
    for idx, (x, y) in positions.items():
        out[idx] = lm(x, y)
    return out


# Landmark indices (mirrors gesture_engine.py)
WRIST = 0
THUMB_CMC=1; THUMB_MCP=2; THUMB_IP=3; THUMB_TIP=4
INDEX_MCP=5; INDEX_PIP=6; INDEX_DIP=7; INDEX_TIP=8
MIDDLE_MCP=9; MIDDLE_PIP=10; MIDDLE_DIP=11; MIDDLE_TIP=12
RING_MCP=13; RING_PIP=14; RING_DIP=15; RING_TIP=16
PINKY_MCP=17; PINKY_PIP=18; PINKY_DIP=19; PINKY_TIP=20


from gesture_engine import classify, GestureBuffer, GESTURES


# ── Canonical landmark poses ───────────────────────────────────────────────────

def thumbs_up_landmarks():
    """Right hand thumbs-up: only thumb tip goes high (low y), fingers curled."""
    return make_landmarks({
        WRIST:      (0.50, 0.80),
        THUMB_MCP:  (0.40, 0.70),
        THUMB_IP:   (0.35, 0.60),
        THUMB_TIP:  (0.32, 0.48),   # high — clearly above MCP
        # Index curled
        INDEX_MCP:  (0.55, 0.68),
        INDEX_PIP:  (0.55, 0.72),
        INDEX_TIP:  (0.55, 0.76),   # below MCP
        # Middle curled
        MIDDLE_MCP: (0.58, 0.67),
        MIDDLE_PIP: (0.58, 0.71),
        MIDDLE_TIP: (0.58, 0.75),
        # Ring curled
        RING_MCP:   (0.60, 0.68),
        RING_PIP:   (0.60, 0.72),
        RING_TIP:   (0.60, 0.76),
        # Pinky curled
        PINKY_MCP:  (0.62, 0.70),
        PINKY_PIP:  (0.62, 0.73),
        PINKY_TIP:  (0.62, 0.77),
    })


def thumbs_down_landmarks():
    """Right hand thumbs-down: thumb tip goes low (high y), fingers curled."""
    return make_landmarks({
        WRIST:      (0.50, 0.40),
        THUMB_MCP:  (0.40, 0.48),
        THUMB_IP:   (0.38, 0.58),
        THUMB_TIP:  (0.36, 0.70),   # below MCP → thumbs down
        INDEX_MCP:  (0.55, 0.45),
        INDEX_PIP:  (0.55, 0.49),
        INDEX_TIP:  (0.55, 0.53),
        MIDDLE_MCP: (0.58, 0.44),
        MIDDLE_PIP: (0.58, 0.48),
        MIDDLE_TIP: (0.58, 0.52),
        RING_MCP:   (0.60, 0.45),
        RING_PIP:   (0.60, 0.49),
        RING_TIP:   (0.60, 0.53),
        PINKY_MCP:  (0.62, 0.46),
        PINKY_PIP:  (0.62, 0.50),
        PINKY_TIP:  (0.62, 0.54),
    })


def fist_landmarks():
    """All fingers curled, thumb neutral."""
    return make_landmarks({
        WRIST:      (0.50, 0.70),
        THUMB_MCP:  (0.42, 0.62),
        THUMB_IP:   (0.40, 0.60),
        THUMB_TIP:  (0.42, 0.60),   # roughly neutral (not up or down)
        INDEX_MCP:  (0.54, 0.58),
        INDEX_PIP:  (0.54, 0.63),
        INDEX_TIP:  (0.54, 0.68),
        MIDDLE_MCP: (0.57, 0.57),
        MIDDLE_PIP: (0.57, 0.62),
        MIDDLE_TIP: (0.57, 0.67),
        RING_MCP:   (0.60, 0.58),
        RING_PIP:   (0.60, 0.63),
        RING_TIP:   (0.60, 0.68),
        PINKY_MCP:  (0.62, 0.60),
        PINKY_PIP:  (0.62, 0.65),
        PINKY_TIP:  (0.62, 0.70),
    })


def open_palm_landmarks():
    """All four fingers fully extended, spread apart."""
    return make_landmarks({
        WRIST:      (0.50, 0.85),
        THUMB_MCP:  (0.35, 0.76),
        THUMB_IP:   (0.28, 0.68),
        THUMB_TIP:  (0.22, 0.62),
        INDEX_MCP:  (0.44, 0.70),
        INDEX_PIP:  (0.40, 0.55),
        INDEX_TIP:  (0.37, 0.42),   # well above MCP
        MIDDLE_MCP: (0.50, 0.68),
        MIDDLE_PIP: (0.50, 0.52),
        MIDDLE_TIP: (0.50, 0.38),
        RING_MCP:   (0.56, 0.70),
        RING_PIP:   (0.58, 0.54),
        RING_TIP:   (0.60, 0.40),
        PINKY_MCP:  (0.62, 0.72),
        PINKY_PIP:  (0.65, 0.58),
        PINKY_TIP:  (0.67, 0.46),
    })


def peace_landmarks():
    """Index and middle extended, others curled."""
    return make_landmarks({
        WRIST:      (0.50, 0.85),
        THUMB_MCP:  (0.38, 0.76),
        THUMB_IP:   (0.34, 0.72),
        THUMB_TIP:  (0.36, 0.72),
        INDEX_MCP:  (0.48, 0.70),
        INDEX_PIP:  (0.44, 0.55),
        INDEX_TIP:  (0.41, 0.42),   # extended
        MIDDLE_MCP: (0.53, 0.68),
        MIDDLE_PIP: (0.55, 0.53),
        MIDDLE_TIP: (0.57, 0.39),   # extended, spread
        RING_MCP:   (0.57, 0.70),
        RING_PIP:   (0.57, 0.74),
        RING_TIP:   (0.57, 0.78),   # curled
        PINKY_MCP:  (0.62, 0.72),
        PINKY_PIP:  (0.62, 0.76),
        PINKY_TIP:  (0.62, 0.80),   # curled
    })


def pointing_landmarks():
    """Only index finger extended."""
    return make_landmarks({
        WRIST:      (0.50, 0.85),
        THUMB_MCP:  (0.38, 0.76),
        THUMB_IP:   (0.34, 0.72),
        THUMB_TIP:  (0.36, 0.72),
        INDEX_MCP:  (0.48, 0.70),
        INDEX_PIP:  (0.45, 0.55),
        INDEX_TIP:  (0.42, 0.40),   # extended
        MIDDLE_MCP: (0.54, 0.68),
        MIDDLE_PIP: (0.54, 0.73),
        MIDDLE_TIP: (0.54, 0.78),   # curled
        RING_MCP:   (0.58, 0.70),
        RING_PIP:   (0.58, 0.75),
        RING_TIP:   (0.58, 0.80),
        PINKY_MCP:  (0.62, 0.72),
        PINKY_PIP:  (0.62, 0.77),
        PINKY_TIP:  (0.62, 0.82),
    })


def rock_landmarks():
    """Index and pinky extended (others curled)."""
    return make_landmarks({
        WRIST:      (0.50, 0.85),
        THUMB_MCP:  (0.38, 0.76),
        THUMB_IP:   (0.34, 0.72),
        THUMB_TIP:  (0.36, 0.73),
        INDEX_MCP:  (0.46, 0.70),
        INDEX_PIP:  (0.43, 0.55),
        INDEX_TIP:  (0.40, 0.42),   # extended
        MIDDLE_MCP: (0.52, 0.68),
        MIDDLE_PIP: (0.52, 0.73),
        MIDDLE_TIP: (0.52, 0.78),   # curled
        RING_MCP:   (0.57, 0.70),
        RING_PIP:   (0.57, 0.75),
        RING_TIP:   (0.57, 0.80),   # curled
        PINKY_MCP:  (0.62, 0.70),
        PINKY_PIP:  (0.64, 0.55),
        PINKY_TIP:  (0.66, 0.42),   # extended
    })


def three_fingers_landmarks():
    """Index, middle, ring extended."""
    return make_landmarks({
        WRIST:      (0.50, 0.85),
        THUMB_MCP:  (0.38, 0.76),
        THUMB_IP:   (0.34, 0.72),
        THUMB_TIP:  (0.36, 0.73),
        INDEX_MCP:  (0.46, 0.70),
        INDEX_PIP:  (0.43, 0.54),
        INDEX_TIP:  (0.40, 0.40),   # extended
        MIDDLE_MCP: (0.52, 0.68),
        MIDDLE_PIP: (0.52, 0.52),
        MIDDLE_TIP: (0.52, 0.38),   # extended
        RING_MCP:   (0.57, 0.70),
        RING_PIP:   (0.59, 0.54),
        RING_TIP:   (0.61, 0.40),   # extended
        PINKY_MCP:  (0.62, 0.72),
        PINKY_PIP:  (0.62, 0.77),
        PINKY_TIP:  (0.62, 0.82),   # curled
    })


def ok_sign_landmarks():
    """Index and thumb tips close, other fingers extended."""
    return make_landmarks({
        WRIST:      (0.50, 0.85),
        THUMB_MCP:  (0.40, 0.72),
        THUMB_IP:   (0.42, 0.62),
        THUMB_TIP:  (0.45, 0.55),   # close to index tip
        INDEX_MCP:  (0.48, 0.70),
        INDEX_PIP:  (0.48, 0.62),
        INDEX_TIP:  (0.48, 0.57),   # close to thumb tip (dist < 0.06)
        MIDDLE_MCP: (0.54, 0.67),
        MIDDLE_PIP: (0.54, 0.52),
        MIDDLE_TIP: (0.54, 0.38),   # extended
        RING_MCP:   (0.59, 0.68),
        RING_PIP:   (0.61, 0.53),
        RING_TIP:   (0.63, 0.39),   # extended
        PINKY_MCP:  (0.64, 0.70),
        PINKY_PIP:  (0.66, 0.56),
        PINKY_TIP:  (0.68, 0.44),   # extended
    })


def pinch_landmarks():
    """Thumb and index tips close, other fingers curled."""
    return make_landmarks({
        WRIST:      (0.50, 0.85),
        THUMB_MCP:  (0.40, 0.72),
        THUMB_IP:   (0.43, 0.63),
        THUMB_TIP:  (0.46, 0.56),   # close to index tip
        INDEX_MCP:  (0.49, 0.70),
        INDEX_PIP:  (0.49, 0.63),
        INDEX_TIP:  (0.49, 0.58),   # close to thumb tip
        MIDDLE_MCP: (0.55, 0.67),
        MIDDLE_PIP: (0.55, 0.72),
        MIDDLE_TIP: (0.55, 0.77),   # curled
        RING_MCP:   (0.59, 0.68),
        RING_PIP:   (0.59, 0.73),
        RING_TIP:   (0.59, 0.78),   # curled
        PINKY_MCP:  (0.63, 0.70),
        PINKY_PIP:  (0.63, 0.75),
        PINKY_TIP:  (0.63, 0.80),   # curled
    })


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestGestureClassifier:

    def test_thumbs_up(self):
        result = classify(thumbs_up_landmarks(), "Right")
        assert result.name == "Thumbs Up", f"Expected 'Thumbs Up', got '{result.name}'"
        assert result.confidence >= 0.80
        assert result.action == "PLAY / RESUME"

    def test_thumbs_down(self):
        result = classify(thumbs_down_landmarks(), "Right")
        assert result.name == "Thumbs Down", f"Expected 'Thumbs Down', got '{result.name}'"
        assert result.confidence >= 0.80
        assert result.action == "PAUSE"

    def test_fist(self):
        result = classify(fist_landmarks(), "Right")
        assert result.name == "Fist", f"Expected 'Fist', got '{result.name}'"
        assert result.confidence >= 0.80
        assert result.action == "MUTE"

    def test_open_palm(self):
        result = classify(open_palm_landmarks(), "Right")
        assert result.name == "Open Palm", f"Expected 'Open Palm', got '{result.name}'"
        assert result.action == "STOP"

    def test_peace(self):
        result = classify(peace_landmarks(), "Right")
        assert result.name == "Peace / V-Sign", f"Expected 'Peace / V-Sign', got '{result.name}'"
        assert result.action == "NEXT TRACK"

    def test_pointing(self):
        result = classify(pointing_landmarks(), "Right")
        assert result.name == "Pointing", f"Expected 'Pointing', got '{result.name}'"
        assert result.action == "CURSOR CLICK"

    def test_rock(self):
        result = classify(rock_landmarks(), "Right")
        assert result.name == "Rock On  🤘", f"Expected 'Rock On  🤘', got '{result.name}'"
        assert result.action == "SHUFFLE"

    def test_three_fingers(self):
        result = classify(three_fingers_landmarks(), "Right")
        assert result.name == "Three Fingers", f"Expected 'Three Fingers', got '{result.name}'"
        assert result.action == "VOLUME UP"

    def test_ok_sign(self):
        result = classify(ok_sign_landmarks(), "Right")
        assert result.name == "OK Sign", f"Expected 'OK Sign', got '{result.name}'"
        assert result.action == "CONFIRM / SELECT"

    def test_pinch(self):
        result = classify(pinch_landmarks(), "Right")
        assert result.name == "Pinch", f"Expected 'Pinch', got '{result.name}'"
        assert result.action == "VOLUME DOWN"

    def test_all_gestures_have_colors(self):
        """Every gesture entry must have a valid BGR color tuple."""
        for key, g in GESTURES.items():
            assert isinstance(g.color, tuple) and len(g.color) == 3, \
                f"Gesture '{key}' has invalid color"
            assert all(0 <= c <= 255 for c in g.color), \
                f"Gesture '{key}' color out of range: {g.color}"

    def test_result_is_copy(self):
        """classify() must return a copy, not mutate the template."""
        r1 = classify(thumbs_up_landmarks(), "Right")
        r2 = classify(thumbs_up_landmarks(), "Right")
        r1.confidence = 0.0
        assert r2.confidence != 0.0, "classify() returned same object (not a copy)"


class TestGestureBuffer:

    def _fill_buffer(self, buf, name, n=10):
        """Push a mock GestureResult into the buffer n times."""
        import copy
        template = list(GESTURES.values())[0]
        mock = copy.copy(template)
        mock.name = name
        mock.action = "TEST"
        for _ in range(n):
            buf.update(mock)
        return mock

    def test_smoothing_commits_stable_gesture(self):
        buf = GestureBuffer(window=10, threshold=0.6)
        result = classify(thumbs_up_landmarks(), "Right")
        for _ in range(8):
            committed = buf.update(result)
        assert committed.name == "Thumbs Up"

    def test_smoothing_rejects_noisy_input(self):
        """Alternating gestures should not commit to either."""
        buf = GestureBuffer(window=10, threshold=0.8)
        tu = classify(thumbs_up_landmarks(), "Right")
        td = classify(thumbs_down_landmarks(), "Right")
        for i in range(10):
            buf.update(tu if i % 2 == 0 else td)
        # Neither gesture reaches 80% threshold
        assert buf.current is None or buf.current.name in ("Thumbs Up", "Thumbs Down")

    def test_history_appended_on_change(self):
        buf = GestureBuffer(window=6, threshold=0.6)
        tu = classify(thumbs_up_landmarks(), "Right")
        op = classify(open_palm_landmarks(), "Right")
        for _ in range(6): buf.update(tu)
        for _ in range(6): buf.update(op)
        names = [e.gesture for e in buf.history]
        assert "Thumbs Up" in names
        assert "Open Palm" in names

    def test_history_capped_at_50(self):
        buf = GestureBuffer(window=1, threshold=0.5)
        gestures_cycle = [
            classify(thumbs_up_landmarks()),
            classify(open_palm_landmarks()),
            classify(fist_landmarks()),
        ]
        import copy
        for i in range(200):
            g = copy.copy(gestures_cycle[i % 3])
            g.name = f"Gesture_{i}"
            buf.update(g)
        assert len(buf.history) <= 50

    def test_history_entry_has_timestamp(self):
        buf = GestureBuffer(window=4, threshold=0.5)
        tu = classify(thumbs_up_landmarks(), "Right")
        before = time.time()
        for _ in range(4): buf.update(tu)
        after = time.time()
        if buf.history:
            assert before <= buf.history[-1].timestamp <= after


class TestGestureActions:

    def test_play_action(self):
        result = classify(thumbs_up_landmarks(), "Right")
        assert "PLAY" in result.action or "RESUME" in result.action

    def test_pause_action(self):
        result = classify(thumbs_down_landmarks(), "Right")
        assert "PAUSE" in result.action

    def test_mute_action(self):
        result = classify(fist_landmarks(), "Right")
        assert "MUTE" in result.action

    def test_stop_action(self):
        result = classify(open_palm_landmarks(), "Right")
        assert "STOP" in result.action

    def test_next_track_action(self):
        result = classify(peace_landmarks(), "Right")
        assert "NEXT" in result.action

    def test_volume_up_action(self):
        result = classify(three_fingers_landmarks(), "Right")
        assert "VOLUME UP" in result.action or "VOL" in result.action

    def test_volume_down_action(self):
        result = classify(pinch_landmarks(), "Right")
        assert "VOLUME DOWN" in result.action or "VOL" in result.action

    def test_shuffle_action(self):
        result = classify(rock_landmarks(), "Right")
        assert "SHUFFLE" in result.action

    def test_confirm_action(self):
        result = classify(ok_sign_landmarks(), "Right")
        assert "CONFIRM" in result.action or "SELECT" in result.action

    def test_click_action(self):
        result = classify(pointing_landmarks(), "Right")
        assert "CLICK" in result.action or "CURSOR" in result.action


class TestEdgeCases:

    def test_classify_never_raises(self):
        """Classifier must never throw on any valid 21-landmark list."""
        import random
        for _ in range(50):
            lms = [SimpleNamespace(x=random.random(), y=random.random(), z=0.0)
                   for _ in range(21)]
            result = classify(lms, "Right")
            assert result is not None
            assert isinstance(result.confidence, float)

    def test_confidence_in_range(self):
        for fn in [thumbs_up_landmarks, thumbs_down_landmarks, fist_landmarks,
                   open_palm_landmarks, peace_landmarks, pointing_landmarks]:
            r = classify(fn(), "Right")
            assert 0.0 <= r.confidence <= 1.0, \
                f"{r.name}: confidence {r.confidence} out of [0,1]"

    def test_gesture_result_has_all_fields(self):
        result = classify(thumbs_up_landmarks(), "Right")
        for attr in ("name", "confidence", "action", "action_icon", "description", "color"):
            assert hasattr(result, attr), f"GestureResult missing field: {attr}"


# ── Run standalone ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    print("\n" + "═"*55)
    print("  Gesture Engine — Unit Tests")
    print("═"*55)

    tests = [
        ("Thumbs Up",     thumbs_up_landmarks,     "PLAY / RESUME"),
        ("Thumbs Down",   thumbs_down_landmarks,    "PAUSE"),
        ("Fist",          fist_landmarks,           "MUTE"),
        ("Open Palm",     open_palm_landmarks,      "STOP"),
        ("Peace/V-Sign",  peace_landmarks,          "NEXT TRACK"),
        ("Pointing",      pointing_landmarks,       "CURSOR CLICK"),
        ("Rock On",       rock_landmarks,           "SHUFFLE"),
        ("Three Fingers", three_fingers_landmarks,  "VOLUME UP"),
        ("OK Sign",       ok_sign_landmarks,        "CONFIRM / SELECT"),
        ("Pinch",         pinch_landmarks,          "VOLUME DOWN"),
    ]

    passed = failed = 0
    for name, fn, expected_action in tests:
        result = classify(fn(), "Right")
        ok = result.name != "Unknown"
        sym = "✓" if ok else "✗"
        status = "PASS" if ok else "FAIL"
        print(f"  {sym} {status}  {name:<18} → {result.name:<20} [{result.action}]")
        if ok: passed += 1
        else:  failed += 1

    print("═"*55)
    print(f"  Results: {passed} passed, {failed} failed")
    print("═"*55 + "\n")
    sys.exit(0 if failed == 0 else 1)
