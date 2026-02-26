"""
gesture_demo.py
───────────────
Real-time hand gesture recognition demo.

Usage:
    python gesture_demo.py                    # default webcam (index 0)
    python gesture_demo.py --camera 1         # alternate camera
    python gesture_demo.py --no-mirror        # disable mirror mode
    python gesture_demo.py --save-log         # save action log to JSON on exit
    python gesture_demo.py --help

Controls (keyboard):
    Q / ESC  — quit
    S        — save screenshot
    H        — toggle help overlay
    L        — toggle gesture log panel
    M        — toggle mirror mode
    +/-      — increase/decrease detection confidence
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

# Local engine
sys.path.insert(0, os.path.dirname(__file__))
from gesture_engine import classify, GestureBuffer, HistoryEntry

# ─── Mediapipe setup ──────────────────────────────────────────────────────────
mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles

# ─── Colours (BGR) ────────────────────────────────────────────────────────────
C_BG       = (15,  17,  22)
C_GREEN    = (80,  220,  80)
C_AMBER    = (30,  180, 240)
C_RED      = (60,   60, 220)
C_CYAN     = (200, 210,  50)
C_WHITE    = (230, 235, 245)
C_DIM      = (90,  100, 120)
C_PANEL    = (22,   26,  34)
C_OVERLAY  = (12,   14,  18)

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO = cv2.FONT_HERSHEY_PLAIN


# ─── Drawing helpers ──────────────────────────────────────────────────────────

def draw_rounded_rect(img, x, y, w, h, r, color, alpha=0.85, border=None, border_color=None):
    overlay = img.copy()
    # corners
    cv2.rectangle(overlay, (x + r, y), (x + w - r, y + h), color, -1)
    cv2.rectangle(overlay, (x, y + r), (x + w, y + h - r), color, -1)
    cv2.ellipse(overlay, (x + r,     y + r    ), (r, r), 180, 0, 90, color, -1)
    cv2.ellipse(overlay, (x + w - r, y + r    ), (r, r), 270, 0, 90, color, -1)
    cv2.ellipse(overlay, (x + r,     y + h - r), (r, r),  90, 0, 90, color, -1)
    cv2.ellipse(overlay, (x + w - r, y + h - r), (r, r),   0, 0, 90, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    if border and border_color:
        cv2.rectangle(img, (x, y), (x + w, y + h), border_color, border)


def put_text(img, text, pos, scale=0.55, color=C_WHITE, thickness=1, font=FONT):
    cv2.putText(img, text, pos, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, color, thickness, cv2.LINE_AA)


def confidence_bar(img, x, y, w, h, value, color):
    """Draw a filled confidence bar."""
    cv2.rectangle(img, (x, y), (x + w, y + h), C_DIM, -1)
    fill = int(w * max(0.0, min(1.0, value)))
    if fill > 0:
        cv2.rectangle(img, (x, y), (x + fill, y + h), color, -1)


# ─── Gesture history sparkline ────────────────────────────────────────────────

def draw_history_panel(img, history: list, x, y, w, h):
    draw_rounded_rect(img, x, y, w, h, 8, C_PANEL, alpha=0.82)
    put_text(img, "GESTURE LOG", (x + 10, y + 18), 0.42, C_DIM)

    visible = history[-10:]
    for i, entry in enumerate(reversed(visible)):
        row_y = y + 36 + i * 22
        age   = time.time() - entry.timestamp
        alpha = max(0.3, 1.0 - age / 15.0)
        col   = tuple(int(c * alpha) for c in C_AMBER)
        ts    = datetime.fromtimestamp(entry.timestamp).strftime("%H:%M:%S")
        put_text(img, f"{ts}  {entry.gesture:<18} {entry.action}", (x + 10, row_y),
                 0.38, col)


# ─── Main demo ────────────────────────────────────────────────────────────────

def run(camera_index: int = 0, mirror: bool = True,
        min_confidence: float = 0.7, save_log: bool = False):

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_index}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf      = GestureBuffer(window=10, threshold=0.55)
    fps_deq  = deque(maxlen=30)
    show_help = False
    show_log  = True
    prev_time = time.time()
    screenshot_count = 0

    # Action flash state
    last_action      = ""
    last_action_time = 0.0
    ACTION_FLASH_DUR = 2.0   # seconds

    print("\n" + "═"*55)
    print("  AXIOM  Hand Gesture Recognition  —  Running")
    print("═"*55)
    print(f"  Camera : {camera_index}   Resolution: {W}×{H}")
    print(f"  Mirror : {mirror}   Min confidence: {min_confidence:.0%}")
    print("  Keys   : Q=quit  S=screenshot  H=help  L=log  M=mirror")
    print("═"*55 + "\n")

    with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=2,
        min_detection_confidence=min_confidence,
        min_tracking_confidence=0.6,
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame grab failed, retrying…")
                time.sleep(0.05)
                continue

            if mirror:
                frame = cv2.flip(frame, 1)

            # ── FPS ─────────────────────────────────────────────────────────
            now = time.time()
            fps_deq.append(1.0 / max(now - prev_time, 1e-6))
            prev_time = now
            fps = sum(fps_deq) / len(fps_deq)

            # ── MediaPipe inference ──────────────────────────────────────────
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            # ── Dark background blend ────────────────────────────────────────
            # Slight desaturation for a "terminal" look
            grey  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grey3 = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
            frame = cv2.addWeighted(frame, 0.80, grey3, 0.20, 0)

            gesture_result = None

            # ── Process hands ────────────────────────────────────────────────
            if results.multi_hand_landmarks:
                for hand_lms, hand_info in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness
                ):
                    handedness = hand_info.classification[0].label  # "Left" or "Right"

                    # Draw skeleton
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_lms,
                        mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(
                            color=(80, 220, 80), thickness=2, circle_radius=4),
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            color=(40, 160, 240), thickness=2),
                    )

                    # Classify
                    raw = classify(hand_lms.landmark, handedness)
                    smoothed = buf.update(raw)

                    # Track action trigger
                    if smoothed and smoothed.name != "Unknown":
                        if smoothed.name != last_action:
                            last_action      = smoothed.action
                            last_action_time = now
                            print(f"  [{datetime.now().strftime('%H:%M:%S')}]  "
                                  f"{smoothed.name:<20}  →  {smoothed.action}")

                    gesture_result = smoothed

                    # ── Landmark bounding box ─────────────────────────────────
                    xs = [lm.x * W for lm in hand_lms.landmark]
                    ys = [lm.y * H for lm in hand_lms.landmark]
                    bx1, by1 = int(min(xs)) - 20, int(min(ys)) - 20
                    bx2, by2 = int(max(xs)) + 20, int(max(ys)) + 20
                    bx1, by1 = max(0, bx1), max(0, by1)
                    bx2, by2 = min(W, bx2), min(H, by2)

                    # Bounding box in gesture colour
                    col = smoothed.color if smoothed else C_DIM
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), col, 1)

                    # Handedness label near wrist
                    wrist_x = int(hand_lms.landmark[0].x * W)
                    wrist_y = int(hand_lms.landmark[0].y * H) + 30
                    put_text(frame, f"{handedness} hand", (wrist_x - 30, wrist_y),
                             0.40, C_DIM)

            # ─────────────────────────────────────────────────────────────────
            # ── UI PANELS
            # ─────────────────────────────────────────────────────────────────

            # ── TOP STATUS BAR ───────────────────────────────────────────────
            draw_rounded_rect(frame, 0, 0, W, 46, 0, C_OVERLAY, alpha=0.78)
            put_text(frame, "AXIOM  GESTURE RECOGNITION",
                     (14, 30), 0.58, C_GREEN, 1)
            put_text(frame, f"FPS: {fps:5.1f}", (W - 110, 30), 0.52, C_CYAN, 1)
            put_text(frame, f"CAM {camera_index}  {W}x{H}",
                     (W - 230, 30), 0.40, C_DIM, 1)

            # ── GESTURE CARD (left side) ──────────────────────────────────────
            card_x, card_y, card_w, card_h = 14, 60, 320, 160
            draw_rounded_rect(frame, card_x, card_y, card_w, card_h, 10,
                               C_PANEL, alpha=0.85)

            if gesture_result:
                col = gesture_result.color
                # Gesture name
                put_text(frame, gesture_result.name.upper(),
                         (card_x + 14, card_y + 30), 0.70, col, 2)
                # Confidence bar
                put_text(frame, "CONFIDENCE", (card_x + 14, card_y + 60), 0.38, C_DIM)
                confidence_bar(frame, card_x + 14, card_y + 68,
                               card_w - 28, 8, gesture_result.confidence, col)
                put_text(frame, f"{gesture_result.confidence:.0%}",
                         (card_x + 14, card_y + 95), 0.45, col)
                # Action
                put_text(frame, "ACTION", (card_x + 14, card_y + 118), 0.38, C_DIM)
                put_text(frame, gesture_result.action,
                         (card_x + 14, card_y + 142), 0.60, C_AMBER, 2)
            else:
                put_text(frame, "NO HAND DETECTED",
                         (card_x + 14, card_y + 60), 0.55, C_DIM)
                put_text(frame, "Point your hand at the camera",
                         (card_x + 14, card_y + 90), 0.40, C_DIM)

            # ── ACTION FLASH ──────────────────────────────────────────────────
            elapsed = now - last_action_time
            if elapsed < ACTION_FLASH_DUR and last_action:
                fade    = 1.0 - elapsed / ACTION_FLASH_DUR
                flash_h = 70
                flash_y = H // 2 - flash_h // 2
                draw_rounded_rect(frame, W // 2 - 200, flash_y, 400, flash_h,
                                  12, C_PANEL, alpha=fade * 0.9)
                alpha_col = tuple(int(c * fade) for c in C_GREEN)
                put_text(frame, last_action,
                         (W // 2 - 150, flash_y + flash_h // 2 + 10),
                         0.80, alpha_col, 2)

            # ── GESTURE LOG (right side) ───────────────────────────────────
            if show_log:
                log_w, log_h = 420, 260
                draw_history_panel(frame, buf.history,
                                   W - log_w - 14, 60, log_w, log_h)

            # ── GESTURE REFERENCE (bottom right) ─────────────────────────────
            ref_x  = W - 220
            ref_y0 = H - 200
            draw_rounded_rect(frame, ref_x - 10, ref_y0 - 10, 230, 200,
                               8, C_PANEL, alpha=0.75)
            put_text(frame, "GESTURE MAP", (ref_x, ref_y0 + 8), 0.40, C_DIM)
            gestures_ref = [
                ("Thumbs Up",     "▶ Play"),
                ("Thumbs Down",   "⏸ Pause"),
                ("Fist",          "🔇 Mute"),
                ("Open Palm",     "⏹ Stop"),
                ("Peace/V",       "⏭ Next"),
                ("Pointing",      "🖱 Click"),
                ("Three Fingers", "🔊 Vol+"),
                ("Pinch",         "🔉 Vol-"),
                ("OK Sign",       "✓ Confirm"),
                ("Rock On",       "🔀 Shuffle"),
            ]
            for i, (g, a) in enumerate(gestures_ref):
                row_y = ref_y0 + 24 + i * 17
                put_text(frame, g, (ref_x, row_y), 0.36, C_WHITE)
                put_text(frame, a, (ref_x + 115, row_y), 0.36, C_AMBER)

            # ── HELP OVERLAY ─────────────────────────────────────────────────
            if show_help:
                hx, hy, hw, hh = W // 2 - 200, H // 2 - 140, 400, 280
                draw_rounded_rect(frame, hx, hy, hw, hh, 12,
                                  C_OVERLAY, alpha=0.92)
                put_text(frame, "KEYBOARD SHORTCUTS", (hx + 20, hy + 30),
                         0.55, C_GREEN, 1)
                shortcuts = [
                    ("Q / ESC", "Quit"),
                    ("S",        "Save screenshot"),
                    ("H",        "Toggle this help"),
                    ("L",        "Toggle gesture log"),
                    ("M",        "Toggle mirror mode"),
                    ("+",        "Increase min confidence"),
                    ("-",        "Decrease min confidence"),
                ]
                for i, (k, v) in enumerate(shortcuts):
                    row_y = hy + 60 + i * 30
                    put_text(frame, k, (hx + 30,  row_y), 0.50, C_CYAN)
                    put_text(frame, v, (hx + 140, row_y), 0.50, C_WHITE)
                put_text(frame, "Press H to close",
                         (hx + 100, hy + hh - 20), 0.40, C_DIM)

            # ── CONFIDENCE INDICATOR ─────────────────────────────────────────
            put_text(frame, f"MIN CONF: {min_confidence:.0%}",
                     (14, H - 14), 0.40, C_DIM)
            put_text(frame, "H=help  L=log  M=mirror  Q=quit",
                     (14 + 140, H - 14), 0.38, C_DIM)

            # ─────────────────────────────────────────────────────────────────
            cv2.imshow("AXIOM — Hand Gesture Recognition", frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):   # Q or ESC
                break
            elif key == ord('s'):
                fname = f"gesture_screenshot_{screenshot_count:03d}.png"
                cv2.imwrite(fname, frame)
                screenshot_count += 1
                print(f"  Screenshot saved: {fname}")
            elif key == ord('h'):
                show_help = not show_help
            elif key == ord('l'):
                show_log = not show_log
            elif key == ord('m'):
                mirror = not mirror
            elif key in (ord('+'), ord('=')):
                min_confidence = min(0.99, min_confidence + 0.05)
                print(f"  Min confidence: {min_confidence:.0%}")
            elif key == ord('-'):
                min_confidence = max(0.30, min_confidence - 0.05)
                print(f"  Min confidence: {min_confidence:.0%}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "═"*55)
    print(f"  Session ended.  Gestures triggered: {len(buf.history)}")

    if save_log and buf.history:
        log_data = [
            {"gesture": e.gesture, "action": e.action,
             "timestamp": datetime.fromtimestamp(e.timestamp).isoformat()}
            for e in buf.history
        ]
        fname = f"gesture_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(fname, "w") as f:
            json.dump(log_data, f, indent=2)
        print(f"  Action log saved: {fname}")

    print("═"*55 + "\n")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AXIOM — Real-time Hand Gesture Recognition Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--camera",     type=int,   default=0,    help="Camera index (default 0)")
    parser.add_argument("--no-mirror",  action="store_true",       help="Disable mirror mode")
    parser.add_argument("--confidence", type=float, default=0.70,  help="Min detection confidence 0.0-1.0")
    parser.add_argument("--save-log",   action="store_true",       help="Save action log to JSON on exit")
    args = parser.parse_args()

    run(
        camera_index=args.camera,
        mirror=not args.no_mirror,
        min_confidence=args.confidence,
        save_log=args.save_log,
    )

if __name__ == "__main__":
    main()
