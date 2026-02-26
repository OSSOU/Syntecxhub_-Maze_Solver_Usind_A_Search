"""
assistant.py
────────────
Main voice assistant — beautiful terminal UI, full conversation loop.

Usage:
    python assistant.py                  # auto-detect mic + TTS
    python assistant.py --text           # keyboard input only (no mic)
    python assistant.py --silent         # no TTS (print only)
    python assistant.py --log            # save conversation log on exit
    python assistant.py --lang en-US     # STT language code
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import signal
import sys
import textwrap
import time
import threading
from typing import Optional

from command_engine import CommandEngine, Response, ConversationTurn
from audio_io import make_stt, make_tts


# ══════════════════════════════════════════════════════════════════════════════
#  Terminal colour / style helpers  (ANSI, degrades gracefully)
# ══════════════════════════════════════════════════════════════════════════════

NO_COLOR = not sys.stdout.isatty() or os.environ.get("NO_COLOR")

def _c(code: str, text: str) -> str:
    if NO_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"

def green(t):   return _c("32",    t)
def cyan(t):    return _c("36",    t)
def yellow(t):  return _c("33",    t)
def magenta(t): return _c("35",    t)
def red(t):     return _c("31",    t)
def bold(t):    return _c("1",     t)
def dim(t):     return _c("2",     t)
def white(t):   return _c("37",    t)
def bg_dark(t): return _c("40",    t)

# Box-drawing chars
H = "─"; V = "│"; TL = "╭"; TR = "╮"; BL = "╰"; BR = "╯"
LT = "├"; RT = "┤"

def box(lines: list[str], width: int = 68, color_fn=cyan) -> str:
    out  = [color_fn(TL + H * (width - 2) + TR)]
    for line in lines:
        # Strip ANSI for length calculation
        import re
        clean = re.sub(r"\033\[[0-9;]*m", "", line)
        pad   = width - 2 - len(clean)
        out.append(color_fn(V) + " " + line + " " * max(0, pad - 1) + color_fn(V))
    out.append(color_fn(BL + H * (width - 2) + BR))
    return "\n".join(out)

def divider(width: int = 68, color_fn=dim) -> str:
    return color_fn(H * width)

def _wrap(text: str, width: int = 62, indent: str = "  ") -> list[str]:
    """Wrap text into lines of given width."""
    return textwrap.wrap(text, width=width, initial_indent=indent,
                         subsequent_indent=indent)


# ══════════════════════════════════════════════════════════════════════════════
#  Typing animation
# ══════════════════════════════════════════════════════════════════════════════

class TypingAnimation:
    """Shows a spinning "thinking" indicator on the same line."""

    FRAMES = ["⠋", "⠙", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, label: str = "Processing"):
        self._label   = label
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if NO_COLOR:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self):
        if NO_COLOR:
            return
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
        # Clear the spinner line
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def _spin(self):
        i = 0
        while self._running:
            frame = self.FRAMES[i % len(self.FRAMES)]
            sys.stdout.write(f"\r  {cyan(frame)} {dim(self._label + '...')}")
            sys.stdout.flush()
            time.sleep(0.08)
            i += 1


# ══════════════════════════════════════════════════════════════════════════════
#  Main UI
# ══════════════════════════════════════════════════════════════════════════════

BANNER = r"""
   ___   _  __ _____  ___  __  ___
  / _ | | |/_//  _/ |/ / |/  |/  /
 / __ |_>  < _/ //    / /|_/ /|_/ /
/_/ |_/_/|_|/___/_/|_/_/  /_/  /_/

"""

def print_banner(stt_name: str, tts_name: str, mic: bool, speech: bool):
    lines = [
        bold(white("AXIOM  Personal Voice Assistant")),
        dim("─" * 46),
        f"  {dim('STT :')} {cyan(stt_name)}{'  ' + green('🎙 Mic active') if mic else '  ' + yellow('⌨  Keyboard mode')}",
        f"  {dim('TTS :')} {cyan(tts_name)}{'  ' + green('🔊 Voice active') if speech else '  ' + yellow('🖨  Print only')}",
        dim("─" * 46),
        f"  {dim('Say')} {yellow('\"help\"')} {dim('for commands ·')} {yellow('\"exit\"')} {dim('to quit')}",
    ]
    print()
    for line in lines:
        print(f"  {line}")
    print()


def print_user_input(text: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    prefix = f"  {dim(ts)}  {yellow('You')}  {dim('›')} "
    print(f"\n{prefix}{bold(white(text))}")


def print_assistant_response(resp: Response, speak: bool):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    intent_tag = f" {dim('[')}{magenta(resp.intent)}{dim(']')}" if resp.intent else ""
    conf_tag   = f" {dim(f'{resp.confidence:.0%}')}" if resp.confidence < 0.8 else ""
    prefix     = f"  {dim(ts)}  {cyan('AXIOM')}{intent_tag}{conf_tag}  {dim('›')} "

    if not resp.success:
        color = yellow
    else:
        color = white

    # Wrap long responses
    max_w = 72 - len(f"  {ts}  AXIOM [{resp.intent}]  › ")
    max_w = max(40, max_w)
    lines = textwrap.wrap(resp.text, width=max_w)

    if lines:
        print(f"\n{prefix}{color(lines[0])}")
        for line in lines[1:]:
            print(f"  {'':>{len(ts) + 14}}{color(line)}")

    if resp.action_taken:
        print(f"  {'':>{len(ts) + 2}}  {dim('↳ ' + resp.action_taken)}")


def print_status(msg: str, kind: str = "info"):
    icons = {"info": dim("ℹ"), "ok": green("✓"), "warn": yellow("⚠"), "error": red("✗")}
    icon  = icons.get(kind, dim("·"))
    print(f"  {icon}  {dim(msg)}")


def print_separator():
    print(f"\n  {dim(H * 64)}\n")


def show_help_panel(engine: CommandEngine):
    """Print a nicely formatted command reference."""
    examples = engine.all_examples()
    print()
    print(f"  {bold(cyan('Command Reference'))}")
    print(f"  {dim(H * 60)}")
    for intent, exs in examples.items():
        print(f"  {magenta(intent.replace('_', ' ').title()):<22} {dim('·')}  {white(exs[0])}")
        for ex in exs[1:]:
            print(f"  {'':22}   {dim(ex)}")
    print(f"  {dim(H * 60)}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  Session log
# ══════════════════════════════════════════════════════════════════════════════

def save_log(history: list[ConversationTurn], path: Optional[str] = None):
    if path is None:
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.expanduser(f"~/axiom_log_{ts}.json")
    data = [
        {"speaker": t.speaker, "text": t.text, "intent": t.intent,
         "ts": t.ts.isoformat()}
        for t in history
    ]
    with open(path, "w") as f:
        json.dump({"session": data}, f, indent=2)
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  Main loop
# ══════════════════════════════════════════════════════════════════════════════

def run(force_text: bool = False, force_silent: bool = False,
        save_on_exit: bool = False, lang: str = "en-US"):

    engine = CommandEngine()
    stt, has_mic    = make_stt(force_terminal=force_text)
    tts, has_speech = make_tts(force_terminal=force_silent)

    # ── Banner ────────────────────────────────────────────────────────────────
    os.system("clear" if os.name != "nt" else "cls")
    for line in BANNER.strip().split("\n"):
        print(cyan(line))
    print_banner(stt.name, tts.name, has_mic, has_speech)

    # ── Greeting ──────────────────────────────────────────────────────────────
    greeting = (
        "AXIOM online. "
        + ("Say a command." if has_mic else "Type a command and press Enter.")
        + " Say 'help' for a list of commands."
    )
    print_status(greeting, "ok")
    if has_speech:
        tts.speak(greeting)

    # ── SIGINT handler ────────────────────────────────────────────────────────
    def _on_sigint(sig, frame):
        print()
        print_status("Interrupted. Goodbye!", "warn")
        if save_on_exit:
            path = save_log(engine.history)
            print_status(f"Session saved → {path}", "info")
        sys.exit(0)

    signal.signal(signal.SIGINT, _on_sigint)

    # ── Main loop ─────────────────────────────────────────────────────────────
    turn = 0
    while True:
        turn += 1

        # ── Prompt ────────────────────────────────────────────────────────────
        if has_mic:
            print(f"\n  {green('🎙')}  {dim('Listening...')}  ", end="", flush=True)
        else:
            print(f"\n  {yellow('⌨ ')}  {dim('You:')} ", end="", flush=True)

        # ── Capture input ─────────────────────────────────────────────────────
        try:
            user_text = stt.listen()
        except KeyboardInterrupt:
            break
        except Exception as e:
            print_status(f"Input error: {e}", "error")
            continue

        if not user_text:
            if has_mic:
                print_status("Didn't catch that. Please try again.", "warn")
            continue

        print_user_input(user_text)

        # ── Inline help command ───────────────────────────────────────────────
        if user_text.lower() in ("list", "commands", "command list", "reference"):
            show_help_panel(engine)
            continue

        # ── Classify + respond ────────────────────────────────────────────────
        spinner = TypingAnimation("Processing")
        spinner.start()

        try:
            resp = engine.process(user_text)
        except Exception as e:
            resp = Response(
                text=f"An unexpected error occurred: {e}. Please try again.",
                intent="error",
                success=False,
            )
        finally:
            spinner.stop()

        print_assistant_response(resp, has_speech)

        if has_speech:
            try:
                tts.speak(resp.text)
            except Exception as e:
                print_status(f"TTS error: {e}", "warn")

        # ── Exit check ────────────────────────────────────────────────────────
        if resp.intent == "goodbye":
            if save_on_exit:
                path = save_log(engine.history)
                print_status(f"Session log saved → {path}", "info")
            print()
            sys.exit(0)

        # ── Optional follow-up (e.g. timer expiry) ────────────────────────────
        if resp.followup and resp.intent == "timer":
            # Parse duration from action_taken "Timer: Xs"
            import re
            m = re.search(r"Timer:\s*(\d+)s", resp.action_taken)
            if m:
                secs = int(m.group(1))
                def _delayed(text, delay):
                    time.sleep(delay)
                    print(f"\n  {green('⏰')}  {bold(cyan(text))}\n")
                    if has_speech:
                        try: tts.speak(text)
                        except: pass
                threading.Thread(
                    target=_delayed, args=(resp.followup, secs), daemon=True
                ).start()


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="AXIOM — Personal Voice Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
          python assistant.py                 # full mic + speech (if deps installed)
          python assistant.py --text          # keyboard input, no mic required
          python assistant.py --silent        # mic input, print responses only
          python assistant.py --text --log    # keyboard + save conversation log
        """),
    )
    parser.add_argument("--text",   action="store_true", help="Use keyboard instead of microphone")
    parser.add_argument("--silent", action="store_true", help="Print responses instead of speaking")
    parser.add_argument("--log",    action="store_true", help="Save conversation log (JSON) on exit")
    parser.add_argument("--lang",   default="en-US",     help="STT language code (default: en-US)")
    args = parser.parse_args()

    run(
        force_text=args.text,
        force_silent=args.silent,
        save_on_exit=args.log,
        lang=args.lang,
    )


if __name__ == "__main__":
    main()
