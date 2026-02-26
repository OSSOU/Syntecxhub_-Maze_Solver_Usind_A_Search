"""
command_engine.py
─────────────────
Intent classification + action routing for the voice assistant.
Completely decoupled from audio — takes a string, returns a Response.
"""

from __future__ import annotations

import datetime
import math
import os
import platform
import random
import re
import subprocess
import sys
import webbrowser
from dataclasses import dataclass, field
from typing import Callable, Optional
import urllib.parse


# ══════════════════════════════════════════════════════════════════════════════
#  Data types
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Response:
    """Everything the assistant wants to say/do after recognising a command."""
    text: str                          # spoken + displayed response
    intent: str = "unknown"            # matched intent label
    confidence: float = 1.0           # 0-1 match confidence
    action_taken: str = ""             # short human-readable action log
    followup: Optional[str] = None     # second TTS utterance (e.g. after delay)
    success: bool = True

@dataclass
class ConversationTurn:
    speaker: str          # "user" | "assistant"
    text: str
    intent: str = ""
    ts: datetime.datetime = field(default_factory=datetime.datetime.now)


# ══════════════════════════════════════════════════════════════════════════════
#  Intent rule definitions
# ══════════════════════════════════════════════════════════════════════════════

class Intent:
    """
    Wraps a list of regex patterns + a handler callable.
    Patterns are tried in order; first match wins.
    """
    def __init__(self, name: str, patterns: list[str],
                 handler: Callable[[re.Match, str], Response],
                 examples: list[str] = None, priority: int = 0):
        self.name     = name
        self.patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
        self.handler  = handler
        self.examples = examples or []
        self.priority = priority   # higher = checked earlier

    def match(self, text: str) -> Optional[re.Match]:
        for pat in self.patterns:
            m = pat.search(text)
            if m:
                return m
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  Individual action handlers
# ══════════════════════════════════════════════════════════════════════════════

def _h_time(m: re.Match, raw: str) -> Response:
    now = datetime.datetime.now()
    t   = now.strftime("%-I:%M %p")          # e.g. "3:42 PM"
    return Response(
        text=f"The current time is {t}.",
        intent="time",
        action_taken=f"Reported time: {t}",
    )


def _h_date(m: re.Match, raw: str) -> Response:
    now  = datetime.datetime.now()
    date = now.strftime("%A, %B %-d, %Y")    # e.g. "Thursday, February 26, 2026"
    return Response(
        text=f"Today is {date}.",
        intent="date",
        action_taken=f"Reported date: {date}",
    )


def _h_datetime(m: re.Match, raw: str) -> Response:
    now  = datetime.datetime.now()
    t    = now.strftime("%-I:%M %p")
    date = now.strftime("%A, %B %-d, %Y")
    return Response(
        text=f"It is {t} on {date}.",
        intent="datetime",
        action_taken="Reported full date-time",
    )


def _h_web_search(m: re.Match, raw: str) -> Response:
    query = (m.group("query") or "").strip()
    if not query:
        query = re.sub(r"(?i)(search(?: for| the web for)?|google|look up|find)\s*", "", raw).strip()
    if not query:
        return Response(text="What would you like me to search for?",
                        intent="web_search", success=False)
    url = "https://www.google.com/search?q=" + urllib.parse.quote_plus(query)
    try:
        webbrowser.open(url)
        action = f"Opened browser: {url}"
    except Exception as e:
        action = f"Could not open browser: {e}"
    return Response(
        text=f"Searching the web for: {query}.",
        intent="web_search",
        action_taken=action,
    )


def _h_wikipedia(m: re.Match, raw: str) -> Response:
    topic = (m.group("topic") or "").strip()
    if not topic:
        topic = re.sub(r"(?i)(wikipedia|wiki|what is|who is|tell me about)\s*", "", raw).strip()
    url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote_plus(topic.replace(" ", "_"))
    try:
        webbrowser.open(url)
        action = f"Opened Wikipedia: {url}"
    except Exception as e:
        action = f"Could not open browser: {e}"
    return Response(
        text=f"Opening Wikipedia page for: {topic}.",
        intent="wikipedia",
        action_taken=action,
    )


def _h_youtube(m: re.Match, raw: str) -> Response:
    query = (m.group("query") or "").strip()
    if not query:
        query = re.sub(r"(?i)(play|youtube|search youtube for|find on youtube)\s*", "", raw).strip()
    url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(query)
    try:
        webbrowser.open(url)
        action = f"Opened YouTube: {url}"
    except Exception as e:
        action = f"Could not open: {e}"
    return Response(
        text=f"Opening YouTube search for: {query}.",
        intent="youtube",
        action_taken=action,
    )


# App launch registry
_APP_MAP = {
    "calculator": {
        "linux":   ["gnome-calculator", "kcalc", "xcalc"],
        "darwin":  ["open", "-a", "Calculator"],
        "windows": ["calc"],
    },
    "notepad": {
        "linux":   ["gedit", "kate", "mousepad", "xed"],
        "darwin":  ["open", "-a", "TextEdit"],
        "windows": ["notepad"],
    },
    "browser": {
        "linux":   ["xdg-open", "https://google.com"],
        "darwin":  ["open", "-a", "Safari"],
        "windows": ["start", "chrome"],
    },
    "terminal": {
        "linux":   ["x-terminal-emulator", "gnome-terminal", "xterm"],
        "darwin":  ["open", "-a", "Terminal"],
        "windows": ["cmd"],
    },
    "file manager": {
        "linux":   ["nautilus", "dolphin", "thunar"],
        "darwin":  ["open", "-a", "Finder"],
        "windows": ["explorer"],
    },
    "spotify": {
        "linux":   ["spotify"],
        "darwin":  ["open", "-a", "Spotify"],
        "windows": ["start", "spotify"],
    },
    "vscode": {
        "linux":   ["code"],
        "darwin":  ["open", "-a", "Visual Studio Code"],
        "windows": ["code"],
    },
    "files": {
        "linux":   ["nautilus", "dolphin", "thunar"],
        "darwin":  ["open", "-a", "Finder"],
        "windows": ["explorer"],
    },
}

def _h_open_app(m: re.Match, raw: str) -> Response:
    app_raw = (m.group("app") or "").strip().lower()
    # Map aliases
    alias = {
        "chrome": "browser", "firefox": "browser", "safari": "browser",
        "text editor": "notepad", "notes": "notepad",
        "music": "spotify", "calculator app": "calculator",
        "vs code": "vscode", "visual studio code": "vscode",
        "finder": "file manager", "file explorer": "file manager",
    }
    app_key = alias.get(app_raw, app_raw)

    if app_key not in _APP_MAP:
        return Response(
            text=f"I don't know how to open '{app_raw}'. Try: calculator, browser, terminal, or notepad.",
            intent="open_app",
            success=False,
            action_taken=f"Unknown app: {app_raw}",
        )

    sys_key = platform.system().lower()  # "linux", "darwin", "windows"
    cmd = _APP_MAP[app_key].get(sys_key, _APP_MAP[app_key].get("linux", []))

    try:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        action = f"Launched: {' '.join(cmd)}"
        text   = f"Opening {app_raw}."
    except (FileNotFoundError, OSError):
        action = f"Could not launch {app_key} (not installed?)"
        text   = f"I tried to open {app_raw} but it doesn't seem to be installed."

    return Response(text=text, intent="open_app", action_taken=action)


def _h_weather(m: re.Match, raw: str) -> Response:
    city = (m.group("city") or "").strip() or "your location"
    url  = f"https://wttr.in/{urllib.parse.quote_plus(city)}"
    try:
        webbrowser.open(url)
        action = f"Opened weather for {city}"
    except Exception as e:
        action = str(e)
    return Response(
        text=f"Opening weather forecast for {city}.",
        intent="weather",
        action_taken=action,
    )


def _h_calculate(m: re.Match, raw: str) -> Response:
    expr_raw = (m.group("expr") or "").strip()
    # Clean up spoken math
    replacements = [
        (r"\bplus\b",      "+"),
        (r"\bminus\b",     "-"),
        (r"\btimes\b",     "*"),
        (r"\bmultiplied by\b", "*"),
        (r"\bdivided by\b","//"),
        (r"\bover\b",      "/"),
        (r"\bsquared\b",   "**2"),
        (r"\bcubed\b",     "**3"),
        (r"\bto the power of\s*(\d+)", r"**\1"),
        (r"\bsquare root of\s*(\d+(?:\.\d+)?)", r"math.sqrt(\1)"),
        (r"\bpi\b",        "math.pi"),
        (r"[^0-9+\-*/().%\s]", ""),
    ]
    expr = expr_raw
    for pat, repl in replacements:
        expr = re.sub(pat, repl, expr, flags=re.IGNORECASE)
    expr = expr.strip()
    if not expr:
        return Response(text="I couldn't parse that calculation. Try: 'calculate 25 times 4'.",
                        intent="calculate", success=False)
    try:
        # Safe eval — restrict to math operations
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        result  = eval(expr, {"__builtins__": {}}, allowed)  # noqa: S307
        result  = round(result, 10) if isinstance(result, float) else result
        # Clean up trailing zeros
        if isinstance(result, float) and result == int(result):
            result = int(result)
        return Response(
            text=f"{expr_raw} equals {result}.",
            intent="calculate",
            action_taken=f"Evaluated: {expr} = {result}",
        )
    except Exception as e:
        return Response(
            text=f"I couldn't compute that. Error: {e}. Please try a simpler expression.",
            intent="calculate",
            success=False,
        )


def _h_timer(m: re.Match, raw: str) -> Response:
    # Extract duration
    hours   = int(m.group("hours")   or 0)
    minutes = int(m.group("minutes") or 0)
    seconds = int(m.group("seconds") or 0)
    total   = hours * 3600 + minutes * 60 + seconds
    if total == 0:
        return Response(text="I couldn't understand the duration. Try: 'set a timer for 5 minutes'.",
                        intent="timer", success=False)
    parts = []
    if hours:   parts.append(f"{hours} hour{'s' if hours > 1 else ''}")
    if minutes: parts.append(f"{minutes} minute{'s' if minutes > 1 else ''}")
    if seconds: parts.append(f"{seconds} second{'s' if seconds > 1 else ''}")
    duration_str = " and ".join(parts)
    return Response(
        text=f"Timer set for {duration_str}. I'll let you know when it's done.",
        intent="timer",
        action_taken=f"Timer: {total}s",
        followup=f"Your {duration_str} timer is up!",
    )


def _h_reminder(m: re.Match, raw: str) -> Response:
    task = (m.group("task") or raw).strip()
    task = re.sub(r"(?i)^(remind me to|set a reminder (to|for)|reminder)\s*", "", task)
    return Response(
        text=f"Got it. I'll remind you to: {task}.",
        intent="reminder",
        action_taken=f"Reminder queued: {task}",
    )


def _h_note(m: re.Match, raw: str) -> Response:
    content = (m.group("content") or "").strip()
    if not content:
        content = re.sub(r"(?i)^(take a note|note down|write down|make a note)\s*(that)?\s*", "", raw).strip()
    # Append to notes file
    ts       = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    note_dir = os.path.expanduser("~")
    note_file = os.path.join(note_dir, "assistant_notes.txt")
    try:
        with open(note_file, "a") as f:
            f.write(f"[{ts}] {content}\n")
        action = f"Saved to {note_file}"
    except Exception as e:
        action = f"Failed to save note: {e}"
    return Response(
        text=f"Note saved: {content}.",
        intent="note",
        action_taken=action,
    )


def _h_greeting(m: re.Match, raw: str) -> Response:
    greetings = [
        "Hello! I'm ready to help. What can I do for you?",
        "Hi there! Say a command or ask me anything.",
        "Hey! Good to hear from you. How can I assist?",
        "Greetings! I'm listening.",
    ]
    return Response(text=random.choice(greetings), intent="greeting")


def _h_how_are_you(m: re.Match, raw: str) -> Response:
    responses = [
        "I'm running smoothly, thanks for asking! What would you like to do?",
        "All systems are go! I'm here and ready to help.",
        "I'm doing great — fully operational and listening!",
    ]
    return Response(text=random.choice(responses), intent="small_talk")


def _h_name(m: re.Match, raw: str) -> Response:
    return Response(
        text="I'm AXIOM, your personal voice assistant. I can help you search the web, "
             "check the time, open apps, do calculations, and much more.",
        intent="name",
    )


def _h_help(m: re.Match, raw: str) -> Response:
    return Response(
        text=(
            "Here's what I can do: "
            "Tell you the time or date. "
            "Search the web or Wikipedia. "
            "Open apps like calculator, browser, or terminal. "
            "Check the weather. "
            "Do calculations. "
            "Set timers and reminders. "
            "Take notes. "
            "Play YouTube videos. "
            "Say 'sample commands' for more examples."
        ),
        intent="help",
        action_taken="Displayed help",
    )


def _h_samples(m: re.Match, raw: str) -> Response:
    return Response(
        text=(
            "Here are some sample commands: "
            "What time is it? "
            "What's today's date? "
            "Search for Python tutorials. "
            "Open calculator. "
            "Weather in New York. "
            "Calculate 128 divided by 4. "
            "Set a timer for 10 minutes. "
            "Remind me to drink water. "
            "Play lo-fi music on YouTube. "
            "Take a note: buy groceries. "
            "Open Wikipedia for artificial intelligence."
        ),
        intent="samples",
    )


def _h_joke(m: re.Match, raw: str) -> Response:
    jokes = [
        "Why do programmers prefer dark mode? Because light attracts bugs!",
        "A SQL query walks into a bar and asks: 'Can I JOIN you?' — The bartender says 'Only if you bring your own keys.'",
        "How many programmers does it take to change a light bulb? None — that's a hardware problem.",
        "Why did the voice assistant break up with the chatbot? Too much talk, not enough action.",
        "I told my computer I needed a break. Now it won't stop sending me Kit-Kat ads.",
        "Why is Python so popular? Because nobody wants to deal with curly brace disputes.",
    ]
    return Response(text=random.choice(jokes), intent="joke")


def _h_goodbye(m: re.Match, raw: str) -> Response:
    farewells = [
        "Goodbye! Have a great day.",
        "See you later! I'll be here whenever you need me.",
        "Farewell! Stay productive.",
        "Bye! Come back anytime.",
    ]
    return Response(text=random.choice(farewells), intent="goodbye")


def _h_volume(m: re.Match, raw: str) -> Response:
    action_word = (m.group("action") or "").lower()
    level       = m.group("level") if "level" in m.groupdict() else None
    sys_key     = platform.system().lower()

    if "mute" in raw.lower():
        cmd_map = {"linux": ["pactl", "set-sink-mute", "@DEFAULT_SINK@", "toggle"],
                   "darwin": ["osascript", "-e", "set volume output muted true"]}
        spoken = "Muting audio."
    elif "up" in action_word or "increase" in action_word or "louder" in raw.lower():
        cmd_map = {"linux": ["pactl", "set-sink-volume", "@DEFAULT_SINK@", "+10%"],
                   "darwin": ["osascript", "-e", "set volume output volume (output volume of (get volume settings) + 10)"]}
        spoken = "Increasing volume."
    else:
        cmd_map = {"linux": ["pactl", "set-sink-volume", "@DEFAULT_SINK@", "-10%"],
                   "darwin": ["osascript", "-e", "set volume output volume (output volume of (get volume settings) - 10)"]}
        spoken = "Decreasing volume."

    cmd = cmd_map.get(sys_key, [])
    action = "Volume command not supported on this OS"
    if cmd:
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            action = f"Ran: {' '.join(cmd)}"
        except Exception as e:
            action = str(e)
    return Response(text=spoken, intent="volume", action_taken=action)


def _h_screenshot(m: re.Match, raw: str) -> Response:
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.expanduser(f"~/screenshot_{ts}.png")
    sys_key = platform.system().lower()
    cmds = {
        "linux":  ["scrot", path],
        "darwin": ["screencapture", path],
        "windows": ["snippingtool"],
    }
    cmd = cmds.get(sys_key, [])
    try:
        if cmd:
            subprocess.Popen(cmd)
            action = f"Screenshot saved to {path}"
            text   = f"Screenshot saved to your home folder."
        else:
            action = "Unsupported OS"
            text   = "Screenshot not supported on this system."
    except Exception as e:
        action = str(e)
        text   = "I couldn't take a screenshot right now."
    return Response(text=text, intent="screenshot", action_taken=action)


def _h_system_info(m: re.Match, raw: str) -> Response:
    info = {
        "OS":     platform.system() + " " + platform.release(),
        "Python": platform.python_version(),
        "CPU":    platform.processor() or "Unknown",
        "Node":   platform.node(),
    }
    parts = [f"{k}: {v}" for k, v in info.items()]
    return Response(
        text="System info: " + ". ".join(parts) + ".",
        intent="system_info",
        action_taken="Reported system info",
    )


def _h_unknown(raw: str) -> Response:
    fallbacks = [
        f"I didn't understand: '{raw}'. Try saying 'help' for a list of commands.",
        f"Hmm, I'm not sure what you meant by '{raw}'. Say 'help' to see what I can do.",
        f"That command didn't match any of my skills. Say 'sample commands' for examples.",
    ]
    return Response(
        text=random.choice(fallbacks),
        intent="unknown",
        confidence=0.0,
        success=False,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Intent registry
# ══════════════════════════════════════════════════════════════════════════════

INTENTS: list[Intent] = [

    # ── High priority: exact/specific ────────────────────────────────────────

    Intent("datetime", [
        r"\b(date and time|time and date|what(?:'s| is) the date and time)\b"
    ], _h_datetime, ["What's the date and time?"], priority=10),

    Intent("time", [
        r"\bwhat(?:'s)?(?:\s+(?:is|the))?\s+time\b",
        r"\b(current time|tell me the time|time right now|time now)\b",
        r"^time$",
    ], _h_time, ["What time is it?", "Current time"], priority=9),

    Intent("date", [
        r"\b(what(?:'s| is)(?: today'?s?)? date|today(?:'s)? date|what day is it|day today)\b",
        r"^date$",
    ], _h_date, ["What's today's date?", "What day is it?"], priority=9),

    Intent("calculate", [
        r"(?:calculate|compute|what(?:'s| is)\s+)?(?P<expr>[\d\s]+(?:(?:plus|minus|times|divided by|multiplied by|over|to the power of|\+|\-|\*|\/|\^)[\d\s\.]+)+(?:squared|cubed)?)",
        r"\b(calculate|compute|math)\b\s+(?P<expr>.+)",
        r"what(?:'s| is)\s+(?P<expr>\d[\d\s\+\-\*\/\.\^\(\)]*(?:times|plus|minus|divided by|squared|cubed|square root)[^\?]*)",
    ], _h_calculate, ["Calculate 25 times 4", "What is 100 divided by 8?"], priority=8),

    Intent("timer", [
        r"(?:set\s+(?:a\s+)?)?timer\s+(?:for\s+)?(?:(?P<hours>\d+)\s*hours?\s*(?:and\s*)?)?(?:(?P<minutes>\d+)\s*minutes?\s*(?:and\s*)?)?(?:(?P<seconds>\d+)\s*seconds?)?",
        r"(?:count|countdown)\s+(?:from\s+)?(?:(?P<hours>\d+)\s*hours?\s*)?(?:(?P<minutes>\d+)\s*minutes?\s*)?(?:(?P<seconds>\d+)\s*seconds?)?",
    ], _h_timer, ["Set a timer for 5 minutes", "Timer for 1 hour 30 minutes"], priority=8),

    Intent("wikipedia", [
        r"(?:open\s+)?wikipedia\s+(?:for\s+|about\s+|on\s+)?(?P<topic>.+)",
        r"(?:search\s+)?wiki(?:pedia)?\s+(?P<topic>.+)",
    ], _h_wikipedia, ["Wikipedia for machine learning", "Open Wikipedia for Python"], priority=7),

    Intent("youtube", [
        r"(?:play|search youtube for|find on youtube|youtube)\s+(?P<query>.+)",
        r"(?:open\s+)?youtube\s+(?:search\s+)?(?:for\s+)?(?P<query>.+)",
    ], _h_youtube, ["Play lo-fi music on YouTube", "Search YouTube for Python tutorials"], priority=7),

    Intent("weather", [
        r"(?:weather|forecast|temperature)\s+(?:in|for|at)\s+(?P<city>[a-z\s]+)",
        r"(?:what'?s?\s+)?(?:the\s+)?weather(?:\s+like)?\s+(?:in|at)\s+(?P<city>[a-z\s]+)",
        r"\b(weather|forecast)\b(?!\s+(?:in|for|at))",
    ], _h_weather, ["Weather in London", "What's the weather in Tokyo?"], priority=7),

    Intent("open_app", [
        r"(?:open|launch|start|run)\s+(?P<app>[a-zA-Z][a-z\s_-]+?)(?:\s+(?:app|application|program))?\s*$",
        r"^(?P<app>calculator|notepad|browser|terminal|spotify|vscode|files|file\s+manager|chrome|firefox)(?:\s+(?:please|now))?\s*$",
    ], _h_open_app, ["Open calculator", "Launch terminal", "Open browser"], priority=7),

    Intent("web_search", [
        r"(?:search(?:\s+(?:the\s+)?web)?(?:\s+for)?|google(?:\s+for)?|look\s+up|find)\s+(?P<query>.+)",
        r"(?P<query>.+)\s+(?:search|google|look up)\s*$",
    ], _h_web_search, ["Search for Python tutorials", "Google the weather", "Look up AI news"], priority=6),

    Intent("reminder", [
        r"(?:set\s+(?:a\s+)?)?reminder\s+(?:to\s+|for\s+)?(?P<task>.+)",
        r"remind\s+me\s+(?:to\s+)?(?P<task>.+)",
    ], _h_reminder, ["Remind me to call John", "Set a reminder to drink water"], priority=6),

    Intent("note", [
        r"(?:take\s+(?:a\s+)?note|note\s+down|write\s+down|make\s+(?:a\s+)?note)(?:\s+that)?\s+(?P<content>.+)",
        r"note\s*:\s*(?P<content>.+)",
    ], _h_note, ["Take a note: buy groceries", "Note down: meeting at 3pm"], priority=6),

    Intent("volume", [
        r"(?P<action>volume\s+up|increase\s+volume|louder|turn\s+it\s+up)",
        r"(?P<action>volume\s+down|decrease\s+volume|quieter|turn\s+it\s+down)",
        r"\b(?P<action>mute|unmute)\b(?:\s+(?:the\s+)?(?:volume|audio|sound))?",
    ], _h_volume, ["Volume up", "Mute", "Decrease volume"], priority=6),

    Intent("screenshot", [
        r"\b(take\s+(?:a\s+)?screenshot|screenshot|capture\s+(?:the\s+)?screen)\b",
    ], _h_screenshot, ["Take a screenshot"], priority=6),

    Intent("system_info", [
        r"\b(system\s+info|system\s+information|about\s+(?:this\s+)?(?:system|computer|device)|what(?:'s| is)\s+(?:my\s+)?os)\b",
    ], _h_system_info, ["System info", "What's my OS?"], priority=5),

    # ── Small talk ────────────────────────────────────────────────────────────

    Intent("samples", [
        r"\b(sample\s+commands?|example\s+commands?|what\s+can\s+you\s+do|show\s+commands?)\b",
    ], _h_samples, ["Sample commands", "What can you do?"], priority=5),

    Intent("help", [
        r"\b(help|commands?|instructions?|guide)\b",
    ], _h_help, ["Help", "Show commands"], priority=5),

    Intent("joke", [
        r"\b(tell\s+(?:me\s+)?(?:a\s+)?joke|make\s+me\s+laugh|something\s+funny|joke)\b",
    ], _h_joke, ["Tell me a joke"], priority=4),

    Intent("how_are_you", [
        r"\b(how\s+are\s+you|how(?:'re|\s+are)\s+you\s+doing|you\s+okay|feeling\s+good)\b",
    ], _h_how_are_you, ["How are you?"], priority=4),

    Intent("name", [
        r"\b(what(?:'s| is)\s+your\s+name|who\s+are\s+you|your\s+name|introduce\s+yourself)\b",
    ], _h_name, ["What's your name?", "Who are you?"], priority=4),

    Intent("greeting", [
        r"^\s*(?:hi|hello|hey|howdy|yo|sup|greetings|good\s+(?:morning|afternoon|evening|day))\s*[!.]?\s*$",
    ], _h_greeting, ["Hello", "Hi", "Hey"], priority=4),

    Intent("goodbye", [
        r"\b(bye|goodbye|exit|quit|farewell|see\s+you|later|stop\s+(?:listening|running))\b",
    ], _h_goodbye, ["Goodbye", "Bye", "Exit"], priority=3),
]


# ══════════════════════════════════════════════════════════════════════════════
#  Classifier
# ══════════════════════════════════════════════════════════════════════════════

class CommandEngine:
    """
    Main entry point.  Call .process(text) to get a Response.
    """

    def __init__(self):
        self._intents = sorted(INTENTS, key=lambda i: -i.priority)
        self.history: list[ConversationTurn] = []

    def process(self, text: str) -> Response:
        """Classify text → Response."""
        text = text.strip()
        if not text:
            return Response(text="I didn't catch anything. Please try again.",
                            intent="empty", success=False)

        # Log user turn
        self.history.append(ConversationTurn("user", text))

        resp = self._classify(text)

        # Log assistant turn
        self.history.append(ConversationTurn("assistant", resp.text, resp.intent))
        return resp

    def _classify(self, text: str) -> Response:
        for intent in self._intents:
            m = intent.match(text)
            if m:
                try:
                    return intent.handler(m, text)
                except Exception as e:
                    return Response(
                        text=f"I encountered an error handling that command: {e}. Please try again.",
                        intent=intent.name,
                        success=False,
                        action_taken=f"Handler error: {e}",
                    )
        return _h_unknown(text)

    def all_examples(self) -> dict[str, list[str]]:
        """Return {intent_name: [example, ...]} for all intents."""
        return {i.name: i.examples for i in self._intents if i.examples}
