"""
test_assistant.py
─────────────────
Tests for the command engine — no audio hardware needed.
Run:  python test_assistant.py
      pytest test_assistant.py -v  (if pytest available)
"""

from __future__ import annotations

import datetime
import math
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))
from command_engine import CommandEngine, Response, _h_unknown


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def make_engine() -> CommandEngine:
    return CommandEngine()


def check(engine: CommandEngine, text: str, expected_intent: str,
          *, expect_success: bool = True, desc: str = "") -> tuple[bool, str]:
    """
    Process `text` and verify the response intent matches `expected_intent`.
    Returns (passed: bool, message: str).
    """
    resp = engine.process(text)
    passed = (resp.intent == expected_intent) and (resp.success == expect_success)
    label = desc or f"'{text}' → intent='{expected_intent}'"
    if not passed:
        msg = f"FAIL  {label}\n       got intent='{resp.intent}' success={resp.success}"
    else:
        msg = f"PASS  {label}"
    return passed, msg


# ══════════════════════════════════════════════════════════════════════════════
#  Test cases
# ══════════════════════════════════════════════════════════════════════════════

CASES: list[tuple[str, str, bool, str]] = [
    # (input, expected_intent, expect_success, description)

    # ── Time / Date ────────────────────────────────────────────────────────────
    ("what time is it",                "time",      True,  "basic time query"),
    ("What's the time?",               "time",      True,  "capitalised time query"),
    ("current time",                   "time",      True,  "current time"),
    ("What's today's date?",           "date",      True,  "basic date query"),
    ("what day is it",                 "date",      True,  "day query"),
    ("What's the date and time?",      "datetime",  True,  "combined datetime"),

    # ── Calculations ──────────────────────────────────────────────────────────
    ("calculate 10 plus 5",            "calculate", True,  "calculate plus"),
    ("what is 20 minus 3",             "calculate", True,  "calculate minus"),
    ("compute 6 times 7",              "calculate", True,  "compute times"),
    ("calculate 100 divided by 4",     "calculate", True,  "calculate divided by"),
    ("what is 9 plus 9",               "calculate", True,  "what is addition"),
    ("calculate 0 plus 0",             "calculate", True,  "zero calculation"),

    # ── Web search ────────────────────────────────────────────────────────────
    ("search for Python tutorials",    "web_search", True, "search for query"),
    ("google machine learning",        "web_search", True, "google query"),
    ("look up best pizza in NYC",      "web_search", True, "look up query"),
    ("find the latest news",           "web_search", True, "find query"),

    # ── Wikipedia ─────────────────────────────────────────────────────────────
    ("wikipedia for neural networks",  "wikipedia", True,  "wikipedia for topic"),
    ("open Wikipedia for Python",      "wikipedia", True,  "open wikipedia"),
    ("wiki artificial intelligence",   "wikipedia", True,  "wiki shorthand"),

    # ── YouTube ───────────────────────────────────────────────────────────────
    ("play lo-fi music on YouTube",    "youtube",   True,  "play youtube"),
    ("youtube Python tutorial",        "youtube",   True,  "youtube direct"),
    ("search YouTube for coding",      "youtube",   True,  "search youtube"),

    # ── Weather ───────────────────────────────────────────────────────────────
    ("weather in London",              "weather",   True,  "weather in city"),
    ("What's the weather in Tokyo?",   "weather",   True,  "what's weather"),
    ("forecast for Paris",             "weather",   True,  "forecast for city"),

    # ── Open app ──────────────────────────────────────────────────────────────
    ("open calculator",                "open_app",  True,  "open calculator"),
    ("launch terminal",                "open_app",  True,  "launch terminal"),
    ("start browser",                  "open_app",  True,  "start browser"),
    ("open notepad",                   "open_app",  True,  "open notepad"),
    ("open unknown_app_xyz",           "open_app",  False, "unknown app → failure"),

    # ── Timer ─────────────────────────────────────────────────────────────────
    ("set a timer for 5 minutes",      "timer",     True,  "timer minutes"),
    ("timer for 1 hour",               "timer",     True,  "timer hours"),
    ("timer for 30 seconds",           "timer",     True,  "timer seconds"),
    ("set a timer for 1 hour 30 minutes", "timer",  True,  "timer hours+minutes"),

    # ── Reminder ──────────────────────────────────────────────────────────────
    ("remind me to call John",         "reminder",  True,  "remind me to"),
    ("set a reminder to drink water",  "reminder",  True,  "set a reminder"),

    # ── Notes ─────────────────────────────────────────────────────────────────
    ("take a note: buy groceries",     "note",      True,  "take a note"),
    ("write down meeting at 3pm",      "note",      True,  "write down"),
    ("note: call dentist",             "note",      True,  "note colon"),

    # ── Volume ────────────────────────────────────────────────────────────────
    ("volume up",                      "volume",    True,  "volume up"),
    ("volume down",                    "volume",    True,  "volume down"),
    ("mute",                           "volume",    True,  "mute"),
    ("increase volume",                "volume",    True,  "increase volume"),

    # ── System info ───────────────────────────────────────────────────────────
    ("system info",                    "system_info", True, "system info"),
    ("what's my OS",                   "system_info", True, "what's my OS"),

    # ── Small talk ────────────────────────────────────────────────────────────
    ("hello",                          "greeting",  True,  "hello"),
    ("hey",                            "greeting",  True,  "hey"),
    ("how are you",                    "small_talk",True,  "how are you"),
    ("what's your name",               "name",      True,  "what's your name"),
    ("tell me a joke",                 "joke",      True,  "tell me a joke"),
    ("help",                           "help",      True,  "help"),
    ("sample commands",                "samples",   True,  "sample commands"),
    ("goodbye",                        "goodbye",   True,  "goodbye"),
    ("bye",                            "goodbye",   True,  "bye"),
    ("exit",                           "goodbye",   True,  "exit"),

    # ── Unknown / fallback ────────────────────────────────────────────────────
    ("asdfghjkl completely nonsense",  "unknown",   False, "nonsense → unknown"),
    ("purple banana flying elephant",  "unknown",   False, "random words → unknown"),
]


# ══════════════════════════════════════════════════════════════════════════════
#  Response quality checks
# ══════════════════════════════════════════════════════════════════════════════

def test_response_fields(engine: CommandEngine) -> tuple[bool, str]:
    """Every response must have non-empty text and valid confidence."""
    inputs = ["hello", "what time is it", "calculate 5 plus 3", "asdf"]
    for text in inputs:
        resp = engine.process(text)
        if not resp.text:
            return False, f"Empty text for input: '{text}'"
        if not (0.0 <= resp.confidence <= 1.0):
            return False, f"Confidence {resp.confidence} out of range for '{text}'"
        if not resp.intent:
            return False, f"Empty intent for '{text}'"
    return True, "Response field validation"


def test_time_response_contains_time(engine: CommandEngine) -> tuple[bool, str]:
    """Time response should mention AM or PM."""
    resp = engine.process("what time is it")
    if "AM" not in resp.text and "PM" not in resp.text:
        return False, f"Time response missing AM/PM: '{resp.text}'"
    return True, "Time response contains AM/PM"


def test_date_response_contains_year(engine: CommandEngine) -> tuple[bool, str]:
    """Date response should contain the current year."""
    resp = engine.process("what's today's date")
    year = str(datetime.datetime.now().year)
    if year not in resp.text:
        return False, f"Date response missing year {year}: '{resp.text}'"
    return True, "Date response contains current year"


def test_calc_arithmetic(engine: CommandEngine) -> tuple[bool, str]:
    """Test several arithmetic results."""
    cases = [
        ("calculate 10 plus 5",      "15"),
        ("calculate 10 minus 3",      "7"),
        ("what is 6 times 7",        "42"),
        ("calculate 100 divided by 4", "25"),
    ]
    for expr, expected in cases:
        resp = engine.process(expr)
        if expected not in resp.text:
            return False, f"Calc '{expr}': expected '{expected}' in '{resp.text}'"
    return True, "Arithmetic calculations correct"


def test_history_logged(engine: CommandEngine) -> tuple[bool, str]:
    """Engine must log user + assistant turns."""
    e = make_engine()
    e.process("hello")
    e.process("what time is it")
    if len(e.history) != 4:  # 2 user + 2 assistant
        return False, f"Expected 4 history entries, got {len(e.history)}"
    if e.history[0].speaker != "user":
        return False, "First history entry should be 'user'"
    if e.history[1].speaker != "assistant":
        return False, "Second history entry should be 'assistant'"
    return True, "Conversation history logging"


def test_response_uniqueness(engine: CommandEngine) -> tuple[bool, str]:
    """Repeated greetings should sometimes vary (randomisation check)."""
    e = make_engine()
    responses = {e.process("hello").text for _ in range(10)}
    # At least 2 distinct greetings in 10 tries (we have 4 variants)
    if len(responses) < 2:
        return False, f"Greeting responses not varied: {responses}"
    return True, "Response randomisation (greetings)"


def test_empty_input(engine: CommandEngine) -> tuple[bool, str]:
    """Empty input should return a graceful failure."""
    resp = engine.process("")
    if resp.success:
        return False, "Empty input should not succeed"
    if not resp.text:
        return False, "Empty input should still return text"
    return True, "Empty input handled gracefully"


def test_whitespace_input(engine: CommandEngine) -> tuple[bool, str]:
    """Whitespace-only input treated like empty."""
    resp = engine.process("   \t  ")
    if resp.success:
        return False, "Whitespace input should not succeed"
    return True, "Whitespace input handled gracefully"


def test_all_intents_have_examples() -> tuple[bool, str]:
    """Every intent (except fallback/unknown) should have examples."""
    from command_engine import INTENTS
    missing = [i.name for i in INTENTS if not i.examples and i.name != "fallback"]
    if missing:
        return False, f"Intents missing examples: {missing}"
    return True, "All intents have example commands"


def test_timer_duration_parsing(engine: CommandEngine) -> tuple[bool, str]:
    """Timer action_taken should mention the correct seconds."""
    cases = [
        ("set a timer for 5 minutes",      300),
        ("timer for 2 minutes 30 seconds", 150),
        ("set a timer for 1 hour",        3600),
    ]
    for text, expected_secs in cases:
        resp = engine.process(text)
        if resp.intent != "timer":
            return False, f"'{text}' → expected timer, got {resp.intent}"
        if str(expected_secs) not in resp.action_taken:
            return False, f"Timer '{text}': expected {expected_secs}s in action '{resp.action_taken}'"
    return True, "Timer duration parsing"


def test_engine_never_raises() -> tuple[bool, str]:
    """The engine must never raise an exception on any input."""
    e = make_engine()
    evil_inputs = [
        "",  "  ", "\n", "\t",
        "a" * 1000,
        "calculate !!!!!",
        "😀🎤🔊",
        "open " + "x" * 200,
        "<script>alert('xss')</script>",
        "'; DROP TABLE intents; --",
        "calculate __import__('os').system('echo HACKED')",
    ]
    for text in evil_inputs:
        try:
            resp = e.process(text)
            assert resp is not None
            assert isinstance(resp.text, str)
        except Exception as ex:
            return False, f"Engine raised on input '{text[:40]}': {ex}"
    return True, "Engine never raises on adversarial inputs"


# ══════════════════════════════════════════════════════════════════════════════
#  Runner
# ══════════════════════════════════════════════════════════════════════════════

def run_all():
    engine = make_engine()
    results = []

    # ── Intent classification cases ───────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  AXIOM Assistant — Test Suite")
    print(f"{'═'*60}\n")
    print("  Intent Classification Tests")
    print(f"  {'─'*56}")

    for text, intent, success, desc in CASES:
        passed, msg = check(engine, text, intent, expect_success=success, desc=desc)
        sym  = "✓" if passed else "✗"
        col  = ""  # no colors in test output by default
        print(f"  {sym}  {msg}")
        results.append(passed)

    # ── Quality tests ─────────────────────────────────────────────────────────
    print(f"\n  Response Quality Tests")
    print(f"  {'─'*56}")

    quality_tests = [
        test_response_fields(engine),
        test_time_response_contains_time(engine),
        test_date_response_contains_year(engine),
        test_calc_arithmetic(engine),
        test_history_logged(engine),
        test_response_uniqueness(engine),
        test_empty_input(engine),
        test_whitespace_input(engine),
        test_all_intents_have_examples(),
        test_timer_duration_parsing(engine),
        test_engine_never_raises(),
    ]

    for passed, desc in quality_tests:
        sym = "✓" if passed else "✗"
        print(f"  {sym}  {'PASS' if passed else 'FAIL'}  {desc}")
        results.append(passed)

    # ── Summary ───────────────────────────────────────────────────────────────
    total  = len(results)
    passed = sum(results)
    failed = total - passed

    print(f"\n{'═'*60}")
    print(f"  Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} failed)")
    else:
        print("  — All tests passed! ✓")
    print(f"{'═'*60}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all())
