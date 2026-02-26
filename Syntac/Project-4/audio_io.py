"""
audio_io.py
───────────
Speech-to-Text and Text-to-Speech adapters.

Each adapter exposes a simple interface:
  STTAdapter.listen()  → str   (transcribed text, or "" on failure)
  TTSAdapter.speak(text)       (blocking until spoken)

Concrete implementations:
  RealSTT   — uses SpeechRecognition + Google / Sphinx
  RealTTS   — tries pyttsx3, falls back to gTTS + playsound
  TerminalSTT / TerminalTTS — keyboard + print fallbacks (zero dependencies)

The assistant auto-selects the best available adapter at startup.
"""

from __future__ import annotations

import io
import os
import sys
import threading
import time
from abc import ABC, abstractmethod


# ══════════════════════════════════════════════════════════════════════════════
#  Base classes
# ══════════════════════════════════════════════════════════════════════════════

class STTAdapter(ABC):
    @abstractmethod
    def listen(self) -> str:
        """Block until a voice command is captured. Return transcribed text."""
        ...

    @property
    @abstractmethod
    def name(self) -> str: ...


class TTSAdapter(ABC):
    @abstractmethod
    def speak(self, text: str) -> None:
        """Convert text to speech and play it (blocking)."""
        ...

    @property
    @abstractmethod
    def name(self) -> str: ...


# ══════════════════════════════════════════════════════════════════════════════
#  Terminal (fallback) adapters — zero external dependencies
# ══════════════════════════════════════════════════════════════════════════════

class TerminalSTT(STTAdapter):
    """Read a command from stdin (keyboard input)."""

    @property
    def name(self) -> str:
        return "Keyboard (Terminal)"

    def listen(self) -> str:
        try:
            text = input()
            return text.strip()
        except (EOFError, KeyboardInterrupt):
            return "exit"


class TerminalTTS(TTSAdapter):
    """Print the response instead of speaking it."""

    @property
    def name(self) -> str:
        return "Print (Terminal)"

    def speak(self, text: str) -> None:
        # Already printed by the main UI — nothing extra needed
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  Real STT — SpeechRecognition
# ══════════════════════════════════════════════════════════════════════════════

class RealSTT(STTAdapter):
    """
    Uses the `SpeechRecognition` library with Google Speech-to-Text.
    Falls back to CMU Sphinx for offline recognition if available.

    Install:  pip install SpeechRecognition pyaudio
    """

    def __init__(self, timeout: int = 5, phrase_limit: int = 10,
                 language: str = "en-US", energy_threshold: int = 4000):
        import speech_recognition as sr
        self._sr       = sr
        self._rec      = sr.Recognizer()
        self._rec.energy_threshold    = energy_threshold
        self._rec.dynamic_energy_threshold = True
        self._timeout      = timeout
        self._phrase_limit = phrase_limit
        self._language     = language
        self._mic          = sr.Microphone()
        # Warm up
        with self._mic as source:
            self._rec.adjust_for_ambient_noise(source, duration=0.5)

    @property
    def name(self) -> str:
        return "SpeechRecognition (Google STT)"

    def listen(self) -> str:
        sr = self._sr
        print("\n  🎙  Listening...", flush=True)
        try:
            with self._mic as source:
                audio = self._rec.listen(
                    source,
                    timeout=self._timeout,
                    phrase_time_limit=self._phrase_limit,
                )
        except sr.WaitTimeoutError:
            return ""

        # Try Google first, fall back to Sphinx
        for engine in ("google", "sphinx"):
            try:
                if engine == "google":
                    text = self._rec.recognize_google(audio, language=self._language)
                else:
                    text = self._rec.recognize_sphinx(audio)
                return text.strip()
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                if engine == "google":
                    continue   # try sphinx
                return ""

        return ""


# ══════════════════════════════════════════════════════════════════════════════
#  Real TTS — pyttsx3 → gTTS fallback chain
# ══════════════════════════════════════════════════════════════════════════════

class Pyttsx3TTS(TTSAdapter):
    """
    Offline TTS using pyttsx3.
    Install:  pip install pyttsx3
    """

    def __init__(self, rate: int = 175, volume: float = 1.0, voice_index: int = 0):
        import pyttsx3
        self._engine = pyttsx3.init()
        self._engine.setProperty("rate",   rate)
        self._engine.setProperty("volume", volume)
        voices = self._engine.getProperty("voices")
        if voices and voice_index < len(voices):
            self._engine.setProperty("voice", voices[voice_index].id)

    @property
    def name(self) -> str:
        return "pyttsx3 (offline TTS)"

    def speak(self, text: str) -> None:
        self._engine.say(text)
        self._engine.runAndWait()


class GttsTTS(TTSAdapter):
    """
    Online TTS using gTTS + playsound (or mpg123 on Linux).
    Install:  pip install gTTS playsound
    """

    def __init__(self, lang: str = "en", slow: bool = False):
        self._lang = lang
        self._slow = slow

    @property
    def name(self) -> str:
        return "gTTS (Google TTS)"

    def speak(self, text: str) -> None:
        from gtts import gTTS
        import tempfile

        tts = gTTS(text=text, lang=self._lang, slow=self._slow)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp = f.name
        try:
            tts.save(tmp)
            self._play(tmp)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def _play(self, path: str):
        import subprocess, sys
        if sys.platform == "linux":
            for player in ("mpg123", "mpg321", "ffplay", "aplay"):
                try:
                    subprocess.run([player, "-q", path], check=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return
                except (FileNotFoundError, subprocess.CalledProcessError):
                    continue
        elif sys.platform == "darwin":
            subprocess.run(["afplay", path], check=True)
        else:  # Windows
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME)


# ══════════════════════════════════════════════════════════════════════════════
#  Auto-detection factory
# ══════════════════════════════════════════════════════════════════════════════

def make_stt(force_terminal: bool = False) -> tuple[STTAdapter, bool]:
    """
    Returns (adapter, is_real_microphone).
    Tries RealSTT first; falls back to TerminalSTT.
    """
    if force_terminal:
        return TerminalSTT(), False

    try:
        adapter = RealSTT()
        return adapter, True
    except ImportError:
        pass
    except Exception:
        pass

    return TerminalSTT(), False


def make_tts(force_terminal: bool = False) -> tuple[TTSAdapter, bool]:
    """
    Returns (adapter, is_real_speech).
    Tries pyttsx3 → gTTS → TerminalTTS.
    """
    if force_terminal:
        return TerminalTTS(), False

    for cls in (Pyttsx3TTS, GttsTTS):
        try:
            adapter = cls()
            return adapter, True
        except (ImportError, Exception):
            continue

    return TerminalTTS(), False
