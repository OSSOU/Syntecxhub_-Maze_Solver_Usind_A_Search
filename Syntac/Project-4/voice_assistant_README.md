# AXIOM — Personal Voice Assistant
### Python 3.8+ · MediaPipe-free · Modular STT/TTS

```
voice_assistant/
├── assistant.py        # Main loop + beautiful terminal UI
├── command_engine.py   # Intent classifier + action handlers (no audio deps)
├── audio_io.py         # STT/TTS adapters (real mic + fallback)
├── test_assistant.py   # 68 unit tests — camera/mic free
└── README.md
```

---

## Quick Start

```bash
# Minimal — keyboard input, printed responses (zero extra deps)
python assistant.py --text --silent

# With microphone only (needs SpeechRecognition + PyAudio)
pip install SpeechRecognition pyaudio
python assistant.py --silent

# With voice output only (needs pyttsx3)
pip install pyttsx3
python assistant.py --text

# Full voice I/O
pip install SpeechRecognition pyaudio pyttsx3
python assistant.py

# Save conversation log on exit
python assistant.py --text --log

# Run tests (no mic/speaker needed)
python test_assistant.py
```

---

## CLI Options

```
python assistant.py [OPTIONS]

  --text     Use keyboard input instead of microphone
  --silent   Print responses instead of speaking them
  --log      Save conversation JSON log on exit
  --lang     STT language code (default: en-US)
  --help     Show this help
```

---

## Supported Commands

| Intent | Example Commands |
|---|---|
| **Time** | "What time is it?", "Current time" |
| **Date** | "What's today's date?", "What day is it?" |
| **Date+Time** | "What's the date and time?" |
| **Search Web** | "Search for Python tutorials", "Google AI news" |
| **Wikipedia** | "Wikipedia for machine learning", "Open wiki for Python" |
| **YouTube** | "Play lo-fi music on YouTube", "Search YouTube for coding" |
| **Weather** | "Weather in London", "What's the weather in Tokyo?" |
| **Open App** | "Open calculator", "Launch terminal", "Start browser" |
| **Calculate** | "Calculate 25 times 4", "What is 100 divided by 8?" |
| **Timer** | "Set a timer for 5 minutes", "Timer for 1 hour 30 minutes" |
| **Reminder** | "Remind me to call John", "Set a reminder to drink water" |
| **Note** | "Take a note: buy groceries", "Write down meeting at 3pm" |
| **Volume** | "Volume up", "Mute", "Decrease volume" |
| **Screenshot** | "Take a screenshot" |
| **System Info** | "System info", "What's my OS?" |
| **Jokes** | "Tell me a joke" |
| **Help** | "Help", "Sample commands", "What can you do?" |
| **Goodbye** | "Goodbye", "Bye", "Exit" |

---

## Architecture

### Adapter Pattern — Zero-dependency core

The command engine is **completely decoupled from audio**. It takes a plain string and returns a `Response` object. This means:
- The entire logic layer is testable without a microphone
- Swapping STT/TTS engines requires changing one line

```
User speaks / types
       │
       ▼
  STTAdapter.listen()          ← RealSTT (SpeechRecognition)
       │                          or TerminalSTT (keyboard)
       ▼
  CommandEngine.process(text)  ← pure Python, no audio deps
       │
       ▼
  Response { text, intent, confidence, action_taken }
       │
       ▼
  TTSAdapter.speak(text)       ← Pyttsx3TTS / GttsTTS
                                  or TerminalTTS (print)
```

### Intent Classification

Rules are evaluated in priority order (highest first). Each `Intent` holds:
- A list of compiled regex patterns
- A handler function `(re.Match, raw_text) → Response`
- A priority integer (higher = checked first)
- Example commands (used in help + tests)

The classifier returns on the **first match**, so specificity (datetime > date > time) is controlled entirely by priority numbers — no ambiguity resolution needed.

### STT Adapter Chain

```python
make_stt()  →  RealSTT (SpeechRecognition + Google)
            →  RealSTT (SpeechRecognition + Sphinx offline)
            →  TerminalSTT (keyboard fallback)
```

`RealSTT` automatically adjusts for ambient noise on startup and applies dynamic energy thresholding for robustness in noisy environments.

### TTS Adapter Chain

```python
make_tts()  →  Pyttsx3TTS  (offline, system voices)
            →  GttsTTS      (Google TTS, online)
            →  TerminalTTS  (print fallback)
```

### Timer Threading

Timers run in a daemon thread — they don't block the main loop. When the timer expires, the assistant prints and speaks the follow-up message regardless of what else is happening.

```python
# In assistant.py — fire-and-forget timer
threading.Thread(
    target=lambda: (time.sleep(secs), tts.speak(followup)),
    daemon=True
).start()
```

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| No speech detected (timeout) | Prints "Didn't catch that", loops |
| Unknown command | Graceful fallback with suggestions |
| App not installed | Friendly error, `success=False` |
| Calculator syntax error | Caught, human-readable message |
| TTS failure | Logged as warning, loop continues |
| STT network error | Falls back to CMU Sphinx (offline) |
| KeyboardInterrupt (Ctrl+C) | Clean exit, optional log save |
| Empty / whitespace input | Handled without crash |
| Adversarial inputs (SQL injection, eval injection) | Safe — eval restricted to math namespace |

---

## Extending the Assistant

### Add a new command

```python
# 1. Write a handler in command_engine.py
def _h_flip_coin(m, raw):
    import random
    result = random.choice(["Heads!", "Tails!"])
    return Response(text=f"I flipped a coin: {result}", intent="flip_coin")

# 2. Register the intent
Intent("flip_coin",
    patterns=[r"\b(flip\s+(?:a\s+)?coin|coin\s+flip|heads\s+or\s+tails)\b"],
    handler=_h_flip_coin,
    examples=["Flip a coin", "Heads or tails?"],
    priority=5,
)
```

That's it — no other changes needed.

### Swap the STT engine (e.g. Whisper)

```python
class WhisperSTT(STTAdapter):
    def __init__(self):
        import whisper
        self._model = whisper.load_model("base")

    @property
    def name(self): return "Whisper (local)"

    def listen(self) -> str:
        import sounddevice as sd, numpy as np
        audio = sd.rec(int(5 * 16000), samplerate=16000, channels=1)
        sd.wait()
        result = self._model.transcribe(audio.flatten())
        return result["text"].strip()
```

Then in `assistant.py`:
```python
stt = WhisperSTT()
```

---

## Requirements

```
# Core (zero extra deps — keyboard + print mode)
Python 3.8+

# For microphone input
SpeechRecognition>=3.10
PyAudio>=0.2.13          # or pipwin install pyaudio on Windows

# For offline TTS
pyttsx3>=2.90

# For online TTS (fallback)
gTTS>=2.3
```

---

*Built with Python · SpeechRecognition · pyttsx3 · gTTS*
