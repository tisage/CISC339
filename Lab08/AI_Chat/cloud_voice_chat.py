#!/usr/bin/env python3
"""
Lab 08 — Cloud LLM Voice Chat Demo  (Groq edition)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STT : Groq Whisper API  (whisper-large-v3-turbo, cloud)
LLM : Groq API          (llama-3.3-70b-versatile, cloud)
TTS : kokoro-onnx       (Kokoro-82M, English + W. European, local)
      → fallback: macOS say / Windows pyttsx3

TESTED ON: macOS (Apple Silicon).  Windows should work via fallbacks.

WHY GROQ?
  • Free tier — no credit card required
  • Students sign up at https://console.groq.com  →  API Keys  →  Create
  • 14 400 requests / day  ·  Each student gets their own key
  • Llama 3.3 70B is stronger than Gemma 4 E4B and runs at 300+ tok/s
  • Groq Whisper endpoint handles STT — no local Whisper install needed

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SETUP  (any laptop — no GPU needed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  pip install groq kokoro-onnx sounddevice soundfile numpy

  # TTS model (kokoro):
  #   place kokoro.onnx  in  ~/.cache/kokoro_onnx/   (macOS / Linux)
  #   place kokoro.onnx  in  %USERPROFILE%\\.cache\\kokoro_onnx\\  (Windows)
  #   download from https://huggingface.co/onnx-community/Kokoro-82M-ONNX

  # Groq API key — set as environment variable:
  #   export GROQ_API_KEY="gsk_..."         (macOS / Linux)
  #   set    GROQ_API_KEY=gsk_...           (Windows CMD)
  #   Or just paste it at the prompt when the script asks.

Run:
  python cloud_voice_chat.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import pathlib
import platform
import subprocess
import tempfile
import textwrap
import threading
import time
import wave
from contextlib import contextmanager
from typing import Optional

# ── ANSI colours ──────────────────────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    CYAN   = "\033[36m"
    BCYAN  = "\033[96m"
    BGREEN = "\033[92m"
    YELLOW = "\033[33m"
    RED    = "\033[31m"
    GRAY   = "\033[90m"
    GREEN  = "\033[32m"

if platform.system() == "Windows":
    os.system("color")

@contextmanager
def spinner(msg: str):
    stop   = threading.Event()
    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    def _spin():
        i = 0
        while not stop.is_set():
            print(f"\r  {C.CYAN}{frames[i % len(frames)]}{C.RESET}  {C.DIM}{msg}{C.RESET}",
                  end="", flush=True)
            time.sleep(0.08)
            i += 1
    t = threading.Thread(target=_spin, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join()
        print(f"\r  {C.BGREEN}✓{C.RESET}  {C.DIM}{msg}{C.RESET}" + " " * 10)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

GROQ_LLM_MODEL = "llama-3.3-70b-versatile"   # change to "gemma2-9b-it" etc. if preferred
GROQ_STT_MODEL = "whisper-large-v3-turbo"
SAMPLE_RATE    = 16_000   # Hz — Whisper works best at 16 kHz

SYSTEM_PROMPT = """\
You are an AI assistant running in the cloud and being demonstrated live in an
undergraduate Artificial Intelligence course (CISC 339).

Your personality:
- Curious and enthusiastic about AI topics
- Witty and occasionally humorous, but never at the expense of clarity
- Concise: aim for 2–4 sentences unless asked to elaborate
- Honest about your limitations

When students ask about AI concepts (neural networks, LLMs, training, etc.),
give a clear intuitive explanation first, then add depth if asked.

Always reply in the same language the user is speaking.\
"""

# Kokoro voice map  { lang_detect_code : (voice, onnx_lang_code) }
KOKORO_VOICES: dict[str, tuple[str, str]] = {
    "en": ("af_heart",  "en-us"),
    "es": ("ef_dora",   "es"),
    "fr": ("ff_siwis",  "fr-fr"),
    "it": ("if_sara",   "it"),
    "pt": ("pf_dora",   "pt-br"),
}
KOKORO_DEFAULT = ("af_heart", "en-us")

SAY_VOICES = {
    "en": "Ava", "es": "Monica", "fr": "Thomas",
    "zh": "Tingting", "ja": "Kyoko",
}

def detect_language(text: str) -> str:
    if not text:
        return "en"
    n = len(text)
    counts = {
        "zh": sum(1 for c in text if "\u4e00" <= c <= "\u9fff"),
        "ja": sum(1 for c in text if "\u3040" <= c <= "\u30ff"),
        "ko": sum(1 for c in text if "\uac00" <= c <= "\ud7a3"),
    }
    for lang, cnt in counts.items():
        if cnt / n > 0.08:
            return lang
    return "en"   # default; Spanish / French detected from context not script

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Groq Backend  (STT + LLM)
# ─────────────────────────────────────────────────────────────────────────────

class GroqBackend:
    """
    Uses the Groq Python client for both:
      • Speech-to-text  : Whisper-large-v3-turbo
      • Chat completion : Llama 3.3 70B (or any other Groq model)
    """

    def __init__(self, api_key: str):
        try:
            from groq import Groq
        except ImportError:
            print("  [Groq] Install: pip install groq")
            raise

        self._client = Groq(api_key=api_key)
        # Quick connectivity check
        try:
            self._client.models.list()
            print(f"  {C.BGREEN}✓{C.RESET}  {C.DIM}Groq API connected{C.RESET}")
        except Exception as e:
            raise RuntimeError(f"Groq API key invalid or no internet: {e}") from e

    def transcribe(self, wav_path: str) -> str:
        """Convert audio file to text via Groq Whisper."""
        with open(wav_path, "rb") as f:
            result = self._client.audio.transcriptions.create(
                model=GROQ_STT_MODEL,
                file=("audio.wav", f, "audio/wav"),
                response_format="text",
            )
        text = result.strip() if isinstance(result, str) else result.text.strip()
        if text:
            print(f"  {C.GRAY}[you said]  {text}{C.RESET}")
        return text

    def chat(self, user_text: str, history: Optional[list] = None) -> str:
        """Send a message with conversation history, return reply text."""
        history = history or []
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] \
                 + history \
                 + [{"role": "user", "content": user_text}]
        completion = self._client.chat.completions.create(
            model=GROQ_LLM_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=512,
        )
        return completion.choices[0].message.content.strip()

# ─────────────────────────────────────────────────────────────────────────────
# 2.  TTS Engine  (kokoro-onnx → say/pyttsx3)
# ─────────────────────────────────────────────────────────────────────────────

class TTSEngine:
    _CACHE_DIR = pathlib.Path.home() / ".cache" / "kokoro_onnx"

    def __init__(self):
        self.plat    = "apple" if platform.system() == "Darwin" else "windows"
        self.backend = self._init()

    def _init(self) -> str:
        try:
            from kokoro_onnx import Kokoro
            import sounddevice as sd

            model_path = next(
                (p for n in ["kokoro-v1.0.onnx", "kokoro.onnx"]
                 if (p := self._CACHE_DIR / n).exists()), None
            )
            if model_path is None:
                print(f"  {C.GRAY}[TTS] kokoro model not found in {self._CACHE_DIR}{C.RESET}")
                raise FileNotFoundError

            # Try new single-file API first, then legacy two-file API
            voices_path = next(
                (p for n in ["voices-v1.0.bin", "voices.json"]
                 if (p := self._CACHE_DIR / n).exists()), None
            )
            try:
                self._kokoro = Kokoro(str(model_path), str(voices_path)) \
                               if voices_path else Kokoro(str(model_path))
            except TypeError:
                self._kokoro = Kokoro(str(model_path))

            self._sd = sd
            print(f"  {C.BGREEN}✓{C.RESET}  {C.DIM}TTS  kokoro-onnx  ({model_path.name}){C.RESET}")
            return "kokoro"

        except (ImportError, FileNotFoundError):
            pass
        except Exception as e:
            print(f"  {C.GRAY}[TTS] kokoro error: {e}{C.RESET}")

        if self.plat == "apple":
            print(f"  {C.GRAY}[TTS] using macOS say  (install kokoro-onnx for better quality){C.RESET}")
            return "say"

        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 170)
            self._pyttsx3 = engine
            return "pyttsx3"
        except Exception:
            return "none"

    def speak(self, text: str) -> None:
        wrapped = textwrap.fill(text, width=66, initial_indent="    ", subsequent_indent="    ")
        print(f"\n  {C.CYAN}{C.BOLD}ai ›{C.RESET}\n{C.YELLOW}{wrapped}{C.RESET}\n")

        lang = detect_language(text)
        voice, onnx_lang = KOKORO_VOICES.get(lang, KOKORO_DEFAULT)

        if self.backend == "kokoro":
            try:
                samples, sr = self._kokoro.create(text, voice=voice, speed=1.0, lang=onnx_lang)
                self._sd.play(samples, sr)
                self._sd.wait()
                return
            except Exception as e:
                print(f"  {C.GRAY}[TTS] kokoro error: {e} — falling back{C.RESET}")

        if self.plat == "apple":
            say_v = SAY_VOICES.get(lang, SAY_VOICES["en"])
            subprocess.run(["say", "-v", say_v, "--", text])
        elif hasattr(self, "_pyttsx3"):
            self._pyttsx3.say(text)
            self._pyttsx3.runAndWait()

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Push-to-Talk Recording
# ─────────────────────────────────────────────────────────────────────────────

def record_push_to_talk(skip_start_prompt: bool = False) -> Optional[str]:
    """
    Record microphone audio.
      1. Press ENTER to start  (skipped when skip_start_prompt=True).
      2. Press ENTER again to stop.
    Returns path to a temporary 16-bit PCM .wav file (caller deletes it).
    """
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError:
        print("  Install: pip install sounddevice numpy")
        return None

    if not skip_start_prompt:
        print(f"  {C.GRAY}Press ENTER to start recording …{C.RESET}", end="", flush=True)
        input()

    frames: list = []
    stop_ev = threading.Event()

    def _cb(indata, _fc, _ti, _st):
        if not stop_ev.is_set():
            frames.append(indata.copy())

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=_cb)
    print(f"  {C.RED}● REC{C.RESET}  …  press ENTER to stop", end="", flush=True)
    with stream:
        input()
        stop_ev.set()

    if not frames:
        print(f"  {C.GRAY}Nothing recorded.{C.RESET}\n")
        return None

    import numpy as np
    audio    = np.concatenate(frames, axis=0)
    duration = len(audio) / SAMPLE_RATE
    print(f"  {C.GRAY}Captured {duration:.1f} s{C.RESET}")

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    tmp.close()
    return tmp.name

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Main
# ─────────────────────────────────────────────────────────────────────────────

def print_banner(api_key_source: str) -> None:
    W   = 54
    bar = f"  {C.CYAN}{'─' * W}{C.RESET}"
    def row(label, value):
        print(f"  {C.GRAY}{label:<8}{C.RESET} {value}")
    print()
    print(bar)
    print(f"  {C.BCYAN}{C.BOLD}  ◈  Cloud AI Voice Chat{C.RESET}"
          f"  {C.DIM}·  Groq + kokoro  ·  CISC 339{C.RESET}")
    print(bar)
    row("llm",    f"{GROQ_LLM_MODEL}  {C.DIM}(Groq cloud){C.RESET}")
    row("stt",    f"{GROQ_STT_MODEL}  {C.DIM}(Groq cloud){C.RESET}")
    row("tts",    "kokoro-onnx  (local)")
    row("key",    api_key_source)
    print(bar)
    print()


def main() -> None:
    # ── API key ──────────────────────────────────────────────────────────────
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if api_key:
        key_source = f"{C.DIM}from $GROQ_API_KEY{C.RESET}"
    else:
        print(f"\n  {C.GRAY}Get a free key at  https://console.groq.com  →  API Keys{C.RESET}")
        try:
            api_key = input("  Paste your Groq API key › ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Goodbye!")
            return
        if not api_key:
            print("  No key provided — exiting.")
            return
        key_source = "entered at prompt"

    print_banner(key_source)

    # ── Init ─────────────────────────────────────────────────────────────────
    try:
        with spinner("Connecting to Groq …"):
            llm = GroqBackend(api_key)
    except Exception as e:
        print(f"\n  {C.RED}✗  {e}{C.RESET}\n")
        return

    tts = TTSEngine()
    print()

    # ── Conversation state ────────────────────────────────────────────────────
    MAX_TURNS = 20
    history:  list = []

    AUDIO_PROMPT = (
        "The user sent a voice message. "
        "Listen carefully, understand it, and reply conversationally. "
        "Reply in the same language the user spoke."
    )

    SEP  = f"  {C.CYAN}{'─' * 34}{C.RESET}"
    HINT = f"{C.GRAY}m · menu   c · clear   q · quit{C.RESET}"

    def _menu():
        n = len(history) // 2
        hist_str = f"{C.DIM}{n} turn{'s' if n != 1 else ''} in memory{C.RESET}"
        print(f"\n{SEP}")
        print(f"  {C.BCYAN}{C.BOLD}t{C.RESET}  text    "
              f"{C.BCYAN}{C.BOLD}a{C.RESET}  audio   "
              f"{C.BCYAN}{C.BOLD}q{C.RESET}  quit    {hist_str}")
        print(SEP)

    # ── Outer menu loop ───────────────────────────────────────────────────────
    while True:
        _menu()
        try:
            cmd = input(f"  {C.BCYAN}›{C.RESET}  ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print(f"\n  {C.DIM}Goodbye!{C.RESET}")
            return

        if cmd == "q":
            print(f"\n  {C.DIM}Goodbye!{C.RESET}\n")
            return
        elif cmd == "c":
            history.clear()
            print(f"  {C.DIM}History cleared.{C.RESET}")
            continue
        elif cmd not in ("t", "a"):
            continue

        mode = cmd
        print(f"\n  {C.DIM}{'text' if mode == 't' else 'audio'} mode  ·  {HINT}{C.RESET}\n")

        # ── Inner conversation loop ───────────────────────────────────────────
        while True:
            turn = len(history) // 2 + 1

            if mode == "t":
                try:
                    raw = input(
                        f"  {C.GRAY}[{turn}]{C.RESET}  {C.BGREEN}you ›{C.RESET}  "
                    ).strip()
                except (KeyboardInterrupt, EOFError):
                    print(f"\n  {C.DIM}Goodbye!{C.RESET}")
                    return
                if raw.lower() == "q":
                    print(f"\n  {C.DIM}Goodbye!{C.RESET}\n")
                    return
                if raw.lower() == "m":
                    break
                if raw.lower() == "c":
                    history.clear()
                    print(f"  {C.DIM}History cleared.{C.RESET}\n")
                    continue
                if not raw:
                    continue
                with spinner("thinking"):
                    response = llm.chat(raw, history=history)
                tts.speak(response)
                history.append({"role": "user",      "content": raw})
                history.append({"role": "assistant", "content": response})

            else:  # audio
                try:
                    pre = input(
                        f"  {C.GRAY}[{turn}]{C.RESET}  "
                        f"{C.RED}↵{C.RESET} {C.DIM}record{C.RESET}  "
                        f"{C.GRAY}·  m  q › {C.RESET}"
                    ).strip().lower()
                except (KeyboardInterrupt, EOFError):
                    print(f"\n  {C.DIM}Goodbye!{C.RESET}")
                    return
                if pre == "q":
                    print(f"\n  {C.DIM}Goodbye!{C.RESET}\n")
                    return
                if pre == "m":
                    break
                if pre == "c":
                    history.clear()
                    print(f"  {C.DIM}History cleared.{C.RESET}\n")
                    continue

                wav_path = record_push_to_talk(skip_start_prompt=True)
                if wav_path is None:
                    continue

                with spinner("transcribing"):
                    transcript = llm.transcribe(wav_path)
                os.unlink(wav_path)

                if not transcript:
                    print(f"  {C.GRAY}[STT] Nothing detected — try again.{C.RESET}\n")
                    continue

                with spinner("thinking"):
                    response = llm.chat(transcript, history=history)
                tts.speak(response)
                history.append({"role": "user",      "content": transcript})
                history.append({"role": "assistant", "content": response})

            if len(history) > MAX_TURNS * 2:
                history = history[-MAX_TURNS * 2:]


if __name__ == "__main__":
    main()
