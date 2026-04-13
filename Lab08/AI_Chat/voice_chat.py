#!/usr/bin/env python3
"""
Lab 08 — Local LLM Voice Chat Demo
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model  : Gemma 4 E4B  (text + native audio input)
Backend: mlx-vlm  on Apple Silicon  (~90–120 tok/s)
         Ollama   on Windows / Linux (~40–60 tok/s)
TTS    : kokoro-onnx (Kokoro-82M, English + Western EU)
         → fallback: macOS say / Windows pyttsx3
Input  : [t] keyboard text  |  [a] push-to-talk recording

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUICK SETUP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Mac (Apple Silicon):
  pip install mlx-vlm kokoro-onnx sounddevice soundfile numpy
  brew install espeak-ng
  # LLM model auto-downloaded on first run (~8 GB)
  # Kokoro models: place kokoro-v1.0.onnx + voices-v1.0.bin
  #   in ~/.cache/kokoro_onnx/

Windows + NVIDIA:
  1. Install Ollama: https://ollama.com
  2. ollama pull gemma4:e4b
  3. pip install kokoro-onnx sounddevice soundfile numpy requests
  # Place kokoro model files in %USERPROFILE%\.cache\kokoro_onnx\

Run:
  python demo_voice_chat.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import pathlib
import sys
import wave
import warnings
import base64
import platform
import subprocess
import tempfile
import textwrap
import threading
from typing import Optional

# Suppress Gemma4 audio processor misconfiguration warning (fixed in mlx-vlm post-PR#906)
warnings.filterwarnings(
    "ignore",
    message=".*At least one mel filter has all zero values.*",
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

MLX_MODEL_ID    = "google/gemma-4-e4b-it"
OLLAMA_MODEL_ID = "gemma4:e4b"
OLLAMA_URL      = "http://localhost:11434/api/chat"
SAMPLE_RATE     = 16_000   # 16 kHz — sufficient for speech

SYSTEM_PROMPT = (
    "You are a helpful, concise AI assistant. "
    "Respond naturally and conversationally. "
    "If the user speaks in a language other than English, reply in that same language."
)

# ── Kokoro voices (Western languages supported by the base model) ─────────────
# Format: lang_code → (kokoro_voice, onnx_lang)
KOKORO_VOICES: dict[str, tuple[str, str]] = {
    "en": ("af_heart",  "en-us"),
    "es": ("ef_dora",   "es"),
    "fr": ("ff_siwis",  "fr-fr"),
    "hi": ("hf_alpha",  "hi"),
    "it": ("if_sara",   "it"),
    "pt": ("pf_dora",   "pt-br"),
}
KOKORO_DEFAULT = ("af_heart", "en-us")

# ── macOS say voice fallback ──────────────────────────────────────────────────
SAY_VOICES = {"en": "Ava", "zh": "Tingting", "ja": "Kyoko", "ko": "Yuna",
              "fr": "Thomas", "es": "Monica"}


def detect_language(text: str) -> str:
    """
    Fast, dependency-free language detection using Unicode block frequencies.
    Returns a BCP-47 language code: 'en' | 'zh' | 'ja' | 'ko' | ...
    Accurate for CJK; for Latin-script languages defaults to 'en'.
    """
    if not text:
        return "en"
    n = len(text)
    counts = {
        "zh": sum(1 for c in text if "\u4e00" <= c <= "\u9fff"   # CJK Unified
                                  or "\u3400" <= c <= "\u4dbf"),  # CJK Extension A
        "ja": sum(1 for c in text if "\u3040" <= c <= "\u30ff"),  # Hiragana + Katakana
        "ko": sum(1 for c in text if "\uac00" <= c <= "\ud7a3"),  # Hangul syllables
        "ar": sum(1 for c in text if "\u0600" <= c <= "\u06ff"),  # Arabic
        "hi": sum(1 for c in text if "\u0900" <= c <= "\u097f"),  # Devanagari
    }
    # Japanese text often contains CJK too; prefer 'ja' when kana present
    best_lang, best_count = "en", 0
    for lang, count in counts.items():
        if count / n > 0.08 and count > best_count:
            best_lang, best_count = lang, count
    return best_lang


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Hardware Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_platform() -> str:
    """
    Detects the current hardware and returns one of:
      'apple_silicon'  — Mac with M-series chip  → use mlx-vlm
      'nvidia'         — Windows/Linux with CUDA  → use Ollama
      'cpu'            — No GPU detected          → use Ollama (slower)
    """
    system = platform.system()

    if system == "Darwin":
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True
        ).stdout.strip()
        return "apple_silicon" if "Apple" in out else "cpu"

    # Windows or Linux: check for NVIDIA GPU via PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            return "nvidia"
    except ImportError:
        pass

    return "cpu"


def print_hardware_info(plat: str) -> None:
    """Prints a startup banner with detected hardware info."""
    system  = platform.system()
    machine = platform.machine()

    print()
    print("━" * 60)
    print("  Lab 08 — Local LLM Voice Chat  │  Gemma 4 E4B")
    print("━" * 60)
    print(f"  OS       : {system} ({machine})")

    if plat == "apple_silicon":
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True
        ).stdout.strip()
        print(f"  Chip     : {chip}")
        try:
            import mlx.core as mx
            mem_gb = mx.device_info()["memory_size"] / 1024 ** 3
            print(f"  GPU Mem  : {mem_gb:.0f} GB unified memory (MLX)")
        except Exception:
            print(f"  GPU Mem  : (mlx not installed yet)")
        print(f"  Inference: mlx-vlm  →  {MLX_MODEL_ID}")
        print(f"  TTS      : mlx-audio (Kokoro-82M, ~33× realtime)")

    elif plat == "nvidia":
        try:
            import torch
            gpu  = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            print(f"  GPU      : {gpu}  ({vram:.0f} GB VRAM)")
        except Exception:
            print(f"  GPU      : NVIDIA (detected via CUDA)")
        print(f"  Inference: Ollama  →  {OLLAMA_MODEL_ID}")
        print(f"  TTS      : pyttsx3 (Windows SAPI)")

    else:
        print(f"  GPU      : none  (CPU mode — responses will be slower)")
        print(f"  Inference: Ollama  →  {OLLAMA_MODEL_ID}")
        print(f"  TTS      : pyttsx3")

    print("━" * 60)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Inference Backends
# ─────────────────────────────────────────────────────────────────────────────

class MLXBackend:
    """
    Apple Silicon backend using mlx-vlm.
    Supports native audio input — no separate STT step required.
    Gemma 4 E4B processes the .wav audio directly.
    """

    def __init__(self):
        try:
            from mlx_vlm import load, generate
            from mlx_vlm.prompt_utils import apply_chat_template
        except ImportError:
            print("ERROR: mlx-vlm not installed.")
            print("  Run: pip install mlx-vlm")
            sys.exit(1)

        print(f"Loading {MLX_MODEL_ID} via mlx-vlm …")
        print("(First run: ~8 GB download — subsequent runs load in seconds)\n")

        self._generate       = generate
        self._apply_template = apply_chat_template
        self.model, self.processor = load(MLX_MODEL_ID)
        print("Model ready.\n")

    def chat(self, user_text: str,
             history: Optional[list] = None,
             audio_path: Optional[str] = None) -> str:
        """
        Send a message to Gemma 4 E4B with full conversation history.
        history = [{"role": "user"|"assistant", "content": "..."}, ...]
        audio_path: path to .wav for the CURRENT turn only.
        """
        history = history or []
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history \
                 + [{"role": "user", "content": user_text}]

        if audio_path:
            prompt = self._apply_template(
                self.processor, self.model.config,
                messages,
                num_audios=1,
            )
            response = self._generate(
                model=self.model,
                processor=self.processor,
                prompt=prompt,
                audio=[audio_path],
                max_tokens=512,
            )
        else:
            prompt = self._apply_template(
                self.processor, self.model.config,
                messages,
            )
            response = self._generate(
                model=self.model,
                processor=self.processor,
                prompt=prompt,
                max_tokens=512,
            )

        text = response.text if hasattr(response, "text") else str(response)
        return text.strip()


class OllamaBackend:
    """
    Windows / Linux backend using the Ollama REST API.

    Audio path A (preferred): native Ollama audio for Gemma 4 E4B.
    Audio path B (fallback):  Whisper local STT → text query to Ollama.
      Install Whisper fallback: pip install openai-whisper
    """

    def __init__(self):
        self._check_ollama_running()
        self._whisper = None   # lazy-loaded only if native audio fails
        print(f"Ollama backend ready  ({OLLAMA_MODEL_ID})\n")

    def _check_ollama_running(self) -> None:
        """Verify Ollama is running and the model is available."""
        import requests
        try:
            r = requests.get("http://localhost:11434/api/version", timeout=5)
            r.raise_for_status()
        except Exception:
            print("ERROR: Ollama is not running or not installed.")
            print("  → Install: https://ollama.com")
            print(f"  → Then run: ollama pull {OLLAMA_MODEL_ID}")
            print("  → Ollama starts automatically after install;")
            print("    if not: run 'ollama serve' in a terminal.")
            sys.exit(1)

    def _call_ollama(self, user_text: str,
                     history: Optional[list] = None,
                     audio_b64: Optional[str] = None) -> str:
        """Raw HTTP call to the Ollama /api/chat endpoint."""
        import requests

        history = history or []
        message: dict = {"role": "user", "content": user_text}
        if audio_b64:
            message["audios"] = [audio_b64]

        payload = {
            "model": OLLAMA_MODEL_ID,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}]
                        + history
                        + [message],
            "stream": False,
        }

        r = requests.post(OLLAMA_URL, json=payload, timeout=180)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()

    def _whisper_transcribe(self, audio_path: str) -> str:
        """Fallback: local Whisper speech-to-text."""
        if self._whisper is None:
            print("  [STT] Loading Whisper (base) for transcription fallback …")
            try:
                import whisper
                self._whisper = whisper.load_model("base")
            except ImportError:
                print("  [STT] Whisper not installed: pip install openai-whisper")
                return "[could not transcribe — install openai-whisper]"
        result = self._whisper.transcribe(audio_path)
        transcript = result["text"].strip()
        print(f"  [STT] Whisper: \"{transcript}\"")
        return transcript

    def chat(self, user_text: str,
             history: Optional[list] = None,
             audio_path: Optional[str] = None) -> str:
        history = history or []
        if not audio_path:
            return self._call_ollama(user_text, history=history)

        try:
            with open(audio_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode()
            return self._call_ollama(user_text, history=history, audio_b64=audio_b64)

        except Exception as e:
            print(f"  [Audio] Ollama native audio failed ({e})")
            print("  [Audio] Falling back to Whisper STT …")
            transcript = self._whisper_transcribe(audio_path)
            return self._call_ollama(transcript, history=history)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  TTS Engine
# ─────────────────────────────────────────────────────────────────────────────

class TTSEngine:
    """
    Text-to-speech engine.
      Primary : kokoro-onnx  (English + Western European languages)
                  pip install kokoro-onnx soundfile sounddevice
                  Place kokoro-v1.0.onnx + voices-v1.0.bin in ~/.cache/kokoro_onnx/
      Mac fallback : macOS built-in `say`
      Win fallback : pyttsx3
    """

    _CACHE_DIR = pathlib.Path.home() / ".cache" / "kokoro_onnx"

    def __init__(self, plat: str):
        self.plat    = plat
        self.backend = self._init(plat)

    def _init(self, plat: str) -> str:
        # ── Try kokoro-onnx ───────────────────────────────────────────────
        try:
            from kokoro_onnx import Kokoro
            import sounddevice as sd

            model_path  = next((p for n in ["kokoro-v1.0.onnx", "kokoro.onnx"]
                                if (p := self._CACHE_DIR / n).exists()), None)
            voices_path = next((p for n in ["voices-v1.0.bin", "voices.json"]
                                if (p := self._CACHE_DIR / n).exists()), None)

            if model_path and voices_path:
                self._kokoro = Kokoro(str(model_path), str(voices_path))
                self._sd     = sd
                print(f"TTS  : kokoro-onnx  ({model_path.name})")
                return "kokoro_onnx"
            else:
                print(f"  [TTS] kokoro model/voices not found in {self._CACHE_DIR}")
        except ImportError:
            print("  [TTS] kokoro-onnx not installed: pip install kokoro-onnx sounddevice")
        except Exception as e:
            print(f"  [TTS] kokoro-onnx init error: {e}")

        # ── macOS fallback ────────────────────────────────────────────────
        if plat == "apple_silicon":
            print("TTS  : macOS say  (install kokoro-onnx for better quality)")
            return "say"

        # ── Windows fallback ──────────────────────────────────────────────
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 170)
            self._pyttsx3 = engine
            print("TTS  : pyttsx3")
            return "pyttsx3"
        except Exception:
            print("TTS  : disabled")
            return "none"

    def speak(self, text: str) -> None:
        """Print the AI response and speak it aloud."""
        wrapped = textwrap.fill(
            text, width=68,
            initial_indent="  AI  ▶  ", subsequent_indent="          "
        )
        print(f"\n{wrapped}\n")

        lang = detect_language(text)
        voice, onnx_lang = KOKORO_VOICES.get(lang, KOKORO_DEFAULT)

        if self.backend == "kokoro_onnx":
            if lang in KOKORO_VOICES:
                try:
                    samples, sr = self._kokoro.create(
                        text, voice=voice, speed=1.0, lang=onnx_lang)
                    self._sd.play(samples, sr)
                    self._sd.wait()
                    return
                except Exception as e:
                    print(f"  [TTS] kokoro-onnx error: {e}")
            # Non-Western language or kokoro failed → system fallback
            self._system_speak(text, lang)

        elif self.backend == "say":
            self._system_speak(text, lang)

        elif self.backend == "pyttsx3":
            try:
                self._pyttsx3.say(text)
                self._pyttsx3.runAndWait()
            except Exception as e:
                print(f"  [TTS] pyttsx3 error: {e}")

    def _system_speak(self, text: str, lang: str) -> None:
        """macOS say fallback."""
        if self.plat == "apple_silicon":
            say_v = SAY_VOICES.get(lang, SAY_VOICES["en"])
            subprocess.run(["say", "-v", say_v, "--", text])


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Push-to-Talk Recording
# ─────────────────────────────────────────────────────────────────────────────

def record_push_to_talk(skip_start_prompt: bool = False) -> Optional[str]:
    """
    Records microphone audio.
      1. Press ENTER to start recording  (skipped when skip_start_prompt=True).
      2. Press ENTER again to stop.
    Returns path to a temporary 16-bit PCM .wav file.
    Caller is responsible for deleting the file after use.
    """
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError:
        print("  [REC] Install: pip install sounddevice numpy")
        return None

    if not skip_start_prompt:
        print("  [REC] Press ENTER to start …", end="", flush=True)
        input()

    frames: list = []
    stop_event = threading.Event()

    def _audio_callback(indata, frame_count, time_info, status):
        if status:
            pass  # ignore overflow warnings during demo
        if not stop_event.is_set():
            frames.append(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        callback=_audio_callback,
    )

    print("  [REC] Recording ▪▪▪  Press ENTER to stop …", end="", flush=True)
    with stream:
        input()          # blocks until user presses ENTER
        stop_event.set()

    if not frames:
        print("  [REC] Nothing recorded.\n")
        return None

    audio = np.concatenate(frames, axis=0)
    duration = len(audio) / SAMPLE_RATE
    print(f"  [REC] Captured {duration:.1f} s of audio")

    # Write 16-bit mono PCM WAV
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)             # 16-bit = 2 bytes per sample
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    tmp.close()
    return tmp.name


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Main Loop
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    plat = detect_platform()
    print_hardware_info(plat)

    # ── Load inference backend ───────────────────────────────────────────────
    llm = MLXBackend()  if plat == "apple_silicon" else OllamaBackend()
    tts = TTSEngine(plat)

    # ── Conversation history ──────────────────────────────────────────────────
    # Each entry: {"role": "user"|"assistant", "content": "..."}
    # Audio turns store "[voice message]" as user content so the model
    # still sees the conversational context even without the audio file.
    MAX_TURNS = 20    # keep last 20 exchanges (40 messages) to avoid OOM
    history: list = []

    AUDIO_PROMPT = (
        "The user sent a voice message. "
        "Listen carefully, understand it, and reply conversationally. "
        "If the speaker uses a language other than English, reply in that language."
    )

    def _hist_label() -> str:
        n = len(history) // 2
        return f"{n} turn{'s' if n != 1 else ''} in history"

    # ── Outer menu loop ───────────────────────────────────────────────────────
    while True:
        print()
        print("  ┌─────────────────────────────────────────────┐")
        print(f"  │  {_hist_label():<43}│")
        print("  │  t  Text mode                               │")
        print("  │  a  Audio mode  (push-to-talk)              │")
        print("  │  c  Clear history                           │")
        print("  │  q  Quit                                    │")
        print("  └─────────────────────────────────────────────┘")
        try:
            cmd = input("  > ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n  Goodbye!")
            return

        if cmd == "q":
            print("  Goodbye!")
            return
        elif cmd == "c":
            history.clear()
            print("  History cleared.")
            continue
        elif cmd not in ("t", "a"):
            continue

        mode = cmd

        if mode == "t":
            print("  Text mode  —  m = menu · c = clear · q = quit\n")
        else:
            print("  Audio mode  —  ENTER = record · ENTER = stop · m = menu · q = quit\n")

        # ── Inner conversation loop ───────────────────────────────────────────
        while True:
            turn = len(history) // 2 + 1

            if mode == "t":
                try:
                    user_text = input(f"  [{turn}] You  ▶  ").strip()
                except (KeyboardInterrupt, EOFError):
                    print("\n  Goodbye!")
                    return
                if user_text.lower() == "q":
                    print("  Goodbye!")
                    return
                if user_text.lower() == "m":
                    break          # back to outer menu
                if user_text.lower() == "c":
                    history.clear()
                    print(f"  History cleared.\n")
                    continue
                if not user_text:
                    continue
                print("  …  thinking")
                response = llm.chat(user_text, history=history)
                tts.speak(response)
                history.append({"role": "user",      "content": user_text})
                history.append({"role": "assistant", "content": response})

            else:  # audio
                try:
                    pre = input(f"  [{turn}] ↵ record  ·  m menu  ·  q quit › ").strip().lower()
                except (KeyboardInterrupt, EOFError):
                    print("\n  Goodbye!")
                    return
                if pre == "q":
                    print("  Goodbye!")
                    return
                if pre == "m":
                    break          # back to outer menu
                if pre == "c":
                    history.clear()
                    print(f"  History cleared.\n")
                    continue
                wav_path = record_push_to_talk(skip_start_prompt=True)
                if wav_path is None:
                    continue
                print("  …  processing audio")
                response = llm.chat(AUDIO_PROMPT, history=history, audio_path=wav_path)
                os.unlink(wav_path)
                tts.speak(response)
                history.append({"role": "user",      "content": "[voice message]"})
                history.append({"role": "assistant", "content": response})

            # Trim history
            if len(history) > MAX_TURNS * 2:
                history = history[-MAX_TURNS * 2:]


if __name__ == "__main__":
    main()
