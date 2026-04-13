#!/usr/bin/env python3
"""
Lab 08 — Local LLM Voice Chat Demo
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model  : Gemma 4 E4B  (text + native audio input)
Backend: mlx-vlm   on Apple Silicon  (primary, ~90–120 tok/s)
         Ollama    on Windows / Linux (primary, ~40–60 tok/s)
TTS    : mlx-audio (Kokoro-82M, 33× realtime) on Mac
         pyttsx3   (Windows SAPI)             on Windows
Input  : [t] keyboard text  |  [a] push-to-talk recording

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUICK SETUP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Mac (Apple Silicon):
  pip install mlx-vlm mlx-audio sounddevice numpy
  # Model is auto-downloaded on first run (~8 GB)

Windows + NVIDIA:
  1. Install Ollama: https://ollama.com
  2. ollama pull gemma4:e4b            (downloads ~5 GB)
  3. pip install sounddevice numpy requests pyttsx3
  # Optional better TTS: pip install kokoro-onnx soundfile

Run:
  python demo_voice_chat.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import pathlib
import sys
import wave
import base64
import platform
import subprocess
import tempfile
import textwrap
import threading
from typing import Optional

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

# ── Kokoro voice library per language ────────────────────────────────────────
# Format: language_code → (kokoro_voice, kokoro_lang_code)
# kokoro_lang_code is used by KokoroPipeline; voice prefix must match it.
# Voices confirmed in Kokoro-82M: af_*, am_*, bf_*, bm_*, ef_*, ff_*,
#   hf_alpha, if_sara, jf_alpha, jf_gongitsune, pf_dora, zf_xiaobei
KOKORO_VOICES: dict[str, tuple[str, str]] = {
    "en": ("af_heart",   "a"),   # American English female
    "zh": ("zf_xiaobei", "z"),   # Mandarin Chinese female
    "ja": ("jf_alpha",   "j"),   # Japanese female
    "ko": ("af_heart",   "a"),   # Korean — no dedicated Kokoro voice, fall back to EN
    "es": ("ef_dora",    "e"),   # Spanish female
    "fr": ("ff_siwis",   "f"),   # French female
    "hi": ("hf_alpha",   "h"),   # Hindi female
    "it": ("if_sara",    "i"),   # Italian female
    "pt": ("pf_dora",    "p"),   # Portuguese female
}
KOKORO_DEFAULT = ("af_heart", "a")   # fallback for unsupported languages


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
            mem_gb = mx.metal.device_info()["memory_size"] / 1024 ** 3
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
    Text-to-speech with automatic platform selection.

    All   : kokoro-onnx  (Kokoro-82M via ONNX, natural neural voice, multilingual)
              pip install kokoro-onnx soundfile sounddevice
              Models auto-downloaded to ~/.cache/kokoro_onnx/ on first run (~340 MB)
    Mac   : mlx-audio  → fallback if kokoro-onnx unavailable
    Mac   : macOS say  → last resort
    Win   : pyttsx3    → fallback if kokoro-onnx unavailable
    """

    # Kokoro-ONNX model file URLs (thewh1teagle/kokoro-onnx releases)
    _MODEL_URLS = {
        "kokoro-v0_19.onnx": (
            "https://github.com/thewh1teagle/kokoro-onnx/releases/"
            "download/model-files-v1.0/kokoro-v0_19.onnx"
        ),
        "voices.json": (
            "https://github.com/thewh1teagle/kokoro-onnx/releases/"
            "download/model-files-v1.0/voices.json"
        ),
    }
    _CACHE_DIR = pathlib.Path.home() / ".cache" / "kokoro_onnx"

    def __init__(self, plat: str):
        self.plat    = plat
        self.backend = self._init(plat)

    # ------------------------------------------------------------------
    # kokoro-onnx: auto-download models, return True on success
    # ------------------------------------------------------------------
    def _init_kokoro_onnx(self) -> bool:
        """Try to initialise kokoro-onnx. Downloads model files if missing."""
        try:
            from kokoro_onnx import Kokoro
            import soundfile as sf
            import sounddevice as sd
        except ImportError:
            print("  [TTS] kokoro-onnx not installed: pip install kokoro-onnx soundfile sounddevice")
            return False

        self._CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for fname, url in self._MODEL_URLS.items():
            dest = self._CACHE_DIR / fname
            if dest.exists():
                continue
            print(f"  [TTS] Downloading {fname} …  (one-time, ~340 MB for the model)")
            try:
                import urllib.request
                def _progress(block, block_size, total):
                    if total > 0:
                        pct = min(100, block * block_size * 100 // total)
                        print(f"\r  [TTS] {fname}: {pct:3d}%", end="", flush=True)
                urllib.request.urlretrieve(url, dest, reporthook=_progress)
                print()  # newline after progress
            except Exception as e:
                print(f"\n  [TTS] Download failed: {e}")
                print(f"  [TTS] Manual download: {url}")
                print(f"  [TTS] Save to: {dest}")
                return False

        try:
            self._kokoro = Kokoro(
                str(self._CACHE_DIR / "kokoro-v0_19.onnx"),
                str(self._CACHE_DIR / "voices.json"),
            )
            self._sf = sf
            self._sd = sd
            print(f"TTS  : kokoro-onnx  (models: {self._CACHE_DIR})")
            return True
        except Exception as e:
            print(f"  [TTS] kokoro-onnx init failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Internal: try every known mlx-audio API pattern, return a callable
    # (text, wav_path) -> None   or   None if nothing works.
    # ------------------------------------------------------------------
    def _discover_mlx_tts(self):
        import importlib, numpy as np
        import sounddevice as sd
        import soundfile as sf

        try:
            import mlx_audio
        except ImportError:
            print("  [TTS diag] mlx_audio not installed: pip install mlx-audio")
            return None

        print(f"  [TTS diag] mlx_audio found, inspecting API …")
        print(f"  [TTS diag] top-level attrs: {[x for x in dir(mlx_audio) if not x.startswith('_')]}")

        # All patterns return a function with signature:
        #   _fn(text: str, wav: str, voice: str, lang_code: str) -> None

        # ── Pattern 1: mlx_audio.tts(text, voice=, output=) ──────────────
        if callable(getattr(mlx_audio, "tts", None)):
            print("  [TTS diag] Pattern 1 matched: mlx_audio.tts() is callable")
            def _fn(text, wav, voice, lang_code):
                mlx_audio.tts(text, voice=voice, output=wav)
            return _fn

        # ── Pattern 2: mlx_audio.tts.generate(text, voice=, output=) ─────
        tts_mod = getattr(mlx_audio, "tts", None)
        if tts_mod is not None:
            print(f"  [TTS diag] mlx_audio.tts is a module; attrs: {[x for x in dir(tts_mod) if not x.startswith('_')]}")
            if callable(getattr(tts_mod, "generate", None)):
                print("  [TTS diag] Pattern 2 matched: mlx_audio.tts.generate()")
                def _fn(text, wav, voice, lang_code):
                    audio, sr = tts_mod.generate(text, voice=voice)
                    sf.write(wav, audio, sr)
                return _fn

        # ── Pattern 3: from mlx_audio.tts import generate ────────────────
        try:
            from mlx_audio.tts import generate as tts_gen
            print("  [TTS diag] Pattern 3 matched: from mlx_audio.tts import generate")
            def _fn(text, wav, voice, lang_code):
                result = tts_gen(text, voice=voice)
                if isinstance(result, tuple):
                    audio, sr = result
                else:
                    audio, sr = result, 24000
                sf.write(wav, audio, sr)
            return _fn
        except Exception as e:
            print(f"  [TTS diag] Pattern 3 failed: {e}")

        # ── Pattern 4: KokoroPipeline (lang_code switches per call) ───────
        try:
            from mlx_audio.tts.models.kokoro import KokoroPipeline
            print("  [TTS diag] Pattern 4 matched: KokoroPipeline")
            # Cache pipelines by lang_code to avoid reloading
            _pipelines: dict = {}
            def _fn(text, wav, voice, lang_code):
                if lang_code not in _pipelines:
                    _pipelines[lang_code] = KokoroPipeline(lang_code=lang_code)
                pipeline = _pipelines[lang_code]
                chunks = [a for _, _, a in pipeline(text, voice=voice, speed=1.0)]
                if chunks:
                    sf.write(wav, np.concatenate(chunks), 24000)
            return _fn
        except Exception as e:
            print(f"  [TTS diag] Pattern 4 failed: {e}")

        print("  [TTS diag] No working mlx-audio TTS API found.")
        return None

    def _init(self, plat: str) -> str:
        # ── 1. kokoro-onnx: best quality, works on all platforms ─────────
        if self._init_kokoro_onnx():
            return "kokoro_onnx"

        # ── 2. macOS fallbacks ────────────────────────────────────────────
        if plat == "apple_silicon":
            tts_fn = self._discover_mlx_tts()
            if tts_fn is not None:
                self._tts_fn = tts_fn
                print("TTS  : mlx-audio (Kokoro-82M via MLX)")
                return "mlx_audio"
            print("TTS  : macOS say  (install kokoro-onnx for better quality)")
            return "say"

        # ── 3. Windows fallback ───────────────────────────────────────────
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 170)
            self._pyttsx3 = engine
            print("TTS  : pyttsx3  (tip: pip install kokoro-onnx soundfile sounddevice)")
            return "pyttsx3"
        except Exception:
            print("TTS  : disabled  (pip install kokoro-onnx or pyttsx3)")
            return "none"

    def speak(self, text: str) -> None:
        """Print the AI response and optionally speak it aloud."""
        # Always print, wrapped nicely
        wrapped = textwrap.fill(
            text, width=68,
            initial_indent    = "  AI  ▶  ",
            subsequent_indent = "          "
        )
        print(f"\n{wrapped}\n")

        # Detect language and pick the right Kokoro voice
        lang = detect_language(text)
        voice, lang_code = KOKORO_VOICES.get(lang, KOKORO_DEFAULT)
        if lang != "en":
            print(f"  [TTS] detected language: {lang}  →  voice: {voice}")

        # macOS say voice map — used as fallback when Kokoro fails
        SAY_VOICES = {
            "en": "Ava",      # macOS neural English (natural quality)
            "zh": "Tingting", "ja": "Kyoko",
            "ko": "Yuna",     "fr": "Thomas", "es": "Monica",
        }

        if self.backend == "mlx_audio":
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            spoken = False
            try:
                self._tts_fn(text, tmp.name, voice, lang_code)
                subprocess.run(["afplay", tmp.name], check=True)
                spoken = True
            except Exception as e:
                err = str(e)
                if "not callable" in err and lang == "en":
                    print(f"  [TTS] English Kokoro failed ({err})")
                    print("  [TTS] Fix: pip install 'misaki[en]'  then restart")
                else:
                    print(f"  [TTS] mlx-audio/{lang} error: {err}")
                print(f"  [TTS] Falling back to macOS say -v {SAY_VOICES.get(lang, 'Ava')}")
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)
            if not spoken:
                say_v = SAY_VOICES.get(lang, "Ava")
                subprocess.run(["say", "-v", say_v, "--", text])

        elif self.backend == "say":
            say_v = SAY_VOICES.get(lang, "Ava")
            try:
                subprocess.run(["say", "-v", say_v, "--", text], check=True)
            except Exception:
                pass

        elif self.backend == "kokoro_onnx":
            try:
                samples, sr = self._kokoro.create(text, voice=voice, speed=1.0)
                self._sd.play(samples, sr)
                self._sd.wait()
            except Exception as e:
                print(f"  [TTS] kokoro-onnx error: {e}")

        elif self.backend == "pyttsx3":
            try:
                self._pyttsx3.say(text)
                self._pyttsx3.runAndWait()
            except Exception as e:
                print(f"  [TTS] pyttsx3 error: {e}")
        # backend == "none": text already printed


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

    # ── Choose mode once ─────────────────────────────────────────────────────
    print()
    print("  Select input mode:")
    print("  t  →  Text")
    print("  a  →  Audio  (push-to-talk: ENTER start / ENTER stop)")
    print()
    while True:
        try:
            mode = input("  Mode [t / a] > ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n  Goodbye!")
            return
        if mode in ("t", "a"):
            break
        print("  Please enter t or a.")

    if mode == "t":
        print("\n  Text mode — type your message, or enter  q  to quit,  c  to clear history.\n")
    else:
        print("\n  Audio mode — press ENTER to record, ENTER again to stop.\n"
              "  Enter  q  to quit,  c  to clear history.\n")

    # ── Conversation loop ────────────────────────────────────────────────────
    AUDIO_PROMPT = (
        "The user sent a voice message. "
        "Listen carefully, understand it, and reply conversationally. "
        "If the speaker uses a language other than English, reply in that language."
    )

    while True:
        turn = len(history) // 2 + 1

        # ── Text mode ────────────────────────────────────────────────────────
        if mode == "t":
            try:
                user_text = input(f"  [{turn}] You  ▶  ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n  Goodbye!")
                break
            if user_text.lower() == "q":
                print("  Goodbye!")
                break
            if user_text.lower() == "c":
                history.clear()
                print("  History cleared.\n")
                continue
            if not user_text:
                continue
            print("  …  thinking")
            response = llm.chat(user_text, history=history)
            tts.speak(response)
            history.append({"role": "user",      "content": user_text})
            history.append({"role": "assistant", "content": response})

        # ── Audio mode ───────────────────────────────────────────────────────
        else:
            try:
                pre = input(f"  [{turn}] Press ENTER to record  (q=quit  c=clear) › ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                print("\n  Goodbye!")
                break
            if pre == "q":
                print("  Goodbye!")
                break
            if pre == "c":
                history.clear()
                print("  History cleared.\n")
                continue
            # user pressed ENTER (or typed anything else) — start recording
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
