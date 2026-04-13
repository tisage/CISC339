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

    def chat(self, user_text: str, audio_path: Optional[str] = None) -> str:
        """
        Send a text prompt (and optionally a .wav file) to Gemma 4 E4B.
        When audio_path is given, Gemma 4 hears the audio and reads the prompt.
        """
        if audio_path:
            # num_audios=1 tells the template to insert one <audio> token
            prompt = self._apply_template(
                self.processor, self.model.config,
                user_text,
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
                user_text,
            )
            response = self._generate(
                model=self.model,
                processor=self.processor,
                prompt=prompt,
                max_tokens=512,
            )

        # generate() returns a GenerationResult object; .text is the string
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

    def _call_ollama(self, user_text: str, audio_b64: Optional[str] = None) -> str:
        """Raw HTTP call to the Ollama /api/chat endpoint."""
        import requests

        message: dict = {"role": "user", "content": user_text}
        if audio_b64:
            # Ollama 0.7+ audio key for Gemma 4 E4B
            message["audios"] = [audio_b64]

        payload = {
            "model": OLLAMA_MODEL_ID,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                message,
            ],
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

    def chat(self, user_text: str, audio_path: Optional[str] = None) -> str:
        if not audio_path:
            return self._call_ollama(user_text)

        # Try native Ollama audio first
        try:
            with open(audio_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode()
            return self._call_ollama(user_text, audio_b64=audio_b64)

        except Exception as e:
            print(f"  [Audio] Ollama native audio failed ({e})")
            print("  [Audio] Falling back to Whisper STT …")
            transcript = self._whisper_transcribe(audio_path)
            return self._call_ollama(transcript)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  TTS Engine
# ─────────────────────────────────────────────────────────────────────────────

class TTSEngine:
    """
    Text-to-speech with automatic platform selection.

    Mac   : mlx-audio  (Kokoro-82M via MLX, ~33× realtime, high quality)
            → fallback: macOS built-in `say` (zero install, lower quality)
    Win   : pyttsx3    (Windows SAPI, built-in voices, no download needed)
            → upgrade:  pip install kokoro-onnx soundfile  (much more natural)
    """

    def __init__(self, plat: str):
        self.plat    = plat
        self.backend = self._init(plat)

    def _init(self, plat: str) -> str:
        if plat == "apple_silicon":
            try:
                import mlx_audio
                # mlx-audio API varies by version:
                #   ≥0.2: mlx_audio.tts is a submodule → use generate() inside it
                #   <0.2: mlx_audio.tts is a direct callable
                if callable(getattr(mlx_audio, "tts", None)):
                    self._mlx_audio = mlx_audio
                    self._tts_mode  = "fn"
                elif hasattr(mlx_audio, "tts") and callable(
                        getattr(mlx_audio.tts, "generate", None)):
                    self._mlx_audio = mlx_audio
                    self._tts_mode  = "submodule"
                else:
                    raise AttributeError(
                        "Cannot find a callable TTS in mlx_audio. "
                        "Run: pip install --upgrade mlx-audio"
                    )
                print("TTS  : mlx-audio (Kokoro-82M)")
                return "mlx_audio"
            except Exception as e:
                print(f"TTS  : macOS say  (mlx-audio unavailable — {e})")
                return "say"

        else:
            # Try kokoro-onnx first (high quality, cross-platform)
            try:
                from kokoro_onnx import Kokoro
                import soundfile as sf
                import sounddevice as sd
                self._kokoro = Kokoro("kokoro-v0_19.onnx", "voices.json")
                self._sf = sf
                self._sd = sd
                print("TTS  : kokoro-onnx (high quality)")
                return "kokoro_onnx"
            except Exception:
                pass

            # Fall back to pyttsx3 (always available on Windows)
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", 170)
                self._pyttsx3 = engine
                print("TTS  : pyttsx3  (tip: pip install kokoro-onnx soundfile for better voice)")
                return "pyttsx3"
            except Exception:
                print("TTS  : disabled  (pip install pyttsx3)")
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

        if self.backend == "mlx_audio":
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            spoken = False
            try:
                if self._tts_mode == "fn":
                    self._mlx_audio.tts(text, voice="af_heart", output=tmp.name)
                else:  # submodule
                    self._mlx_audio.tts.generate(text, voice="af_heart", output=tmp.name)
                subprocess.run(["afplay", tmp.name], check=True)
                spoken = True
            except Exception as e:
                print(f"  [TTS] mlx-audio error: {e} — falling back to say")
            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)
            if not spoken:
                subprocess.run(["say", "--", text])

        elif self.backend == "say":
            try:
                subprocess.run(["say", "--", text], check=True)
            except Exception:
                pass  # text already printed above

        elif self.backend == "kokoro_onnx":
            try:
                samples, sr = self._kokoro.create(text, voice="af_bella", speed=1.0)
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

def record_push_to_talk() -> Optional[str]:
    """
    Records microphone audio.
      1. Press ENTER to start recording.
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

    # ── Instructions ────────────────────────────────────────────────────────
    print()
    print("  Commands")
    print("  ─────────────────────────────────")
    print("  t  →  Type a message")
    print("  a  →  Audio  (push-to-talk: ENTER start / ENTER stop)")
    print("  q  →  Quit")
    print()

    # ── Conversation loop ────────────────────────────────────────────────────
    while True:
        try:
            cmd = input("  Mode [t / a / q] > ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n  Goodbye!")
            break

        # ── Quit ─────────────────────────────────────────────────────────────
        if cmd == "q":
            print("  Goodbye!")
            break

        # ── Text input ───────────────────────────────────────────────────────
        elif cmd == "t":
            user_text = input("  You  ▶  ").strip()
            if not user_text:
                continue
            print("  …  thinking")
            response = llm.chat(
                f"{SYSTEM_PROMPT}\n\nUser: {user_text}"
            )
            tts.speak(response)

        # ── Audio input (push-to-talk) ────────────────────────────────────────
        elif cmd == "a":
            wav_path = record_push_to_talk()
            if wav_path is None:
                continue
            print("  …  processing audio")
            # The prompt tells Gemma 4 to listen and respond conversationally.
            # Gemma 4 E4B understands 140+ languages natively from audio —
            # no separate speech-to-text step is needed on Apple Silicon.
            audio_prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                "The user sent you a voice message. "
                "Listen carefully, understand it, and reply conversationally. "
                "If the speaker uses a language other than English, reply in that language."
            )
            response = llm.chat(audio_prompt, audio_path=wav_path)
            os.unlink(wav_path)    # delete temp file
            tts.speak(response)

        # ── Unknown ──────────────────────────────────────────────────────────
        else:
            print("  Unknown command — use t, a, or q.")


if __name__ == "__main__":
    main()
