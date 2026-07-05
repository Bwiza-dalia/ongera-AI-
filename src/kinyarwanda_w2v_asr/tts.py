from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np
import soundfile as sf
import torch
from transformers import AutoTokenizer, VitsModel

from kinyarwanda_w2v_asr.config import TTS_MODEL_ID

_tts_model: VitsModel | None = None
_tts_tokenizer: AutoTokenizer | None = None
_tts_sample_rate: int = 16_000


def _get_device() -> torch.device:
    forced = os.environ.get("KIN_W2V_DEVICE", "").lower()
    if forced in ("cpu", "cuda", "mps"):
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tts_model() -> tuple[VitsModel, AutoTokenizer, int]:
    """Load Kinyarwanda VITS TTS (singleton)."""
    global _tts_model, _tts_tokenizer, _tts_sample_rate
    if _tts_model is not None and _tts_tokenizer is not None:
        return _tts_model, _tts_tokenizer, _tts_sample_rate

    device = _get_device()
    _tts_tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_ID)
    _tts_model = VitsModel.from_pretrained(TTS_MODEL_ID)
    _tts_model.to(device)
    _tts_model.eval()

    rate = getattr(_tts_model.config, "sampling_rate", None)
    _tts_sample_rate = int(rate) if rate else 16_000
    return _tts_model, _tts_tokenizer, _tts_sample_rate


def synthesize_kinyarwanda(text: str) -> tuple[np.ndarray, int]:
    """Synthesize speech waveform for Kinyarwanda text."""
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("text must not be empty")

    model, tokenizer, sample_rate = load_tts_model()
    device = model.device
    inputs = tokenizer(cleaned, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        waveform = model(**inputs).waveform

    audio = waveform.squeeze().detach().cpu().numpy().astype(np.float32)
    return audio, sample_rate


def synthesize_kinyarwanda_wav_bytes(text: str) -> tuple[bytes, int]:
    """Return WAV bytes and sample rate."""
    audio, sample_rate = synthesize_kinyarwanda(text)
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    return buffer.getvalue(), sample_rate


def _play_wav_bytes(wav_bytes: bytes) -> bool:
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name

        if shutil.which("afplay"):
            subprocess.run(["afplay", tmp_path], check=False, capture_output=True)
            return True

        import sounddevice as sd

        audio, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        sd.play(audio, sample_rate)
        sd.wait()
        return True
    except Exception:
        return False
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def speak_kinyarwanda(text: str) -> bool:
    """Synthesize with Kinyarwanda TTS and play audio locally."""
    try:
        wav_bytes, _ = synthesize_kinyarwanda_wav_bytes(text)
        return _play_wav_bytes(wav_bytes)
    except Exception:
        return False


def tts_info() -> dict[str, Any]:
    """Metadata for API responses."""
    return {
        "model": TTS_MODEL_ID,
        "language": "rw",
        "engine": "mms-vits",
    }
