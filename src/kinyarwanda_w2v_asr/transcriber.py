from __future__ import annotations

import gc
import io
import os
from typing import Any

import librosa
import numpy as np
import torch
from transformers import Wav2Vec2BertForCTC, Wav2Vec2BertProcessor

from kinyarwanda_w2v_asr.audio_utils import validate_audio
from kinyarwanda_w2v_asr.config import (
    CHUNK_DURATION_SEC,
    MAX_DURATION_SEC,
    MODEL_ID,
    SAMPLE_RATE,
)

_processor: Wav2Vec2BertProcessor | None = None
_model: Wav2Vec2BertForCTC | None = None


def _get_device() -> torch.device:
    forced = os.environ.get("KIN_W2V_DEVICE", "").lower()
    if forced in ("cpu", "cuda", "mps"):
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model() -> tuple[Wav2Vec2BertProcessor, Wav2Vec2BertForCTC]:
    """Load model and processor (singleton)."""
    global _processor, _model
    if _processor is not None and _model is not None:
        return _processor, _model

    device = _get_device()
    _processor = Wav2Vec2BertProcessor.from_pretrained(MODEL_ID)
    _model = Wav2Vec2BertForCTC.from_pretrained(MODEL_ID)
    _model.to(device)
    _model.eval()

    return _processor, _model


def warmup_model(*, sample_rate: int = SAMPLE_RATE) -> None:
    """Load weights and run one short forward pass so the first live decode is fast."""
    processor, model = load_model()
    silence = np.zeros(int(0.5 * sample_rate), dtype=np.float32)
    _decode_chunk(processor, model, silence, sample_rate=sample_rate)


def load_audio(path: str, *, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    return audio


def load_audio_bytes(data: bytes, *, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    audio, _ = librosa.load(io.BytesIO(data), sr=sample_rate, mono=True)
    return audio


def _decode_chunk(
    processor: Wav2Vec2BertProcessor,
    model: Wav2Vec2BertForCTC,
    audio: np.ndarray,
    *,
    sample_rate: int,
) -> str:
    device = model.device
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(predicted_ids)[0]


def transcribe(
    audio: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    skip_validation: bool = False,
    chunk_duration: float = CHUNK_DURATION_SEC,
    max_duration: float = MAX_DURATION_SEC,
) -> str:
    """
    Transcribe a 1-D waveform to Kinyarwanda text.

    Long audio is split into chunks (default 30s) and transcripts are joined.
    """
    if not skip_validation:
        validate_audio(audio)

    processor, model = load_model()
    duration = len(audio) / sample_rate

    if duration <= max_duration:
        return _decode_chunk(processor, model, audio, sample_rate=sample_rate).strip()

    chunk_size = int(chunk_duration * sample_rate)
    total_chunks = (len(audio) + chunk_size - 1) // chunk_size
    parts: list[str] = []

    for chunk_idx in range(total_chunks):
        start = chunk_idx * chunk_size
        end = min((chunk_idx + 1) * chunk_size, len(audio))
        chunk = audio[start:end]
        parts.append(_decode_chunk(processor, model, chunk, sample_rate=sample_rate))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return " ".join(part.strip() for part in parts if part.strip())


def transcribe_file(
    path: str,
    *,
    sample_rate: int = SAMPLE_RATE,
    skip_validation: bool = False,
    chunk_duration: float = CHUNK_DURATION_SEC,
    max_duration: float = MAX_DURATION_SEC,
) -> dict[str, Any]:
    """Load a file and return transcript plus simple timing metadata."""
    import time

    audio = load_audio(path, sample_rate=sample_rate)
    duration = len(audio) / sample_rate

    start = time.perf_counter()
    text = transcribe(
        audio,
        sample_rate=sample_rate,
        skip_validation=skip_validation,
        chunk_duration=chunk_duration,
        max_duration=max_duration,
    )
    elapsed = time.perf_counter() - start

    return {
        "file_path": path,
        "duration": duration,
        "transcription": text,
        "processing_time": elapsed,
        "success": True,
    }
