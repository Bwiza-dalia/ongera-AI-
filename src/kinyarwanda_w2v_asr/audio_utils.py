from __future__ import annotations

import numpy as np


class AudioTooQuietError(ValueError):
    """Raised when a recording has no usable speech signal."""


def audio_stats(audio: np.ndarray) -> dict[str, float]:
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio))))
    speech_ratio = float(np.mean(np.abs(audio) > 0.01))
    return {"peak": peak, "rms": rms, "speech_ratio": speech_ratio}


def validate_audio(
    audio: np.ndarray,
    *,
    min_peak: float = 0.02,
    min_rms: float = 0.005,
) -> dict[str, float]:
    """Reject clips that are too quiet for reliable transcription."""
    stats = audio_stats(audio)
    if stats["peak"] < min_peak and stats["rms"] < min_rms:
        raise AudioTooQuietError(
            "Recording is too quiet (likely silence or wrong microphone). "
            f"peak={stats['peak']:.4f}, rms={stats['rms']:.4f}."
        )
    return stats
