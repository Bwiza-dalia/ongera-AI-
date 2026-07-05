"""Kinyarwanda ASR using Wav2Vec2-BERT CTC."""

from kinyarwanda_w2v_asr.transcriber import load_audio, transcribe, transcribe_file

__all__ = ["load_audio", "transcribe", "transcribe_file"]
