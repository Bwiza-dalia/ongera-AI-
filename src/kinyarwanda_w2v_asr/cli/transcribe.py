#!/usr/bin/env python3
"""CLI: transcribe a WAV file with the Kinyarwanda Wav2Vec2-BERT model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kinyarwanda_w2v_asr.audio_utils import AudioTooQuietError
from kinyarwanda_w2v_asr.config import CHUNK_DURATION_SEC, MAX_DURATION_SEC, MODEL_ID
from kinyarwanda_w2v_asr.transcriber import transcribe_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Transcribe Kinyarwanda speech with {MODEL_ID}",
    )
    parser.add_argument("audio_path", help="Path to a .wav file (16 kHz mono recommended)")
    parser.add_argument(
        "--chunk-sec",
        type=float,
        default=CHUNK_DURATION_SEC,
        help=f"Chunk size for long audio in seconds (default: {CHUNK_DURATION_SEC})",
    )
    parser.add_argument(
        "--max-sec",
        type=float,
        default=MAX_DURATION_SEC,
        help=f"Max duration before chunking kicks in (default: {MAX_DURATION_SEC})",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip quiet-audio check",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio_path)
    if not audio_path.is_file():
        print(f"Error: audio file not found: {audio_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    try:
        result = transcribe_file(
            str(audio_path),
            skip_validation=args.skip_validation,
            chunk_duration=args.chunk_sec,
            max_duration=args.max_sec,
        )
    except AudioTooQuietError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    output = {
        "transcript": result["transcription"],
        "duration_sec": round(result["duration"], 3),
        "processing_time_sec": round(result["processing_time"], 3),
        "model": MODEL_ID,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
