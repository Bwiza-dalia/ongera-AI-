#!/usr/bin/env python3
"""Speak Kinyarwanda text using MMS TTS."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kinyarwanda_w2v_asr.tts import _play_wav_bytes, synthesize_kinyarwanda_wav_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description="Speak Kinyarwanda text with MMS TTS")
    parser.add_argument("text", help="Kinyarwanda text to speak")
    parser.add_argument(
        "--save",
        type=Path,
        help="Optional path to save WAV instead of only playing",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Only save file; do not play audio",
    )
    args = parser.parse_args()

    try:
        wav_bytes, sample_rate = synthesize_kinyarwanda_wav_bytes(args.text)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.save:
        args.save.write_bytes(wav_bytes)
        print(f"Saved WAV ({sample_rate} Hz) -> {args.save}")

    if not args.no_play:
        ok = _play_wav_bytes(wav_bytes)
        if not ok:
            print("Could not play audio on this machine.", file=sys.stderr)
            sys.exit(2)


if __name__ == "__main__":
    main()
