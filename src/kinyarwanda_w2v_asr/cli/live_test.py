#!/usr/bin/env python3
"""
Interactive recording tester with countdown, live volume meter, and plots.

Usage:
    kin-w2v-test                          # free speech — live transcript while recording
    kin-w2v-test --seconds 20               # longer recording for multiple sentences
    kin-w2v-test --no-stream                # wait until stop before transcribing
    kin-w2v-test --target amata             # naming exercise with scoring
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import soundfile as sf

from kinyarwanda_w2v_asr.audio_utils import audio_stats, validate_audio
from kinyarwanda_w2v_asr.config import SAMPLE_RATE
from kinyarwanda_w2v_asr.hints import next_hint_for_target
from kinyarwanda_w2v_asr.scoring import score_naming_attempt
from kinyarwanda_w2v_asr.transcriber import transcribe, warmup_model
from kinyarwanda_w2v_asr.tts import _play_wav_bytes, synthesize_kinyarwanda_wav_bytes

DEFAULT_WORDS = ["amafi", "umugati", "amata", "umuceri"]
NAMING_RECORD_SECONDS = 3.0
FREE_SPEECH_SECONDS = 15.0
COUNTDOWN_SECONDS = 3
METER_HZ = 10
STREAM_POLL_SEC = 0.2
STREAM_UPDATE_SEC = 0.7
STREAM_WINDOW_SEC = 2.5
STREAM_FIRST_SEC = 0.5


def beep() -> None:
    try:
        subprocess.run(
            ["afplay", "/System/Library/Sounds/Tink.aiff"],
            check=False,
            capture_output=True,
        )
    except Exception:
        print("\a", end="", flush=True)


def beep_go() -> None:
    try:
        subprocess.run(
            ["afplay", "/System/Library/Sounds/Glass.aiff"],
            check=False,
            capture_output=True,
        )
    except Exception:
        print("\a\a", end="", flush=True)


def draw_volume_bar(rms: float, peak: float, *, speaking: bool) -> str:
    level = min(int(rms * 400), 40)
    bar = "█" * level + "░" * (40 - level)
    status = "SPEAKING" if speaking else "quiet"
    return f"  [{bar}] peak={peak:.3f} rms={rms:.4f}  {status}"


def _print_live_line(text: str) -> None:
    """Print a live transcript line on stdout (works in all terminals)."""
    print(f"  Live: {text}", flush=True)


def _decode_live_window(
    frames: list[np.ndarray],
    frames_lock: threading.Lock,
    sample_rate: int,
    *,
    window_sec: float = STREAM_WINDOW_SEC,
) -> str:
    with frames_lock:
        if not frames:
            return ""
        full = np.concatenate(frames)
    window_samples = int(window_sec * sample_rate)
    segment = full[-window_samples:]
    return transcribe(segment, skip_validation=True).strip()


def record_with_meter(
    duration: float,
    sample_rate: int,
    *,
    live_transcript: bool = False,
) -> tuple[np.ndarray, str | None]:
    """Record audio with a live volume meter; optional streaming transcription."""
    frames: list[np.ndarray] = []
    frames_lock = threading.Lock()
    last_draw = 0.0
    live_lines: list[str] = []
    last_window_text = ""

    if live_transcript:
        print("\n  Live transcript (prints while you speak):\n", flush=True)

    def callback(indata, _frames, _time, status):
        nonlocal last_draw
        if status:
            print(f"\n  [audio] {status}", file=sys.stderr, flush=True)
        chunk = indata[:, 0].copy()
        with frames_lock:
            frames.append(chunk)

        now = time.monotonic()
        if now - last_draw < 1.0 / METER_HZ:
            return
        last_draw = now

        rms = float(np.sqrt(np.mean(chunk**2)))
        peak = float(np.max(np.abs(chunk)))
        speaking = rms > 0.01 or peak > 0.03
        line = draw_volume_bar(rms, peak, speaking=speaking)
        sys.stderr.write("\r" + line + " " * 8)
        sys.stderr.flush()

    if not live_transcript:
        print("\n  Live input meter (updates while you speak):\n", flush=True)

    first_samples = int(STREAM_FIRST_SEC * sample_rate)

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        callback=callback,
        blocksize=int(sample_rate * 0.05),
    ):
        if not live_transcript:
            time.sleep(duration)
        else:
            deadline = time.monotonic() + duration
            last_decode_at = 0.0
            while time.monotonic() < deadline:
                time.sleep(STREAM_POLL_SEC)
                if time.monotonic() - last_decode_at < STREAM_UPDATE_SEC:
                    continue

                with frames_lock:
                    total_samples = sum(len(chunk) for chunk in frames)
                if total_samples < first_samples:
                    continue

                last_decode_at = time.monotonic()
                try:
                    text = _decode_live_window(
                        frames,
                        frames_lock,
                        sample_rate,
                    )
                except Exception:
                    continue

                if not text or text == last_window_text:
                    continue

                last_window_text = text
                live_lines.append(text)
                _print_live_line(text)

    sys.stderr.write("\r" + " " * 72 + "\r")
    sys.stderr.flush()
    print("", flush=True)

    if not frames:
        return np.array([], dtype=np.float32), None

    audio = np.concatenate(frames)
    live_text = " ".join(live_lines) if live_lines else None
    return audio, live_text


def countdown(seconds: int, prompt: str) -> None:
    print("\n" + "=" * 52)
    print(f"  {prompt}")
    print("=" * 52)
    print("\n  Get ready…\n")
    for remaining in range(seconds, 0, -1):
        print(f"    {remaining}…", flush=True)
        beep()
        time.sleep(0.85)
    print("\n  >>> SPEAK NOW! <<<\n", flush=True)
    beep_go()


def show_plots(audio: np.ndarray, sample_rate: int, label: str, out_path: Path) -> None:
    """Waveform, volume envelope, and spectrogram."""
    if len(audio) == 0:
        return

    times = np.arange(len(audio)) / sample_rate
    window = max(int(sample_rate * 0.02), 1)
    envelope = np.convolve(np.abs(audio), np.ones(window) / window, mode="same")

    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle(f"Recording: {label}", fontsize=14, fontweight="bold")

    axes[0].plot(times, audio, color="#2563eb", linewidth=0.8)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Waveform")
    axes[0].axhline(0, color="gray", linewidth=0.5)
    axes[0].set_xlim(0, times[-1])

    axes[1].plot(times, envelope, color="#16a34a", linewidth=1.2)
    axes[1].axhline(0.01, color="orange", linestyle="--", label="speech threshold")
    axes[1].set_ylabel("Level")
    axes[1].set_title("Volume envelope (green = your voice)")
    axes[1].legend(loc="upper right")
    axes[1].set_xlim(0, times[-1])

    axes[2].specgram(audio, Fs=sample_rate, cmap="magma", scale="dB")
    axes[2].set_ylabel("Frequency (Hz)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Spectrogram (frequency over time)")

    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"  Saved plot → {out_path}")
    plt.show(block=False)
    plt.pause(0.5)


def _safe_filename(label: str, *, max_len: int = 60) -> str:
    safe = label.replace(" ", "_").replace("/", "-")
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in safe)
    return safe[:max_len] or "recording"


def run_naming_round(
    target_word: str,
    word_list: list[str],
    *,
    strict: bool,
    save_dir: Path | None,
    record_seconds: float,
    speak_target: bool,
    hints_enabled: bool,
    hint_level: int,
    save_tts: bool = False,
) -> dict:
    tts_wav_path = None
    if speak_target or save_tts:
        print(f"  Kuvuga ijambo ugomba gusubiramo: {target_word}", flush=True)
        try:
            wav_bytes, _tts_rate = synthesize_kinyarwanda_wav_bytes(target_word)
            tts_dir = save_dir or Path("spikes/asr/samples")
            if save_tts:
                tts_dir.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                tts_wav_path = tts_dir / f"tts_{_safe_filename(target_word)}_{stamp}.wav"
                tts_wav_path.write_bytes(wav_bytes)
                print(f"  Saved TTS WAV → {tts_wav_path}")
            if speak_target:
                if not _play_wav_bytes(wav_bytes):
                    print("  [note] TTS ntabwo yabonetse kuri iyi mashini.", flush=True)
        except Exception as exc:
            print(f"  [note] TTS failed: {exc}", flush=True)

    countdown(COUNTDOWN_SECONDS, f"TARGET WORD:  {target_word.upper()}")
    audio, _ = record_with_meter(record_seconds, SAMPLE_RATE)

    stats = audio_stats(audio)
    print(f"  Recording stats: peak={stats['peak']:.4f}  rms={stats['rms']:.4f}")

    wav_path = None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_path = save_dir / f"{_safe_filename(target_word)}_{stamp}.wav"
        sf.write(wav_path, audio, SAMPLE_RATE)
        print(f"  Saved WAV → {wav_path}")

    plot_path = (save_dir or Path(".")) / f"_last_{target_word}_plot.png"
    show_plots(audio, SAMPLE_RATE, target_word, plot_path)

    try:
        validate_audio(audio)
    except Exception as exc:
        return {"error": str(exc), "target_word": target_word, "correct": False}

    print("  Transcribing…")
    transcript = transcribe(audio)
    score = score_naming_attempt(
        transcript,
        target_word,
        strict=strict,
        word_list=word_list,
    )

    result = {
        **score,
        "audio_stats": stats,
        "strict_scoring": strict,
        "saved_wav": str(wav_path) if wav_path else None,
        "saved_tts_wav": str(tts_wav_path) if tts_wav_path else None,
        "feedback_rw": (
            f"Ni byiza cyane! wavuze neza ijambo '{target_word}'."
            if score.get("correct")
            else f"Ongera ugerageze. Ijambo twari dushaka ni '{target_word}'."
        ),
    }
    if hints_enabled and not score.get("correct"):
        hint = next_hint_for_target(target_word, hint_level)
        result["hint"] = {
            "level": hint.level,
            "type": hint.hint_type,
            "message_rw": hint.hint_rw,
        }
    return result


def _default_free_label() -> str:
    return f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def run_free_speech_round(
    label: str,
    *,
    save_dir: Path | None,
    record_seconds: float,
    stream: bool = True,
) -> dict:
    if stream:
        print("  Loading model for live transcription…", flush=True)
        warmup_model()

    countdown(
        COUNTDOWN_SECONDS,
        "SPEAK FREELY  (one or more Kinyarwanda sentences)",
    )
    audio, live_transcript = record_with_meter(
        record_seconds,
        SAMPLE_RATE,
        live_transcript=stream,
    )

    stats = audio_stats(audio)
    print(f"  Recording stats: peak={stats['peak']:.4f}  rms={stats['rms']:.4f}")

    wav_path = None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        safe_name = label.replace(" ", "_").replace("/", "-")[:80]
        wav_path = save_dir / f"{safe_name}.wav"
        sf.write(wav_path, audio, SAMPLE_RATE)
        print(f"  Saved WAV → {wav_path}")

    plot_path = (save_dir or Path(".")) / "_last_free_speech_plot.png"
    show_plots(audio, SAMPLE_RATE, label, plot_path)

    try:
        validate_audio(audio)
    except Exception as exc:
        return {"error": str(exc), "label": label}

    print("  Finalizing transcript…")
    transcript = transcribe(audio)

    return {
        "label": label,
        "transcript": transcript,
        "live_transcript": live_transcript or "",
        "streamed": stream,
        "audio_stats": stats,
        "saved_wav": str(wav_path) if wav_path else None,
        "module": "free_speech",
    }


def print_naming_result(result: dict) -> None:
    print("\n" + "-" * 52)
    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return

    ok = result["correct"]
    icon = "✅ CORRECT" if ok else "❌ INCORRECT"
    print(f"  {icon}")
    print(f"  Target:     {result['target_word']}")
    print(f"  Heard:      {result.get('primary_word') or '(empty)'}")
    print(f"  Transcript: {result['transcript']}")
    if result.get("heard_instead"):
        print(f"  Sounds like: {result['heard_instead']} (score {result['heard_instead_score']})")
    print(f"  Score: {result['score']}  match: {result['match_type']}")
    if result.get("feedback_rw"):
        print(f"  Feedback(RW): {result['feedback_rw']}")
    hint = result.get("hint")
    if hint:
        print(f"  Hint(RW-L{hint['level']}): {hint['message_rw']}")
    print("-" * 52 + "\n")


def print_free_speech_result(result: dict) -> None:
    print("\n" + "-" * 52)
    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return

    if result.get("label") and not result["label"].startswith("recording_"):
        print(f"  Note:       {result['label']}")
    if result.get("streamed") and result.get("live_transcript"):
        print(f"  Live:       {result['live_transcript']}")
    print(f"  Transcript: {result['transcript']}")
    if result.get("saved_wav"):
        print(f"  Saved:      {result['saved_wav']}")
    print("-" * 52 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visual recording tester — countdown, live meter, waveform, transcription",
    )
    parser.add_argument(
        "--target",
        help="Naming mode: practice one word with scoring (omit for free speech)",
    )
    parser.add_argument(
        "--naming",
        action="store_true",
        help="Naming mode loop (prompt for target words; same as --target with no value)",
    )
    parser.add_argument(
        "--sentence",
        action="store_true",
        help="Alias for free speech mode (default when --target is not set)",
    )
    parser.add_argument(
        "--label",
        default="",
        help="Optional note for the recording (filename uses timestamp if omitted)",
    )
    parser.add_argument(
        "--words",
        nargs="*",
        default=DEFAULT_WORDS,
        help="Session vocabulary for confusion detection (naming mode)",
    )
    parser.add_argument(
        "--lenient",
        action="store_true",
        help="Lenient scoring (target anywhere in transcript)",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help=f"Recording length (default: {FREE_SPEECH_SECONDS}s free speech, {NAMING_RECORD_SECONDS}s naming)",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("spikes/asr/samples"),
        help="Directory to save WAV files",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save WAV files",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable live transcription during recording (free speech mode)",
    )
    parser.add_argument(
        "--no-say-target",
        action="store_true",
        help="Naming mode: do not play target word audio before recording",
    )
    parser.add_argument(
        "--no-hints",
        action="store_true",
        help="Naming mode: disable automatic Kinyarwanda hints when answer is wrong",
    )
    parser.add_argument(
        "--save-tts",
        action="store_true",
        help="Naming mode: save TTS prompt audio as WAV (uses --save directory)",
    )
    args = parser.parse_args()

    save_dir = None if args.no_save else args.save
    naming_mode = bool(args.target) or args.naming
    record_seconds = args.seconds
    if record_seconds is None:
        record_seconds = NAMING_RECORD_SECONDS if naming_mode else FREE_SPEECH_SECONDS

    print("\nKinyarwanda W2V recording tester")
    print("  • Countdown before recording")
    print("  • Live volume meter while you speak")
    print("  • Waveform + spectrogram after each attempt")
    print("  • Wav2Vec2-BERT transcription")
    if naming_mode:
        print("  • Naming mode — one word, scored against --target")
        print("  • Target word is spoken before each attempt")
        print("  • Kinyarwanda hints appear after wrong attempts")
    else:
        print("  • Free speech — say sentences, no target needed")
        print("  • Live transcript lines print while recording (~every 0.7s)")
        print(f"  • Recording for {record_seconds:.0f}s (use --seconds 30 for longer)")

    try:
        while True:
            if not naming_mode:
                label = args.label or _default_free_label()
                result = run_free_speech_round(
                    label,
                    save_dir=save_dir,
                    record_seconds=record_seconds,
                    stream=not args.no_stream,
                )
                print_free_speech_result(result)
                print(json.dumps(result, ensure_ascii=False, indent=2))

                again = input("Record again? [Enter=yes, q=quit]: ").strip().lower()
                if again in ("q", "quit", "exit"):
                    break
                continue

            target = args.target
            if not target:
                target = input("\nTarget word (or q to quit): ").strip()
                if target.lower() in ("q", "quit", "exit"):
                    break
            if not target:
                continue

            hint_level = 0
            result = run_naming_round(
                target,
                args.words,
                strict=not args.lenient,
                save_dir=save_dir,
                record_seconds=record_seconds,
                speak_target=not args.no_say_target,
                hints_enabled=not args.no_hints,
                hint_level=hint_level,
                save_tts=args.save_tts,
            )
            print_naming_result(result)
            print(json.dumps(result, ensure_ascii=False, indent=2))

            while not result.get("correct") and not args.target:
                if not args.no_hints and result.get("hint"):
                    hint_level = int(result["hint"]["level"])
                retry = input("Try same word again? [Enter=yes, n=next, q=quit]: ").strip().lower()
                if retry in ("q", "quit", "exit"):
                    return
                if retry in ("n", "next"):
                    break
                result = run_naming_round(
                    target,
                    args.words,
                    strict=not args.lenient,
                    save_dir=save_dir,
                    record_seconds=record_seconds,
                    speak_target=not args.no_say_target,
                    hints_enabled=not args.no_hints,
                    hint_level=hint_level,
                    save_tts=args.save_tts,
                )
                print_naming_result(result)
                print(json.dumps(result, ensure_ascii=False, indent=2))

            if args.target:
                break
            args.target = None

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
