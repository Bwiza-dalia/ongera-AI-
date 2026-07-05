#!/usr/bin/env bash
# Record a short Kinyarwanda sample on macOS.
# Usage: ./record_sample.sh amafi
# Saves to spikes/asr/samples/<word>.wav (16 kHz mono)

set -euo pipefail

WORD="${1:?Usage: $0 <word>   e.g. $0 amafi}"
OUT_DIR="$(cd "$(dirname "$0")/samples" && pwd)"
OUT_FILE="${OUT_DIR}/${WORD}.wav"
MIC_DEVICE="${MIC_DEVICE:-:0}"
DURATION="${RECORD_SECONDS:-4}"

echo "Recording '${WORD}' → ${OUT_FILE}"
echo "Microphone device: ${MIC_DEVICE} (override with MIC_DEVICE=:1 etc.)"
echo ""
echo "List devices: ffmpeg -f avfoundation -list_devices true -i \"\""
echo ""
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Error: install ffmpeg first: brew install ffmpeg"
  exit 1
fi

beep() { afplay /System/Library/Sounds/Tink.aiff 2>/dev/null || printf '\a'; }
beep_go() { afplay /System/Library/Sounds/Glass.aiff 2>/dev/null || printf '\a\a'; }

echo "TARGET: ${WORD}"
echo "Get ready…"
for i in 3 2 1; do
  echo "  ${i}…"
  beep
  sleep 0.85
done
echo ""
echo ">>> SPEAK NOW! <<<"
beep_go
sleep 0.3

if ! ffmpeg -y -loglevel error -f avfoundation -i "${MIC_DEVICE}" -ar 16000 -ac 1 -t "${DURATION}" "${OUT_FILE}"; then
  echo "Error: recording failed. Check mic permissions."
  exit 1
fi

echo ""
echo "Saved ${OUT_FILE}"

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ -f "${SCRIPT_DIR}/.venv/bin/python" ]]; then
  "${SCRIPT_DIR}/.venv/bin/python" - <<PY
import sys
import numpy as np
import librosa

audio, _ = librosa.load("${OUT_FILE}", sr=16000)
peak = float(np.max(np.abs(audio)))
rms = float(np.sqrt(np.mean(audio ** 2)))
print(f"Peak: {peak:.4f}  RMS: {rms:.4f}")
if peak < 0.02 and rms < 0.005:
    print("WARNING: Recording looks too quiet — ASR may fail or repeat.")
    print("Try: MIC_DEVICE=:1 ./spikes/asr/record_sample.sh ${WORD}")
    sys.exit(3)
print("Volume OK.")
PY
  VOL_OK=$?
  if [[ ${VOL_OK} -ne 0 ]]; then
    exit ${VOL_OK}
  fi
fi

echo "Test with:"
echo "  kin-w2v-transcribe ${OUT_FILE}"
echo "  kin-w2v-test --target ${WORD}"
