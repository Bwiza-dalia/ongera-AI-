# kinyarwanda-w2v-asr

Standalone Kinyarwanda automatic speech recognition using the Hugging Face model [`badrex/w2v-bert-2.0-kinyarwanda-asr-1000h`](https://huggingface.co/badrex/w2v-bert-2.0-kinyarwanda-asr-1000h) (Wav2Vec2-BERT CTC, trained on ~1000h of Kinyarwanda).

This project is independent — it does not use or depend on `ongera-ai` or Whisper.

## Setup

```bash
cd kinyarwanda-w2v-asr
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

The first run downloads the model from Hugging Face (~1–2 GB).

## Record and test (like ongera-ai)

**Interactive** — countdown, live volume meter, **live transcript while you speak**, waveform/spectrogram, final transcription:

```bash
# Free speech (default) — live lines print while recording (~every 0.7s)
kin-w2v-test
kin-w2v-test --seconds 25

# Optional note for the saved filename
kin-w2v-test --label "umutima muke wo mu rutiba" --seconds 8

# Transcript only after you stop (old behavior)
kin-w2v-test --no-stream

# Naming exercise — one word, spoken prompt + scoring + Kinyarwanda hints
kin-w2v-test --target amafi
kin-w2v-test --naming   # loop: prompts for target words

# Optional toggles for naming mode
kin-w2v-test --target amata --no-say-target
kin-w2v-test --target amata --no-hints
```

**Quick record** (ffmpeg, macOS):

```bash
chmod +x spikes/asr/record_sample.sh
./spikes/asr/record_sample.sh amafi
```

Samples save to `spikes/asr/samples/`.

## Transcribe a file

```bash
kin-w2v-transcribe path/to/audio.wav
```

Example output:

```json
{
  "transcript": "amazi",
  "duration_sec": 1.2,
  "processing_time_sec": 0.45,
  "model": "badrex/w2v-bert-2.0-kinyarwanda-asr-1000h"
}
```

Long audio (>60s by default) is processed in 30s chunks automatically.

## HTTP API (Postman-ready)

Start API server:

```bash
kin-w2v-api --host 127.0.0.1 --port 8000
```

Swagger docs:

- `http://127.0.0.1:8000/docs`

Core endpoints:

- `GET /health`
- `POST /v1/kinya/prompt/speak`
- `POST /v1/kinya/asr/transcribe` (form-data: `audio_file`, optional `sample_rate`)
- `POST /v1/kinya/score/match`
- `POST /v1/kinya/hints/next`
- `POST /v1/kinya/attempt/evaluate` (single call: ASR + score + hint)

Example Postman JSON body (`/v1/kinya/score/match`):

```json
{
  "target_word": "amata",
  "transcript": "amazi",
  "strict": true,
  "word_list": ["amata", "amazi", "umugati"]
}
```

## Python API

```python
from kinyarwanda_w2v_asr import load_audio, transcribe

audio = load_audio("recording.wav")
text = transcribe(audio)
print(text)
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `KIN_W2V_MODEL` | `badrex/w2v-bert-2.0-kinyarwanda-asr-1000h` | Hugging Face model id |
| `KIN_W2V_DEVICE` | auto (`cuda` / `mps` / `cpu`) | Force inference device |
| `KIN_W2V_CHUNK_SEC` | `30` | Chunk length for long audio |
| `KIN_W2V_MAX_SEC` | `60` | Duration threshold before chunking |
| `KIN_W2V_TTS_MODEL` | `facebook/mms-tts-kin` | Kinyarwanda TTS model id |

## Project layout

```
kinyarwanda-w2v-asr/
├── pyproject.toml
├── README.md
├── spikes/asr/
│   ├── record_sample.sh
│   └── samples/
└── src/kinyarwanda_w2v_asr/
    ├── config.py
    ├── audio_utils.py
    ├── scoring.py
    ├── transcriber.py
    └── cli/
        ├── transcribe.py
        └── live_test.py
```
