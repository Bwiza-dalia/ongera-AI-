# ASR sample audio

Add test recordings here named `<target_word>.wav` or any label for sentences.

## Record on macOS (ffmpeg)

From the `kinyarwanda-w2v-asr` directory:

```bash
chmod +x spikes/asr/record_sample.sh
./spikes/asr/record_sample.sh amafi
```

Requires `ffmpeg` (`brew install ffmpeg`).

## Interactive recording (live meter + plots)

```bash
kin-w2v-test --target amafi
kin-w2v-test --sentence --label "mutima muke wo mu rutiba" --seconds 5
```

Shows countdown, live volume bar while recording, then waveform/spectrogram and transcription.

## Transcribe a saved file

```bash
kin-w2v-transcribe spikes/asr/samples/amafi.wav
```

Requirements: WAV, 16 kHz mono preferred (other rates are resampled automatically).
