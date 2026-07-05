from __future__ import annotations

import base64
import time
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from kinyarwanda_w2v_asr.audio_utils import AudioTooQuietError, audio_stats
from kinyarwanda_w2v_asr.hints import next_hint_for_target
from kinyarwanda_w2v_asr.scoring import score_naming_attempt
from kinyarwanda_w2v_asr.transcriber import load_audio_bytes, transcribe
from kinyarwanda_w2v_asr.tts import (
    _play_wav_bytes,
    synthesize_kinyarwanda_wav_bytes,
    tts_info,
)

app = FastAPI(
    title="Kinyarwanda ASR API",
    version="0.1.0",
    description="ASR, word matching, target prompting, and hints in Kinyarwanda.",
)


class SpeakRequest(BaseModel):
    target_text: str = Field(min_length=1, max_length=200)
    play: bool = Field(
        default=False,
        description="Play audio on the API server machine (optional).",
    )


class HintRequest(BaseModel):
    target_word: str = Field(min_length=1, max_length=120, examples=["indabo zanjye"])
    hint_level: int = Field(default=0, ge=0, le=4)
    semantic_hint_rw: str | None = Field(
        default=None,
        max_length=300,
        description=(
            "Optional lesson-specific semantic hint from content team. "
            "Leave empty to use built-in hints."
        ),
        examples=["Ni interuro ivuga ku ndabo zawe."],
    )


class ScoreRequest(BaseModel):
    target_word: str = Field(min_length=1, max_length=120)
    transcript: str = Field(default="", max_length=500)
    strict: bool = True
    word_list: list[str] | None = None


def _feedback_rw(target_word: str, correct: bool, heard: str | None = None) -> str:
    if correct:
        return f"Ni byiza cyane! wavuze neza ijambo '{target_word}'."
    if heard:
        return f"Ongera ugerageze. Wavuze '{heard}' aho kuba '{target_word}'."
    return f"Ongera ugerageze. Ijambo twari dushaka ni '{target_word}'."


@app.get("/")
def root() -> dict:
    return {
        "service": "Kinyarwanda ASR API",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "speak_target": "POST /v1/kinya/prompt/speak",
            "speak_target_wav": "GET /v1/kinya/prompt/speak/wav?text=amata",
            "transcribe": "POST /v1/kinya/asr/transcribe",
            "score_match": "POST /v1/kinya/score/match",
            "hints": "POST /v1/kinya/hints/next",
            "evaluate_attempt": "POST /v1/kinya/attempt/evaluate",
        },
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _wav_response(wav_bytes: bytes, *, filename: str = "kinya_tts.wav") -> Response:
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@app.get("/v1/kinya/prompt/speak/wav")
def prompt_speak_wav_get(
    text: str = Query(..., min_length=1, max_length=200),
    play: bool = Query(
        default=False,
        description="Play audio on the API server speakers (Mac only).",
    ),
) -> Response:
    """Return WAV audio directly — open this URL in a browser to hear it."""
    try:
        wav_bytes, _ = synthesize_kinyarwanda_wav_bytes(text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS failed: {exc}") from exc

    if play:
        _play_wav_bytes(wav_bytes)
    return _wav_response(wav_bytes)


@app.post("/v1/kinya/prompt/speak")
def prompt_speak(payload: SpeakRequest) -> dict:
    try:
        wav_bytes, sample_rate = synthesize_kinyarwanda_wav_bytes(payload.target_text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS failed: {exc}") from exc

    played = False
    if payload.play:
        played = _play_wav_bytes(wav_bytes)

    listen_path = f"/v1/kinya/prompt/speak/wav?text={quote(payload.target_text)}"
    return {
        "ok": True,
        "target_text": payload.target_text,
        "language": "rw",
        "audio_base64": base64.b64encode(wav_bytes).decode("ascii"),
        "sample_rate": sample_rate,
        "played_on_server": played,
        "listen_url": listen_path,
        "how_to_hear": (
            "Set play=true to hear on this Mac, or open listen_url in your browser."
        ),
        **tts_info(),
        "feedback_rw": "Ijambo ryakinnye mu Kinyarwanda.",
    }


@app.post("/v1/kinya/prompt/speak/wav")
def prompt_speak_wav_post(payload: SpeakRequest) -> Response:
    """Same as GET /wav but accepts JSON body."""
    try:
        wav_bytes, _ = synthesize_kinyarwanda_wav_bytes(payload.target_text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"TTS failed: {exc}") from exc

    if payload.play:
        _play_wav_bytes(wav_bytes)
    return _wav_response(wav_bytes)


@app.post("/v1/kinya/hints/next")
def hints_next(payload: HintRequest) -> dict:
    hint = next_hint_for_target(
        payload.target_word,
        payload.hint_level,
        semantic_hint_rw=payload.semantic_hint_rw,
    )
    return {
        "ok": True,
        "target_word": payload.target_word,
        "hint": {
            "level": hint.level,
            "type": hint.hint_type,
            "message_rw": hint.hint_rw,
        },
    }


@app.post("/v1/kinya/score/match")
def score_match(payload: ScoreRequest) -> dict:
    result = score_naming_attempt(
        payload.transcript,
        payload.target_word,
        strict=payload.strict,
        word_list=payload.word_list,
    )
    return {
        "ok": True,
        **result,
        "feedback_rw": _feedback_rw(
            payload.target_word,
            bool(result.get("correct")),
            result.get("primary_word"),
        ),
    }


@app.post("/v1/kinya/asr/transcribe")
async def asr_transcribe(
    audio_file: UploadFile = File(...),
    sample_rate: int = Form(default=16000),
) -> dict:
    start = time.perf_counter()
    raw = await audio_file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Audio file is empty.")

    try:
        audio = load_audio_bytes(raw, sample_rate=sample_rate)
        stats = audio_stats(audio)
        transcript = transcribe(audio, sample_rate=sample_rate)
    except AudioTooQuietError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Transcription failed: {exc}") from exc

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return {
        "ok": True,
        "language": "rw",
        "transcript": transcript,
        "audio_stats": stats,
        "latency_ms": elapsed_ms,
    }


@app.post("/v1/kinya/attempt/evaluate")
async def attempt_evaluate(
    target_word: str = Form(...),
    audio_file: UploadFile = File(...),
    strict: bool = Form(default=True),
    hint_level: int = Form(default=0),
    sample_rate: int = Form(default=16000),
    word_list_csv: str = Form(default=""),
    semantic_hint_rw: str = Form(default=""),
) -> dict:
    start = time.perf_counter()
    raw = await audio_file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Audio file is empty.")

    words = [w.strip() for w in word_list_csv.split(",") if w.strip()]
    if target_word and target_word not in words:
        words.append(target_word)

    try:
        audio = load_audio_bytes(raw, sample_rate=sample_rate)
        stats = audio_stats(audio)
        transcript = transcribe(audio, sample_rate=sample_rate)
    except AudioTooQuietError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Transcription failed: {exc}") from exc

    score = score_naming_attempt(
        transcript,
        target_word,
        strict=strict,
        word_list=words or None,
    )

    response = {
        "ok": True,
        "target_word": target_word,
        "transcript": transcript,
        "audio_stats": stats,
        "score": score,
        "feedback_rw": _feedback_rw(
            target_word,
            bool(score.get("correct")),
            score.get("primary_word"),
        ),
        "latency_ms": int((time.perf_counter() - start) * 1000),
    }

    if not score.get("correct"):
        hint = next_hint_for_target(
            target_word,
            hint_level,
            semantic_hint_rw=semantic_hint_rw or None,
        )
        response["hint"] = {
            "level": hint.level,
            "type": hint.hint_type,
            "message_rw": hint.hint_rw,
        }

    return response
