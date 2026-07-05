import os

MODEL_ID = os.environ.get(
    "KIN_W2V_MODEL",
    "badrex/w2v-bert-2.0-kinyarwanda-asr-1000h",
)
SAMPLE_RATE = 16_000
CHUNK_DURATION_SEC = float(os.environ.get("KIN_W2V_CHUNK_SEC", "30"))
MAX_DURATION_SEC = float(os.environ.get("KIN_W2V_MAX_SEC", "60"))
DEFAULT_FUZZY_THRESHOLD = int(os.environ.get("KIN_W2V_FUZZY_THRESHOLD", "85"))
TTS_MODEL_ID = os.environ.get("KIN_W2V_TTS_MODEL", "facebook/mms-tts-kin")
