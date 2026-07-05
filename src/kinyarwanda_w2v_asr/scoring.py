import re
import unicodedata

from rapidfuzz import fuzz

from kinyarwanda_w2v_asr.config import DEFAULT_FUZZY_THRESHOLD


def normalize_rw(text: str) -> str:
    """Normalize Kinyarwanda text for comparison."""
    text = unicodedata.normalize("NFC", text.lower().strip())
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def extract_primary_word(transcript: str) -> str:
    """First spoken word in the transcript."""
    words = normalize_rw(transcript).split()
    return words[0] if words else ""


def score_naming_attempt(
    transcript: str,
    target_word: str,
    *,
    threshold: int = DEFAULT_FUZZY_THRESHOLD,
    strict: bool = True,
    word_list: list[str] | None = None,
) -> dict:
    """
    Score a naming exercise attempt against the expected word.

    strict=True: scores only the first word heard (helps with CTC repetitions).
    """
    normalized_target = normalize_rw(target_word)
    if not normalized_target:
        raise ValueError("target_word must not be empty")

    normalized_transcript = normalize_rw(transcript)
    compared = extract_primary_word(transcript) if strict else normalized_transcript

    if not compared:
        return _result(transcript, target_word, False, 0, "none", primary_word="")

    if strict and word_list:
        confusion = _check_confusion(compared, target_word, word_list, threshold)
        if confusion:
            return confusion

    if compared == normalized_target:
        return _result(transcript, target_word, True, 100, "exact", primary_word=compared)

    fuzzy_score = fuzz.ratio(compared, normalized_target)
    if fuzzy_score >= threshold:
        return _result(transcript, target_word, True, fuzzy_score, "fuzzy", primary_word=compared)

    if not strict and normalized_target in normalized_transcript.split():
        return _result(transcript, target_word, True, 95, "token", primary_word=compared)

    return _result(transcript, target_word, False, fuzzy_score, "none", primary_word=compared)


def _check_confusion(
    primary: str,
    target_word: str,
    word_list: list[str],
    threshold: int,
) -> dict | None:
    normalized_target = normalize_rw(target_word)
    target_score = fuzz.ratio(primary, normalized_target)

    best_other = ""
    best_other_score = 0
    for word in word_list:
        normalized = normalize_rw(word)
        if normalized == normalized_target:
            continue
        score = fuzz.ratio(primary, normalized)
        if score > best_other_score:
            best_other_score = score
            best_other = word

    if best_other_score >= threshold and best_other_score > target_score + 5:
        result = _result(
            primary,
            target_word,
            False,
            target_score,
            "confusion",
            primary_word=primary,
        )
        result["heard_instead"] = best_other
        result["heard_instead_score"] = round(best_other_score, 1)
        return result
    return None


def _result(
    transcript: str,
    target_word: str,
    correct: bool,
    score: float,
    match_type: str,
    *,
    primary_word: str = "",
) -> dict:
    return {
        "transcript": transcript,
        "primary_word": primary_word,
        "target_word": target_word,
        "correct": correct,
        "score": round(score, 1),
        "match_type": match_type,
        "module": "naming",
    }
