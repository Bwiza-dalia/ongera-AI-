from __future__ import annotations

import re
from dataclasses import dataclass

VOWELS = set("aeiou")


@dataclass(frozen=True)
class HintStep:
    level: int
    hint_rw: str
    hint_type: str


# Lesson content can be extended; keys are normalized lowercase.
_TARGET_HINTS_RW: dict[str, str] = {
    "amata": "Ni ikinyobwa kiva ku nka, gikunze kunyobwa n'abana.",
    "amazi": "Ni ikintu cy'ingenzi mu buzima, turanywa buri munsi.",
    "umugati": "Ni ifunguro rikorwa mu ifarini, rikunze kuribwa mu gitondo.",
    "amafi": "Ni inyamaswa zo mu mazi, kandi ziraribwa.",
    "umuceri": "Ni ibinyampeke byera, bikunze gutekwa nk'ifunguro nyamukuru.",
    "ibirayi": "Ni ibinyampeke bikunze gutekwa, biterwa mu butaka.",
    "indabo": "Ni ibyatsi bifite amabara meza, bikunze mu busitani.",
    "indabo zanjye": "Ni interuro ivuga ku ndabo zawe — ibyatsi byoroshye bifite amabara.",
    "guhaha ibirayi": "Ni igikorwa cyo kugura ibirayi ku isoko.",
    "gura ibirayi": "Ni igikorwa cyo kugura ibirayi ku isoko.",
    "gura inka": "Ni igikorwa cyo kugura inka ku isoko.",
}

# Swagger / Postman often send these placeholder values unchanged.
_IGNORED_CUSTOM_HINTS = {"string", "example", "null", "none", "n/a", "na"}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _words(text: str) -> list[str]:
    return [w for w in _normalize(text).split() if w]


def _is_phrase(text: str) -> bool:
    return len(_words(text)) > 1


def _syllabify_word(word: str) -> str:
    """Simple Kinyarwanda-friendly syllable spacing for pronunciation hints."""
    syllables: list[str] = []
    current = ""
    for char in word:
        current += char
        if char in VOWELS:
            syllables.append(current)
            current = ""
    if current:
        if syllables:
            syllables[-1] += current
        else:
            syllables.append(current)
    return "-".join(syllables) if syllables else word


def _clean_custom_hint(custom_hint: str | None) -> str | None:
    if not custom_hint:
        return None
    cleaned = custom_hint.strip()
    if not cleaned or cleaned.lower() in _IGNORED_CUSTOM_HINTS:
        return None
    return cleaned


def _semantic_hint(target: str, custom_hint: str | None) -> str:
    lesson_hint = _clean_custom_hint(custom_hint)
    if lesson_hint:
        return lesson_hint
    return _TARGET_HINTS_RW.get(
        _normalize(target),
        "Tekereza ku ishusho cyangwa igikorwa usobanurwa n'ikibazo.",
    )


def next_hint_for_target(
    target_word: str,
    hint_level: int,
    *,
    semantic_hint_rw: str | None = None,
) -> HintStep:
    """
    Progressive cueing ladder in Kinyarwanda.

    Single word:
      L1 semantic -> L2 first syllables -> L3 full pronunciation guide

    Phrase (e.g. "guhaha ibirayi"):
      L1 semantic -> L2 phrase structure -> L3 first-word starter -> L4 full phrase guide
    """
    target = _normalize(target_word)
    words = _words(target)
    if not words:
        return HintStep(level=1, hint_rw="Ongera ugerageze.", hint_type="generic")

    semantic = _semantic_hint(target, semantic_hint_rw)

    if hint_level <= 0:
        return HintStep(
            level=1,
            hint_rw=f"Icyitonderwa: {semantic}",
            hint_type="semantic",
        )

    if _is_phrase(target):
        if hint_level == 1:
            joined = "', '".join(words)
            return HintStep(
                level=2,
                hint_rw=f"Ubutumwa bufite amagambo {len(words)}: '{joined}'.",
                hint_type="structure",
            )
        if hint_level == 2:
            starter = _syllabify_word(words[0])
            return HintStep(
                level=3,
                hint_rw=f"Tangira ukavuga '{starter}...' hanyuma ukomeze.",
                hint_type="starter",
            )
        guide = " ".join(_syllabify_word(word) for word in words)
        return HintStep(
            level=4,
            hint_rw=f"Vuga buhoro: '{guide}'.",
            hint_type="pronunciation",
        )

    # Single-word ladder
    if hint_level == 1:
        starter = _syllabify_word(words[0])
        return HintStep(
            level=2,
            hint_rw=f"Ijambo ritangirira kuri '{starter}...'.",
            hint_type="syllable",
        )

    guide = _syllabify_word(words[0])
    return HintStep(
        level=3,
        hint_rw=f"Ongera witonze: vuga uti '{guide}'.",
        hint_type="pronunciation",
    )
