"""Turn a patient's sentence into a structured claim the engine can test.

This is intentionally a transparent, inspectable lexicon rather than a black box:
when the app says "I understood you as a visual report about a person in the
room", that reasoning should be checkable by a clinician.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from ..models import Modality

_VISUAL_CUES = (
    "see", "seeing", "saw", "look", "looking", "watch", "watching", "staring",
    "standing", "there's", "there is", "there was", "in the corner", "shadow",
    "figure", "silhouette", "face", "eyes", "hand", "apparition", "ghost",
    "camera", "screen", "light", "flash", "blood", "bug", "spider", "snake",
    "insect", "moving", "crawl", "out the window", "behind me",
)

_AUDITORY_CUES = (
    "hear", "hearing", "heard", "voice", "voices", "whisper", "whispering",
    "talking", "speaking", "says", "said", "noise", "music", "buzzing",
    "humming", "screaming", "calling my name", "knocking", "static",
    "tells me", "told me",
)

_COMMAND_CUES = (
    "tells me to", "tell me to", "told me to", "telling me to", "wants me to", "want me to",
    "says i should", "said i should", "commands me", "commanding me",
    "orders me", "is making me", "makes me do", "told me not to tell",
    "said not to tell", "tells me not to tell", "says to hurt",
)

# Entities the vision layer can plausibly detect, plus common benign objects
# that get reported during psychosis. Keys are canonical entity names.
ENTITY_LEXICON = {
    "person": ("person", "man", "woman", "someone", "somebody", "figure", "guy",
               "lady", "child", "kid", "people", "men", "women", "intruder",
               "stranger", "visitor", "apparition", "ghost", "shadow"),
    "face": ("face", "faces", "eyes", "staring", "smiling"),
    "hand": ("hand", "hands", "fingers"),
    "animal": ("dog", "cat", "rat", "mouse", "bird", "snake", "spider", "bug",
               "bugs", "insect", "cockroach", "worm"),
    "camera": ("camera", "cameras", "lens", "webcam"),
    "phone": ("phone", "smartphone", "device"),
    "vehicle": ("car", "van", "truck", "vehicle"),
    "object": ("box", "bag", "chair", "door", "window", "mirror", "tv",
               "television", "computer", "lamp"),
}

_CHECK_CUES = (
    "is this real", "is that real", "check the room", "can you see",
    "can you hear", "am i seeing", "am i hearing", "is it there",
    "reality check", "check reality",
)

_WORD_RE = re.compile(r"[a-z0-9']+")

#: One alternation per cue family instead of one regex search per cue. This is
#: behaviourally identical to ``any(cue in text for cue in cues)`` (and, for the
#: entity lexicon, to ``any(\bword\b in text ...)``); it just replaces ~50 separate
#: searches per check with one, which is the only hot loop in claim parsing.

def _any_cue(cues: tuple) -> "re.Pattern[str]":
    ordered = sorted(cues, key=len, reverse=True)
    return re.compile("|".join(re.escape(cue) for cue in ordered))


def _whole_word(words: tuple) -> "re.Pattern[str]":
    ordered = sorted(words, key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(word) for word in ordered) + r")\b")


_VISUAL_RE = _any_cue(_VISUAL_CUES)
_AUDITORY_RE = _any_cue(_AUDITORY_CUES)
_COMMAND_RE = _any_cue(_COMMAND_CUES)
_CHECK_RE = _any_cue(_CHECK_CUES)
_ENTITY_RES = {
    canonical: _whole_word(synonyms) for canonical, synonyms in ENTITY_LEXICON.items()
}


_ENTITY_TO_LABEL = {
    "person": "person",
    "face": "person",
    "hand": "person",
    "animal": "animal",
    "camera": "camera",
    "phone": "phone",
    "vehicle": "vehicle",
    "object": "object",
}


@dataclass
class Claim:
    text: str
    modality: Modality
    entities: List[str] = field(default_factory=list)
    is_command: bool = False
    asks_for_check: bool = False
    word_count: int = 0

    @property
    def sensor_target(self) -> List[str]:
        """Vision labels that would corroborate this claim."""
        return sorted({_ENTITY_TO_LABEL[e] for e in self.entities if e in _ENTITY_TO_LABEL})

    def public(self) -> dict:
        return {
            "text": self.text,
            "modality": self.modality.value,
            "entities": list(self.entities),
            "is_command": self.is_command,
            "sensor_target": self.sensor_target,
        }


def parse_claim(text: str) -> Claim:
    lowered = (text or "").lower()
    words = _WORD_RE.findall(lowered)

    visual = _VISUAL_RE.search(lowered) is not None
    audio = _AUDITORY_RE.search(lowered) is not None

    if visual and audio:
        modality = Modality.MIXED
    elif audio:
        modality = Modality.AUDITORY
    elif visual:
        modality = Modality.VISUAL
    else:
        modality = Modality.OTHER

    entities: List[str] = [
        canonical
        for canonical, pattern in _ENTITY_RES.items()
        if pattern.search(lowered) is not None
    ]

    is_command = _COMMAND_RE.search(lowered) is not None
    asks_for_check = _CHECK_RE.search(lowered) is not None

    return Claim(
        text=(text or "").strip(),
        modality=modality,
        entities=entities,
        is_command=is_command,
        asks_for_check=asks_for_check or not text.strip(),
        word_count=len(words),
    )
