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
    "tells me to", "told me to", "telling me to", "wants me to", "want me to",
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
    words = re.findall(r"[a-z0-9']+", lowered)

    visual_hits = sum(1 for cue in _VISUAL_CUES if cue in lowered)
    audio_hits = sum(1 for cue in _AUDITORY_CUES if cue in lowered)

    if visual_hits and audio_hits:
        modality = Modality.MIXED
    elif audio_hits:
        modality = Modality.AUDITORY
    elif visual_hits:
        modality = Modality.VISUAL
    else:
        modality = Modality.OTHER

    entities: List[str] = []
    for canonical, synonyms in ENTITY_LEXICON.items():
        if any(re.search(rf"\b{re.escape(s)}\b", lowered) for s in synonyms):
            entities.append(canonical)

    is_command = any(cue in lowered for cue in _COMMAND_CUES)
    asks_for_check = any(
        phrase in lowered
        for phrase in ("is this real", "is that real", "check the room", "can you see",
                       "can you hear", "am i seeing", "am i hearing", "is it there",
                       "reality check", "check reality")
    )

    return Claim(
        text=(text or "").strip(),
        modality=modality,
        entities=entities,
        is_command=is_command,
        asks_for_check=asks_for_check or not text.strip(),
        word_count=len(words),
    )
