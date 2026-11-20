"""Transparent cue lexicons. All matching is whole-word / whole-phrase (regex word boundaries),
fixing the v2 substring bug where "no" matched "know" and "real" matched "really".

Each category has a strength in [0, 1] used by the hook / cliffhanger detectors.
"""

from __future__ import annotations

import re
from functools import lru_cache

CUES = {
    # physical confrontation — usually in ACTION lines
    "physical": (1.0, [
        "slap", "slaps", "slapped", "strike", "strikes", "struck", "punch", "punches", "punched",
        "shove", "shoves", "shoved", "push", "pushes", "pushed", "grab", "grabs", "grabbed",
        "seizes", "seized", "chokes", "throttles", "stab", "stabs", "stabbed", "draws his sword",
        "draws her sword", "draws a dagger", "knife", "dagger", "blade", "sword", "gun", "pistol",
        "blood", "bleeding", "collapses", "falls to the floor", "kicks", "kicked", "hits", "attack", "attacks",
        "wrenches", "yanks", "drags", "throws",
    ]),
    "threat": (0.9, [
        "i will kill", "i'll kill", "kill you", "kill him", "kill her", "you will pay", "you'll pay",
        "you will regret", "you'll regret", "or else", "you will die", "you'll die", "dead by", "destroy you",
        "ruin you", "i will ruin", "i'll ruin", "watch your back", "you have until", "make you suffer",
        "nobody will save", "no one will save", "if you tell", "you're finished", "you are finished",
        "i will make sure", "you will never", "you'll never", "get rid of", "poison", "throw you out",
        "cast you out", "take everything",
    ]),
    "betrayal": (0.9, [
        "betray", "betrayed", "betrayal", "traitor", "liar", "lied to me", "you lied", "behind my back",
        "you swore", "you promised", "how could you", "how dare you", "how dare", "cheated", "affair",
        "deceived", "deceive", "used me", "all along", "framed", "abandon", "abandoned",
    ]),
    "revelation": (0.85, [
        "the truth", "truth is", "secret", "secrets", "i know who", "i know what you", "you knew",
        "he knew", "she knew", "found out", "it was you", "it was him", "it was her", "is alive",
        "still alive", "not dead", "is dead", "pregnant", "your real", "my real", "real father",
        "real mother", "true identity", "inheritance", "confess", "confession",
        "never told", "hidden", "disguise", "in disguise", "revealed", "reveal", "exposed", "proof",
    ]),
    "secret_setup": (0.9, [
        "i need to tell you", "i have to tell you", "there is something", "there's something",
        "you need to know", "you don't know who", "you have no idea who", "you have no idea what",
        "if you knew", "what i'm about to", "what i am about to", "the reason is", "listen to me",
        "do you know who", "do you know what", "who i really am", "who he really is", "who she really is",
        "the truth about", "what really happened", "before i tell", "let me explain",
    ]),
    "shock": (0.7, [
        "what?!", "no!", "impossible", "it can't be", "it cannot be", "you!", "how is this possible",
        "what have you done", "what is this", "what is the meaning of this", "oh my god", "dear god",
        "good heavens", "gasps", "gasp", "freezes", "stunned", "shocked", "horrified", "speechless",
        "stops dead", "frozen", "turns pale", "shatters",
    ]),
    "arrival": (0.75, [
        "bursts in", "bursts through", "enters", "steps in", "steps out", "appears", "emerges",
        "storms in", "arrives", "doors swing open", "doors burst open", "doors fly open",
        "a figure", "silhouette", "footsteps", "someone is watching", "watching from",
    ]),
    "discovery": (0.6, [
        "discovers", "notices", "sees", "spots", "realizes", "realises", "recognizes", "recognises",
        "finds", "uncovers", "reads the letter", "the letter", "the ring", "the pendant", "the seal",
        "the document", "the mark", "the scar", "birthmark",
    ]),
    "stakes": (0.6, [
        "marry", "married", "marriage", "wedding", "husband", "wife", "bride", "fiance", "fiancé",
        "sister", "brother", "father", "mother", "son", "daughter", "child", "baby", "family",
        "duke", "duchess", "title", "estate", "fortune", "inheritance", "love", "die", "death",
        "kill", "life", "honor", "honour", "disgrace", "shame", "throne", "war", "soldier",
    ]),
}

# phrases that signal the situation is being resolved/closed — bad promo endings
RESOLUTION = [
    "thank you", "thanks", "goodbye", "good night", "goodnight", "it's alright", "it is alright",
    "it's all right", "it's okay", "it is okay", "all is well", "everything is fine", "everything will be fine",
    "i understand", "i forgive you", "never mind", "as you wish", "very well", "of course", "rest now",
    "let's go home", "we are safe", "we're safe", "it's over", "it is over", "welcome", "you're welcome",
]

# language that only makes sense with knowledge of earlier episodes
CONTEXT_DEPENDENT = [
    "last night", "yesterday", "remember when", "you remember", "as i said", "like i said",
    "as you said", "like before", "as before", "earlier", "the other day", "that night", "back then",
    "once more", "after what happened", "after everything",
]

PRONOUN_STARTS = ("he ", "she ", "they ", "him ", "her ", "it ", "that ", "this ", "those ", "them ", "his ", "their ")
CONJUNCTION_STARTS = ("and ", "but ", "so ", "because ", "or ", "then ", "...", "…")

# words in the script's emotion tags, e.g. "(Cold, Commanding)", and the intensity they imply
TAG_INTENSITY = {
    3: ["furious", "enraged", "screaming", "shrieking", "horrified", "terrified", "hysterical", "murderous",
        "desperate", "panicked", "anguished", "devastated", "explosive", "livid", "seething", "venomous",
        "menacing", "threatening", "shocked", "stunned", "betrayed", "sobbing", "broken", "wrathful", "roaring"],
    2: ["angry", "cold", "cruel", "contemptuous", "defiant", "bitter", "accusing", "urgent", "fearful",
        "afraid", "gloating", "mocking", "sneering", "icy", "hostile", "tearful", "pleading", "trembling",
        "breathless", "gasping", "commanding", "fierce", "vicious", "heartbroken", "disgusted", "smug",
        "ruthless", "callous", "haughty", "outraged", "piercing", "grim", "dangerous", "warning", "alarmed"],
    1: ["confused", "surprised", "worried", "anxious", "firm", "sharp", "tense", "suspicious", "sad",
        "hurt", "quiet", "pained", "hesitant", "nervous", "uneasy", "stern", "serious", "determined"],
}


@lru_cache(maxsize=None)
def _pattern(phrase: str) -> re.Pattern:
    esc = re.escape(phrase.lower())
    left = r"(?<![\w'])" if phrase[0].isalnum() else ""
    right = r"(?![\w'])" if phrase[-1].isalnum() else ""
    return re.compile(left + esc + right)


def find_phrases(text: str, phrases) -> list:
    low = text.lower().replace("’", "'")
    return [p for p in phrases if _pattern(p).search(low)]


def cue_hits(text: str) -> dict:
    """category -> list of matched phrases (only non-empty categories)."""
    out = {}
    for cat, (_, phrases) in CUES.items():
        hits = find_phrases(text, phrases)
        if hits:
            out[cat] = hits
    return out


# Physical verbs are reliable in ACTION lines ("slaps her") but often figurative or harmless in speech
# ("let me grab my ledger"), so they count for less in dialogue.
DIALOGUE_PHYSICAL_STRENGTH = 0.5

# de-escalation in ACTION lines — a scene settling down is not a cliffhanger
ACTION_RESOLUTION = [
    "lower their weapon", "lowers his weapon", "lowers her weapon", "lower their weapons", "sheathes", "embraces",
    "embrace", "hugs", "relief", "relaxes", "smiles warmly", "smiles softly", "falls asleep", "kisses her forehead",
    "settles", "calm returns", "peacefully",
]


def cue_strength(hits: dict, exclude=("stakes",), kind: str = "action") -> float:
    best = 0.0
    for cat in hits:
        if cat in exclude:
            continue
        w = CUES[cat][0]
        if cat == "physical" and kind == "dialogue":
            w = DIALOGUE_PHYSICAL_STRENGTH
        best = max(best, w)
    return best


def tag_intensity(tag: str) -> float:
    """Map the script's own emotion annotation to [0, 1]."""
    if not tag:
        return 0.0
    words = re.findall(r"[a-z]+", tag.lower())
    best = 0
    for level, vocab in TAG_INTENSITY.items():
        if any(w in vocab for w in words):
            best = max(best, level)
    return best / 3.0


def is_question(text: str) -> bool:
    return "?" in text


def is_interrupted(text: str) -> bool:
    t = text.rstrip().rstrip('"”')
    return t.endswith(("...", "…", "—", "--", "-"))


def starts_dangling(text: str) -> bool:
    low = text.lower().lstrip(" .…\"'")
    return low.startswith(PRONOUN_STARTS) or text.lower().lstrip().startswith(CONJUNCTION_STARTS)
