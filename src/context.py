"""Show-level context shared by the detectors: episode order, line features, what comes before/after a cut."""

from __future__ import annotations

import re
from typing import Optional

from src.features import LineFeatures, line_key
from src.models import Episode, Utterance

STOPWORDS = set("""a an the and or but if then so of to in on at by for with from as is are was were be been being
i you he she it we they me him her us them my your his its our their this that these those there here what which
who whom whose when where why how not no nor do does did done have has had will would shall should can could may
might must am just only very too also than into out up down over under again further once all any both each few
more most other some such own same s t don now ll re ve d m o y ma""".split())


def content_words(text: str) -> set:
    return {w for w in re.findall(r"[a-z']{3,}", text.lower()) if w not in STOPWORDS}


class ShowContext:
    def __init__(self, show: str, episodes: list, feats: dict):
        self.show = show
        self.episodes: list = sorted(episodes, key=lambda e: (e.number is None, e.number or 0, e.episode_id))
        self.by_id = {e.episode_id: e for e in self.episodes}
        self.order = {e.episode_id: i for i, e in enumerate(self.episodes)}
        self.feats: dict = feats
        self.content = {e.episode_id: e.content for e in self.episodes}
        self.pos = {}
        for eid, lines in self.content.items():
            for p, u in enumerate(lines):
                self.pos[line_key(u)] = p
        # character name tokens known for this show (from speaker labels), for name-mention detection
        names = set()
        for e in self.episodes:
            for c in e.characters:
                for tok in re.findall(r"[A-Za-z']+", c):
                    if len(tok) > 2 and tok.upper() not in {"LADY", "LORD", "THE", "MAN", "WOMAN", "OLD", "YOUNG", "SIR", "MRS", "MR", "MOTHER", "FATHER", "DUKE", "DUCHESS", "DOWAGER", "BLOND", "NOBLEMAN", "MYSTERIOUS", "BEARDED", "GUARD", "SERVANT", "MAID", "VOICE", "UNKNOWN", "SUBTITLE"}:
                        names.add(tok.lower())
        self.name_tokens = names
        self.logline_words = {e.episode_id: content_words(e.logline) for e in self.episodes}

    def f(self, u: Utterance) -> LineFeatures:
        return self.feats.get(line_key(u)) or LineFeatures()

    def episode(self, eid: str) -> Episode:
        return self.by_id[eid]

    def lines_after(self, u: Utterance, n: int = 3, cross_episode: bool = True) -> list:
        """Next n content lines in the source after u (continuing into the next episode if u is last)."""
        lines = self.content[u.episode_id]
        p = self.pos.get(line_key(u))
        out = [] if p is None else [x for x in lines[p + 1:] if x.start >= u.start][:n]
        if len(out) < n and cross_episode:
            nxt = self.next_episode(u.episode_id)
            if nxt is not None:
                out += self.content[nxt.episode_id][: n - len(out)]
        return out

    def line_before(self, u: Utterance) -> Optional[Utterance]:
        p = self.pos.get(line_key(u))
        if not p:
            return None
        return self.content[u.episode_id][p - 1]

    def is_episode_final(self, u: Utterance, tail: int = 1) -> bool:
        lines = self.content[u.episode_id]
        return bool(lines) and u in lines[-tail:]

    def next_episode(self, eid: str) -> Optional[Episode]:
        i = self.order.get(eid)
        if i is None or i + 1 >= len(self.episodes):
            return None
        return self.episodes[i + 1]

    def mentions_names(self, text: str) -> set:
        return {w.lower() for w in re.findall(r"\b[A-Z][a-z']+\b", text) if w.lower() in self.name_tokens}
