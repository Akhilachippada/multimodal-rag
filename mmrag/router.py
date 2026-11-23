"""Query-intent routing (the light 'agentic' step): decide which modalities are worth retrieving."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import Config
from .datasets import data_intent

VISUAL_RE = re.compile(r"\b(chart|graph|plot|figure|fig\.|diagram|image|picture|photo|illustration|trend|curve|"
                       r"axis|bar|pie|heatmap|shown|visual|looks? like|color|colour)\b", re.I)
TABLE_RE = re.compile(r"\b(table|row|column|value|values|total|sum|average|mean|percent\w*|number|how many|"
                      r"compare|comparison|rate|cost|revenue|score|accuracy)\b", re.I)


@dataclass
class Route:
    k_text: int
    k_visual: int
    want_visual: bool = False
    want_table: bool = False
    want_data: bool = False
    intents: list[str] = field(default_factory=list)


class QueryRouter:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def route(self, question: str, has_datasets: bool = False, has_image: bool = False) -> Route:
        kt, kv = self.cfg.top_k_text, self.cfg.top_k_visual
        r = Route(kt, kv)
        if VISUAL_RE.search(question) or has_image:
            r.want_visual = True
            r.k_visual = kv + 2
            r.intents.append("visual")
        if TABLE_RE.search(question):
            r.want_table = True
            r.k_text = kt + 2
            r.intents.append("table")
        if has_datasets and data_intent(question):
            r.want_data = True
            r.intents.append("dataset")
        if not r.intents:
            r.intents.append("text")
        return r
