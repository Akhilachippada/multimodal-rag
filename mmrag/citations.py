"""Provenance tracking (report eq. 26): link every generated sentence to its best supporting evidence."""
from __future__ import annotations

import re

import numpy as np

from .schema import Evidence
from .utils import split_sentences

TAG = re.compile(r"\[E(\d+)\]")


def attribute(answer: str, evidence: list[Evidence], text_emb, min_sim: float = 0.25) -> list[dict]:
    valid = {e.eid for e in evidence}
    with_text = [e for e in evidence if e.text.strip()]
    ev_vecs = text_emb.encode([e.text[:600] for e in with_text]) if with_text else None
    out = []
    for s in split_sentences(answer):
        if s.startswith("*") and s.endswith("*"):
            continue
        tags = [f"E{n}" for n in TAG.findall(s) if f"E{n}" in valid]
        method = "model-cited"
        if not tags and ev_vecs is not None and len(s) > 20:
            sv = text_emb.encode([s])[0]
            sims = ev_vecs @ sv
            j = int(np.argmax(sims))
            if sims[j] >= min_sim:
                tags, method = [with_text[j].eid], "similarity"
        if tags:
            out.append({"sentence": TAG.sub("", s).strip(), "evidence": tags, "method": method})
    return out
