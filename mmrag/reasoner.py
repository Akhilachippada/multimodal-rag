"""Cross-modal reasoning and grounded response generation (report sections 3.2.5-3.2.6)."""
from __future__ import annotations

import re

import numpy as np

from .backends import Backend
from .config import Config
from .schema import Evidence, VISUAL_MODALITIES
from .utils import split_sentences

SYSTEM = (
    "You are a careful multimodal document analyst. Answer ONLY from the numbered evidence supplied "
    "(text passages, tables, charts, images, computed dataset results). Combine evidence across modalities when the "
    "question needs it. Read numbers from charts and tables carefully and show the arithmetic you use. "
    "Cite every claim with the evidence id in square brackets, e.g. [E2]. If the evidence is insufficient or "
    "contradictory, say so plainly instead of guessing."
)


def truncate(t: str, n: int) -> str:
    t = t.strip()
    return t if len(t) <= n else t[:n] + " ..."


def build_parts(question: str, evidence: list[Evidence], supports_vision: bool, max_images: int):
    parts = [("text", "EVIDENCE:\n")]
    used_images = 0
    ranked_visual = sorted((e for e in evidence if e.asset and e.modality in VISUAL_MODALITIES),
                           key=lambda e: (e.modality == "page", -e.score))
    chosen = {e.eid for e in ranked_visual[:max_images]} if supports_vision else set()
    for e in evidence:
        head = f"[{e.eid}] {e.modality.upper()}"
        if e.where():
            head += f" - {e.where()}"
        if e.source == "computed":
            head += " (computed from the uploaded dataset)"
        elif e.source == "user":
            head += " (image supplied by the user with the question)"
        body = truncate(e.text, 1500) if e.text else ""
        parts.append(("text", f"{head}\n{body}\n"))
        if e.eid in chosen and used_images < max_images:
            parts.append(("image", e.asset))
            used_images += 1
    parts.append(("text", f"\nQUESTION: {question}\n\nANSWER (cite evidence ids like [E1]):"))
    return parts


class Reasoner:
    def __init__(self, backend: Backend, text_emb, cfg: Config):
        self.backend, self.text_emb, self.cfg = backend, text_emb, cfg

    def answer(self, question: str, evidence: list[Evidence]) -> str:
        if not evidence:
            return "I could not find any relevant evidence in the indexed documents. Upload a document or dataset first."
        if self.backend.name == "extractive":
            return self.extractive(question, evidence)
        parts = build_parts(question, evidence, self.backend.supports_vision, self.cfg.max_images_to_vlm)
        return self.backend.generate(parts, system=SYSTEM, max_new_tokens=self.cfg.max_new_tokens)

    # -- model-free fallback ----------------------------------------------------------
    def extractive(self, question: str, evidence: list[Evidence], max_sent: int = 4) -> str:
        lines = ["*Extractive mode - no vision-language model configured, so evidence is quoted rather than reasoned over.*", ""]
        for e in evidence:
            if e.source == "computed":
                head = [ln for ln in e.text.splitlines() if ln and not ln.startswith("|") and ln != "Data:"][:3]
                lines.append(" ".join(head) + f" [{e.eid}]")
        sents = []
        for e in evidence:
            if e.source == "computed" or not e.text:
                continue
            for s in split_sentences(e.text):
                if len(s) >= 25 and not s.startswith("|"):
                    sents.append((s, e.eid))
        if sents:
            sv = self.text_emb.encode([s for s, _ in sents])
            qv = self.text_emb.encode([question])[0]
            for i in np.argsort(-(sv @ qv))[:max_sent]:
                lines.append(f"{sents[i][0]} [{sents[i][1]}]")
        for e in evidence:
            if e.modality in ("chart", "image", "page") and e.source != "computed" and not e.text:
                lines.append(f"A relevant {e.modality} was retrieved from {e.where()} [{e.eid}]; "
                             "switch to a vision-language backend to have it interpreted.")
        return "\n\n".join(lines)
