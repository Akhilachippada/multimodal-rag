"""Hybrid multimodal retrieval (report section 3.2.4, eq. 17-22)."""
from __future__ import annotations

from typing import Iterable, Optional

from .config import Config
from .router import Route
from .schema import Evidence
from .vectorstore import VectorStore


class HybridRetriever:
    def __init__(self, text_store: VectorStore, vis_store: VectorStore, text_emb, clip_emb, cfg: Config):
        self.text_store, self.vis_store = text_store, vis_store
        self.text_emb, self.clip_emb, self.cfg = text_emb, clip_emb, cfg

    def retrieve(self, question: str, route: Route, doc_ids: Optional[Iterable[str]] = None,
                 query_image: Optional[str] = None) -> list[Evidence]:
        allowed = set(doc_ids) if doc_ids else None
        where = (lambda r: r["meta"].get("doc_id") in allowed) if allowed else None

        # q_t and q_v: the same question mapped into both spaces (eq. 17-18)
        qt = self.text_emb.encode([question])[0]
        qv = self.clip_emb.encode_text([question])[0]

        text_hits = self.text_store.search(qt, route.k_text, where)
        vis_hits = self.vis_store.search(qv, route.k_visual, where)
        if query_image:                                  # image-to-image retrieval for a user-supplied picture
            qi = self.clip_emb.encode_images([query_image])[0]
            vis_hits += self.vis_store.search(qi, route.k_visual, where)

        merged: dict[str, Evidence] = {}

        def add(score: float, rec: dict, from_visual: bool):
            key = rec.get("asset") or rec["id"]
            if key in merged:
                e = merged[key]
                e.score = max(e.score, score)
                if not e.text and rec.get("text"):
                    e.text = rec["text"]
                return
            merged[key] = Evidence("", rec["modality"], rec.get("text", ""), rec.get("asset"), rec["meta"], score)

        for s, r in text_hits:
            add(s, r, False)
        for s, r in vis_hits:
            add(s, r, True)

        # context expansion: figures/tables that sit on the same page as retrieved text (fights context fragmentation)
        if self.cfg.expand_neighbors > 0:
            pages = {(e.meta.get("doc_id"), e.meta.get("page")) for e in merged.values()
                     if e.modality == "text" and e.meta.get("page")}
            extra = 0
            for rec in self.vis_store.records:
                if extra >= self.cfg.expand_neighbors:
                    break
                m = rec["meta"]
                if rec["modality"] in ("chart", "table", "image") and (m.get("doc_id"), m.get("page")) in pages:
                    key = rec.get("asset") or rec["id"]
                    if key not in merged and (not allowed or m.get("doc_id") in allowed):
                        merged[key] = Evidence("", rec["modality"], rec.get("text", ""), rec.get("asset"), m, 0.0)
                        extra += 1

        ev = list(merged.values())
        text_first = [e for e in ev if e.modality in ("text", "table") or (e.text and not e.asset)]
        rest = [e for e in ev if e not in text_first]
        ordered = sorted(text_first, key=lambda e: -e.score) + sorted(rest, key=lambda e: -e.score)
        for i, e in enumerate(ordered, 1):
            e.eid = f"E{i}"
        return ordered
