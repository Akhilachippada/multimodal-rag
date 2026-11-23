from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

VISUAL_MODALITIES = {"image", "chart", "table", "page"}


@dataclass
class Element:
    """One extracted unit of a document (report eq. 1: e_m in T u I u A u C)."""
    modality: str                      # text | table | image | chart | page
    text: str = ""                     # raw text / markdown table / caption
    asset: Optional[str] = None        # path of the image file (crop or render)
    meta: dict = field(default_factory=dict)   # page, bbox, caption ... (eq. 16)


@dataclass
class Evidence:
    eid: str
    modality: str
    text: str
    asset: Optional[str]
    meta: dict
    score: float = 0.0
    source: str = "retrieved"          # retrieved | computed | user

    def where(self) -> str:
        name = self.meta.get("doc_name", "")
        page = self.meta.get("page")
        return f"{name}, p.{page}" if page else name

    def short(self, n: int = 160) -> str:
        t = " ".join(self.text.split())
        return t[:n] + ("..." if len(t) > n else "")


def evidence_to_dict(e: Evidence) -> dict[str, Any]:
    return {"id": e.eid, "modality": e.modality, "where": e.where(), "score": round(e.score, 4),
            "text": e.text, "asset": e.asset, "source": e.source}
