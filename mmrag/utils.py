from __future__ import annotations

import re
from typing import Iterable


def fmt(v) -> str:
    if v is None:
        return ""
    try:
        import numpy as np
        if isinstance(v, (float, np.floating)):
            if v != v:
                return ""
            return f"{v:.4g}"
    except Exception:
        pass
    return str(v).replace("\n", " ").replace("|", "/").strip()


def rows_to_markdown(rows: Iterable[Iterable]) -> str:
    rows = [[fmt(c) for c in r] for r in rows if r is not None]
    rows = [r for r in rows if any(r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def df_to_md(df, max_rows: int = 30) -> str:
    d = df.head(max_rows)
    return rows_to_markdown([list(map(str, d.columns))] + d.values.tolist())


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()[:40] or "doc"


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]
