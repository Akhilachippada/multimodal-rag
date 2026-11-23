"""Vector database (report section 3.2.3).

One store per space (Omega_t text space, Omega_v visual space). Each record is the report's tuple
(e_n, tau_n, rho_n, mu_n): embedding, modality tag, pointer to the raw asset, spatial metadata.

Uses FAISS (IndexFlatIP, or IndexIVFPQ for very large stores) and falls back to NumPy if FAISS is absent.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


class VectorStore:
    def __init__(self, dim: int, path: Optional[str] = None, ivfpq_above: int = 20000):
        self.dim = dim
        self.path = path
        self.ivfpq_above = ivfpq_above
        self.vecs = np.zeros((0, dim), dtype="float32")
        self.records: list[dict] = []
        self._index = None
        if path and Path(path + ".npy").exists() and Path(path + ".json").exists():
            self.load()

    def __len__(self) -> int:
        return len(self.records)

    # ---- mutation
    def add(self, vecs: np.ndarray, records: list[dict]) -> None:
        if len(records) == 0:
            return
        assert vecs.shape == (len(records), self.dim), f"bad shape {vecs.shape}"
        self.vecs = np.vstack([self.vecs, vecs.astype("float32")])
        self.records.extend(records)
        self._index = None

    def remove_doc(self, doc_id: str) -> int:
        keep = [i for i, r in enumerate(self.records) if r["meta"].get("doc_id") != doc_id]
        removed = len(self.records) - len(keep)
        if removed:
            self.vecs = self.vecs[keep]
            self.records = [self.records[i] for i in keep]
            self._index = None
        return removed

    # ---- search (maximum inner product search, eq. 20-21)
    def _build(self):
        n = len(self.records)
        if faiss is None:
            return None
        if n >= self.ivfpq_above:
            m = next(c for c in (64, 48, 32, 16, 8, 4, 2, 1) if self.dim % c == 0)
            nlist = int(4 * math.sqrt(n))
            index = faiss.IndexIVFPQ(faiss.IndexFlatIP(self.dim), self.dim, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
            index.train(self.vecs)
            index.add(self.vecs)
            index.nprobe = 16
        else:
            index = faiss.IndexFlatIP(self.dim)
            index.add(self.vecs)
        return index

    def search(self, q: np.ndarray, k: int, where: Optional[Callable[[dict], bool]] = None):
        n = len(self.records)
        if n == 0 or k <= 0:
            return []
        q = np.asarray(q, dtype="float32").reshape(1, -1)
        kk = min(n, k) if where is None else (n if n <= 5000 else min(n, k * 50))
        if faiss is not None:
            if self._index is None:
                self._index = self._build()
            scores, idx = self._index.search(q, kk)
            scores, idx = scores[0], idx[0]
        else:
            sims = (self.vecs @ q.T).ravel()
            idx = np.argsort(-sims)[:kk]
            scores = sims[idx]
        out = []
        for s, i in zip(scores, idx):
            if i < 0:
                continue
            r = self.records[int(i)]
            if where and not where(r):
                continue
            out.append((float(s), r))
            if len(out) >= k:
                break
        return out

    # ---- persistence
    def save(self) -> None:
        if not self.path:
            return
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        np.save(self.path + ".npy", self.vecs)
        Path(self.path + ".json").write_text(json.dumps(self.records), encoding="utf-8")

    def load(self) -> None:
        self.vecs = np.load(self.path + ".npy")
        self.records = json.loads(Path(self.path + ".json").read_text(encoding="utf-8"))
        self._index = None
