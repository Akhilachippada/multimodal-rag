"""End-to-end orchestration: ingest -> embed -> index -> route -> retrieve -> (compute) -> reason -> cite."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .citations import attribute
from .config import Config
from .datasets import (ChartPlanner, ChartResult, dataset_elements, execute_spec, load_tables, pick_dataset)
from .ingestion import IMAGE_EXT, copy_into, ingest_image, ingest_pdf, ingest_textfile
from .reasoner import Reasoner
from .retriever import HybridRetriever
from .router import QueryRouter, Route
from .schema import Element, Evidence
from .utils import slug

DATA_EXT = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".parquet"}
TEXT_EXT = {".txt", ".md"}

DESCRIBE_PROMPT = (
    "Describe this {kind} for a search index. State its type, title, axis labels and units, legend/series, "
    "and the key values, trends, maxima/minima (for charts list approximate data points). "
    "Transcribe any visible text. Do not speculate beyond what is visible."
)


@dataclass
class Answer:
    text: str
    evidence: list[Evidence]
    citations: list[dict]
    route: Route
    chart_path: Optional[str] = None
    spec: Optional[dict] = None


class MMRAG:
    def __init__(self, cfg: Optional[Config] = None, text_embedder=None, clip_embedder=None, backend=None):
        self.cfg = cfg or Config()
        self.cfg.ensure_dirs()
        self._text_emb, self._clip, self._backend = text_embedder, clip_embedder, backend
        self._text_store = self._vis_store = None
        self._datasets: Optional[dict] = None          # table name -> (DataFrame, doc_id)
        self.router = QueryRouter(self.cfg)
        self._registry_path = self.cfg.path("registry.json")
        self.registry: dict = json.loads(self._registry_path.read_text()) if self._registry_path.exists() else {}

    # ------------------------------------------------------------ lazy components
    @property
    def text_emb(self):
        if self._text_emb is None:
            from .embeddings import TextEmbedder
            self._text_emb = TextEmbedder(self.cfg.text_model)
        return self._text_emb

    @property
    def clip(self):
        if self._clip is None:
            from .embeddings import ClipEmbedder
            self._clip = ClipEmbedder(self.cfg.clip_model)
        return self._clip

    @property
    def backend(self):
        if self._backend is None:
            from .backends import make_backend
            self._backend = make_backend(self.cfg)
        return self._backend

    @property
    def text_store(self):
        if self._text_store is None:
            from .vectorstore import VectorStore
            self._text_store = VectorStore(self.text_emb.dim, str(self.cfg.path("index", "text")))
        return self._text_store

    @property
    def vis_store(self):
        if self._vis_store is None:
            from .vectorstore import VectorStore
            self._vis_store = VectorStore(self.clip.dim, str(self.cfg.path("index", "visual")))
        return self._vis_store

    @property
    def datasets(self) -> dict:
        if self._datasets is None:
            self._datasets = {}
            for doc_id, info in self.registry.items():
                if info.get("type") == "dataset" and Path(info["file"]).exists():
                    for name, df in load_tables(info["file"]).items():
                        self._datasets[name] = (df, doc_id)
        return self._datasets

    # ------------------------------------------------------------ ingestion
    @staticmethod
    def _doc_id(path: str) -> str:
        h = hashlib.sha1(Path(path).read_bytes()).hexdigest()[:8]
        return f"{slug(Path(path).stem)}-{h}"

    def add_file(self, path: str, label: Optional[str] = None,
                 progress: Optional[Callable[[str], None]] = None) -> dict:
        """Index a PDF, image/chart picture, dataset (csv/xlsx/json) or text file. `label` may force 'chart'/'image'."""
        say = progress or (lambda m: None)
        p = Path(path)
        ext = p.suffix.lower()
        doc_id, name = self._doc_id(path), p.name
        if doc_id in self.registry:
            self.remove(doc_id)
        asset_dir = self.cfg.path("assets", doc_id)
        info = {"name": name, "added": time.strftime("%Y-%m-%d %H:%M")}

        say(f"Extracting modalities from {name} ...")
        if ext == ".pdf":
            els, info["type"] = ingest_pdf(str(p), doc_id, asset_dir, self.cfg), "pdf"
        elif ext in IMAGE_EXT:
            els, info["type"] = ingest_image(str(p), asset_dir, label), "image"
        elif ext in TEXT_EXT:
            els, info["type"] = ingest_textfile(str(p), self.cfg), "text"
        elif ext in DATA_EXT:
            stored = copy_into(str(p), self.cfg.path("datasets") / doc_id)
            tables = load_tables(str(stored))
            els = []
            for tname, df in tables.items():
                els += dataset_elements(tname, df, asset_dir, self.cfg)
                self.datasets[tname] = (df, doc_id)
            info.update(type="dataset", file=str(stored), tables=list(tables),
                        rows={t: len(d) for t, d in tables.items()})
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        n_text, n_vis = self._index_elements(els, doc_id, name, say)
        info.update(n_text=n_text, n_visual=n_vis,
                    counts={m: sum(1 for e in els if e.modality == m) for m in ("text", "table", "image", "chart", "page")})
        self.registry[doc_id] = info
        self.save()
        say(f"Indexed {name}: {n_text} text records, {n_vis} visual records.")
        return {"doc_id": doc_id, **info}

    def _describe_visual(self, el: Element) -> str:
        kind = "chart or graph" if el.modality == "chart" else "figure or image"
        try:
            return self.backend.generate([("text", DESCRIBE_PROMPT.format(kind=kind)), ("image", el.asset)],
                                         max_new_tokens=350)
        except Exception:
            return ""

    def _index_elements(self, els: list[Element], doc_id: str, name: str, say) -> tuple[int, int]:
        text_recs, vis_recs, vis_paths = [], [], []
        can_caption = self.cfg.caption_visuals and getattr(self.backend, "supports_vision", False)
        for i, el in enumerate(els):
            meta = {**el.meta, "doc_id": doc_id, "doc_name": name}
            if can_caption and el.modality in ("image", "chart") and el.asset and not el.text:
                say(f"Captioning {el.modality} {i + 1}/{len(els)} with the VLM ...")
                el.text = self._describe_visual(el)
            if el.text:
                text_recs.append({"id": f"{doc_id}:{i}t", "modality": el.modality, "text": el.text,
                                  "asset": el.asset, "meta": meta})
            if el.asset:
                vis_recs.append({"id": f"{doc_id}:{i}v", "modality": el.modality, "text": el.text,
                                 "asset": el.asset, "meta": meta})
                vis_paths.append(el.asset)
        say("Embedding text (Sentence Transformers) ...")
        self.text_store.add(self.text_emb.encode([r["text"] for r in text_recs]), text_recs)
        say("Embedding visuals (CLIP) ...")
        self.vis_store.add(self.clip.encode_images(vis_paths), vis_recs)
        return len(text_recs), len(vis_recs)

    def remove(self, doc_id: str) -> None:
        self.text_store.remove_doc(doc_id)
        self.vis_store.remove_doc(doc_id)
        shutil.rmtree(self.cfg.path("assets", doc_id), ignore_errors=True)
        shutil.rmtree(self.cfg.path("datasets", doc_id), ignore_errors=True)
        if self._datasets is not None:
            self._datasets = {k: v for k, v in self._datasets.items() if v[1] != doc_id}
        self.registry.pop(doc_id, None)
        self.save()

    def save(self) -> None:
        if self._text_store is not None:
            self.text_store.save()
        if self._vis_store is not None:
            self.vis_store.save()
        self._registry_path.write_text(json.dumps(self.registry, indent=2), encoding="utf-8")

    # ------------------------------------------------------------ charts from user input
    def _planner(self) -> ChartPlanner:
        return ChartPlanner(self.backend)

    def make_chart(self, table: str, spec: dict) -> ChartResult:
        """User-specified chart (chart-builder UI or API). `spec` keys: chart,x,y,group,agg,top_n,filters,title."""
        df, _ = self.datasets[table]
        return execute_spec(df, spec, self.cfg.path("generated"), table)

    def chart_from_request(self, table: str, request: str) -> ChartResult:
        df, _ = self.datasets[table]
        spec = self._planner().plan(request, df, table)
        if spec is None:
            raise ValueError("Could not work out a chart from that request - try naming the columns.")
        return execute_spec(df, spec, self.cfg.path("generated"), table)

    def _data_branch(self, question: str, table: Optional[str], doc_ids) -> Optional[tuple[Evidence, ChartResult]]:
        pool = {k: v[0] for k, v in self.datasets.items() if not doc_ids or v[1] in set(doc_ids)}
        if not pool:
            return None
        table = table if table in pool else pick_dataset(question, pool)
        spec = self._planner().plan(question, pool[table], table)
        if spec is None:
            return None
        try:
            res = execute_spec(pool[table], spec, self.cfg.path("generated"), table)
        except Exception:
            return None
        ev = Evidence("", "chart", res.description, res.path,
                      {"doc_name": table, "page": None, "spec": res.spec}, 1.0, "computed")
        return ev, res

    # ------------------------------------------------------------ question answering
    def ask(self, question: str, doc_ids: Optional[list[str]] = None, dataset: Optional[str] = None,
            query_image: Optional[str] = None) -> Answer:
        route = self.router.route(question, has_datasets=bool(self.datasets), has_image=bool(query_image))
        retriever = HybridRetriever(self.text_store, self.vis_store, self.text_emb, self.clip, self.cfg)
        evidence = retriever.retrieve(question, route, doc_ids, query_image)

        extra: list[Evidence] = []
        chart_path, spec = None, None
        if self.datasets and (dataset or route.want_data):
            got = self._data_branch(question, dataset, doc_ids)
            if got:
                extra.append(got[0])
                chart_path, spec = got[1].path, got[1].spec
        if query_image:
            extra.append(Evidence("", "image", "", query_image, {"doc_name": "user upload", "page": None}, 1.0, "user"))

        evidence = extra + evidence
        for i, e in enumerate(evidence, 1):
            e.eid = f"E{i}"

        reasoner = Reasoner(self.backend, self.text_emb, self.cfg)
        text = reasoner.answer(question, evidence)
        cites = attribute(text, evidence, self.text_emb)
        return Answer(text, evidence, cites, route, chart_path, spec)

    def stats(self) -> dict:
        return {"documents": len(self.registry), "text_records": len(self.text_store),
                "visual_records": len(self.vis_store), "datasets": list(self.datasets)}
