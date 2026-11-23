from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Config:
    # ---- storage ----
    storage_dir: str = field(default_factory=lambda: _env("MMRAG_STORAGE", "storage"))

    # ---- encoders (report section 3.2.2) ----
    text_model: str = field(default_factory=lambda: _env("MMRAG_TEXT_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    clip_model: str = field(default_factory=lambda: _env("MMRAG_CLIP_MODEL", "clip-ViT-B-32"))

    # ---- chunking (eq. 4: at most L_max "tokens", overlap delta; words are used as the unit) ----
    chunk_words: int = 180
    chunk_overlap: int = 30

    # ---- PDF extraction ----
    render_dpi: int = 150          # figure / table crops
    page_dpi: int = 110            # whole-page renders (also catches vector charts)
    index_pages: bool = True       # index each page as an image too (VisRAG / ColPali style)
    min_image_px: int = 60         # ignore icons / logos smaller than this (PDF points)

    # ---- retrieval (eq. 20-22) ----
    top_k_text: int = 5
    top_k_visual: int = 3
    expand_neighbors: int = 2      # pull figures/tables from the same page as retrieved text
    max_images_to_vlm: int = 4

    # ---- generation backend ----
    # "extractive"    : no model needed (demo / tests). Quotes best-matching sentences.
    # "openai_compat" : any OpenAI-compatible server (Ollama, vLLM, LM Studio...) serving Qwen2.5-VL
    # "qwen_local"    : load Qwen2.5-VL in-process with transformers (GPU recommended)
    backend: str = field(default_factory=lambda: _env("MMRAG_BACKEND", "extractive"))
    qwen_model: str = field(default_factory=lambda: _env("MMRAG_QWEN_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"))
    load_4bit: bool = field(default_factory=lambda: _env("MMRAG_4BIT", "0") == "1")
    api_base: str = field(default_factory=lambda: _env("MMRAG_API_BASE", "http://localhost:11434/v1"))
    api_model: str = field(default_factory=lambda: _env("MMRAG_API_MODEL", "qwen2.5vl:7b"))
    api_key: str = field(default_factory=lambda: _env("MMRAG_API_KEY", "ollama"))
    max_new_tokens: int = 600

    # ---- VLM captions for extracted images/charts (makes figures text-searchable) ----
    caption_visuals: bool = True

    # ---- datasets ----
    max_row_chunks: int = 150      # row-group chunks indexed per table
    rows_per_chunk: int = 25
    max_auto_charts: int = 5

    # ---- helpers ----
    def path(self, *parts: str) -> Path:
        return Path(self.storage_dir).joinpath(*parts)

    def ensure_dirs(self) -> None:
        for sub in ("assets", "datasets", "generated", "uploads", "index"):
            self.path(sub).mkdir(parents=True, exist_ok=True)
