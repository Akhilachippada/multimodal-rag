"""Dual-encoder embeddings (report section 3.2.2): Sentence Transformers for text, CLIP ViT for visuals.

All vectors are L2-normalised so cosine similarity == inner product (eq. 10, 14, 19).
Models are loaded lazily so that importing the package stays cheap.
"""
from __future__ import annotations

import numpy as np


class TextEmbedder:
    def __init__(self, model_name: str, device: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts, batch_size: int = 32) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        v = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False)
        return v.astype("float32")


class ClipEmbedder:
    """OpenAI CLIP through sentence-transformers: images and text land in the same space."""

    def __init__(self, model_name: str, device: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode_images(self, paths, batch_size: int = 16) -> np.ndarray:
        from PIL import Image
        paths = list(paths)
        if not paths:
            return np.zeros((0, self.dim), dtype="float32")
        out = []
        for i in range(0, len(paths), batch_size):
            ims = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
            v = self.model.encode(ims, batch_size=batch_size, normalize_embeddings=True,
                                  convert_to_numpy=True, show_progress_bar=False)
            out.append(v)
            for im in ims:
                im.close()
        return np.vstack(out).astype("float32")

    def encode_text(self, texts) -> np.ndarray:
        texts = [" ".join(t.split()[:50]) for t in texts]       # CLIP text tower: 77 tokens max
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        v = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return v.astype("float32")
