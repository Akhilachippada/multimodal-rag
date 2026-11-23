"""Offline tests: fake encoders stand in for the real models, so no downloads/GPU are needed.
Run:  python -m unittest discover -s tests -v
"""
import hashlib
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mmrag import Config, MMRAG                      # noqa: E402
from mmrag.backends import ExtractiveBackend          # noqa: E402
from mmrag.datasets import ChartPlanner, execute_spec, load_tables  # noqa: E402
from mmrag.ingestion import chunk_text                # noqa: E402

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data"


def _hash_vec(text: str, dim: int) -> np.ndarray:
    v = np.zeros(dim, dtype="float32")
    for w in text.lower().split():
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1
    n = np.linalg.norm(v)
    return v / n if n else v


class FakeText:
    dim = 64

    def encode(self, texts, batch_size=32):
        return np.array([_hash_vec(t, self.dim) for t in texts], dtype="float32").reshape(-1, self.dim)


class FakeClip:
    dim = 32

    def encode_text(self, texts):
        return np.array([_hash_vec(t, self.dim) for t in texts], dtype="float32").reshape(-1, self.dim)

    def encode_images(self, paths):
        from PIL import Image
        out = []
        for p in paths:
            a = np.asarray(Image.open(p).convert("RGB").resize((4, 4)), dtype="float32").ravel()
            v = np.resize(a, self.dim)
            out.append(v / (np.linalg.norm(v) or 1))
        return np.array(out, dtype="float32").reshape(-1, self.dim)


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = Config(storage_dir=self.tmp, backend="extractive")
        self.rag = MMRAG(self.cfg, FakeText(), FakeClip(), ExtractiveBackend())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_chunking_overlap_and_limit(self):
        t = " ".join(f"Sentence {i} has a few words." for i in range(80))
        chunks = chunk_text(t, 40, 8)
        self.assertTrue(all(len(c.split()) <= 40 for c in chunks))
        self.assertGreater(len(chunks), 5)

    def test_dataset_chart_planning(self):
        df = list(load_tables(str(SAMPLE / "sales.csv")).values())[0]
        spec = ChartPlanner().plan("plot total revenue by region", df, "sales")
        self.assertEqual((spec["chart"], spec["x"], spec["y"], spec["agg"]), ("bar", "region", "revenue", "sum"))
        res = execute_spec(df, spec, Path(self.tmp) / "g", "sales")
        self.assertTrue(Path(res.path).exists())
        self.assertAlmostEqual(res.table.iloc[:, 1].sum(), df["revenue"].sum(), places=1)
        trend = ChartPlanner().plan("monthly revenue trend", df, "sales")
        self.assertEqual(trend["chart"], "line")

    def test_upload_dataset_and_ask(self):
        info = self.rag.add_file(str(SAMPLE / "sales.csv"))
        self.assertEqual(info["type"], "dataset")
        ans = self.rag.ask("Show total revenue by product as a chart")
        self.assertIsNotNone(ans.chart_path)
        self.assertTrue(any(e.source == "computed" for e in ans.evidence))
        self.assertIn("[E1]", ans.text)

    def test_upload_chart_image_and_text(self):
        self.rag.add_file(str(SAMPLE / "quarterly_revenue.png"), label="chart")
        txt = Path(self.tmp) / "notes.txt"
        txt.write_text("The quarterly revenue chart shows that revenue peaked in the fourth quarter of 2025. " * 3)
        self.rag.add_file(str(txt))
        ans = self.rag.ask("What does the quarterly revenue chart show?")
        mods = {e.modality for e in ans.evidence}
        self.assertIn("text", mods)
        self.assertIn("chart", mods)

    def test_persistence_and_removal(self):
        info = self.rag.add_file(str(SAMPLE / "sales.csv"))
        rag2 = MMRAG(self.cfg, FakeText(), FakeClip(), ExtractiveBackend())
        self.assertIn("sales", rag2.datasets)
        self.assertGreater(len(rag2.text_store), 0)
        rag2.remove(info["doc_id"])
        self.assertEqual(len(rag2.text_store), 0)

    @unittest.skipUnless(__import__("importlib").util.find_spec("fitz"), "PyMuPDF not installed")
    def test_pdf_pipeline(self):
        pdf = SAMPLE / "sales_report.pdf"
        if not pdf.exists():
            self.skipTest("run scripts/make_samples.py first")
        info = self.rag.add_file(str(pdf))
        self.assertGreater(info["counts"]["text"], 0)
        self.assertGreater(info["counts"]["page"], 0)


if __name__ == "__main__":
    unittest.main()
