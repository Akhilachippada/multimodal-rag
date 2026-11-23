# Multimodal RAG for Cross-Modal Reasoning

Implementation of the major-project report *"Multimodal Retrieval-Augmented Generation for Cross-Modal Reasoning"*
(RGUKT Nuzvid), extended so that it also handles **graphs / charts** and **user-uploaded datasets**.

Ask a question about a PDF report, a chart image, a table or a spreadsheet. The system retrieves text, tables, figures
and charts, lets a Vision-Language Model reason over them together, and answers with citations like `[E2]` that point
to the evidence (page, figure, table or dataset).

## What is inside

| Report section | Code |
|---|---|
| 3.2.1 Document processing & modality extraction (T, I, A, C) | `mmrag/ingestion.py` (PyMuPDF: text chunks with overlap, tables as markdown **and** image, figures, chart detection, captions, full-page renders) |
| 3.2.2 Multimodal embeddings (Sentence Transformers + CLIP, L2-normalised) | `mmrag/embeddings.py` |
| 3.2.3 Vector database, 4-tuple records `(e, tau, rho, mu)`, FAISS Flat / IVFPQ | `mmrag/vectorstore.py` |
| 3.2.4 Hybrid retrieval, `q_t`/`q_v`, MIPS top-k, fused context `C` | `mmrag/retriever.py`, `mmrag/router.py` |
| 3.2.5 Cross-modal reasoning with Qwen2.5-VL (interleaved text + image tokens) | `mmrag/backends.py`, `mmrag/reasoner.py` |
| 3.2.6 Grounded generation, attribution `argmax Sim(h(y_s), h(e))`, provenance | `mmrag/citations.py`, `mmrag/reasoner.py` |
| Future work in the report: agentic retrieval, chart/table understanding, explainability, multi-source | query router, chart engine, VLM captions, attribution (all implemented in a first form) |

### Extensions beyond the report

* **Dataset upload** (CSV, TSV, XLSX with several sheets, JSON, Parquet). Each dataset is profiled, split into row-group
  chunks, and given auto-generated charts whose captions contain real numbers, so datasets are searchable like documents.
* **Charts from user input**
  * Ask in plain language: *"plot monthly revenue by product"*. The planner (the VLM, or a built-in heuristic when no VLM is
    configured) produces a **JSON chart spec**. The spec is validated against the real columns and rendered with
    matplotlib. **No model-written code is ever executed.** The aggregated numbers are fed to the model as evidence.
  * A **Chart builder** tab with dropdowns: bar, line, pie, scatter, histogram, box plot, correlation heatmap, with
    aggregation, grouping, top-N and filters. PNG and CSV download.
* **Uploaded chart / graph images**: indexed with CLIP and, with a VLM backend, captioned (type, axes, series, approximate
  values) so they are retrievable by text and by image.
* **Attach an image to your question**: used for image-to-image retrieval and shown to the VLM as evidence.
* **Context expansion**: figures and tables on the same page as a retrieved passage are pulled in, which addresses the
  report's *context fragmentation* failure mode.
* **Page-level visual index** (VisRAG / ColPali style) so charts drawn as vectors, with no embedded bitmap, are still retrievable.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/make_samples.py                           # demo CSV, chart PNG and report PDF
streamlit run app.py
```

The first run downloads `all-MiniLM-L6-v2` (about 90 MB) and CLIP `ViT-B/32` (about 600 MB).
In the sidebar upload `sample_data/sales_report.pdf`, `sample_data/sales.csv` and `sample_data/quarterly_revenue.png`,
click **Index files**, then try:

* *What does the quarterly revenue chart show?*
* *Which product has the highest revenue and which region sells the most?*
* *Plot monthly revenue by product* (a computed chart appears)

### Choose a reasoning backend

| Backend | Needs | Use it for |
|---|---|---|
| `extractive` (default) | nothing | trying the pipeline, tests, low-end laptops (quotes evidence, no real reasoning) |
| `openai_compat` | Ollama / vLLM / LM Studio serving Qwen2.5-VL | **recommended**, no PyTorch in this project |
| `qwen_local` | `torch`, `transformers`, `accelerate`, GPU ≥ 16 GB (or 4-bit ≈ 8 GB) | the report's setup: Qwen2.5-VL in-process |

```bash
# Option A - Ollama
ollama pull qwen2.5vl:7b                       # or qwen2.5vl:3b on small machines
export MMRAG_BACKEND=openai_compat             # Windows PowerShell: $env:MMRAG_BACKEND="openai_compat"
streamlit run app.py                           # defaults: http://localhost:11434/v1, model qwen2.5vl:7b

# Option B - in-process
pip install torch transformers accelerate
export MMRAG_BACKEND=qwen_local MMRAG_QWEN_MODEL=Qwen/Qwen2.5-VL-7B-Instruct   # add MMRAG_4BIT=1 for 4-bit
```

You can also pick the backend in the Streamlit sidebar. Other env vars: `MMRAG_STORAGE`, `MMRAG_TEXT_MODEL`,
`MMRAG_CLIP_MODEL`, `MMRAG_API_BASE`, `MMRAG_API_MODEL`, `MMRAG_API_KEY`. Everything else is in `mmrag/config.py`.

### Command line

```bash
python cli.py add sample_data/sales_report.pdf sample_data/sales.csv
python cli.py ask "Which region has the highest revenue?" --show-evidence
python cli.py ask "Is this chart consistent with the report?" --image my_chart.png
python cli.py chart sales "average units per product by region"
```

### Python API

```python
from mmrag import MMRAG
rag = MMRAG()
rag.add_file("report.pdf"); rag.add_file("data.xlsx"); rag.add_file("chart.png", label="chart")
ans = rag.ask("Does the revenue table agree with the trend chart?")
print(ans.text); print(ans.citations); print([e.eid for e in ans.evidence])
res = rag.make_chart("sales", {"chart": "bar", "x": "region", "y": "revenue", "agg": "sum", "top_n": 3})
```

## How a question flows

1. **Route**: keywords decide whether to widen visual / table retrieval and whether the dataset branch is needed.
2. **Retrieve**: the question is embedded twice, with Sentence Transformers (`q_t`) and with the CLIP text tower (`q_v`).
   Top-k text/table/caption records come from the text index and top-k figures/charts/pages from the visual index.
   Hits are merged, and neighbours on the same page are added.
3. **Compute** (datasets): the planner proposes a chart spec, the engine aggregates the data and draws the chart, and
   the numeric result becomes evidence `[E1]` marked *computed*.
4. **Reason**: evidence is serialised in order with images interleaved, and Qwen2.5-VL answers using only that evidence.
5. **Cite**: `[E#]` tags are validated, and uncited sentences are linked to the most similar evidence (eq. 26).

## Evaluate

```bash
python evaluate.py eval_questions.json      # page hit@k, modality hit@k, keyword recall, citation rate
```

Edit `eval_questions.json` with questions for your own documents. For the report's objective 7, also run the pipeline on
public benchmarks such as ScienceQA, MMLongBench-Doc, ViDoRe or DocVQA.

## Tests

```bash
python -m unittest discover -s tests -v    # offline, uses fake encoders, no downloads
```

## Known limitations (good material for your report's "future work")

* CLIP's text tower is weak at reading text inside charts, so charts are also indexed through VLM captions. With the
  `extractive` backend there are no captions.
* Chart/vector-graphic detection relies on the page render plus a caption/colour heuristic, not a trained classifier.
* Scanned PDFs need OCR before ingestion (not included).
* Chunk size is measured in words, not model tokens.
* The router is rule-based. An LLM or learned router (UniversalRAG / RouteDRAG style) is a natural extension.
* FAISS runs in-process. Use a server (Qdrant, Milvus) for very large collections.

## Project layout

```
app.py              Streamlit UI (Ask / Chart builder / Library)
cli.py              command-line interface
evaluate.py         retrieval + answer evaluation
mmrag/              the library (config, ingestion, embeddings, vectorstore, router, retriever,
                    datasets, backends, reasoner, citations, pipeline)
scripts/make_samples.py   demo data
tests/              offline unit tests
```
