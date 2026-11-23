"""Evaluate retrieval and answers on a small QA file (report objective 7).

  python evaluate.py eval_questions.json

Each item: {"question": str, "doc": optional file name, "expected_pages": [int], "expected_modalities": [str],
            "expected_keywords": [str]}
Metrics: page hit@k, modality hit@k, keyword recall in the answer, and grounding rate (answers with citations).
"""
import json
import sys

from mmrag import MMRAG


def main(path: str):
    items = json.loads(open(path, encoding="utf-8").read())
    rag = MMRAG()
    names = {v["name"]: k for k, v in rag.registry.items()}
    tot = {"page": [0, 0], "modality": [0, 0], "kw": [0, 0], "cited": [0, 0]}
    for it in items:
        docs = [names[it["doc"]]] if it.get("doc") in names else None
        ans = rag.ask(it["question"], doc_ids=docs)
        pages = {e.meta.get("page") for e in ans.evidence}
        mods = {e.modality for e in ans.evidence}
        if it.get("expected_pages"):
            tot["page"][1] += 1
            tot["page"][0] += bool(pages & set(it["expected_pages"]))
        if it.get("expected_modalities"):
            tot["modality"][1] += 1
            tot["modality"][0] += set(it["expected_modalities"]) <= mods
        for kw in it.get("expected_keywords", []):
            tot["kw"][1] += 1
            tot["kw"][0] += kw.lower() in ans.text.lower()
        tot["cited"][1] += 1
        tot["cited"][0] += bool(ans.citations)
        print(f"- {it['question'][:70]!r}: pages={sorted(p for p in pages if p)} modalities={sorted(mods)}")
    print()
    for k, (hit, n) in tot.items():
        print(f"{k:9}: {hit}/{n} = {hit / n:.2%}" if n else f"{k:9}: n/a")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval_questions.json")
