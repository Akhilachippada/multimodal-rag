"""Command line interface.

  python cli.py add sample_data/sales_report.pdf sample_data/sales.csv
  python cli.py ask "Which region has the highest revenue?" --show-evidence
  python cli.py chart sales "average units per product by region"
  python cli.py list
"""
import argparse

from mmrag import Config, MMRAG


def main():
    ap = argparse.ArgumentParser(description="Multimodal RAG")
    ap.add_argument("--backend", choices=["extractive", "openai_compat", "qwen_local"], default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add"); a.add_argument("files", nargs="+"); a.add_argument("--label", choices=["chart", "image"])
    q = sub.add_parser("ask"); q.add_argument("question"); q.add_argument("--dataset"); q.add_argument("--image")
    q.add_argument("--show-evidence", action="store_true")
    c = sub.add_parser("chart"); c.add_argument("dataset"); c.add_argument("request")
    sub.add_parser("list")
    args = ap.parse_args()

    cfg = Config()
    if args.backend:
        cfg.backend = args.backend
    rag = MMRAG(cfg)

    if args.cmd == "add":
        for f in args.files:
            print(rag.add_file(f, label=args.label, progress=print))
    elif args.cmd == "ask":
        ans = rag.ask(args.question, dataset=args.dataset, query_image=args.image)
        print(ans.text)
        if ans.chart_path:
            print(f"\n[chart saved to {ans.chart_path}]")
        if args.show_evidence:
            print("\n--- evidence ---")
            for e in ans.evidence:
                print(f"[{e.eid}] {e.modality:6} {e.where():35} score={e.score:.3f} {e.short(90)}")
    elif args.cmd == "chart":
        r = rag.chart_from_request(args.dataset, args.request)
        print(r.description)
        print(f"\n[chart saved to {r.path}]")
    elif args.cmd == "list":
        print(rag.stats())
        for k, v in rag.registry.items():
            print(k, v["type"], v["name"], v.get("counts"))


if __name__ == "__main__":
    main()
