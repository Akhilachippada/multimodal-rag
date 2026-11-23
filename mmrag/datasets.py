"""Dataset upload + chart engine.

* load_tables()      - CSV / TSV / XLSX / JSON / Parquet -> DataFrames
* dataset_elements() - profile text, row-group chunks and auto-generated charts for indexing
* ChartPlanner       - question -> safe JSON chart spec (LLM if available, heuristic otherwise)
* execute_spec()     - spec -> aggregated table + matplotlib PNG (no LLM-written code is ever executed)
"""
from __future__ import annotations

import json
import re
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import Config  # noqa: E402
from .schema import Element  # noqa: E402
from .utils import df_to_md  # noqa: E402

AGGS = {"sum", "mean", "count", "median", "max", "min"}
CHARTS = {"bar", "line", "scatter", "hist", "pie", "box", "heatmap"}


# ------------------------------------------------------------------ loading
def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        if not (pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c])):
            continue
        s = df[c].dropna()
        if s.empty:
            continue
        num = pd.to_numeric(s.astype(str).str.replace(",", "").str.replace("%", "").str.strip(), errors="coerce")
        if num.notna().mean() >= 0.9:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "").str.replace("%", "").str.strip(),
                                  errors="coerce")
            continue
        looks_date = s.astype(str).str.match(r"^\d{1,4}[-/.]\d{1,2}([-/.]\d{1,4})?").mean() > 0.8 or \
            re.search(r"date|time|month|day", c, re.I)
        if looks_date:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dt = pd.to_datetime(df[c], errors="coerce")
            if dt.notna().mean() >= 0.9:
                df[c] = dt
    return df


def load_tables(path: str) -> dict[str, pd.DataFrame]:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".csv":
        raw = {p.stem: pd.read_csv(p)}
    elif ext == ".tsv":
        raw = {p.stem: pd.read_csv(p, sep="\t")}
    elif ext in (".xlsx", ".xls"):
        sheets = pd.read_excel(p, sheet_name=None)
        raw = {(p.stem if len(sheets) == 1 else f"{p.stem}:{k}"): v for k, v in sheets.items()}
    elif ext == ".json":
        try:
            raw = {p.stem: pd.read_json(p)}
        except ValueError:
            raw = {p.stem: pd.json_normalize(json.loads(p.read_text(encoding="utf-8")))}
    elif ext == ".parquet":
        raw = {p.stem: pd.read_parquet(p)}
    else:
        raise ValueError(f"Unsupported dataset type: {ext}")
    return {k: _coerce(v) for k, v in raw.items() if len(v.columns)}


def col_kinds(df: pd.DataFrame) -> dict[str, list[str]]:
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    dt = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    cat = [c for c in df.columns if c not in num and c not in dt]
    return {"numeric": num, "datetime": dt, "categorical": cat}


# ------------------------------------------------------------------ profiling
def profile_text(name: str, df: pd.DataFrame) -> str:
    k = col_kinds(df)
    lines = [f"Dataset '{name}': {len(df)} rows x {len(df.columns)} columns."]
    for c in df.columns:
        s = df[c]
        if c in k["numeric"]:
            lines.append(f"- {c} (numeric): min {s.min():.4g}, max {s.max():.4g}, mean {s.mean():.4g}, "
                         f"sum {s.sum():.4g}, missing {int(s.isna().sum())}")
        elif c in k["datetime"]:
            lines.append(f"- {c} (date): {s.min().date()} to {s.max().date()}")
        else:
            top = s.astype(str).value_counts().head(5)
            lines.append(f"- {c} (category): {s.nunique()} unique; most common: "
                         + ", ".join(f"{i} ({n})" for i, n in top.items()))
    return "\n".join(lines)


def describe_result(spec: dict, res: pd.DataFrame, note: str, name: str) -> str:
    chart = spec["chart"]
    parts = [f"{chart.capitalize()} chart from dataset '{name}': {spec['title']}."]
    if chart in ("bar", "line", "pie") and res.shape[1] == 2 and len(res):
        xc, yc = res.columns
        r = res.dropna()
        if len(r):
            hi, lo = r.loc[r[yc].idxmax()], r.loc[r[yc].idxmin()]
            parts.append(f"{len(r)} points; x='{xc}', y='{yc}'. Highest: {hi[xc]} = {hi[yc]:.4g}. "
                         f"Lowest: {lo[xc]} = {lo[yc]:.4g}."
                         + (f" Total of plotted values: {r[yc].sum():.4g}." if spec["agg"] in ("sum", "count") else ""))
            if chart == "line" and len(r) >= 2 and r[yc].iloc[0]:
                a, b = r[yc].iloc[0], r[yc].iloc[-1]
                parts.append(f"From {r[xc].iloc[0]} ({a:.4g}) to {r[xc].iloc[-1]} ({b:.4g}): "
                             f"{'up' if b > a else 'down'} {abs(b - a) / abs(a) * 100:.1f}%.")
    if note:
        parts.append(note)
    parts.append("Data:\n" + df_to_md(res, 30))
    return "\n".join(p for p in parts if p)


# ------------------------------------------------------------------ spec handling
def _fix_col(df: pd.DataFrame, c):
    if c is None:
        return None
    c = str(c)
    if c in df.columns:
        return c
    low = {x.lower(): x for x in df.columns}
    return low.get(c.lower())


def validate_spec(spec, df: pd.DataFrame):
    if not isinstance(spec, dict):
        return None
    k = col_kinds(df)
    chart = str(spec.get("chart", "bar")).lower()
    chart = chart if chart in CHARTS else "bar"
    x, y, g = _fix_col(df, spec.get("x")), _fix_col(df, spec.get("y")), _fix_col(df, spec.get("group"))
    agg = str(spec.get("agg", "sum")).lower()
    agg = agg if agg in AGGS else "sum"
    filters = []
    for f in spec.get("filters") or []:
        if isinstance(f, dict) and _fix_col(df, f.get("col")):
            filters.append({"col": _fix_col(df, f["col"]), "op": f.get("op", "=="), "value": f.get("value")})
    try:
        top_n = max(1, min(50, int(spec.get("top_n") or 0))) if spec.get("top_n") else None
    except (TypeError, ValueError):
        top_n = None

    if chart == "hist":
        x = x if x in k["numeric"] else (y if y in k["numeric"] else None)
        if not x:
            return None
        y = None
    elif chart == "scatter":
        if x not in k["numeric"] or y not in k["numeric"]:
            return None
    elif chart == "box":
        if y not in k["numeric"]:
            y, x = (x, None) if x in k["numeric"] else (None, x)
        if y not in k["numeric"]:
            return None
    elif chart == "heatmap":
        if len(k["numeric"]) < 2:
            return None
    else:  # bar / line / pie
        if not x:
            x = (k["datetime"] or k["categorical"] or [None])[0]
        if not x:
            return None
        if y not in k["numeric"]:
            y, agg = None, "count"
    title = spec.get("title") or _auto_title(chart, x, y, g, agg)
    return {"chart": chart, "x": x, "y": y, "group": g, "agg": agg, "top_n": top_n,
            "filters": filters, "title": str(title)}


def _auto_title(chart, x, y, g, agg):
    if chart == "hist":
        return f"Distribution of {x}"
    if chart == "scatter":
        return f"{y} vs {x}"
    if chart == "heatmap":
        return "Correlation heatmap"
    if chart == "box":
        return f"{y}" + (f" by {x}" if x else "") + " (box plot)"
    what = f"{agg} of {y}" if y else "count of rows"
    return f"{what} by {x}" + (f" and {g}" if g else "")


def apply_filters(df: pd.DataFrame, filters) -> pd.DataFrame:
    for f in filters or []:
        c, op, v = f["col"], f.get("op", "=="), f.get("value")
        s = df[c]
        try:
            if op in ("==", "="):
                df = df[s.astype(str).str.lower() == str(v).lower()]
            elif op == "!=":
                df = df[s.astype(str).str.lower() != str(v).lower()]
            elif op in (">", ">=", "<", "<="):
                if pd.api.types.is_datetime64_any_dtype(s):
                    a, b = s, pd.to_datetime(v)
                else:
                    a, b = pd.to_numeric(s, errors="coerce"), float(v)
                df = df[{">": a > b, ">=": a >= b, "<": a < b, "<=": a <= b}[op]]
            elif op == "in":
                vals = [str(i).lower() for i in (v if isinstance(v, list) else [v])]
                df = df[s.astype(str).str.lower().isin(vals)]
            elif op == "contains":
                df = df[s.astype(str).str.contains(str(v), case=False, na=False)]
        except Exception:
            continue
    return df


@dataclass
class ChartResult:
    table: pd.DataFrame
    note: str
    path: str
    spec: dict
    description: str


def execute_spec(df: pd.DataFrame, spec: dict, out_dir: Path, name: str = "dataset") -> ChartResult:
    spec = validate_spec(spec, df)
    if spec is None:
        raise ValueError("Could not build a valid chart from that request/spec.")
    df = apply_filters(df, spec["filters"])
    if df.empty:
        raise ValueError("No rows match the filters.")
    chart, x, y, g, agg = spec["chart"], spec["x"], spec["y"], spec["group"], spec["agg"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=130)
    note, res = "", pd.DataFrame()

    if chart in ("bar", "line", "pie"):
        xs = df[x]
        d = df.copy()
        is_dt = pd.api.types.is_datetime64_any_dtype(xs)
        if is_dt:
            span = (xs.max() - xs.min()).days
            d["_x"] = xs.dt.to_period("M").dt.to_timestamp() if span > 90 else xs.dt.normalize()
        elif chart == "line" and pd.api.types.is_numeric_dtype(xs):
            d["_x"] = xs
        else:
            d["_x"] = xs.astype(str)
        keys = ["_x"] + ([g] if g and chart != "pie" else [])
        grp = d.groupby(keys)[y].agg(agg) if y else d.groupby(keys).size()
        ycol = f"{agg}_{y}" if y else "count"
        res = grp.rename(ycol).reset_index()
        if chart == "line" or is_dt:
            res = res.sort_values("_x")
        elif not g:
            res = res.sort_values(ycol, ascending=False)
        if spec["top_n"] and not g:
            res = res.head(spec["top_n"])
        elif chart == "pie":
            res = res.head(8)
        if g and chart != "pie":
            wide = res.pivot(index="_x", columns=g, values=ycol)
            if chart == "bar":
                wide.plot(kind="bar", ax=ax)
            else:
                wide.plot(ax=ax, marker="o")
            ax.legend(title=g, fontsize=8)
        elif chart == "bar":
            ax.bar(res["_x"].astype(str), res[ycol], color="#3b6fb6")
        elif chart == "line":
            ax.plot(res["_x"], res[ycol], marker="o", color="#3b6fb6")
        else:
            ax.pie(res[ycol].clip(lower=0), labels=res["_x"].astype(str), autopct="%1.0f%%", textprops={"fontsize": 8})
        if chart != "pie":
            ax.set_xlabel(x)
            ax.set_ylabel(ycol)
            plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
            ax.grid(alpha=0.25)
        res = res.rename(columns={"_x": x})

    elif chart == "scatter":
        cols = [x, y] + ([g] if g else [])
        d = df[cols].dropna()
        if g:
            for key, sub in d.groupby(g):
                ax.scatter(sub[x], sub[y], s=14, alpha=0.7, label=str(key))
            ax.legend(title=g, fontsize=8)
        else:
            ax.scatter(d[x], d[y], s=14, alpha=0.7, color="#3b6fb6")
        r = d[x].corr(d[y]) if len(d) > 2 else float("nan")
        note = f"Pearson correlation r = {r:.3f} over {len(d)} points."
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(alpha=0.25)
        res = d[[x, y]].head(30)

    elif chart == "hist":
        s = df[x].dropna()
        ax.hist(s, bins=20, color="#3b6fb6", edgecolor="white")
        ax.set_xlabel(x)
        ax.set_ylabel("frequency")
        st = s.describe()
        res = pd.DataFrame({"statistic": st.index, "value": st.values})
        note = f"Mean {s.mean():.4g}, median {s.median():.4g}, std {s.std():.4g}."

    elif chart == "box":
        if x:
            top = df[x].astype(str).value_counts().head(10).index
            groups = [df.loc[df[x].astype(str) == k, y].dropna() for k in top]
            ax.boxplot(groups, labels=[str(k) for k in top])
            plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
            res = df[df[x].astype(str).isin(top)].groupby(df[x].astype(str))[y].describe().reset_index()
        else:
            ax.boxplot(df[y].dropna())
            res = df[y].describe().reset_index()
            res.columns = ["statistic", "value"]
        ax.set_ylabel(y)

    elif chart == "heatmap":
        nums = col_kinds(df)["numeric"][:12]
        corr = df[nums].corr()
        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(nums)), nums, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(nums)), nums, fontsize=8)
        for i in range(len(nums)):
            for j in range(len(nums)):
                ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)
        pairs = corr.where(~np.eye(len(nums), dtype=bool)).abs().stack()
        if len(pairs):
            (a, b), v = pairs.idxmax(), pairs.max()
            note = f"Strongest correlation: {a} vs {b} (r = {corr.loc[a, b]:.2f})."
        res = corr.round(2).reset_index().rename(columns={"index": "column"})

    ax.set_title(spec["title"], fontsize=10)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"chart_{uuid.uuid4().hex[:10]}.png"
    fig.savefig(path)
    plt.close(fig)
    return ChartResult(res, note, str(path), spec, describe_result(spec, res, note, name))


# ------------------------------------------------------------------ planning from a question
DATA_INTENT = re.compile(
    r"\b(plot|chart|graph|visuali[sz]e|trend|over time|total|sum|average|mean|median|count|how many|"
    r"highest|lowest|maximum|minimum|top \d+|per|by|compare|comparison|distribution|correlat\w*|share|"
    r"proportion|percentage|growth|histogram|scatter|pie|bar)\b", re.I)


def data_intent(question: str) -> bool:
    return bool(DATA_INTENT.search(question))


def _norm(s: str) -> str:
    return re.sub(r"[_\-]+", " ", str(s).lower()).strip()


def match_columns(question: str, df: pd.DataFrame) -> list[str]:
    q = _norm(question)
    hits = []
    for c in df.columns:
        n = _norm(c)
        m = re.search(rf"\b{re.escape(n)}s?\b", q)
        if m:
            hits.append((m.start(), c))
    return [c for _, c in sorted(hits)]


def pick_dataset(question: str, datasets: dict[str, pd.DataFrame]) -> str | None:
    if not datasets:
        return None
    best, best_score = None, -1
    q = _norm(question)
    for name, df in datasets.items():
        score = len(match_columns(question, df)) * 2 + (3 if _norm(name.split(":")[0]) in q else 0)
        if score > best_score:
            best, best_score = name, score
    return best


def heuristic_spec(question: str, df: pd.DataFrame):
    q = question.lower()
    k = col_kinds(df)
    cols = match_columns(question, df)
    nums = [c for c in cols if c in k["numeric"]]
    dts = [c for c in cols if c in k["datetime"]]
    cats = [c for c in cols if c in k["categorical"]]

    if re.search(r"\bpie\b|share of|proportion|percentage of", q):
        chart = "pie"
    elif re.search(r"histogram|distribution", q):
        chart = "hist"
    elif re.search(r"scatter|correlat|relationship| vs\.? | versus ", q):
        chart = "scatter" if len(nums) >= 2 else ("heatmap" if "correlat" in q else "bar")
    elif re.search(r"box ?plot", q):
        chart = "box"
    elif re.search(r"trend|over time|monthly|daily|yearly|growth|line", q) and (dts or k["datetime"]):
        chart = "line"
    else:
        chart = "bar"
    if re.search(r"average|mean|avg", q):
        agg = "mean"
    elif re.search(r"how many|count|number of", q):
        agg = "count"
    elif re.search(r"median", q):
        agg = "median"
    elif re.search(r"maximum|highest|largest|max\b", q):
        agg = "max"
    elif re.search(r"minimum|lowest|smallest|min\b", q):
        agg = "min"
    else:
        agg = "sum"
    top = re.search(r"top (\d+)", q)
    spec = {"chart": chart, "agg": agg, "top_n": int(top.group(1)) if top else None}
    if chart in ("scatter",):
        spec.update(x=nums[0], y=nums[1]) if len(nums) >= 2 else None
        if len(cats) >= 1:
            spec["group"] = cats[0]
    elif chart == "hist":
        spec["x"] = (nums or k["numeric"] or [None])[0]
    elif chart == "heatmap":
        pass
    elif chart == "line":
        spec["x"] = (dts or k["datetime"])[0]
        spec["y"] = (nums or k["numeric"][:1] or [None])[0]
        if cats:
            spec["group"] = cats[0]
    else:
        spec["x"] = (cats or dts or k["categorical"][:1] or [None])[0]
        spec["y"] = (nums or k["numeric"][:1] or [None])[0] if not (agg == "count" and not nums) else None
        if len(cats) > 1 and chart == "bar":
            spec["group"] = cats[1]
    # simple value filters: "in North", "for 2023" -> match category values mentioned in the question
    filters = []
    for c in k["categorical"]:
        if df[c].nunique() <= 50:
            for v in df[c].dropna().astype(str).unique():
                if len(v) > 2 and re.search(rf"\b{re.escape(v.lower())}\b", q) and c not in (spec.get("x"), spec.get("group")):
                    filters.append({"col": c, "op": "==", "value": v})
    spec["filters"] = filters
    return validate_spec(spec, df)


PLANNER_SYSTEM = ("You convert a user's request about a table into a JSON chart specification. "
                  "Reply with ONE JSON object only, no prose, no markdown fences.")


def schema_for_prompt(df: pd.DataFrame, name: str) -> str:
    k = col_kinds(df)
    lines = [f"Table '{name}' ({len(df)} rows). Columns:"]
    for c in df.columns:
        kind = "numeric" if c in k["numeric"] else "date" if c in k["datetime"] else "category"
        ex = ", ".join(map(str, df[c].dropna().astype(str).unique()[:4]))
        lines.append(f"- {c} [{kind}] e.g. {ex}")
    return "\n".join(lines)


class ChartPlanner:
    def __init__(self, backend=None):
        self.backend = backend

    def plan(self, question: str, df: pd.DataFrame, name: str):
        spec = None
        if self.backend is not None and getattr(self.backend, "name", "") != "extractive":
            prompt = (schema_for_prompt(df, name) + f"\n\nRequest: {question}\n\n"
                      'JSON keys: "chart" (bar|line|scatter|hist|pie|box|heatmap), "x", "y", "group" (optional column), '
                      '"agg" (sum|mean|count|median|max|min), "top_n" (optional int), "title", '
                      '"filters" (list of {"col","op","value"}, op in ==,!=,>,>=,<,<=,in,contains). '
                      "Use only the column names listed above. Use null where not needed.")
            try:
                raw = self.backend.generate([("text", prompt)], system=PLANNER_SYSTEM, max_new_tokens=300)
                m = re.search(r"\{.*\}", raw, re.S)
                spec = validate_spec(json.loads(m.group(0)), df) if m else None
            except Exception:
                spec = None
        return spec or heuristic_spec(question, df)


# ------------------------------------------------------------------ indexing helpers
def auto_specs(df: pd.DataFrame, limit: int) -> list[dict]:
    k = col_kinds(df)
    specs = []
    if k["datetime"] and k["numeric"]:
        for n in k["numeric"][:2]:
            specs.append({"chart": "line", "x": k["datetime"][0], "y": n, "agg": "sum"})
    cats = [c for c in k["categorical"] if 2 <= df[c].nunique() <= 30]
    for c in cats[:2]:
        for n in k["numeric"][:2]:
            specs.append({"chart": "bar", "x": c, "y": n, "agg": "sum" if n == k["numeric"][0] else "mean"})
    for n in k["numeric"][:2]:
        specs.append({"chart": "hist", "x": n})
    if len(k["numeric"]) >= 3:
        specs.append({"chart": "heatmap"})
    return specs[:limit]


def dataset_elements(name: str, df: pd.DataFrame, asset_dir: Path, cfg: Config) -> list[Element]:
    els = [Element("table", profile_text(name, df), None, {"page": None, "caption": f"Profile of {name}", "table": name})]
    n = cfg.rows_per_chunk
    for i, start in enumerate(range(0, len(df), n)):
        if i >= cfg.max_row_chunks:
            break
        chunk = df.iloc[start:start + n]
        els.append(Element("table", f"Rows {start + 1}-{start + len(chunk)} of dataset '{name}':\n" + df_to_md(chunk, n),
                           None, {"page": None, "caption": f"rows {start + 1}-{start + len(chunk)}", "table": name}))
    for spec in auto_specs(df, cfg.max_auto_charts):
        try:
            r = execute_spec(df, spec, asset_dir, name)
        except Exception:
            continue
        els.append(Element("chart", r.description, r.path, {"page": None, "caption": r.spec["title"], "table": name}))
    return els
