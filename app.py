"""Streamlit UI:  streamlit run app.py"""
import tempfile
from pathlib import Path

import streamlit as st

from mmrag import Config, MMRAG
from mmrag.datasets import col_kinds

st.set_page_config(page_title="Multimodal RAG - Cross-Modal Reasoning", page_icon="🧩", layout="wide")


@st.cache_resource(show_spinner="Loading encoders and index ...")
def get_engine(backend: str, api_base: str, api_model: str, qwen_model: str, load_4bit: bool) -> MMRAG:
    cfg = Config(backend=backend, api_base=api_base, api_model=api_model, qwen_model=qwen_model, load_4bit=load_4bit)
    return MMRAG(cfg)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🧩 Multimodal RAG")
    st.caption("Text + tables + images + charts + datasets")
    _opts = ["extractive", "openai_compat", "qwen_local"]
    backend = st.selectbox("Reasoning backend", _opts,
                       index=_opts.index(Config().backend) if Config().backend in _opts else 0,
                       help="extractive = no model. openai_compat = Ollama/vLLM serving Qwen2.5-VL. "
                            "qwen_local = load Qwen2.5-VL in this process (GPU).")
    api_base = api_model = qwen_model = ""
    load_4bit = False
    defaults = Config()
    if backend == "openai_compat":
        api_base = st.text_input("API base", defaults.api_base)
        api_model = st.text_input("Model name", defaults.api_model)
    elif backend == "qwen_local":
        qwen_model = st.text_input("HF model", defaults.qwen_model)
        load_4bit = st.checkbox("4-bit quantisation (NVIDIA GPU)")
    rag = get_engine(backend, api_base or defaults.api_base, api_model or defaults.api_model,
                     qwen_model or defaults.qwen_model, load_4bit)

    st.divider()
    st.subheader("Upload")
    files = st.file_uploader("PDF reports, chart/figure images, datasets (CSV/XLSX/JSON), text",
                             type=["pdf", "png", "jpg", "jpeg", "webp", "csv", "tsv", "xlsx", "xls", "json", "txt", "md"],
                             accept_multiple_files=True)
    image_kind = st.radio("Uploaded pictures are...", ["auto-detect", "charts / graphs", "photos / diagrams"],
                          horizontal=False)
    if st.button("Index files", type="primary", disabled=not files):
        label = {"charts / graphs": "chart", "photos / diagrams": "image"}.get(image_kind)
        log = st.empty()
        for f in files:
            dest = rag.cfg.path("uploads") / f.name
            dest.write_bytes(f.getbuffer())
            try:
                rag.add_file(str(dest), label=label, progress=lambda m: log.info(m))
            except Exception as ex:
                st.error(f"{f.name}: {ex}")
        log.success("Done.")
        st.rerun()

docs = rag.registry
tab_ask, tab_chart, tab_lib = st.tabs(["💬 Ask", "📊 Chart builder", "📚 Library"])

# ------------------------------------------------------------------ Ask
with tab_ask:
    if "history" not in st.session_state:
        st.session_state.history = []
    c1, c2 = st.columns([3, 2])
    with c1:
        question = st.text_area("Your question", placeholder="e.g. Which region had the highest revenue, and does the "
                                "chart in the report agree?  /  Plot monthly revenue by product", height=90)
    with c2:
        names = {v["name"]: k for k, v in docs.items()}
        scope = st.multiselect("Restrict to documents (optional)", list(names))
        ds = st.selectbox("Dataset for computed answers", ["(auto)"] + list(rag.datasets))
        qimg = st.file_uploader("Attach an image/chart to the question (optional)",
                                type=["png", "jpg", "jpeg", "webp"], key="qimg")
    if st.button("Ask", type="primary", disabled=not question.strip()):
        qpath = None
        if qimg is not None:
            qpath = str(Path(tempfile.mkdtemp()) / qimg.name)
            Path(qpath).write_bytes(qimg.getbuffer())
        with st.spinner("Retrieving evidence and reasoning ..."):
            try:
                ans = rag.ask(question, [names[n] for n in scope] or None,
                              None if ds == "(auto)" else ds, qpath)
                st.session_state.history.insert(0, (question, ans))
            except Exception as ex:
                st.error(f"Failed: {ex}")

    for q, ans in st.session_state.history[:5]:
        st.markdown(f"#### ❓ {q}")
        st.caption("Routing: " + ", ".join(ans.route.intents))
        left, right = st.columns([3, 2])
        with left:
            st.markdown(ans.text)
            if ans.citations:
                with st.expander("Sentence → source attribution"):
                    for c in ans.citations:
                        st.markdown(f"- {c['sentence']}  \n  `{', '.join(c['evidence'])}` ({c['method']})")
        with right:
            if ans.chart_path:
                st.image(ans.chart_path, caption="Chart generated from your dataset")
        st.markdown("**Evidence used**")
        for e in ans.evidence:
            with st.expander(f"[{e.eid}] {e.modality} · {e.where()} · {e.source} · score {e.score:.2f}"):
                if e.asset:
                    st.image(e.asset, width=420)
                if e.text:
                    st.text(e.text[:1500])
        st.divider()

# ------------------------------------------------------------------ Chart builder
with tab_chart:
    if not rag.datasets:
        st.info("Upload a CSV / Excel / JSON dataset in the sidebar to build charts.")
    else:
        table = st.selectbox("Dataset", list(rag.datasets), key="cb_table")
        df = rag.datasets[table][0]
        st.caption(f"{len(df)} rows × {len(df.columns)} columns")
        st.dataframe(df.head(20), use_container_width=True)

        st.markdown("##### Describe the chart in plain language")
        req = st.text_input("e.g. “average units per product by region as a bar chart”", key="cb_req")
        if st.button("Create from description", disabled=not req):
            try:
                st.session_state.cb_result = rag.chart_from_request(table, req)
            except Exception as ex:
                st.error(str(ex))

        st.markdown("##### …or choose the fields yourself")
        k = col_kinds(df)
        cols = list(df.columns)
        a, b, c, d = st.columns(4)
        chart = a.selectbox("Chart type", ["bar", "line", "pie", "scatter", "hist", "box", "heatmap"])
        x = b.selectbox("X", [None] + cols)
        y = c.selectbox("Y (numeric)", [None] + k["numeric"])
        g = d.selectbox("Group by", [None] + k["categorical"])
        e1, e2, e3 = st.columns(3)
        agg = e1.selectbox("Aggregate", ["sum", "mean", "count", "median", "max", "min"])
        top_n = e2.number_input("Top N (0 = all)", 0, 50, 0)
        fcol = e3.selectbox("Filter column", [None] + cols)
        fval = st.text_input("Filter value (equals)") if fcol else None
        if st.button("Create chart"):
            spec = {"chart": chart, "x": x, "y": y, "group": g, "agg": agg, "top_n": top_n or None,
                    "filters": [{"col": fcol, "op": "==", "value": fval}] if fcol and fval else []}
            try:
                st.session_state.cb_result = rag.make_chart(table, spec)
            except Exception as ex:
                st.error(str(ex))

        res = st.session_state.get("cb_result")
        if res:
            st.image(res.path)
            st.markdown(res.description.split("Data:")[0])
            st.dataframe(res.table, use_container_width=True)
            st.download_button("Download PNG", Path(res.path).read_bytes(), "chart.png", "image/png")
            st.download_button("Download data (CSV)", res.table.to_csv(index=False), "chart_data.csv", "text/csv")

# ------------------------------------------------------------------ Library
with tab_lib:
    st.write(rag.stats())
    for doc_id, info in docs.items():
        c1, c2 = st.columns([6, 1])
        c1.markdown(f"**{info['name']}** · {info['type']} · added {info['added']} · "
                    f"{info.get('n_text', 0)} text / {info.get('n_visual', 0)} visual records · "
                    f"`{info.get('counts', {})}`")
        if c2.button("Delete", key=f"del_{doc_id}"):
            rag.remove(doc_id)
            st.rerun()
