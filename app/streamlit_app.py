"""
Streamlit chatbot UI for the Internal Knowledge Navigator.
 
Run from the project root:
    streamlit run app/streamlit_app.py
 
Features:
    - Standard chat interface (question in, answer out, history preserved)
    - Per-answer cost (USD) and latency shown under each assistant message
    - Sources list with freshness flags (OUTDATED/DEPRECATED)
    - A "View trace in Langfuse" link per answer, so you can jump straight
      into the full trace (retrieval, MCP freshness check, generation,
      guardrails) for that specific question
    - A running total cost in the sidebar across the whole session
    - A Live Evaluation tab that scores every question you actually ask,
      in real time, using cheap local proxies (no extra LLM calls):
        - faithfulness_proxy: the same lexical-overlap groundedness check
          used by the OUTPUT guardrail, surfaced as a number
        - relevancy_proxy: cosine similarity between question/answer
          embeddings
      You can export this live history to run through the real, LLM-judged
      RAGAS pipeline (eval/run_ragas_eval.py) whenever you want a rigorous
      score instead of the fast local proxy.
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path
 
import numpy as np
 
# ---------------------------------------------------------------------
# Make sure the project root (the folder that CONTAINS the `app` package)
# is on sys.path. Streamlit runs this script directly — unlike
# `python -m app.main`, it does NOT automatically add the project root,
# so `from app.rag_pipeline import ...` would otherwise fail with
# "ModuleNotFoundError: No module named 'app'".
# THIS BLOCK MUST COME BEFORE `from app... import ...` BELOW.
# ---------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
 
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
 
from app import config  # noqa: E402
from app.rag_pipeline import answer_query  # noqa: E402
from app.guardrails_setup import _groundedness_score  # noqa: E402
from app.embeddings import Embedder  # noqa: E402
 
 
st.set_page_config(
    page_title="Internal Knowledge Navigator",
    page_icon="📚",
    layout="centered",
)
 
LIVE_EVAL_EXPORT_PATH = Path(PROJECT_ROOT) / "eval" / "live_eval_dataset.jsonl"
 
st.title("📚 Internal Knowledge Navigator")
st.caption("Ask a question about the Nimbus Cloud Systems wiki.")
 
# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role", "content", "meta"?}
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0
if "live_eval_rows" not in st.session_state:
    st.session_state.live_eval_rows = []  # one row per real question you've asked
 
# ---------------------------------------------------------------------
# Sidebar — session stats + What's New
# ---------------------------------------------------------------------
with st.sidebar:
    st.subheader("Session stats")
    st.metric("Total cost this session", f"${st.session_state.total_cost:.5f}")
    st.metric("Questions asked", len([m for m in st.session_state.messages if m["role"] == "user"]))
    if st.session_state.live_eval_rows:
        faith_vals = [r["faithfulness_proxy"] for r in st.session_state.live_eval_rows
                      if r.get("faithfulness_proxy") is not None]
        if faith_vals:
            st.metric("Avg live faithfulness proxy", f"{sum(faith_vals) / len(faith_vals):.2f}")
        st.caption("See 📊 Live Evaluation tab for full scoring.")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.total_cost = 0.0
        st.session_state.live_eval_rows = []
        st.rerun()
 
 
def render_meta(meta: dict):
    """Renders the cost/latency/sources/trace-link footer under an answer."""
    cols = st.columns(3)
    cols[0].caption(f"💵 Cost: ${meta['cost_usd']:.5f}")
    cols[1].caption(f"⏱️ Latency: {meta['latency_seconds']}s")
    cols[2].caption("🚫 Blocked" if meta["blocked"] else "✅ Passed guardrails")
 
    live = meta.get("live_scores")
    if live:
        score_cols = st.columns(2)
        faith = live.get("faithfulness_proxy")
        rel = live.get("relevancy_proxy")
        score_cols[0].caption(
            f"🧭 Faithfulness proxy: {faith:.2f}" if faith is not None else "🧭 Faithfulness proxy: n/a"
        )
        score_cols[1].caption(
            f"🎯 Relevancy proxy: {rel:.2f}" if rel is not None else "🎯 Relevancy proxy: n/a"
        )
 
    if meta.get("sources"):
        with st.expander(f"📄 Sources ({len(meta['sources'])})"):
            for s in meta["sources"]:
                flag = "" if s["status"] == "current" else f"  ⚠️ **{s['status'].upper()}**"
                st.markdown(f"- **[{s['page_id']}]** {s['title']} → {s['heading']}{flag}")
 
    if meta.get("trace_url"):
        st.markdown(f"[🔍 View trace in Langfuse]({meta['trace_url']})")
    elif meta.get("trace_id") is None:
        st.caption("_Langfuse not configured — no trace available._")
 
 
# =======================================================================
# Live evaluation (scores YOUR actual chat questions, in real time)
# =======================================================================
_live_embedder = None
 
 
def _get_live_embedder():
    """Lazy singleton so we don't reload the embedding backend on every
    chat turn. Reuses the same Embedder class the retriever already uses,
    so no new dependency is introduced.
    """
    global _live_embedder
    if _live_embedder is None:
        try:
            _live_embedder = Embedder.load_for_query()
        except Exception as e:  # pragma: no cover - eval scoring must never break the chat
            print(f"[live-eval] embedder unavailable, relevancy proxy disabled: {e}")
            _live_embedder = False  # sentinel: "tried and failed"
    return _live_embedder or None
 
 
def _cosine_similarity(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0
 
 
def compute_live_scores(question: str, answer: str, contexts: list[str]) -> dict:
    """Cheap, dependency-free proxy metrics computed synchronously in the
    request path — NOT the same as the LLM-judged RAGAS metrics in
    eval/run_ragas_eval.py, but close enough to flag a bad answer live.
 
    - faithfulness_proxy: reuses guardrails_setup._groundedness_score, the
      exact lexical-overlap check already run inside the OUTPUT guardrail
      — just surfaced as a number here instead of a pass/fail.
    - relevancy_proxy: cosine similarity between question and answer
      embeddings (same embedder the retriever uses).
    """
    scores = {"faithfulness_proxy": None, "relevancy_proxy": None}
 
    if contexts:
        try:
            scores["faithfulness_proxy"] = round(_groundedness_score(answer, contexts), 3)
        except Exception as e:  # pragma: no cover
            print(f"[live-eval] faithfulness proxy failed: {e}")
 
    embedder = _get_live_embedder()
    if embedder is not None and answer:
        try:
            q_vec = embedder.embed_query(question)
            a_vec = embedder.embed_query(answer)
            scores["relevancy_proxy"] = round(_cosine_similarity(q_vec, a_vec), 3)
        except Exception as e:  # pragma: no cover
            print(f"[live-eval] relevancy proxy failed: {e}")
 
    return scores
 
 
def render_live_eval_section():
    st.subheader("🟢 Live Session Evaluations")
    st.caption(
        "Every question you've asked in the Chat tab this session, scored in real "
        "time with cheap local proxies (no extra LLM calls)."
    )
 
    rows = st.session_state.live_eval_rows
    if not rows:
        st.info("No live questions scored yet — ask something in the **💬 Chat** tab first.")
        return
 
    live_df = pd.DataFrame(rows)
 
    agg_cols = st.columns(4)
    agg_cols[0].metric("Questions asked", len(live_df))
    if "faithfulness_proxy" in live_df:
        avg_faith = live_df["faithfulness_proxy"].dropna().mean()
        agg_cols[1].metric("Avg faithfulness proxy", f"{avg_faith:.2f}" if pd.notna(avg_faith) else "n/a")
    if "relevancy_proxy" in live_df:
        avg_rel = live_df["relevancy_proxy"].dropna().mean()
        agg_cols[2].metric("Avg relevancy proxy", f"{avg_rel:.2f}" if pd.notna(avg_rel) else "n/a")
    if "blocked" in live_df:
        agg_cols[3].metric("Guardrail-blocked", f"{int(live_df['blocked'].sum())} / {len(live_df)}")
 
    display_cols = [c for c in [
        "timestamp", "question", "answer", "faithfulness_proxy", "relevancy_proxy",
        "blocked", "block_reason", "cost_usd", "latency_seconds",
    ] if c in live_df.columns]
    st.dataframe(
        live_df[display_cols].sort_values("timestamp", ascending=False),
        hide_index=True, use_container_width=True,
    )
 
    low_faith_live = live_df[live_df["faithfulness_proxy"].fillna(1.0) < 0.5] if "faithfulness_proxy" in live_df else pd.DataFrame()
    if len(low_faith_live):
        st.warning(f"⚠️ {len(low_faith_live)} live answer(s) scored below 0.5 on the faithfulness proxy — "
                   f"possibly under-grounded in retrieved context.")
 
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("💾 Export live Q&A for full RAGAS scoring"):
            LIVE_EVAL_EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(LIVE_EVAL_EXPORT_PATH, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps({
                        "question": r["question"],
                        "answer": r["answer"],
                        "contexts": r.get("contexts", []) or [""],
                        # No human-written ground_truth for live questions —
                        # faithfulness/answer_relevancy don't need it, but
                        # context_precision/context_recall will be unreliable
                        # without one. Fill in ground truths by hand if you
                        # want those two metrics to mean anything here.
                        "ground_truth": "",
                    }) + "\n")
            st.success(f"Wrote {len(rows)} live questions to `{LIVE_EVAL_EXPORT_PATH.relative_to(PROJECT_ROOT)}`. "
                       f"Point eval/run_ragas_eval.py's DATASET_PATH at this file and run it in "
                       f"ragas-env for real LLM-judged scores.")
    with col_b:
        if st.button("🗑️ Clear live eval history"):
            st.session_state.live_eval_rows = []
            st.rerun()
 
 
# =======================================================================
# Main layout: Chat + Live Evaluation tabs
# =======================================================================
chat_tab, eval_tab = st.tabs(["💬 Chat", "📊 Live Evaluation"])
 
with chat_tab:
    # -------------------------------------------------------------
    # Render chat history
    # -------------------------------------------------------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "meta" in msg:
                render_meta(msg["meta"])
 
    # -------------------------------------------------------------
    # Chat input
    # -------------------------------------------------------------
    if prompt := st.chat_input("Ask a question about the internal wiki..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
 
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = answer_query(prompt, user_id="streamlit-user")
                contexts = result.get("_contexts", [])
                live_scores = compute_live_scores(prompt, result["answer"], contexts)
 
            st.markdown(result["answer"])
            meta = {
                "cost_usd": result["cost_usd"],
                "latency_seconds": result["latency_seconds"],
                "blocked": result["blocked"],
                "sources": result.get("sources", []),
                "trace_id": result.get("trace_id"),
                "trace_url": result.get("trace_url"),
                "live_scores": live_scores,
            }
            render_meta(meta)
 
        st.session_state.total_cost += result["cost_usd"]
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "meta": meta,
        })
        st.session_state.live_eval_rows.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": prompt,
            "answer": result["answer"],
            "contexts": contexts,
            "faithfulness_proxy": live_scores.get("faithfulness_proxy"),
            "relevancy_proxy": live_scores.get("relevancy_proxy"),
            "blocked": result["blocked"],
            "block_reason": result.get("block_reason"),
            "cost_usd": result["cost_usd"],
            "latency_seconds": result["latency_seconds"],
        })
 
with eval_tab:
    render_live_eval_section()
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 