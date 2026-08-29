"""Shared UI helper functions, metric formatters, and preview renderers."""

import base64
import streamlit as st
import pandas as pd

# ===========================================================================
# Attachment Preview Helpers
# ===========================================================================

def _render_attachment_preview(
    filename: str,
    data: bytes | None = None,
    mime_type: str | None = None,
    label: str = "👁️ Preview Attached Document",
    loader_fn=None,
    key_prefix: str = "preview",
):
    """Render an inline preview for PDF, images, or text files directly inside a collapsed expander."""
    if not filename:
        return
    fn_lower = filename.lower()
    mime = mime_type or ""

    with st.expander(label, expanded=False):
        file_bytes = data
        if file_bytes is None and loader_fn is not None:
            file_bytes = loader_fn()

        if not file_bytes:
            st.warning("Document content unavailable.")
            return

        if fn_lower.endswith(".pdf") or "pdf" in mime:
            base64_pdf = base64.b64encode(file_bytes).decode("utf-8")
            pdf_display = (
                f'<iframe src="data:application/pdf;base64,{base64_pdf}" '
                f'width="100%" height="480" type="application/pdf" '
                f'style="border-radius: 6px; border: 1px solid #ccc;"></iframe>'
            )
            st.markdown(pdf_display, unsafe_allow_html=True)
        elif fn_lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")) or "image" in mime:
            st.image(file_bytes, caption=filename, use_container_width=True)
        elif fn_lower.endswith((".txt", ".log", ".json", ".md", ".yaml", ".yml", ".py", ".sql")) or "text" in mime:
            try:
                text_content = bytes(file_bytes).decode("utf-8", errors="ignore")
                st.code(text_content, language="text")
            except Exception:
                st.write("Unable to decode text content.")
        else:
            st.caption(f"Binary file ({len(file_bytes)} bytes). Use the download button to inspect.")


# ===========================================================================
# Metric Quality & Formatting Helpers
# ===========================================================================

# Per-metric thresholds based on industry production RAG system standards.
# Faithfulness is strictest (hallucination prevention is critical).
# Answer Relevancy targets >= 0.85 for good UX.
# Context Precision/Recall vary more by domain but >= 0.8 is a solid target.

METRIC_THRESHOLDS = {
    "faithfulness":      {"good": 0.90, "ok": 0.70},
    "answer_relevancy":  {"good": 0.85, "ok": 0.70},
    "context_precision": {"good": 0.80, "ok": 0.60},
    "context_recall":    {"good": 0.80, "ok": 0.60},
}
_DEFAULT_THRESHOLDS = {"good": 0.80, "ok": 0.60}


def _quality_icon(value, metric_key: str = "") -> str:
    """Return a colored circle icon based on per-metric thresholds."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "⚪"
    thresholds = METRIC_THRESHOLDS.get(metric_key, _DEFAULT_THRESHOLDS)
    if value >= thresholds["good"]:
        return "🟢"
    if value >= thresholds["ok"]:
        return "🟡"
    return "🔴"


def _format_metric(value, metric_key: str = "") -> str:
    """Format a metric value with its quality icon."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "⚪ N/A"
    return f"{_quality_icon(value, metric_key)} {value:.3f}"


METRIC_INFO = {
    "faithfulness": (
        "**Faithfulness** measures whether the generated answer is factually grounded in the retrieved context — "
        "i.e., the LLM didn't hallucinate information.\n\n"
        "**Example:** If the context says *\"Python 3.12 was released in Oct 2023\"* and the answer says "
        "*\"Python 3.12 was released in Oct 2023\"*, faithfulness = 1.0. If the answer adds "
        "*\"…and it introduced pattern matching\"* (which is not in the context), faithfulness drops.\n\n"
        "🟢 ≥ 0.90 — Production-ready &nbsp;|&nbsp; 🟡 ≥ 0.70 — Acceptable, flag for review &nbsp;|&nbsp; 🔴 < 0.70 — Block / investigate\n\n"
        "*Strictest threshold — hallucination prevention is critical in production.*"
    ),
    "answer_relevancy": (
        "**Answer Relevancy** measures how directly and concisely the answer addresses the original question, "
        "penalizing vague or off-topic responses.\n\n"
        "**Example:** Q: *\"What is the capital of France?\"* → A: *\"Paris\"* = high relevancy. "
        "A: *\"France is a country in Europe with many cities, rivers, and a rich history…\"* = lower relevancy.\n\n"
        "🟢 ≥ 0.85 — Good &nbsp;|&nbsp; 🟡 ≥ 0.70 — Acceptable &nbsp;|&nbsp; 🔴 < 0.70 — Poor\n\n"
        "*Directly impacts user experience — vague answers erode trust.*"
    ),
    "context_precision": (
        "**Context Precision** measures whether the retrieved context chunks that are actually relevant are ranked "
        "higher than irrelevant ones — i.e., the retrieval ranking quality.\n\n"
        "**Example:** If 2 out of 5 retrieved chunks are relevant, and both appear in positions 1 and 2, "
        "precision is high. If they appear at positions 3 and 5, precision is low.\n\n"
        "🟢 ≥ 0.80 — Good &nbsp;|&nbsp; 🟡 ≥ 0.60 — Acceptable &nbsp;|&nbsp; 🔴 < 0.60 — Poor\n\n"
        "*Low precision means noise in the context — tune your reranker or embedding model.*"
    ),
    "context_recall": (
        "**Context Recall** measures how much of the expected ground-truth answer can be attributed to the "
        "retrieved context — i.e., did we retrieve enough relevant information?\n\n"
        "**Example:** If the reference answer has 3 key facts, and the retrieved context covers all 3, "
        "recall = 1.0. If it only covers 1 of the 3, recall ≈ 0.33.\n\n"
        "🟢 ≥ 0.80 — Good &nbsp;|&nbsp; 🟡 ≥ 0.60 — Acceptable &nbsp;|&nbsp; 🔴 < 0.60 — Poor\n\n"
        "*Low recall is a frequent cause of hallucinations — the LLM fills gaps when context is missing.*"
    ),
}

