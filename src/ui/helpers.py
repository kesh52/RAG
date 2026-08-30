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


# ===========================================================================
# Plotly Chart Helpers for Professional Evaluation Dashboards
# ===========================================================================

def create_radar_comparison_chart(
    runs_data: list[dict],
    title: str = "RAG Multi-Dimensional Quality Radar",
):
    """Generate an interactive Plotly radar chart comparing 1 or more runs across the 4 RAGAS metrics."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    categories = ["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall"]
    # Close the radar loop
    categories_closed = categories + [categories[0]]

    fig = go.Figure()

    colors = [
        {"line": "rgb(31, 119, 180)", "fill": "rgba(31, 119, 180, 0.2)"},
        {"line": "rgb(46, 204, 113)", "fill": "rgba(46, 204, 113, 0.25)"},
        {"line": "rgb(231, 76, 60)", "fill": "rgba(231, 76, 60, 0.2)"},
        {"line": "rgb(155, 89, 182)", "fill": "rgba(155, 89, 182, 0.2)"},
    ]

    # Add Target Benchmark Polygon (0.85 Target)
    target_vals = [0.85, 0.85, 0.80, 0.80, 0.85]
    fig.add_trace(
        go.Scatterpolar(
            r=target_vals,
            theta=categories_closed,
            mode="lines",
            name="Target Quality (≥0.80)",
            line=dict(color="rgba(128, 128, 128, 0.6)", dash="dash", width=1.5),
            hoverinfo="name+r",
        )
    )

    for i, item in enumerate(runs_data):
        color = colors[i % len(colors)]
        r_vals = [
            item.get("faithfulness", 0) or 0,
            item.get("answer_relevancy", 0) or 0,
            item.get("context_precision", 0) or 0,
            item.get("context_recall", 0) or 0,
        ]
        # Close loop
        r_vals_closed = r_vals + [r_vals[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=r_vals_closed,
                theta=categories_closed,
                fill="toself",
                fillcolor=color["fill"],
                name=item.get("name", f"Run {i+1}"),
                line=dict(color=color["line"], width=2.5),
                hoverinfo="name+theta+r",
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1.0],
                tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=["0.2", "0.4", "0.6", "0.8 (Target)", "1.0"],
                gridcolor="rgba(128, 128, 128, 0.2)",
            ),
            angularaxis=dict(
                gridcolor="rgba(128, 128, 128, 0.2)",
                linecolor="rgba(128, 128, 128, 0.3)",
            ),
        ),
        showlegend=True,
        margin=dict(l=40, r=40, t=40, b=30),
        height=320,
    )
    return fig


def create_metric_sparkline_chart(
    run_labels: list[str],
    values: list[float | None],
    metric_name: str,
    target_threshold: float = 0.80,
    line_color: str = "#2ecc71",
):
    """Generate a clean Plotly sparkline with a target line for individual metric cards."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    clean_vals = [v if v is not None else 0.0 for v in values]

    fig = go.Figure()

    # Target line
    fig.add_hline(
        y=target_threshold,
        line_dash="dot",
        line_color="rgba(128, 128, 128, 0.5)",
        annotation_text=f"Target ({target_threshold})",
        annotation_position="bottom right",
        annotation_font_size=9,
    )

    # Historical trend line with filled gradient
    fig.add_trace(
        go.Scatter(
            x=run_labels,
            y=clean_vals,
            mode="lines+markers",
            name=metric_name,
            line=dict(color=line_color, width=2.5),
            marker=dict(size=6, color=line_color),
            fill="tozeroy",
            fillcolor=f"rgba{tuple(list(bytes.fromhex(line_color.lstrip('#'))) + [0.12])}" if line_color.startswith("#") else "rgba(46, 204, 113, 0.12)",
            hovertemplate="<b>%{x}</b><br>Score: %{y:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        height=130,
        margin=dict(l=10, r=10, t=10, b=15),
        yaxis=dict(range=[0, 1.05], tickvals=[0, 0.5, 1.0], gridcolor="rgba(128, 128, 128, 0.15)"),
        xaxis=dict(showgrid=False, tickfont=dict(size=9)),
        showlegend=False,
    )
    return fig


