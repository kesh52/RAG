"""Tab 1: Confluence ETL Pipeline & Interactive Chunking Strategy Playground."""

import logging
import os
from io import StringIO
import streamlit as st
import pandas as pd
import numpy as np

from src.utils.config import config
from src.etl.chunking import (
    BaseChunker,
    RecursiveTextChunker,
    SemanticChunker,
    get_chunker,
)

logger = logging.getLogger(__name__)

SAMPLE_TEXTS = {
    "Confluence: Spring Batch Runbook (SOP-402)": (
        "# SOP-402: Spring Batch Memory Isolation & Chunk-Oriented Processing Runbook\n\n"
        "## 1. Overview & Objective\n"
        "This Standard Operating Procedure defines configuration standards for high-throughput batch ETL jobs running in Kubernetes clusters. "
        "Spring Batch jobs must prevent JVM heap exhaustion and database lock escalation during large-scale data ingestion.\n\n"
        "## 2. Architecture: Chunk-Oriented Processing\n"
        "Spring Batch implements chunk-based execution where `ItemReader` reads records one-by-one, gathers them into a chunk list up to the `commit-interval`, and passes them to `ItemProcessor` and `ItemWriter`. "
        "The entire chunk is committed in a single database transaction boundary.\n\n"
        "Key Configuration Directives:\n"
        "- Default commit-interval: 500 items (do NOT exceed 1,000 without architecture review).\n"
        "- Clear Hibernate Session L1 cache at the end of each chunk using `ChunkListener.afterChunk()`.\n"
        "- Always enable skip and retry policies for transient network timeouts.\n\n"
        "## 3. Incident Remediation: Handling OutOfMemoryError (OOM)\n"
        "If a batch step crashes with `java.lang.OutOfMemoryError: Java heap space`:\n"
        "1. Check the active `commit-interval` in `application.yml`.\n"
        "2. If commit interval is > 1,000 items, reduce to 500 and trigger a retry.\n"
        "3. Verify that container memory limits match JVM `-Xmx` (e.g. 4GB container with `-Xmx3072m`).\n"
        "4. Inspect heap dump with Eclipse Memory Analyzer to ensure entity caches are freed after each chunk transaction."
    ),
    "Confluence: Cloud SQL pgvector Architecture Spec": (
        "# Cloud SQL PostgreSQL Vector Search Architecture & Indexing Standards\n\n"
        "## 1. System Architecture\n"
        "Our RAG architecture relies on Google Cloud SQL PostgreSQL 16 with the `pgvector` extension enabled. "
        "Vector representations of internal documentation are stored in the `documents` table alongside full-text search metadata.\n\n"
        "## 2. Vector Index Design (HNSW)\n"
        "Hierarchical Navigable Small World (HNSW) graphs are constructed using cosine distance (`vector_cosine_ops`). "
        "HNSW provides logarithmic search latency and superior recall compared to IVF-Flat on multi-million row datasets.\n\n"
        "Index parameters:\n"
        "- `m = 16`: Number of bi-directional links per node.\n"
        "- `ef_construction = 64`: Build-time search queue depth.\n"
        "- `ef_search = 40`: Query-time search depth set per session.\n\n"
        "## 3. Maintenance & Vacuuming SOP\n"
        "Vector tables subject to heavy re-indexing require regular autovacuum tuning. "
        "Execute `VACUUM ANALYZE documents;` every 24 hours to prevent index dead tuple bloat."
    ),
    "Confluence: Zero-Trust IAM & Cloud Run Security Spec": (
        "# Security Policy: Zero-Trust Authentication for Internal AI Microservices\n\n"
        "## 1. Scope & Compliance\n"
        "All AI backend microservices deployed on Google Cloud Run must enforce strict IAM authentication and service-to-service mutual validation.\n\n"
        "## 2. Token Exchange & Workload Identity\n"
        "Clients authenticate by presenting short-lived Google OpenID Connect (OIDC) ID tokens with the target audience set to the Cloud Run service URL. "
        "Service accounts utilize IAM Workload Identity Federation without long-lived JSON service account keys.\n\n"
        "## 3. Audit & Access Logs\n"
        "All RAG queries and administrative operations are recorded in Cloud Logging with retention set to 365 days."
    ),
    "Custom / Pasted Confluence Page": "",
}


def _get_ui_embedding_service():
    """Helper to lazily initialize embedding service for UI chunking test."""
    try:
        from src.embeddings.vertex import VertexEmbeddingService
        from google import genai

        gcp_project = config.get("gcp.project")
        gcp_location = config.get("gcp.location")
        model_name = config.get("models.embedding", "text-embedding-005")
        ai_client = genai.Client(vertexai=True, project=gcp_project, location=gcp_location)
        return VertexEmbeddingService(client=ai_client, model_name=model_name)
    except Exception as e:
        logger.warning(f"Could not initialize live Vertex AI embedding service: {e}")
        return None


def render_etl_tab():
    """Render the ETL Pipeline and Chunking Strategy Playground tab."""
    st.header("ETL Pipeline & Chunking Strategies")
    st.markdown(
        "Explore and test different **text chunking strategies** (Recursive vs. Semantic Chunking) on **Confluence pages**, "
        "tune boundaries and distance thresholds, and execute live Confluence ETL ingestion runs."
    )

    etl_subtab_playground, etl_subtab_ingest = st.tabs([
        "✂️ Chunking Playground & Visualizer",
        "🚀 Confluence Ingestion Pipeline",
    ])

    # =======================================================================
    # Subtab 1: Chunking Playground & Visualizer
    # =======================================================================
    with etl_subtab_playground:
        st.subheader("Interactive Chunking Strategy Tester (Confluence & Docs)")
        st.caption(
            "Evaluate how **Recursive Chunking** (hierarchical delimiter splitting) and **Semantic Chunking** (embedding topic transition detection) "
            "segment structured Confluence pages, runbooks, and SOP documents."
        )

        # Initialize default content in session_state if not already present
        if "confluence_doc_content" not in st.session_state:
            st.session_state["confluence_doc_content"] = SAMPLE_TEXTS["Confluence: Spring Batch Runbook (SOP-402)"]

        # Live Confluence Fetcher Expander
        with st.expander("🌐 Fetch Live Confluence Page for Chunking Test", expanded=False):
            st.caption("Fetch real-time page content directly from Confluence to evaluate chunking strategies without storing to the database.")
            col_conf_url, col_conf_btn = st.columns([3, 1])
            with col_conf_url:
                conf_test_target = st.text_input(
                    "Confluence Page ID or URL",
                    placeholder="e.g. 557057 or https://your-domain.atlassian.net/wiki/spaces/DEV/pages/557057/...",
                    key="conf_playground_target_input",
                )
            with col_conf_btn:
                st.write("")
                st.write("")
                btn_fetch_conf = st.button("📥 Fetch Page Content", key="btn_fetch_conf_playground")

            if btn_fetch_conf:
                if not conf_test_target.strip():
                    st.error("Please provide a Confluence Page ID or URL.")
                else:
                    domain = os.getenv("CONFLUENCE_DOMAIN", "")
                    username = os.getenv("CONFLUENCE_USERNAME", "")
                    api_token = os.getenv("CONFLUENCE_API_TOKEN", "")

                    if not domain or not username or not api_token:
                        st.error(
                            "Missing Confluence credentials! Please ensure CONFLUENCE_DOMAIN, "
                            "CONFLUENCE_USERNAME, and CONFLUENCE_API_TOKEN are set in your environment / .env."
                        )
                    else:
                        with st.spinner(f"Fetching page {conf_test_target} from Confluence..."):
                            try:
                                from src.etl.confluence import APIConfluenceClient
                                client = APIConfluenceClient(domain=domain, username=username, api_token=api_token)
                                page_info = client.fetch_page_details(conf_test_target.strip())
                                st.session_state["confluence_doc_content"] = page_info["text"]
                                st.session_state["fetched_conf_title"] = page_info["title"]
                                st.success(f"✅ Successfully loaded Confluence Page: **'{page_info['title']}'** ({len(page_info['text'])} chars)")
                                st.rerun()
                            except Exception as conf_err:
                                st.error(f"Failed to fetch Confluence page: {conf_err}")

        # Controls & Presets Header
        col_mode, col_preset, col_preset_btn, col_clear_btn = st.columns([2, 3, 1, 1])
        with col_mode:
            view_mode = st.radio(
                "Inspection Mode",
                ["Single Strategy", "Side-by-Side Comparison"],
                horizontal=True,
                key="chunk_view_mode",
            )
        with col_preset:
            sample_choice = st.selectbox(
                "Sample Confluence Runbooks",
                options=[k for k in SAMPLE_TEXTS.keys() if SAMPLE_TEXTS[k]],
                index=0,
                key="chunk_sample_choice_sel",
            )
        with col_preset_btn:
            st.write("")
            st.write("")
            if st.button("📋 Load Sample", key="btn_load_sample_choice"):
                if SAMPLE_TEXTS.get(sample_choice):
                    st.session_state["confluence_doc_content"] = SAMPLE_TEXTS[sample_choice]
                    st.rerun()
        with col_clear_btn:
            st.write("")
            st.write("")
            if st.button("🗑️ Clear", key="btn_clear_playground_doc"):
                st.session_state["confluence_doc_content"] = ""
                st.rerun()

        # Document Text Area (Bound directly to session_state key)
        input_text = st.text_area(
            "Confluence Page Content to Chunk (Paste text, markdown, or Confluence XHTML)",
            height=220,
            placeholder="Paste your Confluence page text, SOP, runbook, or documentation here...",
            key="confluence_doc_content",
        )

        # Auto-parse HTML/XHTML if pasted
        cleaned_input = input_text
        if input_text and ("<p" in input_text.lower() or "<table" in input_text.lower() or "<h1" in input_text.lower() or "<div" in input_text.lower()):
            try:
                from src.etl.confluence import ConfluenceHTMLParser
                parser = ConfluenceHTMLParser(base_url="https://confluence.local")
                parser.feed(input_text)
                parsed_t, _, _, _ = parser.get_parsed_data()
                if parsed_t and len(parsed_t) > 10:
                    cleaned_input = parsed_t
                    st.caption("ℹ️ *Detected HTML/XHTML tags — cleaned into structured text for chunking.*")
            except Exception:
                cleaned_input = input_text

        st.divider()

        if view_mode == "Single Strategy":
            _render_single_strategy_view(cleaned_input)
        else:
            _render_side_by_side_view(cleaned_input)


    # =======================================================================
    # Subtab 2: Confluence Ingestion Pipeline
    # =======================================================================
    with etl_subtab_ingest:
        _render_confluence_ingestion_section()


def _render_single_strategy_view(input_text: str):
    """Render single chunking strategy controls and detailed diagnostics."""
    col_strat, col_params = st.columns([1, 2])

    with col_strat:
        strategy = st.selectbox(
            "🎯 Chunking Strategy",
            ["Semantic", "Recursive"],
            index=0,
            help="Select the chunking algorithm to evaluate.",
            key="single_strat_sel",
        )

    with col_params:
        if strategy == "Recursive":
            st.markdown("**Recursive Delimiter Parameters**")
            c1, c2 = st.columns(2)
            with c1:
                chunk_size = st.number_input(
                    "Chunk Size (Characters)",
                    min_value=50,
                    max_value=5000,
                    value=int(config.get("pipeline.chunk_size", 500)),
                    step=50,
                    key="single_rec_cs",
                )
            with c2:
                chunk_overlap = st.number_input(
                    "Chunk Overlap (Characters)",
                    min_value=0,
                    max_value=1000,
                    value=int(config.get("pipeline.chunk_overlap", 50)),
                    step=10,
                    key="single_rec_co",
                )
        else:
            st.markdown("**Semantic Chunking Parameters**")
            c1, c2, c3 = st.columns(3)
            with c1:
                thresh_type = st.selectbox(
                    "Threshold Type",
                    ["percentile", "standard_deviation", "interquartile", "gradient", "fixed"],
                    index=0,
                    key="single_sem_tt",
                    help=(
                        "Method used to calculate cosine distance cutoff:\n"
                        "• percentile: Splits at top N-th percentile distance (recommended).\n"
                        "• standard_deviation: Cutoff at mean + k * std_dev.\n"
                        "• interquartile: Cutoff at Q3 + k * IQR (resilient against outliers).\n"
                        "• gradient: Detects sharp anomaly slopes.\n"
                        "• fixed: Absolute cosine distance cutoff (e.g. 0.35)."
                    ),
                )
            with c2:
                if thresh_type == "percentile":
                    thresh_amt = st.slider(
                        "Percentile",
                        min_value=50.0,
                        max_value=99.0,
                        value=85.0,
                        step=1.0,
                        key="single_sem_p",
                        help="Higher percentile (e.g. 90-95%) = only largest topic shifts trigger splits (fewer, larger chunks). Lower = more granular chunks.",
                    )
                elif thresh_type == "standard_deviation":
                    thresh_amt = st.slider(
                        "Std Dev Multiplier (k)",
                        min_value=0.1,
                        max_value=3.0,
                        value=1.0,
                        step=0.1,
                        key="single_sem_sd",
                        help="Multiplier k for cutoff = mean(dist) + k * std(dist). Higher k creates fewer splits.",
                    )
                elif thresh_type == "interquartile":
                    thresh_amt = st.slider(
                        "IQR Multiplier (k)",
                        min_value=0.1,
                        max_value=3.0,
                        value=1.5,
                        step=0.1,
                        key="single_sem_iqr",
                        help="Multiplier k for cutoff = Q3 + k * (Q3 - Q1).",
                    )
                elif thresh_type == "gradient":
                    thresh_amt = st.slider(
                        "Gradient Percentile",
                        min_value=50.0,
                        max_value=99.0,
                        value=85.0,
                        step=1.0,
                        key="single_sem_g",
                        help="Percentile of slope rates-of-change between adjacent sentences.",
                    )
                else:
                    thresh_amt = st.slider(
                        "Fixed Distance Cutoff",
                        min_value=0.05,
                        max_value=1.5,
                        value=0.35,
                        step=0.05,
                        key="single_sem_f",
                        help="Absolute cosine distance cutoff. 0.35 distance corresponds to ~0.65 cosine similarity.",
                    )
            with c3:
                buffer_sz = st.number_input(
                    "Context Buffer (Sentences)",
                    min_value=0,
                    max_value=4,
                    value=1,
                    key="single_sem_buf",
                    help="Combines adjacent +/- k sentences before and after each sentence to smooth out local noise before computing vector embeddings.",
                )

            c4, c5 = st.columns(2)
            with c4:
                min_sz = st.number_input(
                    "Min Chunk Size (chars)",
                    min_value=0,
                    max_value=500,
                    value=50,
                    step=25,
                    key="single_sem_min",
                    help="Chunks smaller than this character limit are merged with neighboring chunks to prevent micro-fragments.",
                )
            with c5:
                max_sz = st.number_input(
                    "Max Chunk Size (chars)",
                    min_value=200,
                    max_value=5000,
                    value=1500,
                    step=100,
                    key="single_sem_max",
                    help="Hard ceiling in characters. Any chunk exceeding this limit is split using recursive delimiter fallback.",
                )

            with st.expander("ℹ️ Semantic Chunking Parameter Guide & Best Practices", expanded=False):
                st.markdown(
                    """
                    **How Semantic Chunking Works:**
                    1. **Sentence Splitting:** Breaks your Confluence document into individual sentences while preserving paragraph and header blocks.
                    2. **Context Buffering (`buffer_size`):** Combines neighboring sentences ($S_{i-k} \dots S_{i+k}$) to form smooth, contextual sentence windows.
                    3. **Vector Distance:** Computes dense embeddings using Vertex AI and calculates cosine distance ($1 - \text{similarity}$) between adjacent sentence embeddings.
                    4. **Breakpoint Thresholding:** Identifies topic shifts where distance exceeds the threshold.
                    5. **Boundary Merging & Capping:** Merges chunks under `min_chunk_size` and splits chunks over `max_chunk_size`.

                    ---
                    **Recommended Settings for Confluence Pages:**
                    - **Runbooks & Incident Reports:** `Threshold: percentile (85-90%)`, `Buffer: 1`, `Min Size: 50`, `Max Size: 1500`
                    - **Long Architectural Specs:** `Threshold: standard_deviation (k=1.0-1.2)`, `Buffer: 1`, `Min Size: 80`, `Max Size: 2000`
                    - **Granular FAQ / Q&A Tables:** `Threshold: percentile (75-80%)`, `Buffer: 0`, `Min Size: 30`, `Max Size: 800`
                    """
                )

    if st.button("✂️ Run & Visualize Chunking", type="primary", key="btn_run_single_chunk"):
        if not input_text.strip():
            st.error("Please enter or load some text to chunk.")
            return

        with st.spinner("Executing chunking algorithm..."):
            if strategy == "Recursive":
                chunker = RecursiveTextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                details = chunker.split_text_with_details(input_text)
            else:
                emb_service = _get_ui_embedding_service()
                chunker = SemanticChunker(
                    embedding_service=emb_service,
                    breakpoint_threshold_type=thresh_type,
                    breakpoint_threshold_amount=thresh_amt,
                    buffer_size=buffer_sz,
                    min_chunk_size=min_sz,
                    max_chunk_size=max_sz,
                )
                details = chunker.split_text_with_details(input_text)

        _display_chunk_results(details, strategy)


def _render_side_by_side_view(input_text: str):
    """Render side-by-side comparison between Recursive and Semantic Chunking."""
    st.markdown("### ⚖️ Side-by-Side Strategy Comparison")
    
    col_rec_cfg, col_sem_cfg = st.columns(2)

    with col_rec_cfg:
        st.markdown("#### 📐 Recursive Config")
        cr1, cr2 = st.columns(2)
        with cr1:
            rec_cs = st.number_input("Chunk Size", min_value=50, max_value=3000, value=400, step=50, key="cmp_rec_cs")
        with cr2:
            rec_co = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=50, step=10, key="cmp_rec_co")

    with col_sem_cfg:
        st.markdown("#### 🧠 Semantic Config")
        cs1, cs2 = st.columns(2)
        with cs1:
            sem_type = st.selectbox("Threshold Type", ["percentile", "standard_deviation", "fixed"], index=0, key="cmp_sem_type")
        with cs2:
            sem_amt = st.slider("Threshold Amount", min_value=0.1 if sem_type == "fixed" else 50.0, max_value=1.5 if sem_type == "fixed" else 99.0, value=0.35 if sem_type == "fixed" else 85.0, key="cmp_sem_amt")

    if st.button("⚖️ Compare Strategies Side-by-Side", type="primary", key="btn_cmp_chunks"):
        if not input_text.strip():
            st.error("Please enter or load text to compare.")
            return

        with st.spinner("Computing Recursive and Semantic chunks..."):
            rec_chunker = RecursiveTextChunker(chunk_size=rec_cs, chunk_overlap=rec_co)
            rec_details = rec_chunker.split_text_with_details(input_text)

            emb_service = _get_ui_embedding_service()
            sem_chunker = SemanticChunker(
                embedding_service=emb_service,
                breakpoint_threshold_type=sem_type,
                breakpoint_threshold_amount=sem_amt,
                buffer_size=1,
            )
            sem_details = sem_chunker.split_text_with_details(input_text)

        rec_chunks = rec_details.get("chunks", [])
        sem_chunks = sem_details.get("chunks", [])

        # Summary comparison KPIs
        st.divider()
        kpi_rec, kpi_sem = st.columns(2)

        with kpi_rec:
            st.markdown(f"### 📐 Recursive Results ({len(rec_chunks)} chunks)")
            avg_len_r = int(np.mean([len(c) for c in rec_chunks])) if rec_chunks else 0
            max_len_r = max([len(c) for c in rec_chunks]) if rec_chunks else 0
            min_len_r = min([len(c) for c in rec_chunks]) if rec_chunks else 0
            st.caption(f"Avg Chars: **{avg_len_r}** | Min: **{min_len_r}** | Max: **{max_len_r}**")

            for idx, c in enumerate(rec_chunks, 1):
                with st.expander(f"📦 Chunk #{idx} ({len(c)} chars, ~{len(c.split())} words)", expanded=True):
                    st.text_area(f"rec_chunk_{idx}", c, height=120, disabled=True, label_visibility="collapsed")

        with kpi_sem:
            st.markdown(f"### 🧠 Semantic Results ({len(sem_chunks)} chunks)")
            avg_len_s = int(np.mean([len(c) for c in sem_chunks])) if sem_chunks else 0
            max_len_s = max([len(c) for c in sem_chunks]) if sem_chunks else 0
            min_len_s = min([len(c) for c in sem_chunks]) if sem_chunks else 0
            st.caption(f"Avg Chars: **{avg_len_s}** | Min: **{min_len_s}** | Max: **{max_len_s}**")

            for idx, c in enumerate(sem_chunks, 1):
                with st.expander(f"🧠 Chunk #{idx} ({len(c)} chars, ~{len(c.split())} words)", expanded=True):
                    st.text_area(f"sem_chunk_{idx}", c, height=120, disabled=True, label_visibility="collapsed")


def _display_chunk_results(details: dict, strategy: str):
    """Display chunk cards, diagnostic charts, and length breakdown."""
    chunks = details.get("chunks", [])
    total_chunks = len(chunks)

    st.divider()
    st.subheader(f"📊 Chunking Output ({total_chunks} Chunks Generated)")

    if total_chunks == 0:
        st.warning("No chunks produced.")
        return

    # Aggregate Metrics
    lengths = [len(c) for c in chunks]
    words = [len(c.split()) for c in chunks]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Chunks", total_chunks)
    m2.metric("Avg Chunk Length", f"{int(np.mean(lengths))} chars")
    m3.metric("Min Length", f"{min(lengths)} chars")
    m4.metric("Max Length", f"{max(lengths)} chars")

    # If Semantic Chunking: render distance plot with threshold line
    distances = details.get("distances", [])
    threshold = details.get("threshold", 0.0)

    if strategy == "Semantic" and distances:
        st.markdown("#### 📈 Sentence-to-Sentence Cosine Distance Curve")
        st.caption(
            "Points above the dashed cutoff threshold trigger a new chunk boundary (semantic transition / topic shift)."
        )

        chart_data = []
        for i, d in enumerate(distances):
            chart_data.append({
                "Sentence Transition": f"S{i+1} → S{i+2}",
                "Cosine Distance": round(d, 4),
                "Breakpoint Threshold": round(threshold, 4),
                "Is Split Boundary": "✂️ Cut" if d > threshold else "—",
            })

        df_chart = pd.DataFrame(chart_data)
        st.line_chart(
            df_chart,
            x="Sentence Transition",
            y=["Cosine Distance", "Breakpoint Threshold"],
            color=["#FF4B4B", "#0083B8"],
        )

        with st.expander("🔍 Inspect Sentence Distance Values & Breakpoints", expanded=False):
            st.dataframe(df_chart, use_container_width=True, hide_index=True)

    # Render Chunks Cards
    st.markdown("#### 📦 Generated Text Chunks")
    for idx, chunk in enumerate(chunks, 1):
        char_count = len(chunk)
        word_count = len(chunk.split())
        with st.expander(f"Chunk #{idx} — {char_count} chars ({word_count} words)", expanded=True):
            st.text_area(f"chunk_content_{idx}", chunk, height=130, disabled=True, label_visibility="collapsed")


def _render_confluence_ingestion_section():
    """Render the Confluence ETL ingestion controls with chunking strategy selection."""
    col_input, col_params = st.columns([2, 1])

    with col_input:
        root_page = st.text_input(
            "Confluence Page ID or URL",
            placeholder="e.g. 123456 or https://your-domain.atlassian.net/wiki/spaces/...",
            key="etl_root_page_input",
        )

    with col_params:
        max_depth = st.slider("Crawl Depth Limit", min_value=1, max_value=5, value=1, key="etl_max_depth_slider")

    st.subheader("Chunking Strategy Configuration")
    col_strat_sel, col_strat_cfg = st.columns([1, 2])

    with col_strat_sel:
        ingest_strategy = st.selectbox(
            "Ingestion Strategy",
            ["Recursive", "Semantic"],
            index=0 if config.get("pipeline.chunking_strategy", "recursive") == "recursive" else 1,
            key="etl_ingest_strategy_sel",
        )

    with col_strat_cfg:
        if ingest_strategy == "Recursive":
            c_cs, c_co = st.columns(2)
            with c_cs:
                chunk_size = st.number_input(
                    "Chunk Size",
                    min_value=50,
                    max_value=5000,
                    value=int(config.get("pipeline.chunk_size", 500)),
                    key="etl_rec_cs",
                )
            with c_co:
                chunk_overlap = st.number_input(
                    "Chunk Overlap",
                    min_value=0,
                    max_value=500,
                    value=int(config.get("pipeline.chunk_overlap", 50)),
                    key="etl_rec_co",
                )
        else:
            c_tt, c_ta = st.columns(2)
            with c_tt:
                sem_thresh_type = st.selectbox(
                    "Breakpoint Type",
                    ["percentile", "standard_deviation", "interquartile", "gradient", "fixed"],
                    index=0,
                    key="etl_sem_tt",
                )
            with c_ta:
                sem_thresh_amt = st.number_input(
                    "Threshold Value",
                    min_value=0.1,
                    max_value=100.0,
                    value=90.0 if sem_thresh_type == "percentile" else 1.0,
                    key="etl_sem_ta",
                )

    # Show current config info
    with st.expander("⚙️ Current System Configuration", expanded=False):
        st.json({
            "chunking_strategy": ingest_strategy.lower(),
            "embedding_model": config.get("models.embedding"),
            "gcp_project": config.get("gcp.project"),
            "gcp_location": config.get("gcp.location"),
            "database_type": config.get("database.type"),
            "database_name": config.get("database.name"),
        })

    st.divider()

    if st.button("🚀 Start Ingestion Run", type="primary", key="btn_start_etl"):
        if not root_page.strip():
            st.error("Please enter a valid Confluence Page ID or URL.")
            return

        etl_status = st.empty()
        etl_progress = st.progress(0, text="Initializing...")

        # Capture logs
        log_buffer = StringIO()
        log_handler = logging.StreamHandler(log_buffer)
        log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        root_logger = logging.getLogger()
        root_logger.addHandler(log_handler)
        prev_level = root_logger.level
        root_logger.setLevel(logging.INFO)

        try:
            from google import genai
            from src.etl.confluence import APIConfluenceClient
            from src.embeddings.vertex import VertexEmbeddingService
            from src.etl.pipeline import ConfluenceETLPipeline

            # Resolve credentials
            domain = os.getenv("CONFLUENCE_DOMAIN", "")
            username = os.getenv("CONFLUENCE_USERNAME", "")
            api_token = os.getenv("CONFLUENCE_API_TOKEN", "")

            if not domain or not username or not api_token:
                st.warning(
                    "⚠️ Confluence credentials (CONFLUENCE_DOMAIN, CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN) "
                    "are not set in your environment. Crawl may fail unless running in mocked test mode."
                )

            etl_status.info("📐 Step 1/3: Initializing AI & Embedding Services...")
            etl_progress.progress(15, text="Connecting to Vertex AI...")

            gcp_project = config.get("gcp.project")
            gcp_location = config.get("gcp.location")
            model_name = config.get("models.embedding", "text-embedding-005")
            ai_client = genai.Client(vertexai=True, project=gcp_project, location=gcp_location)
            embedding_service = VertexEmbeddingService(client=ai_client, model_name=model_name)

            # Initialize selected chunker
            if ingest_strategy == "Recursive":
                chunker = RecursiveTextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            else:
                chunker = SemanticChunker(
                    embedding_service=embedding_service,
                    breakpoint_threshold_type=sem_thresh_type,
                    breakpoint_threshold_amount=sem_thresh_amt,
                )

            client = APIConfluenceClient(domain=domain, username=username, api_token=api_token)
            pipeline = ConfluenceETLPipeline(
                confluence_client=client,
                chunker=chunker,
                embedding_service=embedding_service,
            )

            etl_status.info(f"🕷️ Step 2/3: Crawling Confluence and processing content with {ingest_strategy} chunker...")
            etl_progress.progress(40, text="Crawling and transforming pages...")

            inserted_chunks = pipeline.run(root_identifier=root_page.strip(), max_depth=max_depth)

            etl_progress.progress(100, text="Ingestion Complete!")
            etl_status.success(
                f"✅ ETL finished successfully! Ingested **{inserted_chunks}** chunks into PostgreSQL vector store using **{ingest_strategy} Chunking**."
            )

        except Exception as e:
            etl_status.error(f"❌ ETL pipeline execution failed: {e}")
        finally:
            root_logger.removeHandler(log_handler)
            root_logger.setLevel(prev_level)
            log_output = log_buffer.getvalue()
            if log_output:
                with st.expander("📋 Execution Logs", expanded=True):
                    st.code(log_output, language="text")


