"""Tab 0: Interactive User Guide, Operational Workflow & Decision Matrices."""

import streamlit as st
import pandas as pd


def render_guide_tab():
    """Render the RAG Admin Dashboard User Guide and Workflow tab."""
    st.header("📖 RAG Pipeline Admin Guide & Operational Workflow")
    st.markdown(
        "Welcome to the **RAG Pipeline Admin Dashboard** — your mission control center for ingesting, "
        "inspecting, evaluating, and continuously optimizing your enterprise Retrieval-Augmented Generation (RAG) system."
    )

    # ---------------------------------------------------------------------------
    # 1. Executive Summary & Value Proposition
    # ---------------------------------------------------------------------------
    col_v1, col_v2, col_v3, col_v4 = st.columns(4)
    with col_v1:
        st.info("⚙️ **1. Ingest & Chunk**\nClean Confluence XHTML and test chunking strategies.")
    with col_v2:
        st.info("🗄️ **2. Audit Knowledge**\nInspect 2D vector topology, outliers, and duplicates.")
    with col_v3:
        st.info("💬 **3. Test & Diagnose**\nQuery SOPs, upload scan reports, and find knowledge gaps.")
    with col_v4:
        st.info("📊 **4. Evaluate Quality**\nBenchmark with RAGAS metrics & track score regressions.")

    st.divider()

    # ---------------------------------------------------------------------------
    # 2. Interactive Navigation Subtabs
    # ---------------------------------------------------------------------------
    guide_tab_workflow, guide_tab_steps, guide_tab_matrices, guide_tab_quickstart = st.tabs([
        "🔄 5-Step Operational Workflow",
        "🔍 Detailed Step-by-Step Rationale",
        "⚖️ Strategy & Decision Cheatsheet",
        "🚀 New User Quickstart Checklist",
    ])

    # =======================================================================
    # Subtab 1: Operational Workflow Map
    # =======================================================================
    with guide_tab_workflow:
        st.subheader("The Continuous RAG Optimization Loop")
        st.markdown(
            "An enterprise RAG system is not a static setup — it is a **continuous improvement loop**. "
            "The Admin Dashboard provides the complete tooling necessary to navigate every stage of this lifecycle:"
        )

        workflow_mermaid = """
        graph LR
            subgraph "1. Ingestion Layer"
                A["Confluence Docs / SOPs"] --> B["✂️ ETL & Chunking<br/>(Recursive / Semantic)"]
                B --> C["Dense Embeddings<br/>(768D Vertex AI)"]
                C --> D[("PostgreSQL<br/>pgvector")]
            end

            subgraph "2. Quality & Inspection Layer"
                D --> E["🗄️ Database Explorer<br/>(2D PCA / t-SNE)"]
                E --> F["🤖 AI Topology Audit<br/>(Outliers & Duplicates)"]
            end

            subgraph "3. Execution & Triage Layer"
                D --> G["💬 Playground<br/>(Hybrid Search + Reranker)"]
                G --> H["Incident & Audit Reports<br/>(PDF / Logs / JSON)"]
                H --> I["⚠️ Documentation Gaps<br/>Detection"]
            end

            subgraph "4. Feedback & Benchmarking Layer"
                G --> J["⭐ Rating & Root-Cause Tags<br/>(Recall/Precision/Hallucination)"]
                J --> K["✨ Ideal Ground Truth<br/>Correction"]
                K --> L["📁 Benchmark Datasets<br/>(JSON)"]
                L --> M["📊 RAGAS Evaluation<br/>(Faithfulness, Recall, Precision)"]
                M -.->|Tune Chunk Size / Reranker| B
            end

            classDef primary fill:#2563eb,stroke:#1d4ed8,stroke-width:2px,color:#fff;
            classDef success fill:#16a34a,stroke:#15803d,stroke-width:2px,color:#fff;
            classDef warning fill:#d97706,stroke:#b45309,stroke-width:2px,color:#fff;
            classDef db fill:#475569,stroke:#334155,stroke-width:2px,color:#fff;

            class B,G primary;
            class E,F,M success;
            class J,K,I warning;
            class D db;
        """
        st.markdown(f"```mermaid\n{workflow_mermaid}\n```")

        st.caption("💡 *Diagram: Data flows from left to right, feeding back into pipeline tuning and benchmark expansion.*")

        st.markdown("---")
        st.markdown("### Why Do We Need This Loop?")
        col_why1, col_why2 = st.columns(2)
        with col_why1:
            st.markdown(
                """
                **The Problem with Static RAG:**
                - ❌ **Garbage In, Garbage Out**: Poorly split documents break sentence context and degrade search recall.
                - ❌ **Invisible Blind Spots**: Without vector space visualization, missing documentation topics remain unnoticed until an outage occurs.
                - ❌ **Silent Hallucinations**: In production, LLMs may generate plausible-sounding but ungrounded answers.
                """
            )
        with col_why2:
            st.markdown(
                """
                **How the Admin Dashboard Solves It:**
                - ✅ **Visual Chunking Strategy Tester**: Test and compare delimiters and semantic breakpoints before ingesting.
                - ✅ **Proactive Vector Audits**: Detect isolated topic outliers and redundant duplicates with AI diagnostics.
                - ✅ **Quantitative RAGAS Benchmarks**: Validate pipeline tuning with statistical metrics (Faithfulness, Precision, Recall).
                """
            )

    # =======================================================================
    # Subtab 2: Detailed Step-by-Step Rationale
    # =======================================================================
    with guide_tab_steps:
        st.subheader("Understanding Each Step: Why It's Needed & How the UI Helps")

        # Step 1
        with st.expander("⚙️ STEP 1: ETL Pipeline & Chunking Strategy Playground", expanded=True):
            st.markdown("#### 🎯 Purpose: Transform Unstructured Confluence Pages into High-Quality Chunks")
            col_s1_why, col_s1_how = st.columns(2)
            with col_s1_why:
                st.markdown("**Why this step is critical:**")
                st.markdown(
                    """
                    Confluence pages contain nested headings, bullet lists, XML macros, and varying document lengths.
                    If you slice text naively (e.g. fixed character cuts):
                    - Sentences and code blocks get cut in half.
                    - Headings get separated from their relevant paragraphs.
                    - The vector embedding represents fragmented noise, causing **Recall Failure** during search.
                    """
                )
            with col_s1_how:
                st.markdown("**How this tab helps you:**")
                st.markdown(
                    """
                    1. **Interactive Chunking Playground**:
                       - Paste any SOP text or fetch live Confluence pages.
                       - Test **Recursive Chunking** (hierarchical `\\n\\n` $\\to$ `\\n` $\\to$ ` ` delimiters).
                       - Test **Semantic Chunking** (detects topic shifts using embedding cosine distance).
                    2. **Side-by-Side Comparison**: Visually inspect chunk counts, lengths, and boundaries.
                    3. **Live Ingestion Pipeline**: Ingest Confluence spaces directly into PostgreSQL `pgvector`.
                    """
                )

        # Step 2
        with st.expander("🗄️ STEP 2: Database Explorer & Vector Space Topology", expanded=False):
            st.markdown("#### 🎯 Purpose: Inspect Ingested Documents, Vector Geometry & Knowledge Quality")
            col_s2_why, col_s2_how = st.columns(2)
            with col_s2_why:
                st.markdown("**Why this step is critical:**")
                st.markdown(
                    """
                    Vector databases are typically opaque "black boxes". You cannot easily tell if:
                    - Ingested documents are concentrated in one topic while leaving other areas empty (**Coverage Blind Spots**).
                    - Duplicate or outdated runbooks exist with >95% similarity, polluting search results.
                    - Individual chunks are undersized (<50 chars) or oversized (>2000 chars).
                    """
                )
            with col_s2_how:
                st.markdown("**How this tab helps you:**")
                st.markdown(
                    """
                    1. **Browse & Full-Text Search**: Filter documents by type, search content, and view raw metadata/embeddings.
                    2. **2D Embedding Space Projections**:
                       - **PCA**: Visualizes maximum global variance across all 768 dimensions.
                       - **t-SNE**: Projects local semantic neighborhoods and tight topic clusters.
                    3. **🤖 AI Knowledge Base Quality Audit**: 1-click automated diagnosis identifying outliers (>2σ distance from corpus centroid) and near-duplicate pairs.
                    """
                )

        # Step 3
        with st.expander("💬 STEP 3: Interactive Playground & Report Analysis", expanded=False):
            st.markdown("#### 🎯 Purpose: Real-Time Testing with Hybrid Search, Reranking & Report Attachments")
            col_s3_why, col_s3_how = st.columns(2)
            with col_s3_why:
                st.markdown("**Why this step is critical:**")
                st.markdown(
                    """
                    Engineers during production incidents or audits don't just ask short questions — they have **scan reports (PDFs, vulnerability logs, JSON dumps)**.
                    Pure dense vector search often struggles with exact error codes or function names, while pure keyword search misses conceptual matches.
                    """
                )
            with col_s3_how:
                st.markdown("**How this tab helps you:**")
                st.markdown(
                    """
                    1. **Document Attachments**: Upload PDFs, TXT, JSON, or images directly with your query.
                    2. **Hybrid Search (Vector + FTS)**: Combines pgvector semantic search and PostgreSQL Full-Text Search via Reciprocal Rank Fusion (RRF).
                    3. **Semantic Cross-Encoder Reranker**: Reranks top candidates using Vertex AI Semantic Ranker to filter out false positives.
                    4. **Automated Documentation Gaps**: Automatically extracts missing SOP instructions identified by the LLM into a dedicated knowledge backlog.
                    """
                )

        # Step 4
        with st.expander("⭐ STEP 4: Feedback Curation & Continuous Dataset Building", expanded=False):
            st.markdown("#### 🎯 Purpose: Turn Production Failures into Permanent Benchmark Cases")
            col_s4_why, col_s4_how = st.columns(2)
            with col_s4_why:
                st.markdown("**Why this step is critical:**")
                st.markdown(
                    """
                    Without feedback collection, errors repeat indefinitely.
                    When a model gives an incomplete or hallucinated answer, subject matter experts (SMEs) need a fast way to provide the **ground-truth reference** so the system can be continuously evaluated against it.
                    """
                )
            with col_s4_how:
                st.markdown("**How this tab helps you:**")
                st.markdown(
                    """
                    1. **Rate & Tag Failures**: Flag responses as 👍 Good (5★) or 👎 Bad (1★) and categorize root causes (*Missing Context*, *Poor Ranking*, *Hallucination*).
                    2. **Ground-Truth Correction**: Write the ideal verified answer right in the UI.
                    3. **1-Click Promote to Benchmark**: Export curated feedback directly into your benchmark test suites (`default.json` or custom datasets) for automated regression testing.
                    """
                )

        # Step 5
        with st.expander("📊 STEP 5: RAGAS Benchmark Evaluation & Metric Dashboards", expanded=False):
            st.markdown("#### 🎯 Purpose: Scientifically Validate Pipeline Improvements with Statistical Metrics")
            col_s5_why, col_s5_how = st.columns(2)
            with col_s5_why:
                st.markdown("**Why this step is critical:**")
                st.markdown(
                    """
                    Changing a pipeline parameter (e.g. switching chunk size from 500 to 1000, or enabling a reranker) might fix one query while breaking three others.
                    You need objective, automated statistical metrics before deploying any configuration change.
                    """
                )
            with col_s5_how:
                st.markdown("**How this tab helps you:**")
                st.markdown(
                    r"""
                    1. **Automated RAGAS Evaluation**: Compute Faithfulness, Answer Relevancy, Context Precision, and Context Recall across standard test suites.
                    2. **Run History & Persistence**: Save evaluation runs with all pipeline hyperparameters to the database.
                    3. **Side-by-Side Run Comparison & Quality Radar**: Compare Run A vs Run B with automated delta ($\Delta$) indicators and Plotly spider charts.
                    4. **1-Click Production Promotion**: Promote winning evaluation hyperparameters directly to the active system defaults with a single click.
                    """
                )

        # Step 6
        with st.expander("⚙️ STEP 6: System Configuration & Live Production Policy", expanded=False):
            st.markdown("#### 🎯 Purpose: Synchronize Winning Parameters to the Chat App with Zero Downtime")
            col_s6_why, col_s6_how = st.columns(2)
            with col_s6_why:
                st.markdown("**Why this step is critical:**")
                st.markdown(
                    """
                    Once you find a high-performing configuration (e.g. *Strict Grounding prompt + Semantic Reranker + Top K=3*), modifying `.env` files manually or redeploying cloud containers is slow and error-prone.
                    You need centralized, dynamic overrides stored in PostgreSQL that sync live across all user-facing chat apps and API servers.
                    """
                )
            with col_s6_how:
                st.markdown("**How this tab helps you:**")
                st.markdown(
                    """
                    1. **Live RAG Policy Management**: Modify default retrieval modes, candidate pool sizes, and prompt presets without container restarts.
                    2. **Immediate Chat App Sync**: All new queries in `chat_app.py` and FastAPI endpoints inherit the updated policy instantly.
                    3. **1-Click Promotion**: Apply the exact settings of top-scoring benchmark runs directly from the Evaluation History tab.
                    """
                )

    # =======================================================================
    # Subtab 3: Strategy & Decision Cheatsheet
    # =======================================================================
    with guide_tab_matrices:
        st.subheader("Engineering Decision Matrices & Best Practices")

        st.markdown("### 1. Chunking Strategy Decision Matrix")
        chunking_data = [
            {
                "Strategy": "Recursive Chunking",
                "How it Splits": "Hierarchical delimiters (\\n\\n -> \\n -> space)",
                "Best For": "Technical SOPs, code runbooks, API specs, numbered step procedures",
                "Pros": "Fast, deterministic, preserves document structure & formatting",
                "Recommended Settings": "Chunk Size: 400–600 chars | Overlap: 50 chars",
            },
            {
                "Strategy": "Semantic Chunking",
                "How it Splits": "Embedding cosine distance transitions between adjacent sentences",
                "Best For": "Long prose, incident post-mortems, architectural overviews, policy docs",
                "Pros": "Self-adapting boundaries, groups cohesive thoughts regardless of length",
                "Recommended Settings": "Percentile: 85–90% | Buffer: 1 sentence | Min: 50 | Max: 1500",
            },
        ]
        st.dataframe(pd.DataFrame(chunking_data), use_container_width=True, hide_index=True)

        st.divider()

        st.markdown("### 2. Search & Retrieval Configuration Matrix")
        search_data = [
            {
                "Mode": "Pure Dense Vector",
                "Mechanism": "Cosine similarity over 768D pgvector embeddings",
                "When to Use": "Conceptual / conversational questions with varied wording",
                "Limitation": "Can miss specific error codes, table names, or UUIDs",
            },
            {
                "Mode": "Hybrid Search (Vector + FTS)",
                "Mechanism": "pgvector cosine distance + PostgreSQL tsvector merged via RRF",
                "When to Use": "Standard enterprise documentation (Recommended Default)",
                "Limitation": "Slightly higher retrieval query latency (+5-15ms)",
            },
            {
                "Mode": "Hybrid + Cross-Encoder Reranker",
                "Mechanism": "Stage 1 candidate pool (e.g. 5-10) re-scored by Vertex AI Semantic Ranker",
                "When to Use": "High-precision incident remediation & regulatory compliance queries",
                "Limitation": "Additional API call latency (+50-100ms)",
            },
        ]
        st.dataframe(pd.DataFrame(search_data), use_container_width=True, hide_index=True)

        st.divider()

        st.markdown("### 3. Generation Prompt Strategy Matrix (Faithfulness vs. Comprehensiveness)")
        prompt_matrix_data = [
            {
                "Prompt Preset": "Structured Multi-Section (Default)",
                "Grounding Behavior": "Balanced: verified runbook steps + clearly marked general knowledge",
                "Best Used For": "General engineering support, incident triage, user queries",
                "Impact on RAGAS": "High Relevancy (~0.85), Good Faithfulness (~0.88), logs doc gaps",
            },
            {
                "Prompt Preset": "Strict Grounding (High Faithfulness)",
                "Grounding Behavior": "Ultra-strict: rejects any unstated claim or extrapolation",
                "Best Used For": "Compliance, security audits, automated validation pipelines",
                "Impact on RAGAS": "Maximizes Faithfulness ($\\ge 0.95$), prevents hallucinations",
            },
            {
                "Prompt Preset": "Concise Technical Q&A",
                "Grounding Behavior": "Concise bullet points, zero conversational boilerplate",
                "Best Used For": "CLI assistants, Slack bots, fast developer Q&A",
                "Impact on RAGAS": "Maximizes Answer Relevancy ($\\ge 0.90$), minimizes latency",
            },
            {
                "Prompt Preset": "Detailed SOP Remediation",
                "Grounding Behavior": "Operational structure: Root cause $\\to$ Procedure $\\to$ Rollback",
                "Best Used For": "Production outage runbooks, SRE incident response",
                "Impact on RAGAS": "High Context Utilization & structured procedure validation",
            },
        ]
        st.dataframe(pd.DataFrame(prompt_matrix_data), use_container_width=True, hide_index=True)

        st.divider()

        st.markdown("### 4. RAGAS Quality Metrics & Target Thresholds")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(
                """
                | Metric | What It Measures | Target | Quality Formula |
                | :--- | :--- | :---: | :--- |
                | **Faithfulness** | Are claims in the response grounded in the retrieved context? | **$\\ge 0.85$** | $\\frac{\\text{Grounded Claims}}{\\text{Total Claims}}$ |
                | **Answer Relevancy** | Does the answer directly address the user prompt without filler? | **$\\ge 0.80$** | Cosine sim of generated questions |
                """
            )
        with col_m2:
            st.markdown(
                """
                | Metric | What It Measures | Target | Quality Formula |
                | :--- | :--- | :---: | :--- |
                | **Context Precision** | Are the most relevant chunks at top rank positions? | **$\\ge 0.80$** | Mean Average Precision of contexts |
                | **Context Recall** | Did the retriever find all info needed to answer the reference? | **$\\ge 0.80$** | $\\frac{\\text{Attributed Ref Sentences}}{\\text{Total Ref Sentences}}$ |
                """
            )

        st.caption("🟢 **$\\ge 0.80$**: Production Ready | 🟡 **0.60 – 0.80**: Requires Tuning | 🔴 **$< 0.60$**: Retrieval / Generation Failure")

    # =======================================================================
    # Subtab 4: Quickstart Checklist
    # =======================================================================
    with guide_tab_quickstart:
        st.subheader("🚀 New User Quickstart Checklist")
        st.markdown("Follow this 4-step sequence to get up and running with the Admin Dashboard:")

        col_q1, col_q2 = st.columns([1, 1])

        with col_q1:
            st.markdown("#### ✅ Phase 1: Ingestion & Verification")
            st.markdown(
                """
                1. **Check Existing Knowledge Base**:
                   - Go to **🗄️ Database Explorer** $\\to$ Select `documents` table.
                   - Verify total document chunk count.
                2. **Run an AI Quality Audit**:
                   - Click **'🚀 Run AI Quality Audit & Recommendations'** in Database Explorer.
                   - Check for near-duplicates or isolated outliers.
                3. **Ingest or Test New Content**:
                   - Go to **⚙️ ETL Pipeline** $\\to$ **✂️ Chunking Playground**.
                   - Load sample runbooks or paste Confluence content to preview chunk boundaries.
                """
            )

        with col_q2:
            st.markdown("#### ✅ Phase 2: Testing & Continuous Evaluation")
            st.markdown(
                """
                4. **Execute Real-Time Queries**:
                   - Go to **💬 Playground & Feedback** $\\to$ Ask questions or upload incident reports.
                   - Verify Hybrid Search and Reranker results.
                5. **Rate & Triage**:
                   - Give 👍/👎 rating, add issue tags, and write ideal ground-truth reference answers.
                   - Click **'⭐ Save & Promote to Benchmark'**.
                6. **Run RAGAS Benchmarks**:
                   - Go to **📊 Evaluation** $\\to$ Click **'▶️ Run Evaluation'**.
                   - Save the run and compare against previous configurations!
                """
            )

        st.success("🎉 You are ready to explore! Click any of the tabs above to begin.")
