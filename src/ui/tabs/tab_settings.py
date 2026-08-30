"""Tab 6: System Configuration & RAG Pipeline Production Defaults."""

import streamlit as st
import pandas as pd

from src.utils.config import config
from src.db.settings_store import (
    get_setting,
    get_all_settings,
    set_many_settings,
    delete_setting,
)
from src.pipeline.prompts import (
    PROMPT_PRESETS,
    DEFAULT_STANDARD_PROMPT,
    DEFAULT_ATTACHED_REPORT_PROMPT,
    list_prompt_presets,
    get_prompt_preset,
)


def render_settings_tab():
    """Render the System Configuration and Production Defaults tab."""
    st.header("⚙️ System Configuration & Production Defaults")
    st.markdown(
        "Manage global defaults for the RAG pipeline. Configuration saved here is stored in **PostgreSQL** "
        "and overrides `config.yaml` with zero-downtime, applying immediately to the **💬 Chat Application** "
        "and **FastAPI backend** without requiring container redeployments or service restarts."
    )

    # 3-Tier Hierarchy Explanation Alert
    with st.expander("ℹ️ Configuration Resolution Hierarchy & Architecture", expanded=False):
        st.markdown(
            """
            ```
            ┌────────────────────────────────────────────────────────┐
            │  Tier 1: PostgreSQL `system_settings` (Admin UI)       │ ◄── Dynamic overrides (applied live)
            ├────────────────────────────────────────────────────────┤
            │  Tier 2: Environment Variables (`.env`)                 │ ◄── Secrets (API keys, DB credentials)
            ├────────────────────────────────────────────────────────┤
            │  Tier 3: `config.yaml`                                 │ ◄── Static codebase baseline
            └────────────────────────────────────────────────────────┘
            ```
            - **Live Synchronization**: When you click **Save Configuration**, new queries in `chat_app.py` and the API immediately adopt the new prompt template, model, and reranking parameters.
            - **Multi-Instance Support**: Shared across all background workers and cloud container instances.
            """
        )

    st.divider()

    # Load current dynamic values (fallback to config.yaml)
    current_hybrid = config.get_dynamic("pipeline.use_hybrid", config.get("pipeline.use_hybrid", True))
    current_reranker = config.get_dynamic("pipeline.use_reranker", config.get("pipeline.use_reranker", True))
    current_pool = config.get_dynamic("pipeline.pool_size", config.get("pipeline.pool_size", 5))
    current_top_k = config.get_dynamic("pipeline.final_top_k", config.get("pipeline.final_top_k", 2))

    current_gen_model = config.get_dynamic("models.generation", config.get("models.generation", "gemini-2.5-flash"))
    current_emb_model = config.get_dynamic("models.embedding", config.get("models.embedding", "text-embedding-005"))
    current_rerank_model = config.get_dynamic("models.rerank", config.get("models.rerank", "semantic-ranker-512@latest"))

    current_prompt_preset = config.get_dynamic("pipeline.prompt_preset", "Structured Multi-Section (Default)")
    current_prompt_tpl = config.get_dynamic("pipeline.prompt_template", DEFAULT_STANDARD_PROMPT)
    current_attached_tpl = config.get_dynamic("pipeline.attached_prompt_template", DEFAULT_ATTACHED_REPORT_PROMPT)

    current_chunk_strategy = config.get_dynamic("pipeline.chunking_strategy", config.get("pipeline.chunking_strategy", "recursive"))
    current_chunk_size = config.get_dynamic("pipeline.chunk_size", config.get("pipeline.chunk_size", 500))
    current_chunk_overlap = config.get_dynamic("pipeline.chunk_overlap", config.get("pipeline.chunk_overlap", 50))

    # Form
    with st.form("system_settings_form"):
        # -------------------------------------------------------------------
        # 1. Retrieval Strategy
        # -------------------------------------------------------------------
        st.subheader("1. 🔍 Default Retrieval Strategy")
        c_ret1, c_ret2 = st.columns(2)

        with c_ret1:
            cfg_hybrid = st.toggle("Enable Hybrid Search (pgvector + FTS via RRF)", value=bool(current_hybrid))
            cfg_reranker = st.toggle("Enable Semantic Cross-Encoder Reranker (Vertex AI)", value=bool(current_reranker))

        with c_ret2:
            cfg_pool = st.number_input(
                "Stage 1 Candidate Pool Size",
                min_value=1,
                max_value=50,
                value=int(current_pool),
                help="Number of initial candidate chunks retrieved by Vector/FTS search before reranking.",
            )
            cfg_top_k = st.number_input(
                "Final Top K Context Chunks for LLM",
                min_value=1,
                max_value=20,
                value=int(current_top_k),
                help="Final number of top-ranked context chunks passed into the generation prompt.",
            )

        st.divider()

        # -------------------------------------------------------------------
        # 2. Generation Models & Grounding Strategy
        # -------------------------------------------------------------------
        st.subheader("2. 🤖 Default Generation Model & Grounding Strategy")
        c_gen1, c_gen2 = st.columns(2)

        with c_gen1:
            model_options = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"]
            m_idx = model_options.index(current_gen_model) if current_gen_model in model_options else 0
            cfg_gen_model = st.selectbox("Default Generation Model", model_options, index=m_idx)

        with c_gen2:
            preset_options = list_prompt_presets() + ["Custom Template"]
            p_idx = preset_options.index(current_prompt_preset) if current_prompt_preset in preset_options else 0
            cfg_prompt_preset = st.selectbox(
                "Default Prompt Grounding Preset",
                preset_options,
                index=p_idx,
                help="Determines default grounding strictness, section headers, and hallucination guardrails.",
            )

        cfg_prompt_tpl = st.text_area(
            "Default Standard System Prompt (`{context}`, `{query}`)",
            value=current_prompt_tpl,
            height=160,
            help="Prompt used for standard user questions without attachments.",
        )

        with st.expander("📎 Edit Multimodal Report Attachment Prompt Template", expanded=False):
            cfg_attached_tpl = st.text_area(
                "Attached Document Prompt Template (`{context}`, `{query}`)",
                value=current_attached_tpl,
                height=160,
                help="Prompt used when users upload external scan reports or audit documents.",
            )

        st.divider()

        # -------------------------------------------------------------------
        # 3. Ingestion & Chunking Defaults
        # -------------------------------------------------------------------
        st.subheader("3. ⚙️ Ingestion & Chunking Defaults")
        c_chk1, c_chk2 = st.columns(2)

        with c_chk1:
            chunk_strategies = ["recursive", "semantic"]
            cs_idx = chunk_strategies.index(current_chunk_strategy) if current_chunk_strategy in chunk_strategies else 0
            cfg_chunk_strategy = st.selectbox("Default Ingestion Chunking Strategy", chunk_strategies, index=cs_idx)

        with c_chk2:
            c_sz1, c_sz2 = st.columns(2)
            with c_sz1:
                cfg_chunk_size = st.number_input("Target Chunk Size (chars)", min_value=100, max_value=4000, value=int(current_chunk_size))
            with c_sz2:
                cfg_chunk_overlap = st.number_input("Chunk Overlap (chars)", min_value=0, max_value=500, value=int(current_chunk_overlap))

        st.divider()

        # Submit button
        col_btn_save, col_btn_info = st.columns([1.5, 3])
        with col_btn_save:
            btn_save = st.form_submit_button("💾 Save Production Configuration", type="primary", use_container_width=True)

        if btn_save:
            # If user selected a preset other than Custom, update the template if it was left untouched or matches preset
            final_tpl = cfg_prompt_tpl
            if cfg_prompt_preset in PROMPT_PRESETS and not final_tpl.strip():
                final_tpl = PROMPT_PRESETS[cfg_prompt_preset]

            updates = {
                "pipeline.use_hybrid": cfg_hybrid,
                "pipeline.use_reranker": cfg_reranker,
                "pipeline.pool_size": int(cfg_pool),
                "pipeline.final_top_k": int(cfg_top_k),
                "models.generation": cfg_gen_model,
                "pipeline.prompt_preset": cfg_prompt_preset,
                "pipeline.prompt_template": final_tpl,
                "pipeline.attached_prompt_template": cfg_attached_tpl,
                "pipeline.chunking_strategy": cfg_chunk_strategy,
                "pipeline.chunk_size": int(cfg_chunk_size),
                "pipeline.chunk_overlap": int(cfg_chunk_overlap),
            }

            success = set_many_settings(updates, updated_by="Admin Dashboard")
            if success:
                st.success("✅ Production RAG Configuration saved! Applied immediately across Chat App and API.")
                st.rerun()
            else:
                st.error("Failed to persist configuration to PostgreSQL.")

    # -----------------------------------------------------------------------
    # 4. Active Database Overrides Table & Reset Action
    # -----------------------------------------------------------------------
    st.divider()
    st.subheader("📋 Active Database Overrides")

    all_active = get_all_settings()
    if not all_active:
        st.info("No database overrides active. The pipeline is currently using static defaults from `config.yaml`.")
    else:
        table_rows = []
        for k, v in all_active.items():
            val_display = str(v["value"])
            if len(val_display) > 60:
                val_display = val_display[:57] + "..."
            table_rows.append({
                "Setting Key": k,
                "Active Value": val_display,
                "Last Updated": v.get("updated_at"),
                "Updated By": v.get("updated_by"),
            })

        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        col_reset, _ = st.columns([1.5, 3])
        with col_reset:
            if st.button("🔄 Reset to `config.yaml` Baseline Defaults", type="secondary"):
                for k in all_active.keys():
                    delete_setting(k)
                st.success("✅ All database overrides cleared. System restored to config.yaml defaults.")
                st.rerun()

