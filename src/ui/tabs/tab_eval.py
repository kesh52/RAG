import re
import json
import logging
from io import StringIO
import streamlit as st
import pandas as pd

from src.utils.config import config
from src.ui.helpers import (
    _format_metric,
    _quality_icon,
    METRIC_THRESHOLDS,
    METRIC_INFO,
    create_radar_comparison_chart,
    create_metric_sparkline_chart,
)
from evaluation.evaluation_store import (
    save_evaluation_run,
    list_evaluation_runs,
    get_evaluation_run,
    delete_evaluation_run,
)


def _extract_prompt_preset_from_notes(notes: str | None) -> str:
    """Extract the prompt preset name from evaluation run notes."""
    if not notes:
        return "Default"
    match = re.search(r"\[Prompt Preset:\s*([^\]]+)\]", notes)
    if match:
        return match.group(1).strip()
    return "Default"


def _extract_prompt_template_from_notes(notes: str | None) -> str | None:
    """Extract the full prompt template text from evaluation run notes if present."""
    if not notes:
        return None
    match = re.search(r"\[Prompt Template\]:\s*\n([\s\S]+)", notes)
    if match:
        return match.group(1).strip()
    return None


def render_eval_tab():
    """Render the Evaluation tab."""
    st.header("RAG Evaluation")

    eval_tab_run, eval_tab_history = st.tabs(["▶️ Run Evaluation", "📜 History"])

    # =======================================================================
    # Subtab 1: Run Evaluation
    # =======================================================================
    with eval_tab_run:
        st.subheader("Run Configuration")

        col_toggles, col_params_eval = st.columns(2)

        with col_toggles:
            eval_use_hybrid = st.toggle("Hybrid Search", value=True)
            eval_use_reranker = st.toggle("Semantic Reranker", value=True)

        with col_params_eval:
            eval_pool_size = st.number_input("Pool Size (Stage 1 candidates)", min_value=1, max_value=50, value=5, key="eval_pool")
            eval_top_k = st.number_input("Final Top K", min_value=1, max_value=20, value=2, key="eval_topk")

        # Prompt configuration
        from src.pipeline.prompts import PROMPT_PRESETS, list_prompt_presets

        with st.expander("📝 Generation Prompt & Grounding Strategy", expanded=False):
            st.caption("Select a prompt template preset or customize the grounding instructions for RAGAS evaluation.")
            col_ep_preset, col_ep_reset = st.columns([3, 1])
            with col_ep_preset:
                eval_preset_options = list_prompt_presets() + ["Custom Template"]
                eval_selected_preset = st.selectbox(
                    "Prompt Preset",
                    options=eval_preset_options,
                    index=0,
                    key="eval_prompt_preset_sel",
                )
            with col_ep_reset:
                st.write("")
                st.write("")
                if st.button("🔄 Reset Prompt", key="btn_reset_eval_prompt"):
                    if eval_selected_preset in PROMPT_PRESETS:
                        st.session_state["eval_custom_prompt_text"] = PROMPT_PRESETS[eval_selected_preset]
                        st.rerun()

            if "eval_custom_prompt_text" not in st.session_state or st.session_state.get("last_eval_preset") != eval_selected_preset:
                st.session_state["last_eval_preset"] = eval_selected_preset
                if eval_selected_preset in PROMPT_PRESETS:
                    st.session_state["eval_custom_prompt_text"] = PROMPT_PRESETS[eval_selected_preset]

            eval_active_prompt = st.text_area(
                "Prompt Template (`{context}`, `{query}`)",
                value=st.session_state.get("eval_custom_prompt_text", PROMPT_PRESETS["Structured Multi-Section (Default)"]),
                height=160,
                key="eval_custom_prompt_text",
            )

        # Dataset selector
        st.subheader("Dataset")

        from evaluation.evaluate_ragas import list_available_datasets, load_dataset

        available_datasets = list_available_datasets()
        if not available_datasets:
            available_datasets = ["default"]

        selected_dataset = st.selectbox("Benchmark Dataset", available_datasets)

        with st.expander("Preview dataset cases"):
            try:
                cases = load_dataset(selected_dataset)
                cases_df = pd.DataFrame(cases)
                st.dataframe(cases_df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Failed to load dataset: {e}")
                cases = []

        st.subheader("Run Metadata")
        col_name, col_notes = st.columns(2)
        with col_name:
            prompt_slug = eval_selected_preset.split()[0].lower()
            run_name = st.text_input("Run Name", value=f"{'hybrid' if eval_use_hybrid else 'dense'}+{'reranker' if eval_use_reranker else 'topk'}+{prompt_slug}")
        with col_notes:
            run_notes = st.text_area("Notes", placeholder="Optional notes about this evaluation run...", height=80)

        st.divider()

        if st.button("▶️ Run Evaluation", type="primary"):
            if not cases:
                st.error("No benchmark cases loaded. Check your dataset selection.")
            else:
                eval_status = st.empty()
                eval_progress = st.progress(0, text="Initializing pipeline...")

                # Capture logs during evaluation
                eval_log_buffer = StringIO()
                eval_log_handler = logging.StreamHandler(eval_log_buffer)
                eval_log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
                root_logger = logging.getLogger()
                root_logger.addHandler(eval_log_handler)
                prev_level = root_logger.level
                root_logger.setLevel(logging.INFO)

                try:
                    from src.pipeline import get_default_pipeline
                    from evaluation.evaluate_ragas import run_pipeline_evaluation

                    eval_status.info("⏳ Initializing RAG pipeline...")
                    pipeline = get_default_pipeline()

                    def progress_cb(current, total, message):
                        if total > 0:
                            pct = int((current / total) * 80) + 10  # 10-90% for queries
                            eval_progress.progress(min(pct, 95), text=message)

                    eval_status.info("🏃 Evaluating benchmark cases against pipeline...")
                    eval_results = run_pipeline_evaluation(
                        pipeline=pipeline,
                        label=run_name,
                        benchmark_cases=cases,
                        use_hybrid=eval_use_hybrid,
                        use_reranker=eval_use_reranker,
                        pool_size=eval_pool_size,
                        final_top_k=eval_top_k,
                        progress_callback=progress_cb,
                        prompt_template=eval_active_prompt,
                    )

                    eval_progress.progress(100, text="Evaluation complete!")
                    eval_status.success("✅ Evaluation finished successfully!")

                    # Store results in session state
                    st.session_state["eval_results"] = eval_results
                    st.session_state["eval_run_name"] = run_name
                    st.session_state["eval_prompt_preset"] = eval_selected_preset
                    st.session_state["eval_prompt_template"] = eval_active_prompt
                    st.session_state["eval_run_notes"] = run_notes
                    st.session_state["eval_dataset_name"] = selected_dataset
                    st.session_state["eval_cases"] = cases
                    st.session_state["eval_saved"] = False

                except Exception as e:
                    eval_status.error(f"❌ Evaluation failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                finally:
                    root_logger.removeHandler(eval_log_handler)
                    root_logger.setLevel(prev_level)
                    eval_log_out = eval_log_buffer.getvalue()
                    if eval_log_out:
                        with st.expander("📋 Evaluation Logs", expanded=False):
                            st.code(eval_log_out, language="text")

        # Display results if available in session_state
        if "eval_results" in st.session_state:
            res = st.session_state["eval_results"]
            avg_scores = res.get("aggregated_scores", {})

            st.divider()
            st.subheader("📊 Aggregate Metrics")

            # KPI metric cards with quality thresholds
            c1, c2, c3, c4 = st.columns(4)
            metric_cols = [
                ("Faithfulness", "faithfulness", c1),
                ("Answer Relevancy", "answer_relevancy", c2),
                ("Context Precision", "context_precision", c3),
                ("Context Recall", "context_recall", c4),
            ]

            for label, key, col in metric_cols:
                val = avg_scores.get(key)
                with col:
                    st.metric(
                        label=label,
                        value=_format_metric(val, key),
                    )

            # Metric interpretation expander
            with st.expander("ℹ️ How to interpret these metrics & quality thresholds"):
                for mkey, minfo in METRIC_INFO.items():
                    st.markdown(f"### {mkey.replace('_', ' ').title()}")
                    st.markdown(minfo)
                    st.divider()

            # Detailed results per question
            st.subheader("Per-Question Results")
            details = res.get("detailed_results", [])
            if details:
                detail_df = pd.DataFrame(details)
                display_cols = ["user_input", "response", "reference"]
                score_cols = [k for k in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"] if k in detail_df.columns]

                # Format score columns in dataframe
                formatted_df = detail_df[display_cols + score_cols].copy()
                for sc in score_cols:
                    formatted_df[sc] = formatted_df[sc].apply(
                        lambda v, m=sc: _format_metric(v, m) if v is not None and not (isinstance(v, float) and pd.isna(v)) else "⚪ N/A"
                    )

                st.dataframe(formatted_df, use_container_width=True, hide_index=True)

            # Save / Export actions
            st.divider()
            col_save_db, col_export_json = st.columns(2)

            with col_save_db:
                if not st.session_state.get("eval_saved"):
                    if st.button("💾 Save Run to Evaluation History", type="primary"):
                        try:
                            preset_used = st.session_state.get("eval_prompt_preset", "Structured Multi-Section (Default)")
                            template_used = st.session_state.get("eval_prompt_template", "")
                            user_n = st.session_state.get("eval_run_notes", "") or ""
                            combined_notes = f"[Prompt Preset: {preset_used}]"
                            if user_n.strip():
                                combined_notes += f" {user_n.strip()}"
                            if template_used and template_used.strip():
                                combined_notes += f"\n\n[Prompt Template]:\n{template_used.strip()}"

                            run_id = save_evaluation_run(
                                run_name=st.session_state.get("eval_run_name", "unnamed_run"),
                                dataset_name=st.session_state.get("eval_dataset_name", "default"),
                                dataset_cases=st.session_state.get("eval_cases", []),
                                use_hybrid=eval_use_hybrid,
                                use_reranker=eval_use_reranker,
                                pool_size=eval_pool_size,
                                final_top_k=eval_top_k,
                                chunk_size=config.get("pipeline.chunk_size", 500),
                                chunk_overlap=config.get("pipeline.chunk_overlap", 50),
                                embedding_model=config.get("models.embedding", "text-embedding-004"),
                                rerank_model=config.get("models.reranking", "semantic-ranker-512"),
                                generation_model=config.get("models.generation", "gemini-2.5-flash"),
                                avg_faithfulness=avg_scores.get("faithfulness"),
                                avg_answer_relevancy=avg_scores.get("answer_relevancy"),
                                avg_context_precision=avg_scores.get("context_precision"),
                                avg_context_recall=avg_scores.get("context_recall"),
                                detailed_results=res.get("detailed_results", []),
                                notes=combined_notes,
                            )
                            st.session_state["eval_saved"] = True
                            st.success(f"✅ Run saved to database with ID #{run_id}!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to save run: {e}")
                else:
                    st.success("✅ This run has been saved to evaluation history.")

            with col_export_json:
                export_json_str = json.dumps(res, indent=2, default=str)
                st.download_button(
                    label="📥 Export Results as JSON",
                    data=export_json_str,
                    file_name=f"eval_results_{st.session_state.get('eval_run_name', 'run')}.json",
                    mime="application/json",
                )

    # =======================================================================
    # Subtab 2: History & Comparison
    # =======================================================================
    with eval_tab_history:
        st.subheader("Historical Evaluation Runs")

        try:
            from evaluation.evaluation_store import (
                list_evaluation_runs,
                get_evaluation_run,
                delete_evaluation_run,
            )

            runs = list_evaluation_runs()

            if not runs:
                st.info("No saved evaluation runs found. Run an evaluation and click 'Save Run to History'.")
            else:
                # -----------------------------------------------------------
                # 1. Continuous Performance Trajectory & Sparklines (Stage-Separated)
                # -----------------------------------------------------------
                st.markdown("### 📈 Continuous Quality Trajectories & Component Health")
                st.caption(
                    "Monitor whether pipeline and prompt changes are improving (🟢) or regressing (🔴) generation grounding and retrieval accuracy."
                )

                # Chronological order (oldest to newest)
                chrono_runs = list(reversed(runs))
                run_chart_labels = [f"#{r['id']}" for r in chrono_runs]
                latest_run = runs[0]
                prev_run = runs[1] if len(runs) > 1 else None

                col_gen_stage, col_ret_stage = st.columns(2)

                with col_gen_stage:
                    st.markdown("#### 🤖 Generation Stage (LLM & Grounding)")
                    gc1, gc2 = st.columns(2)

                    # Faithfulness
                    f_curr = latest_run.get("avg_faithfulness")
                    f_prev = prev_run.get("avg_faithfulness") if prev_run else None
                    f_delta = f"{(f_curr - f_prev):+.3f}" if f_curr is not None and f_prev is not None else None
                    with gc1:
                        st.metric("Faithfulness", _format_metric(f_curr, "faithfulness"), delta=f_delta)
                        if len(chrono_runs) >= 2:
                            f_vals = [r.get("avg_faithfulness") for r in chrono_runs]
                            f_fig = create_metric_sparkline_chart(run_chart_labels, f_vals, "Faithfulness", target_threshold=0.90, line_color="#2ecc71")
                            if f_fig:
                                st.plotly_chart(f_fig, use_container_width=True, config={"displayModeBar": False})

                    # Answer Relevancy
                    r_curr = latest_run.get("avg_answer_relevancy")
                    r_prev = prev_run.get("avg_answer_relevancy") if prev_run else None
                    r_delta = f"{(r_curr - r_prev):+.3f}" if r_curr is not None and r_prev is not None else None
                    with gc2:
                        st.metric("Answer Relevancy", _format_metric(r_curr, "answer_relevancy"), delta=r_delta)
                        if len(chrono_runs) >= 2:
                            r_vals = [r.get("avg_answer_relevancy") for r in chrono_runs]
                            r_fig = create_metric_sparkline_chart(run_chart_labels, r_vals, "Answer Relevancy", target_threshold=0.85, line_color="#3498db")
                            if r_fig:
                                st.plotly_chart(r_fig, use_container_width=True, config={"displayModeBar": False})

                with col_ret_stage:
                    st.markdown("#### 🔍 Retrieval Stage (Search & Reranker)")
                    rc1, rc2 = st.columns(2)

                    # Context Precision
                    cp_curr = latest_run.get("avg_context_precision")
                    cp_prev = prev_run.get("avg_context_precision") if prev_run else None
                    cp_delta = f"{(cp_curr - cp_prev):+.3f}" if cp_curr is not None and cp_prev is not None else None
                    with rc1:
                        st.metric("Context Precision", _format_metric(cp_curr, "context_precision"), delta=cp_delta)
                        if len(chrono_runs) >= 2:
                            cp_vals = [r.get("avg_context_precision") for r in chrono_runs]
                            cp_fig = create_metric_sparkline_chart(run_chart_labels, cp_vals, "Context Precision", target_threshold=0.80, line_color="#9b59b6")
                            if cp_fig:
                                st.plotly_chart(cp_fig, use_container_width=True, config={"displayModeBar": False})

                    # Context Recall
                    cr_curr = latest_run.get("avg_context_recall")
                    cr_prev = prev_run.get("avg_context_recall") if prev_run else None
                    cr_delta = f"{(cr_curr - cr_prev):+.3f}" if cr_curr is not None and cr_prev is not None else None
                    with rc2:
                        st.metric("Context Recall", _format_metric(cr_curr, "context_recall"), delta=cr_delta)
                        if len(chrono_runs) >= 2:
                            cr_vals = [r.get("avg_context_recall") for r in chrono_runs]
                            cr_fig = create_metric_sparkline_chart(run_chart_labels, cr_vals, "Context Recall", target_threshold=0.80, line_color="#e67e22")
                            if cr_fig:
                                st.plotly_chart(cr_fig, use_container_width=True, config={"displayModeBar": False})

                st.divider()

                # -----------------------------------------------------------
                # 2. Historical Summary Table
                # -----------------------------------------------------------
                st.markdown("### 📋 All Saved Evaluation Runs")
                runs_df = pd.DataFrame(runs)
                runs_df["prompt_preset"] = runs_df["notes"].apply(_extract_prompt_preset_from_notes)

                metric_display_cols = ["avg_faithfulness", "avg_answer_relevancy", "avg_context_precision", "avg_context_recall"]
                
                # Format metric columns in summary table
                runs_table = runs_df.copy()
                for mc in metric_display_cols:
                    if mc in runs_table.columns:
                        metric_key = mc.replace("avg_", "")
                        runs_table[mc] = runs_table[mc].apply(
                            lambda v, m=metric_key: _format_metric(v, m) if v is not None and not (isinstance(v, float) and pd.isna(v)) else "⚪ N/A"
                        )

                table_cols = ["id", "created_at", "run_name", "prompt_preset", "dataset_name"] + metric_display_cols + ["generation_model"]
                st.dataframe(
                    runs_table[[c for c in table_cols if c in runs_table.columns]],
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()

                # -----------------------------------------------------------
                # 3. Single Run Inspector
                # -----------------------------------------------------------
                st.subheader("🔍 Inspect Single Run")
                run_ids_avail = [r["id"] for r in runs]
                run_id_detail = st.selectbox(
                    "Select Run to Inspect",
                    run_ids_avail,
                    format_func=lambda x: f"Run #{x} — {next((r['run_name'] for r in runs if r['id'] == x), '')} [{_extract_prompt_preset_from_notes(next((r.get('notes') for r in runs if r['id'] == x), ''))}]",
                )

                if run_id_detail:
                    run_detail = get_evaluation_run(run_id_detail)
                    if run_detail:
                        preset_inspected = _extract_prompt_preset_from_notes(run_detail.get("notes"))
                        template_inspected = _extract_prompt_template_from_notes(run_detail.get("notes"))

                        st.markdown(f"### Run #{run_detail['id']}: `{run_detail['run_name']}`")
                        st.caption(f"Created: {run_detail.get('created_at')} | Dataset: `{run_detail.get('dataset_name')}` | Prompt Preset: **{preset_inspected}**")

                        col_single_kpis, col_single_radar = st.columns([1.2, 1.8])

                        with col_single_kpis:
                            k1, k2 = st.columns(2)
                            k1.metric("Faithfulness", _format_metric(run_detail.get("avg_faithfulness"), "faithfulness"))
                            k2.metric("Answer Relevancy", _format_metric(run_detail.get("avg_answer_relevancy"), "answer_relevancy"))
                            k3, k4 = st.columns(2)
                            k3.metric("Context Precision", _format_metric(run_detail.get("avg_context_precision"), "context_precision"))
                            k4.metric("Context Recall", _format_metric(run_detail.get("avg_context_recall"), "context_recall"))

                            # Prompt & Grounding Strategy Card
                            with st.expander(f"📝 Prompt & Grounding ({preset_inspected})", expanded=False):
                                st.markdown(f"**Active Preset:** `{preset_inspected}`")
                                if template_inspected:
                                    st.text_area("Prompt Template Used", template_inspected, height=130, disabled=True, key=f"tpl_insp_{run_id_detail}")
                                else:
                                    st.caption("Used default preset instructions.")

                            # Pipeline Configuration
                            with st.expander("⚙️ Pipeline Hyperparameters", expanded=False):
                                st.json({
                                    "prompt_preset": preset_inspected,
                                    "use_hybrid": run_detail.get("use_hybrid"),
                                    "use_reranker": run_detail.get("use_reranker"),
                                    "pool_size": run_detail.get("pool_size"),
                                    "final_top_k": run_detail.get("final_top_k"),
                                    "chunk_size": run_detail.get("chunk_size"),
                                    "chunk_overlap": run_detail.get("chunk_overlap"),
                                    "embedding_model": run_detail.get("embedding_model"),
                                    "rerank_model": run_detail.get("rerank_model"),
                                    "generation_model": run_detail.get("generation_model"),
                                    "notes": run_detail.get("notes"),
                                })

                        with col_single_radar:
                            radar_single_data = [
                                {
                                    "name": f"Run #{run_detail['id']}",
                                    "faithfulness": run_detail.get("avg_faithfulness"),
                                    "answer_relevancy": run_detail.get("avg_answer_relevancy"),
                                    "context_precision": run_detail.get("avg_context_precision"),
                                    "context_recall": run_detail.get("avg_context_recall"),
                                }
                            ]
                            fig_single_radar = create_radar_comparison_chart(radar_single_data, title=f"Quality Balance: Run #{run_detail['id']}")
                            if fig_single_radar:
                                st.plotly_chart(fig_single_radar, use_container_width=True)

                        # Detailed question breakdown
                        details_list = run_detail.get("detailed_results", [])
                        if details_list:
                            st.markdown(f"**Question-by-Question Breakdown ({len(details_list)} test cases):**")
                            detail_df = pd.DataFrame(details_list)
                            hist_metric_cols = [k for k in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"] if k in detail_df.columns]
                            for mc in hist_metric_cols:
                                detail_df[mc] = detail_df[mc].apply(
                                    lambda v, m=mc: _format_metric(v, m) if v is not None and not (isinstance(v, float) and pd.isna(v)) else "⚪ N/A"
                                )
                            detail_cols = [c for c in ["user_input"] + hist_metric_cols if c in detail_df.columns]
                            st.dataframe(detail_df[detail_cols], use_container_width=True, hide_index=True)

                        # Actions: Promote to Production or Delete
                        col_act1, col_act2 = st.columns([2, 1])
                        with col_act1:
                            if st.button(f"🚀 Apply Run #{run_id_detail} as Production Default", type="primary", key=f"apply_prod_{run_id_detail}"):
                                from src.db.settings_store import set_many_settings
                                from src.pipeline.prompts import PROMPT_PRESETS

                                prod_updates = {
                                    "pipeline.use_hybrid": bool(run_detail.get("use_hybrid", True)),
                                    "pipeline.use_reranker": bool(run_detail.get("use_reranker", True)),
                                    "pipeline.pool_size": int(run_detail.get("pool_size", 5)),
                                    "pipeline.final_top_k": int(run_detail.get("final_top_k", 2)),
                                    "models.generation": run_detail.get("generation_model", "gemini-2.5-flash"),
                                }
                                if run_detail.get("chunk_size"):
                                    prod_updates["pipeline.chunk_size"] = int(run_detail["chunk_size"])
                                if run_detail.get("chunk_overlap"):
                                    prod_updates["pipeline.chunk_overlap"] = int(run_detail["chunk_overlap"])

                                preset_ins = _extract_prompt_preset_from_notes(run_detail.get("notes"))
                                template_ins = _extract_prompt_template_from_notes(run_detail.get("notes"))
                                if preset_ins:
                                    prod_updates["pipeline.prompt_preset"] = preset_ins
                                if template_ins:
                                    prod_updates["pipeline.prompt_template"] = template_ins
                                elif preset_ins in PROMPT_PRESETS:
                                    prod_updates["pipeline.prompt_template"] = PROMPT_PRESETS[preset_ins]

                                set_many_settings(prod_updates, updated_by=f"Promoted from Run #{run_id_detail}")
                                st.success(f"✅ Run #{run_id_detail} promoted to Production Default! Applied immediately to Chat App and API.")
                                st.rerun()

                        with col_act2:
                            if st.button(f"🗑️ Delete Run #{run_id_detail}", key=f"del_run_{run_id_detail}"):
                                delete_evaluation_run(run_id_detail)
                                st.success(f"Deleted run #{run_id_detail}.")
                                st.rerun()

                st.divider()

                # -----------------------------------------------------------
                # 4. Side-by-Side Comparison & Multi-Dimensional Radar
                # -----------------------------------------------------------
                st.subheader("⚖️ Side-by-Side Run Comparison & Radar Analysis")
                st.caption("Overlay Run A (Baseline) vs. Run B (Candidate) on a multi-dimensional radar chart to identify trade-offs.")

                if len(runs) >= 2:
                    col_a, col_b = st.columns(2)
                    run_ids = [r["id"] for r in runs]
                    run_labels = {r["id"]: f"#{r['id']} — {r['run_name']} [{_extract_prompt_preset_from_notes(r.get('notes'))}]" for r in runs}

                    with col_a:
                        compare_a = st.selectbox("Run A (Baseline)", run_ids, index=min(1, len(run_ids) - 1), format_func=lambda x: run_labels[x], key="cmp_a")
                    with col_b:
                        compare_b = st.selectbox("Run B (Candidate)", run_ids, index=0, format_func=lambda x: run_labels[x], key="cmp_b")

                    if st.button("📊 Compare Runs Side-by-Side", type="primary", key="btn_cmp_runs"):
                        detail_a = get_evaluation_run(compare_a)
                        detail_b = get_evaluation_run(compare_b)

                        if detail_a and detail_b:
                            col_radar_cmp, col_table_cmp = st.columns([1.3, 1.2])

                            with col_radar_cmp:
                                radar_cmp_data = [
                                    {
                                        "name": f"Run A (#{compare_a}): {detail_a.get('run_name', '')[:15]}",
                                        "faithfulness": detail_a.get("avg_faithfulness"),
                                        "answer_relevancy": detail_a.get("avg_answer_relevancy"),
                                        "context_precision": detail_a.get("avg_context_precision"),
                                        "context_recall": detail_a.get("avg_context_recall"),
                                    },
                                    {
                                        "name": f"Run B (#{compare_b}): {detail_b.get('run_name', '')[:15]}",
                                        "faithfulness": detail_b.get("avg_faithfulness"),
                                        "answer_relevancy": detail_b.get("avg_answer_relevancy"),
                                        "context_precision": detail_b.get("avg_context_precision"),
                                        "context_recall": detail_b.get("avg_context_recall"),
                                    },
                                ]
                                fig_radar = create_radar_comparison_chart(radar_cmp_data, title="Run A vs Run B Quality Radar")
                                if fig_radar:
                                    st.plotly_chart(fig_radar, use_container_width=True)

                            with col_table_cmp:
                                metric_keys = ["avg_faithfulness", "avg_answer_relevancy", "avg_context_precision", "avg_context_recall"]
                                comparison_data = []
                                for key in metric_keys:
                                    val_a = detail_a.get(key)
                                    val_b = detail_b.get(key)
                                    metric_key = key.replace("avg_", "")
                                    diff = None
                                    if val_a is not None and val_b is not None:
                                        diff = val_b - val_a
                                    diff_str = "N/A"
                                    if diff is not None:
                                        indicator = "📈" if diff > 0.005 else ("📉" if diff < -0.005 else "➡️")
                                        diff_str = f"{indicator} {diff:+.3f}"
                                    comparison_data.append({
                                        "Metric": key.replace("avg_", "").replace("_", " ").title(),
                                        f"Run A (#{compare_a})": _format_metric(val_a, metric_key),
                                        f"Run B (#{compare_b})": _format_metric(val_b, metric_key),
                                        "Δ (B − A)": diff_str,
                                    })

                                st.markdown("#### 📋 Score Deltas")
                                st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

                            # Parameter diff
                            preset_a = _extract_prompt_preset_from_notes(detail_a.get("notes"))
                            preset_b = _extract_prompt_preset_from_notes(detail_b.get("notes"))

                            param_diff = [
                                {
                                    "Parameter": "Prompt Preset",
                                    f"Run A (#{compare_a})": preset_a,
                                    f"Run B (#{compare_b})": preset_b,
                                    "Changed": "✅" if preset_a != preset_b else "",
                                }
                            ]
                            param_keys = [
                                "use_hybrid", "use_reranker", "pool_size", "final_top_k",
                                "chunk_size", "chunk_overlap", "embedding_model", "rerank_model", "generation_model"
                            ]
                            for key in param_keys:
                                va = detail_a.get(key)
                                vb = detail_b.get(key)
                                param_diff.append({
                                    "Parameter": key,
                                    f"Run A (#{compare_a})": str(va),
                                    f"Run B (#{compare_b})": str(vb),
                                    "Changed": "✅" if va != vb else "",
                                })
                            st.markdown("#### ⚙️ Parameter & Prompt Differences")
                            st.dataframe(pd.DataFrame(param_diff), use_container_width=True, hide_index=True)
                else:
                    st.info("Save at least 2 evaluation runs to compare them.")

        except Exception as e:
            st.error(f"Failed to load evaluation history: {e}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())


