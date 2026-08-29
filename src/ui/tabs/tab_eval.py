"""Tab 3: RAGAS Benchmark Evaluation & Metric Dashboards."""

import json
import logging
from io import StringIO
import streamlit as st
import pandas as pd

from src.ui.helpers import _format_metric, _quality_icon, METRIC_THRESHOLDS, METRIC_INFO


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
            run_name = st.text_input("Run Name", value=f"{'hybrid' if eval_use_hybrid else 'dense'}+{'reranker' if eval_use_reranker else 'topk'}")
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
                        cases=cases,
                        use_hybrid=eval_use_hybrid,
                        use_reranker=eval_use_reranker,
                        pool_size=eval_pool_size,
                        final_top_k=eval_top_k,
                        progress_callback=progress_cb,
                    )

                    eval_progress.progress(100, text="Evaluation complete!")
                    eval_status.success("✅ Evaluation finished successfully!")

                    # Store results in session state
                    st.session_state["eval_results"] = eval_results
                    st.session_state["eval_run_name"] = run_name
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
                            from evaluation.evaluation_store import save_evaluation_run

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
                                notes=st.session_state.get("eval_run_notes"),
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
                runs_df = pd.DataFrame(runs)
                metric_display_cols = ["avg_faithfulness", "avg_answer_relevancy", "avg_context_precision", "avg_context_recall"]
                
                # Format metric columns in summary table
                runs_table = runs_df.copy()
                for mc in metric_display_cols:
                    if mc in runs_table.columns:
                        metric_key = mc.replace("avg_", "")
                        runs_table[mc] = runs_table[mc].apply(
                            lambda v, m=metric_key: _format_metric(v, m) if v is not None and not (isinstance(v, float) and pd.isna(v)) else "⚪ N/A"
                        )

                table_cols = ["id", "created_at", "run_name", "dataset_name"] + metric_display_cols + ["generation_model"]
                st.dataframe(
                    runs_table[[c for c in table_cols if c in runs_table.columns]],
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()

                # --- Single Run Inspector ---
                st.subheader("Inspect Single Run")
                run_ids_avail = [r["id"] for r in runs]
                run_id_detail = st.selectbox("Select Run to Inspect", run_ids_avail, format_func=lambda x: f"Run #{x} — {next((r['run_name'] for r in runs if r['id'] == x), '')}")

                if run_id_detail:
                    run_detail = get_evaluation_run(run_id_detail)
                    if run_detail:
                        st.markdown(f"### Run #{run_detail['id']}: `{run_detail['run_name']}`")
                        st.caption(f"Created: {run_detail.get('created_at')} | Dataset: `{run_detail.get('dataset_name')}`")

                        # KPI Cards
                        k1, k2, k3, k4 = st.columns(4)
                        k1.metric("Faithfulness", _format_metric(run_detail.get("avg_faithfulness"), "faithfulness"))
                        k2.metric("Answer Relevancy", _format_metric(run_detail.get("avg_answer_relevancy"), "answer_relevancy"))
                        k3.metric("Context Precision", _format_metric(run_detail.get("avg_context_precision"), "context_precision"))
                        k4.metric("Context Recall", _format_metric(run_detail.get("avg_context_recall"), "context_recall"))

                        # Pipeline Configuration
                        with st.expander("⚙️ Pipeline Configuration", expanded=False):
                            st.json({
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

                        # Delete button
                        if st.button(f"🗑️ Delete Run #{run_id_detail}", key=f"del_run_{run_id_detail}"):
                            delete_evaluation_run(run_id_detail)
                            st.success(f"Deleted run #{run_id_detail}.")
                            st.rerun()

                st.divider()

                # --- Side-by-side comparison ---
                st.subheader("Compare Two Runs")

                if len(runs) >= 2:
                    col_a, col_b = st.columns(2)
                    run_ids = [r["id"] for r in runs]
                    run_labels = {r["id"]: f"#{r['id']} — {r['run_name']}" for r in runs}

                    with col_a:
                        compare_a = st.selectbox("Run A", run_ids, index=0, format_func=lambda x: run_labels[x], key="cmp_a")
                    with col_b:
                        compare_b = st.selectbox("Run B", run_ids, index=min(1, len(run_ids) - 1), format_func=lambda x: run_labels[x], key="cmp_b")

                    if st.button("📊 Compare"):
                        detail_a = get_evaluation_run(compare_a)
                        detail_b = get_evaluation_run(compare_b)

                        if detail_a and detail_b:
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

                            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

                            # Parameter diff
                            param_keys = ["use_hybrid", "use_reranker", "pool_size", "final_top_k",
                                          "chunk_size", "chunk_overlap", "embedding_model", "rerank_model", "generation_model"]
                            param_diff = []
                            for key in param_keys:
                                va = detail_a.get(key)
                                vb = detail_b.get(key)
                                param_diff.append({
                                    "Parameter": key,
                                    f"Run A (#{compare_a})": str(va),
                                    f"Run B (#{compare_b})": str(vb),
                                    "Changed": "✅" if va != vb else "",
                                })
                            st.markdown("**Parameter Differences**")
                            st.dataframe(pd.DataFrame(param_diff), use_container_width=True, hide_index=True)
                else:
                    st.info("Save at least 2 evaluation runs to compare them.")

        except Exception as e:
            st.error(f"Failed to load evaluation history: {e}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())

