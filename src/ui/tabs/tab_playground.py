"""Tab 4: Interactive Playground, Report Attachments & Feedback Curation."""

import os
import re
import time
import base64
import streamlit as st
import pandas as pd

from src.utils.config import config
from src.ui.helpers import _render_attachment_preview


def render_playground_tab():
    """Render the Playground & Feedback tab."""
    st.header("Interactive RAG Playground & Feedback")
    st.markdown(
        "Ask arbitrary questions to test pipeline responses in real-time, inspect retrieved chunks, "
        "and record feedback (ratings, root-cause tags, ground-truth corrections) to build continuous evaluation datasets."
    )

    play_tab_ask, play_tab_analytics = st.tabs(["🎯 Ask & Rate", "📈 Feedback Explorer & Benchmark Export"])

    # =======================================================================
    # Subtab 1: Ask & Rate
    # =======================================================================
    with play_tab_ask:
        col_q_input, col_q_opts = st.columns([3, 1])

        with col_q_input:
            user_question = st.text_area(
                "Question / Remediation Request",
                placeholder="e.g. How do we remediate the vulnerabilities in this report according to our standard operating procedures?",
                height=100,
                key="playground_q_input",
            )

            # Document Upload for Audits / Vulnerability Reports
            with st.expander("📎 Attach Vulnerability Report / Audit File (PDF, TXT, Image, JSON)", expanded=False):
                st.caption("Upload an external scan report, policy document, or error log to query directly with RAG guidance.")
                uploaded_file = st.file_uploader(
                    "Choose report file",
                    type=["pdf", "txt", "log", "json", "md", "png", "jpg", "jpeg"],
                    key="rag_doc_upload",
                    help="Extract text from uploaded documents and augment your query against company SOP knowledge bases.",
                )
                if uploaded_file is not None:
                    file_details = {
                        "Filename": uploaded_file.name,
                        "FileType": uploaded_file.type,
                        "FileSize": f"{uploaded_file.size / 1024:.1f} KB",
                    }
                    st.json(file_details)

        with col_q_opts:
            st.markdown("**Pipeline Configuration**")
            play_use_hybrid = st.toggle("Hybrid Search", value=True, key="play_hybrid")
            play_use_reranker = st.toggle("Semantic Reranker", value=True, key="play_rerank")
            play_pool_size = st.number_input("Pool Size", min_value=1, max_value=50, value=5, key="play_pool")
            play_top_k = st.number_input("Final Top K", min_value=1, max_value=20, value=2, key="play_topk")
            play_model = st.selectbox(
                "Generation Model",
                [
                    "gemini-2.5-flash",
                    "gemini-2.5-pro",
                    "gemini-2.0-flash",
                ],
                index=0,
                key="play_gen_model",
            )

        if st.button("🚀 Ask Pipeline", type="primary", key="btn_ask_play"):
            if not user_question.strip() and uploaded_file is None:
                st.error("Please enter a question or upload a document report.")
            else:
                with st.spinner("Processing query through RAG pipeline..."):
                    from src.pipeline import get_default_pipeline

                    pipeline = get_default_pipeline()

                    # Extract text content from attachment if present
                    attached_text = ""
                    raw_attachment_bytes = None
                    attachment_mime = None
                    if uploaded_file is not None:
                        try:
                            raw_attachment_bytes = uploaded_file.getvalue()
                            attachment_mime = uploaded_file.type
                            fn_lower = uploaded_file.name.lower()

                            if fn_lower.endswith(".pdf"):
                                try:
                                    import pypdf
                                    from io import BytesIO
                                    reader = pypdf.PdfReader(BytesIO(raw_attachment_bytes))
                                    pdf_pages_text = [page.extract_text() for page in reader.pages if page.extract_text()]
                                    attached_text = "\n\n".join(pdf_pages_text)
                                except Exception as pdf_err:
                                    attached_text = f"[PDF Parsing Notice: {pdf_err}]"
                            elif fn_lower.endswith((".txt", ".log", ".json", ".md")):
                                attached_text = raw_attachment_bytes.decode("utf-8", errors="ignore")
                            elif fn_lower.endswith((".png", ".jpg", ".jpeg")):
                                attached_text = f"[Attached Image: {uploaded_file.name} ({uploaded_file.size} bytes)]"
                            else:
                                attached_text = f"[Attached File: {uploaded_file.name}]"
                        except Exception as parse_err:
                            st.warning(f"Could not extract text from attachment: {parse_err}")

                    # Compose full augmented query
                    augmented_query = user_question.strip()
                    if attached_text:
                        if augmented_query:
                            augmented_query = f"{augmented_query}\n\n--- Attached Report Content ({uploaded_file.name}) ---\n{attached_text}"
                        else:
                            augmented_query = f"Please analyze and propose remediation for the following attached report:\n\n--- Attached Report Content ({uploaded_file.name}) ---\n{attached_text}"

                    t0 = time.time()
                    try:
                        resp_obj = pipeline.query(
                            query_text=augmented_query,
                            use_hybrid=play_use_hybrid,
                            use_reranker=play_use_reranker,
                            pool_size=play_pool_size,
                            final_top_k=play_top_k,
                            model_name=play_model,
                        )
                        latency = int((time.time() - t0) * 1000)

                        # Extract structured documentation gaps from response
                        raw_response_text = resp_obj.response
                        extracted_gaps = None

                        # Pattern 1: HTML Comment delimiter <!-- DOCUMENTATION_GAPS --> ... <!-- END_DOCUMENTATION_GAPS -->
                        gap_html_match = re.search(
                            r"<!--\s*DOCUMENTATION_GAPS\s*-->([\s\S]*?)<!--\s*END_DOCUMENTATION_GAPS\s*-->",
                            raw_response_text,
                            re.IGNORECASE,
                        )
                        if gap_html_match:
                            gaps_content = gap_html_match.group(1).strip()
                            if gaps_content and not gaps_content.lower().startswith("none"):
                                extracted_gaps = gaps_content

                        # Pattern 2: Markdown Section ### ⚠️ Internal Documentation Gaps or ### 3. Internal Documentation Gaps
                        if not extracted_gaps:
                            gap_md_match = re.search(
                                r"(?:###\s*(?:\d+\.\s*)?⚠️\s*Internal Documentation Gaps[\s\S]*?)(?=(?:###|\Z))",
                                raw_response_text,
                                re.IGNORECASE,
                            )
                            if gap_md_match:
                                gap_block = gap_md_match.group(0).strip()
                                # Clean off the header itself
                                cleaned_gap = re.sub(r"^###\s*(?:\d+\.\s*)?⚠️\s*Internal Documentation Gaps\s*", "", gap_block, flags=re.IGNORECASE).strip()
                                if cleaned_gap and not cleaned_gap.lower().startswith("none"):
                                    extracted_gaps = cleaned_gap

                        st.session_state["play_last_result"] = {
                            "query": user_question.strip() if user_question.strip() else f"Analyze attached: {uploaded_file.name}",
                            "response": resp_obj.response,
                            "retrieved_contexts": [c.text for c in resp_obj.retrieved_chunks],
                            "sources": resp_obj.sources,
                            "use_hybrid": play_use_hybrid,
                            "use_reranker": play_use_reranker,
                            "pool_size": play_pool_size,
                            "final_top_k": play_top_k,
                            "generation_model": play_model,
                            "latency_ms": latency,
                            "attached_filename": uploaded_file.name if uploaded_file is not None else None,
                            "attached_file_bytes": raw_attachment_bytes,
                            "attached_file_mime": attachment_mime,
                            "documentation_gaps": extracted_gaps,
                        }
                    except Exception as e:
                        st.error(f"Pipeline execution failed: {e}")

        # Show Output and Feedback Form
        if "play_last_result" in st.session_state:
            res = st.session_state["play_last_result"]

            st.divider()
            st.subheader("💡 Pipeline Response")

            # Inline Preview for Attached Document (if present)
            if res.get("attached_filename") and res.get("attached_file_bytes"):
                _render_attachment_preview(
                    filename=res["attached_filename"],
                    data=res["attached_file_bytes"],
                    mime_type=res.get("attached_file_mime"),
                    label=f"👁️ Preview Attached Document ({res['attached_filename']})",
                    key_prefix="play_preview",
                )

            st.markdown(res["response"])

            # Highlight Knowledge / Documentation Gaps prominently if discovered
            if res.get("documentation_gaps"):
                st.warning(
                    f"⚠️ **Internal Documentation Gap (Knowledge Backlog Alert):**\n\n"
                    f"{res['documentation_gaps']}\n\n"
                    f"*This query exposed a gap in the internal SOP knowledge base. The missing topic has been flagged for documentation triage.*"
                )

            # Metadata info badges
            st.caption(
                f"⏱️ Latency: **{res['latency_ms']} ms** | "
                f"🤖 Model: `{res['generation_model']}` | "
                f"🔍 Mode: `{'Hybrid' if res['use_hybrid'] else 'Dense'}` + `{'Reranker' if res['use_reranker'] else 'Direct'}` | "
                f"📎 Attachment: `{res.get('attached_filename') or 'None'}` | "
                f"📚 Sources: {', '.join(res['sources']) if res['sources'] else 'None'}"
            )

            # Retrieved chunks expander
            if res["retrieved_contexts"]:
                with st.expander(f"🔍 Inspect Retrieved Contexts ({len(res['retrieved_contexts'])} chunks)", expanded=False):
                    for i, ctx in enumerate(res["retrieved_contexts"], 1):
                        st.markdown(f"**Chunk #{i}:**")
                        st.code(ctx, language="text")

            # --- Feedback & Rating Card ---
            st.divider()
            st.subheader("⭐ Rate Answer Quality & Collect Feedback")
            st.caption(
                "Feedback is saved to the database and can be used to curate benchmark datasets, "
                "identify retrieval failure patterns, and write ideal ground-truth reference answers."
            )

            col_rating, col_tags = st.columns([1, 2])

            with col_rating:
                rating_choice = st.radio(
                    "Rating",
                    options=["👍 Good (Accurate & Grounded)", "👎 Bad (Hallucination / Incomplete)"],
                    index=0,
                    key="play_rating_choice",
                )
                rating_val = 5 if "Good" in rating_choice else 1

            with col_tags:
                issue_tag_options = [
                    "Hallucination / Ungrounded Claim",
                    "Missing Context (Recall Failure)",
                    "Poor Ranking (Precision Failure)",
                    "Vague / Incomplete Answer",
                    "Incorrect Remediation Guidance",
                    "Formatting / Tone Issue",
                ]
                issue_tags_selected = st.multiselect(
                    "Failure Root-Cause Tags (if any)",
                    options=issue_tag_options,
                    default=[] if rating_val == 5 else ["Missing Context (Recall Failure)"],
                    key="play_issue_tags",
                )

            col_fb_correct, col_fb_comment = st.columns(2)
            with col_fb_correct:
                corrected_answer = st.text_area(
                    "Ideal Reference Answer (Optional ground truth for benchmark)",
                    placeholder="If the answer was incomplete or wrong, enter the ideal ground-truth answer here to promote into benchmark datasets...",
                    height=100,
                    key="play_corrected_ref",
                )
            with col_fb_comment:
                feedback_comment = st.text_area(
                    "Comments / Notes (Optional)",
                    placeholder="Add context or rationale for this rating...",
                    height=100,
                    key="play_comment",
                )

            if st.button("💾 Submit Feedback & Save to DB", type="primary", key="btn_save_fb"):
                try:
                    from src.feedback import save_feedback

                    user_comment_final = feedback_comment.strip() if feedback_comment.strip() else None
                    if res.get("attached_filename"):
                        file_note = f"[Attached: {res['attached_filename']}]"
                        user_comment_final = f"{file_note} {user_comment_final}" if user_comment_final else file_note

                    fb_id = save_feedback(
                        query=res["query"],
                        response=res["response"],
                        retrieved_contexts=res["retrieved_contexts"],
                        sources=res["sources"],
                        use_hybrid=res["use_hybrid"],
                        use_reranker=res["use_reranker"],
                        pool_size=res["pool_size"],
                        final_top_k=res["final_top_k"],
                        generation_model=res["generation_model"],
                        latency_ms=res["latency_ms"],
                        rating=rating_val,
                        issue_tags=issue_tags_selected,
                        corrected_reference=corrected_answer.strip() if corrected_answer.strip() else None,
                        user_comment=user_comment_final,
                        attached_filename=res.get("attached_filename"),
                        attached_file_data=res.get("attached_file_bytes"),
                        attached_file_mime=res.get("attached_file_mime"),
                        documentation_gaps=res.get("documentation_gaps"),
                    )
                    st.session_state["feedback_submitted_id"] = fb_id
                    st.success(f"✅ Feedback successfully recorded in DB with ID **#{fb_id}**!")
                except Exception as e:
                    st.error(f"Failed saving feedback: {e}")

    # =======================================================================
    # Subtab 2: Feedback Explorer & Benchmark Export
    # =======================================================================
    with play_tab_analytics:
        st.subheader("📊 Feedback Analytics & Continuous Benchmark Curation")

        col_fb_refresh, _ = st.columns([1, 5])
        with col_fb_refresh:
            if st.button("🔄 Refresh Feedback", key="btn_ref_fb_analytics"):
                st.rerun()

        try:
            from src.feedback import (
                list_feedback,
                get_feedback_by_id,
                get_feedback_analytics,
                delete_feedback,
                promote_to_benchmark,
            )
            from evaluation.evaluate_ragas import list_available_datasets

            stats = get_feedback_analytics()

            if not stats.get("table_exists", True):
                st.warning(
                    "⚠️ Database table `query_feedback` has not been created yet. "
                    "Run `python3 scripts/run_migrations.py` in your terminal to apply migrations."
                )

            # KPI metrics
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Total Rated Queries", stats["total_ratings"])
            m_col2.metric("Satisfaction Rate", f"{stats['satisfaction_rate']:.1f}%", f"{stats['positive_count']} 👍 / {stats['negative_count']} 👎")
            m_col3.metric("Negative Feedback", stats["negative_count"])
            m_col4.metric("Avg Latency", f"{stats['avg_latency_ms']:.0f} ms")

            # Tag breakdown
            if stats["tag_breakdown"]:
                st.markdown("**Common Failure Modes (Issue Tags)**")
                tag_df = pd.DataFrame([
                    {"Failure Tag": k, "Occurrences": v} for k, v in stats["tag_breakdown"].items()
                ])
                st.dataframe(tag_df, use_container_width=True, hide_index=True)

            st.divider()

            # Filter feedback entries
            st.subheader("Browse Past Ratings")
            col_filt_rating, col_filt_limit = st.columns([2, 1])
            with col_filt_rating:
                filter_rating_ui = st.selectbox("Filter by Rating", ["All", "👍 Positive (Rating 5)", "👎 Negative (Rating 1)"])
                rating_arg = None
                if "Positive" in filter_rating_ui:
                    rating_arg = 5
                elif "Negative" in filter_rating_ui:
                    rating_arg = 1

            with col_filt_limit:
                fb_limit = st.number_input("Max entries", min_value=10, max_value=500, value=50, step=25)

            feedback_rows = list_feedback(limit=fb_limit, rating_filter=rating_arg)

            if not feedback_rows:
                st.info("No feedback entries found. Ask questions in the Playground and rate them to see history here.")
            else:
                fb_display_df = pd.DataFrame(feedback_rows)
                fb_display_df["rating"] = fb_display_df["rating"].apply(lambda r: "👍 Good" if r == 5 else "👎 Bad")
                fb_display_df["is_promoted_to_benchmark"] = fb_display_df["is_promoted_to_benchmark"].apply(lambda p: "⭐ Promoted" if p else "")
                if "documentation_gaps" in fb_display_df.columns:
                    fb_display_df["knowledge_gap"] = fb_display_df["documentation_gaps"].apply(lambda g: "⚠️ Gap" if g else "")

                cols_to_show = [
                    "id", "created_at", "query", "response_preview", "rating",
                    "issue_tags", "latency_ms", "is_promoted_to_benchmark"
                ]
                if "attached_filename" in fb_display_df.columns:
                    cols_to_show.insert(4, "attached_filename")
                if "knowledge_gap" in fb_display_df.columns:
                    cols_to_show.append("knowledge_gap")

                st.dataframe(
                    fb_display_df[[c for c in cols_to_show if c in fb_display_df.columns]],
                    use_container_width=True,
                    hide_index=True,
                )

                st.divider()

                # Detailed inspection, review, and benchmark promotion
                st.subheader("🔍 Review Feedback & Set Ideal Reference Answer")
                fb_id_selected = st.selectbox(
                    "Select Feedback Entry to inspect, edit, or export",
                    options=[r["id"] for r in feedback_rows],
                    format_func=lambda x: (
                        f"#{x} — {'👍' if next((r['rating'] for r in feedback_rows if r['id'] == x), '') == '👍 Good' else '👎'} "
                        f"{'📎 [' + next((r['attached_filename'] for r in feedback_rows if r['id'] == x), '') + '] ' if next((r.get('attached_filename') for r in feedback_rows if r['id'] == x), None) else ''}"
                        f"{'⚠️ [Gap] ' if next((r.get('documentation_gaps') for r in feedback_rows if r['id'] == x), None) else ''}"
                        f"{next((r['query'][:60] for r in feedback_rows if r['id'] == x), '')}..."
                    ),
                    key="sel_fb_inspect",
                )

                if fb_id_selected:
                    fb_detail = get_feedback_by_id(fb_id_selected)
                    if fb_detail:
                        # Resolve attached file details with smart fallbacks
                        attached_fn = fb_detail.get("attached_filename")
                        attached_data = fb_detail.get("attached_file_data")
                        attached_mime = fb_detail.get("attached_file_mime")

                        # Fallback 1: check if filename was noted in user_comment or query
                        if not attached_fn:
                            full_text = f"{fb_detail.get('user_comment') or ''} {fb_detail.get('query') or ''}"
                            match = re.search(r"\[Attached(?:\s+Document)?:\s*([^\]]+)\]", full_text)
                            if match:
                                attached_fn = match.group(1).strip()

                        # Fallback 2: if we have filename but no binary data in DB, check local assets/sample_reports/
                        if attached_fn and not attached_data:
                            sample_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets", "sample_reports", attached_fn)
                            if os.path.isfile(sample_path):
                                try:
                                    with open(sample_path, "rb") as f:
                                        attached_data = f.read()
                                except Exception:
                                    pass

                        col_dt_q, col_dt_resp = st.columns(2)
                        with col_dt_q:
                            st.markdown(f"**Question / Remediation Request:**\n> {fb_detail['query']}")
                            st.markdown(f"**Rating:** {'👍 Good' if fb_detail['rating'] == 5 else '👎 Bad'}")
                            if fb_detail.get("issue_tags"):
                                st.markdown(f"**Issue Tags:** `{', '.join(fb_detail['issue_tags'])}`")

                            # Attachment details card
                            if attached_fn:
                                st.info(f"📎 **Attached Report File:** `{attached_fn}`")
                                if attached_data:
                                    mime_val = attached_mime or ("application/pdf" if attached_fn.lower().endswith(".pdf") else "application/octet-stream")
                                    st.download_button(
                                        label=f"📥 Download Report ({attached_fn})",
                                        data=attached_data,
                                        file_name=attached_fn,
                                        mime=mime_val,
                                        key=f"dl_fb_attach_{fb_id_selected}",
                                    )
                                    from src.feedback import get_feedback_attachment
                                    _render_attachment_preview(
                                        filename=attached_fn,
                                        data=attached_data,
                                        mime_type=mime_val,
                                        label=f"👁️ View Document Preview Directly in Browser ({attached_fn})",
                                        loader_fn=lambda fid=fb_id_selected: get_feedback_attachment(fid)[0],
                                        key_prefix=f"rev_{fb_id_selected}",
                                    )
                            else:
                                st.caption("📎 *No file was attached to this query.*")

                            # Resolve documentation gaps with smart fallback for older entries
                            doc_gaps = fb_detail.get("documentation_gaps")
                            if not doc_gaps:
                                raw_resp = fb_detail.get("response") or ""
                                legacy_gap_m = re.search(r"(?:###\s*(?:\d+\.\s*)?⚠️\s*Internal Documentation Gaps[\s\S]*?)(?=(?:###|\Z))", raw_resp, re.IGNORECASE)
                                if legacy_gap_m:
                                    g_block = legacy_gap_m.group(0).strip()
                                    g_text = re.sub(r"^###\s*(?:\d+\.\s*)?⚠️\s*Internal Documentation Gaps\s*", "", g_block, flags=re.IGNORECASE).strip()
                                    if g_text and not g_text.lower().startswith("none"):
                                        doc_gaps = g_text
                                else:
                                    html_gap_m = re.search(r"<!--\s*DOCUMENTATION_GAPS\s*-->([\s\S]*?)<!--\s*END_DOCUMENTATION_GAPS\s*-->", raw_resp, re.IGNORECASE)
                                    if html_gap_m:
                                        g_text = html_gap_m.group(1).strip()
                                        if g_text and not g_text.lower().startswith("none"):
                                            doc_gaps = g_text

                            if doc_gaps:
                                st.warning(f"⚠️ **Internal Documentation Gap (Knowledge Backlog):**\n\n{doc_gaps}")
                            else:
                                st.caption("ℹ️ *No internal documentation gaps logged for this query.*")

                            st.caption(f"⚙️ Hybrid: `{fb_detail.get('use_hybrid')}` | Rerank: `{fb_detail.get('use_reranker')}` | Latency: `{fb_detail.get('latency_ms', 0)} ms`")
                            if fb_detail.get("is_promoted_to_benchmark"):
                                st.success("⭐ Already included in benchmark dataset")

                        with col_dt_resp:
                            st.markdown("**Generated LLM Output:**")
                            st.markdown(fb_detail["response"])
                            if fb_detail.get("retrieved_contexts"):
                                with st.expander(f"📚 Retrieved Chunks ({len(fb_detail['retrieved_contexts'])})", expanded=False):
                                    for c_idx, chunk in enumerate(fb_detail["retrieved_contexts"], 1):
                                        st.caption(f"Chunk #{c_idx}")
                                        st.code(chunk, language="text")

                        st.divider()
                        st.markdown("### ✏️ Triage & Ground-Truth Correction")
                        st.caption(
                            "Review the bad answer, write the ideal ground-truth reference, and save it to the database "
                            "or export directly to your benchmark dataset for automated regression testing."
                        )

                        col_edit_ref, col_edit_comment = st.columns([3, 2])
                        with col_edit_ref:
                            review_corrected_ref = st.text_area(
                                "✨ Ideal / Corrected Reference Answer",
                                value=fb_detail.get("corrected_reference") or "",
                                placeholder="Enter the 100% correct, verified answer for this question...",
                                height=110,
                                key=f"ref_edit_{fb_id_selected}",
                            )
                        with col_edit_comment:
                            review_comment = st.text_area(
                                "Reviewer Notes / Context",
                                value=fb_detail.get("user_comment") or "",
                                placeholder="Add notes on why the model failed or what was fixed...",
                                height=110,
                                key=f"comment_edit_{fb_id_selected}",
                            )

                        # Promotion actions
                        col_prom_ds, col_save_db, col_prom_btn, col_del_fb = st.columns([2, 2, 2, 1])
                        with col_prom_ds:
                            datasets_avail = list_available_datasets() or ["default"]
                            target_dataset = st.selectbox("Target Benchmark", datasets_avail, key="prom_target_ds")

                        with col_save_db:
                            st.write("")
                            st.write("")
                            if st.button("💾 Save Reference to DB", key=f"btn_save_ref_{fb_id_selected}"):
                                try:
                                    from src.feedback import update_feedback_correction
                                    updated = update_feedback_correction(
                                        feedback_id=fb_id_selected,
                                        corrected_reference=review_corrected_ref,
                                        user_comment=review_comment,
                                    )
                                    if updated:
                                        st.success("✅ Saved updated reference to database.")
                                        st.rerun()
                                    else:
                                        st.info("No changes detected.")
                                except Exception as e:
                                    st.error(f"Save failed: {e}")

                        with col_prom_btn:
                            st.write("")
                            st.write("")
                            if st.button("⭐ Save & Promote to Benchmark", type="primary", key="btn_prom_bm"):
                                try:
                                    from src.feedback import promote_to_benchmark
                                    res_prom = promote_to_benchmark(
                                        feedback_id=fb_id_selected,
                                        dataset_name=target_dataset,
                                        override_reference=review_corrected_ref if review_corrected_ref.strip() else None,
                                    )
                                    if res_prom["is_new"]:
                                        st.success(f"🎉 Promoted to `{target_dataset}.json` as a new test case! Total cases: {res_prom['total_cases']}")
                                    else:
                                        st.success(f"✅ Updated existing test case in `{target_dataset}.json`! Total cases: {res_prom['total_cases']}")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Promotion failed: {e}")

                        with col_del_fb:
                            st.write("")
                            st.write("")
                            if st.button(f"🗑️ Delete", key=f"del_fb_{fb_id_selected}"):
                                delete_feedback(fb_id_selected)
                                st.success(f"Deleted feedback #{fb_id_selected}.")
                                st.rerun()

        except Exception as e:
            st.error(f"Failed to load feedback analytics: {e}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())
