"""Tab 2: Database Explorer."""

import json
from contextlib import closing
from datetime import datetime
import streamlit as st
import pandas as pd

from src.db import get_connection
from src.ui.helpers import _render_attachment_preview, _format_metric
from src.ui.db_helpers import (
    _list_database_tables,
    _get_table_count,
    _fetch_table_schema,
    _get_doc_count,
    _get_doc_type_breakdown,
    _fetch_documents,
    _fetch_embeddings_for_viz,
    _fetch_generic_table_data,
    run_embedding_quality_audit,
)


def render_db_tab():
    """Render the Database Explorer tab."""
    st.header("Database Explorer")
    st.markdown("Inspect, search, and manage PostgreSQL tables, feedback records, evaluation benchmarks, and vector embeddings.")

    # --- Table Selection Header ---
    avail_tables = _list_database_tables()
    col_tbl_sel, col_tbl_meta = st.columns([2, 1])

    with col_tbl_sel:
        selected_table = st.selectbox(
            "🗄️ Select Database Table to Explore",
            options=avail_tables,
            index=0,
            key="db_explorer_table_sel",
        )

    with col_tbl_meta:
        tbl_row_count = _get_table_count(selected_table)
        st.metric(f"Total Rows in `{selected_table}`", tbl_row_count)

    # --- Table Schema Inspector ---
    schema_df = _fetch_table_schema(selected_table)
    with st.expander(f"📐 Inspect `{selected_table}` Table Schema & Columns ({len(schema_df)} columns)", expanded=False):
        if not schema_df.empty:
            st.dataframe(schema_df, use_container_width=True, hide_index=True)
        else:
            st.caption("Schema information unavailable.")

    st.divider()

    # -----------------------------------------------------------------------
    # Case 1: `documents` table (Knowledge Base Chunks & Vector Embeddings)
    # -----------------------------------------------------------------------
    if selected_table == "documents":
        # --- Summary stats ---
        col_total, col_breakdown = st.columns([1, 2])

        with col_total:
            total_docs = _get_doc_count()
            st.metric("Total Ingested Documents", total_docs)

        with col_breakdown:
            breakdown = _get_doc_type_breakdown()
            if breakdown:
                breakdown_df = pd.DataFrame(
                    [{"Type": k, "Count": v} for k, v in breakdown.items()]
                )
                st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
            else:
                st.info("No documents found in the database.")

        st.divider()

        # --- Search and filter ---
        st.subheader("Browse Document Chunks")
        col_search, col_filter, col_limit = st.columns([3, 1, 1])

        with col_search:
            search_q = st.text_input("🔍 Full-text search", placeholder="Search document contents...", key="doc_fts_input")

        with col_filter:
            type_options = ["All"] + list(breakdown.keys()) if breakdown else ["All"]
            type_filter = st.selectbox("Filter by type", type_options, key="doc_type_sel")

        with col_limit:
            row_limit = st.number_input("Max rows", min_value=10, max_value=1000, value=100, step=50, key="doc_max_rows")

        docs_df = _fetch_documents(limit=row_limit, search_query=search_q, doc_type_filter=type_filter)

        if not docs_df.empty:
            doc_all_cols = list(docs_df.columns)
            selected_doc_cols = st.multiselect(
                "👁️ Visible Columns",
                options=doc_all_cols,
                default=doc_all_cols,
                key="doc_cols_visibility_picker",
                help="Select which table columns to display in the grid.",
            )
            st.dataframe(
                docs_df[selected_doc_cols if selected_doc_cols else doc_all_cols],
                use_container_width=True,
                hide_index=True,
            )

            # --- Expandable detail view ---
            with st.expander("📝 View full document detail", expanded=False):
                avail_doc_ids = docs_df["id"].tolist()
                sel_doc_id = st.selectbox("Select Document ID to inspect", options=avail_doc_ids, key="doc_id_select_detail")
                if sel_doc_id:
                    try:
                        with closing(get_connection()) as conn:
                            with conn.cursor() as cur:
                                cur.execute(
                                    "SELECT id, content, metadata, embedding::text FROM documents WHERE id = %s;",
                                    (sel_doc_id,),
                                )
                                row = cur.fetchone()
                                if row:
                                    st.markdown(f"**Document ID:** `{row[0]}`")
                                    st.text_area("Content", row[1], height=200, disabled=True)
                                    st.markdown("**Metadata:**")
                                    st.json(row[2])
                                    if row[3]:
                                        vec = [float(x) for x in row[3].strip("[]").split(",")]
                                        st.caption(f"Embedding dimensions: {len(vec)}")
                                        with st.expander("Raw embedding vector"):
                                            st.code(row[3][:500] + "..." if len(row[3]) > 500 else row[3])
                                else:
                                    st.warning(f"No document found with ID {sel_doc_id}.")
                    except Exception as e:
                        st.error(f"Error loading document: {e}")

        else:
            st.info("No documents match the current filter.")

        st.divider()

        # --- Embedding visualization ---
        st.subheader("🌌 Embedding Space Visualization (2D Projection)")
        st.markdown(
            "Project the **768-dimensional dense vector embeddings** (`text-embedding-005`) into 2D space "
            "to visually inspect semantic clustering, topic distribution, and identify coverage gaps or outliers across your knowledge base."
        )

        with st.expander("ℹ️ How to interpret Embedding Space Projections & Axis Scales", expanded=False):
            st.markdown(
                """
                - **Semantic Proximity**: Points located close together in 2D space share similar technical concepts, error domains, or runbook procedures.
                - **Clusters**: Tight groupings indicate well-documented topics with high knowledge density (e.g. database failover, auth errors).
                - **Outliers & Sparse Areas**: Isolated points represent unique topics or potential documentation gaps in the knowledge base.
                
                ---
                #### 📏 Understanding the X and Y Scales:
                - **When using PCA (Principal Component Analysis)**:
                  - **X-axis (`PC1` - Principal Component 1)**: The single linear direction capturing the **maximum variance** across all 768 embedding dimensions (e.g. general architectural concepts vs operational runtime errors).
                  - **Y-axis (`PC2` - Principal Component 2)**: The orthogonal (perpendicular) direction capturing the **second largest variance**, completely uncorrelated with PC1.
                  - *Units*: Normalized coordinates centered at `0.0`. Greater distance along an axis reflects greater conceptual divergence along that principal variation vector.
                - **When using t-SNE (t-Distributed Stochastic Neighbor Embedding)**:
                  - **X-axis & Y-axis (`t-SNE Dim 1`, `t-SNE Dim 2`)**: Non-linear manifold coordinates optimized to place nearest neighbors close together.
                  - *Units*: **Arbitrary coordinates**. The exact numerical values have no physical units—only the **relative distance between local points** is meaningful for topic clustering.
                """
            )

        col_method, col_info_box = st.columns([1, 2])
        with col_method:
            viz_method = st.radio(
                "Dimensionality Reduction Method",
                ["PCA", "t-SNE"],
                horizontal=True,
                key="doc_viz_method",
                help="PCA preserves global geometric variance; t-SNE reveals local semantic clusters."
            )
        with col_info_box:
            if viz_method == "PCA":
                st.caption("📐 **PCA Mode:** Orthogonal linear transformation from 768D to 2D (PC1 = primary variance, PC2 = secondary variance).")
            else:
                st.caption("🔍 **t-SNE Mode:** Probabilistic non-linear manifold mapping for tight semantic clusters (arbitrary scale).")

        if st.button("🎨 Generate 2D Embedding Projection", type="primary", key="btn_gen_viz"):
            ids, embeddings, types_ = _fetch_embeddings_for_viz()
            if len(embeddings) < 2:
                st.warning("Need at least 2 documents with embeddings to visualize.")
            else:
                import numpy as np
                from sklearn.decomposition import PCA
                from sklearn.manifold import TSNE

                X = np.array(embeddings)

                with st.spinner(f"Computing {viz_method} projection for {len(X)} document embeddings (768D → 2D)..."):
                    explained_var_text = ""
                    if viz_method == "PCA":
                        x_label = "Principal Component 1 (PC1)"
                        y_label = "Principal Component 2 (PC2)"
                        reducer = PCA(n_components=2, random_state=42)
                        coords = reducer.fit_transform(X)
                        var_ratio = reducer.explained_variance_ratio_
                        explained_var_text = f"Explained Variance: PC1 = **{var_ratio[0]*100:.1f}%**, PC2 = **{var_ratio[1]*100:.1f}%** (Total Captured: **{(var_ratio[0]+var_ratio[1])*100:.1f}%**)"
                    else:
                        x_label = "t-SNE Dimension 1"
                        y_label = "t-SNE Dimension 2"
                        perplexity = min(30, max(2, len(X) - 1))
                        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                        coords = reducer.fit_transform(X)

                viz_df = pd.DataFrame({
                    x_label: coords[:, 0],
                    y_label: coords[:, 1],
                    "ID": ids,
                    "Type": types_,
                })

                # Display summary metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Visualized Documents", len(viz_df))
                m2.metric("Original Dimension", "768D")
                m3.metric("Projected Dimension", "2D Plane")

                if explained_var_text:
                    st.caption(f"📊 {explained_var_text}")

                st.scatter_chart(
                    viz_df,
                    x=x_label,
                    y=y_label,
                    color="Type",
                    size=25,
                )

                # Axis & Scale explanation card directly under the chart
                with st.container():
                    col_ax_x, col_ax_y = st.columns(2)
                    if viz_method == "PCA":
                        with col_ax_x:
                            st.info(f"↔️ **X-Axis ({x_label}):** Captures the single largest axis of semantic variance ({var_ratio[0]*100:.1f}% of total information).")
                        with col_ax_y:
                            st.info(f"↕️ **Y-Axis ({y_label}):** Captures the second largest independent axis of variance ({var_ratio[1]*100:.1f}% of total information).")
                    else:
                        with col_ax_x:
                            st.info(f"↔️ **X-Axis ({x_label}):** Non-linear coordinate mapping local semantic neighborhoods.")
                        with col_ax_y:
                            st.info(f"↕️ **Y-Axis ({y_label}):** Non-linear coordinate. Note: scale values are arbitrary relative distances.")

                st.caption("💡 *Hover over points on the chart to inspect coordinates and document types.*")

        # --- AI Knowledge Base Topology & Quality Audit Section ---
        st.markdown("---")
        st.subheader("🤖 AI Knowledge Base Quality & Topology Audit")
        st.markdown(
            "Run an automated AI diagnostic across the vector database to analyze chunk quality, "
            "detect outliers and documentation blind spots, find redundant near-duplicates, and receive prioritized recommendations."
        )

        if st.button("🚀 Run AI Quality Audit & Recommendations", type="primary", key="btn_run_ai_quality_audit"):
            with st.spinner("Analyzing knowledge base topology, calculating redundancy & running AI audit..."):
                audit_res = run_embedding_quality_audit()

            if "error" in audit_res:
                st.error(audit_res["error"])
            else:
                m = audit_res["metrics"]
                st.success("✅ AI Audit Completed Successfully!")

                # Key Metric Tiles
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Total Chunks", m["total_documents"])
                k2.metric("Avg Chunk Length", f"{m['avg_char_length']} chars")
                k3.metric("Detected Outliers", m["outliers_count"], help="Documents >2σ distance from the corpus centroid.")
                k4.metric("Near-Duplicate Pairs", m["near_duplicate_pairs"], help="Document pairs with >95% cosine similarity.")

                # Diagnostic details
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.caption(f"📏 **Chunk Size Bounds:** Min `{m['min_char_length']}` chars | Max `{m['max_char_length']}` chars")
                    st.caption(f"⚠️ **Undersized (<100 chars):** `{m['short_chunks_count']}` | **Oversized (>1800 chars):** `{m['long_chunks_count']}`")
                with col_d2:
                    st.caption(f"🌐 **Corpus Centroid Dispersion:** `{m['mean_centroid_distance']}`")
                    st.caption(f"📂 **Document Types Breakdown:** {json.dumps(m['type_breakdown'])}")

                # Full AI Review & Actionable Recommendations Card
                st.markdown("### 📋 AI Audit Report & Actionable Recommendations")
                st.markdown(audit_res["ai_review"])

                # Expandable diagnostics for Outliers and Duplicates
                if audit_res["outliers"]:
                    with st.expander(f"📍 Inspect {len(audit_res['outliers'])} Outlier Documents (Isolated in Embedding Space)", expanded=False):
                        for o in audit_res["outliers"]:
                            st.markdown(f"**Document ID `{o['id']}`** | Type: `{o['doc_type']}` | Distance: `{o['distance']:.3f}`")
                            st.caption(f"Source: `{o['source']}`")
                            st.text_area("Snippet", o["snippet"], height=70, key=f"outlier_snip_{o['id']}", disabled=True)
                            st.markdown("---")

                if audit_res["near_duplicates"]:
                    with st.expander(f"🔄 Inspect {len(audit_res['near_duplicates'])} Near-Duplicate Overlap Pairs (>95% Similarity)", expanded=False):
                        for d in audit_res["near_duplicates"]:
                            st.markdown(f"**Pair `{d['id1']}` ↔ `{d['id2']}`** | Cosine Similarity: **{d['similarity']*100:.1f}%**")
                            c1, c2 = st.columns(2)
                            with c1:
                                st.caption(f"Doc `{d['id1']}` ({d['source1']})")
                                st.text_area("Doc 1", d["snippet1"], height=70, key=f"dup1_{d['id1']}_{d['id2']}", disabled=True)
                            with c2:
                                st.caption(f"Doc `{d['id2']}` ({d['source2']})")
                                st.text_area("Doc 2", d["snippet2"], height=70, key=f"dup2_{d['id1']}_{d['id2']}", disabled=True)
                            st.markdown("---")

        st.divider()

        # --- Bulk actions ---
        st.subheader("⚠️ Bulk Actions")
        col_del_id, col_clear = st.columns(2)

        with col_del_id:
            del_id = st.number_input("Delete document by ID", min_value=1, step=1, key="del_doc_id")
            if st.button("🗑️ Delete Document", key="btn_del_doc_id"):
                try:
                    with closing(get_connection()) as conn:
                        with conn.cursor() as cur:
                            cur.execute("DELETE FROM documents WHERE id = %s;", (del_id,))
                        conn.commit()
                    st.success(f"Deleted document ID {del_id}.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

        with col_clear:
            st.warning("This will permanently delete ALL documents.")
            if st.button("🗑️ Clear All Documents", key="btn_clear_all_docs"):
                st.session_state["confirm_clear"] = True

            if st.session_state.get("confirm_clear"):
                st.error("Are you sure? This cannot be undone.")
                col_yes, col_no = st.columns(2)
                with col_yes:
                    if st.button("Yes, delete everything", type="primary", key="btn_confirm_clear_docs"):
                        try:
                            with closing(get_connection()) as conn:
                                with conn.cursor() as cur:
                                    cur.execute("DELETE FROM documents;")
                                conn.commit()
                            st.success("All documents deleted.")
                            st.session_state["confirm_clear"] = False
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
                with col_no:
                    if st.button("Cancel", key="btn_cancel_clear_docs"):
                        st.session_state["confirm_clear"] = False
                        st.rerun()

    # -----------------------------------------------------------------------
    # Case 2: `query_feedback` table (User Ratings, Tags, Attachments & Gaps)
    # -----------------------------------------------------------------------
    elif selected_table == "query_feedback":
        st.subheader("Browse `query_feedback` Table")

        # Summary KPIs
        try:
            with closing(get_connection()) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT 
                            COUNT(*) AS total_count,
                            COUNT(*) FILTER (WHERE rating = 5) AS pos_count,
                            COUNT(*) FILTER (WHERE rating = 1) AS neg_count,
                            AVG(latency_ms) AS avg_lat
                        FROM query_feedback;
                        """
                    )
                    tot, pos, neg, avg_lat = cur.fetchone()
                    tot = tot or 0
                    pos = pos or 0
                    neg = neg or 0
                    sat_rate = (pos / tot * 100) if tot > 0 else 0.0

                    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                    kpi1.metric("Total Ratings", tot)
                    kpi2.metric("Positive (5★)", pos)
                    kpi3.metric("Negative (1★)", neg)
                    kpi4.metric("Satisfaction Rate", f"{sat_rate:.1f}%")
        except Exception:
            pass

        st.divider()

        col_fb_search, col_fb_filter, col_fb_limit = st.columns([3, 1, 1])
        with col_fb_search:
            fb_search_q = st.text_input("🔍 Search Feedback", placeholder="Search query, comments, gaps, tags...", key="fb_search_input")
        with col_fb_filter:
            fb_filter_sel = st.selectbox("Rating Filter", ["All", "5 ★ (Positive)", "1 ★ (Negative)"], key="fb_rating_filter_sel")
        with col_fb_limit:
            fb_limit = st.number_input("Max rows", min_value=10, max_value=1000, value=100, step=50, key="fb_max_rows")

        # Fetch feedback rows
        try:
            with closing(get_connection()) as conn:
                with conn.cursor() as cur:
                    where_clauses = []
                    params = []

                    if fb_filter_sel == "5 ★ (Positive)":
                        where_clauses.append("rating = 5")
                    elif fb_filter_sel == "1 ★ (Negative)":
                        where_clauses.append("rating = 1")

                    if fb_search_q.strip():
                        where_clauses.append("(query ILIKE %s OR response ILIKE %s OR user_comment ILIKE %s OR documentation_gaps ILIKE %s)")
                        term = f"%{fb_search_q.strip()}%"
                        params.extend([term, term, term, term])

                    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
                    params.append(fb_limit)

                    cur.execute(f"SELECT * FROM query_feedback {where_sql} ORDER BY created_at DESC LIMIT %s;", params)
                    rows = cur.fetchall()
                    cols = [desc[0] for desc in cur.description]

                    formatted_rows = []
                    for row in rows:
                        formatted_row = []
                        for val in row:
                            if isinstance(val, (bytes, memoryview)):
                                raw_b = bytes(val) if isinstance(val, memoryview) else val
                                b_len = len(raw_b)
                                formatted_row.append(f"📦 <binary {b_len} bytes>" if b_len > 0 else "<empty binary>")
                            elif isinstance(val, (dict, list)):
                                formatted_row.append(json.dumps(val))
                            elif isinstance(val, datetime):
                                formatted_row.append(val.isoformat())
                            else:
                                formatted_row.append(val)
                        formatted_rows.append(formatted_row)
                    table_df = pd.DataFrame(formatted_rows, columns=cols)
        except Exception as e:
            st.error(f"Error loading query_feedback: {e}")
            table_df = pd.DataFrame()

        if not table_df.empty:
            tbl_all_cols = list(table_df.columns)
            selected_tbl_cols = st.multiselect(
                "👁️ Visible Columns",
                options=tbl_all_cols,
                default=[c for c in ["id", "created_at", "query", "rating", "user_comment", "documentation_gaps", "attached_filename", "latency_ms"] if c in tbl_all_cols] or tbl_all_cols,
                key="fb_cols_picker",
            )
            st.dataframe(
                table_df[selected_tbl_cols if selected_tbl_cols else tbl_all_cols],
                use_container_width=True,
                hide_index=True,
            )

            # Record Inspector
            with st.expander("🔍 Inspect Full Feedback Record & Attachment", expanded=False):
                fb_ids = table_df["id"].tolist()
                selected_fb_id = st.selectbox("Select Feedback ID", fb_ids, key="fb_row_inspector_sel")
                if selected_fb_id:
                    try:
                        with closing(get_connection()) as conn:
                            with conn.cursor() as cur:
                                cur.execute("SELECT * FROM query_feedback WHERE id = %s;", (selected_fb_id,))
                                fb_row = cur.fetchone()
                                fb_cols = [d[0] for d in cur.description]
                                if fb_row:
                                    fb_data = dict(zip(fb_cols, fb_row))
                                    
                                    col_q_r1, col_q_r2 = st.columns([1, 1])
                                    with col_q_r1:
                                        st.markdown(f"**Query (ID #{fb_data.get('id')}):**")
                                        st.info(fb_data.get("query", ""))
                                        st.markdown(f"**Rating:** {'⭐' * int(fb_data.get('rating', 0))} ({fb_data.get('rating')}/5)")
                                        if fb_data.get("issue_tags"):
                                            st.markdown(f"**Issue Tags:** `{fb_data.get('issue_tags')}`")
                                        if fb_data.get("user_comment"):
                                            st.markdown(f"**User Comment:** {fb_data.get('user_comment')}")
                                        if fb_data.get("documentation_gaps"):
                                            st.markdown(f"**Documentation Gaps:**\n{fb_data.get('documentation_gaps')}")

                                    with col_q_r2:
                                        st.markdown("**Generated Response:**")
                                        st.text_area("Response", fb_data.get("response", ""), height=180, disabled=True)
                                        if fb_data.get("corrected_reference"):
                                            st.markdown("**Corrected Reference Answer:**")
                                            st.success(fb_data.get("corrected_reference"))

                                    # Attachment Preview & Download
                                    att_fn = fb_data.get("attached_filename")
                                    att_data = fb_data.get("attached_file_data")
                                    att_mime = fb_data.get("attached_file_mime")
                                    if att_fn and att_data:
                                        raw_bytes = bytes(att_data) if isinstance(att_data, memoryview) else att_data
                                        st.markdown(f"📎 **Attached File:** `{att_fn}` ({len(raw_bytes)} bytes)")
                                        st.download_button(
                                            label=f"⬇️ Download Attached File ({att_fn})",
                                            data=raw_bytes,
                                            file_name=att_fn,
                                            mime=att_mime or "application/octet-stream",
                                            key=f"dl_fb_att_{selected_fb_id}",
                                        )
                                        _render_attachment_preview(
                                            filename=att_fn,
                                            data=raw_bytes,
                                            mime_type=att_mime,
                                            label=f"👁️ Preview Attached File ({att_fn})",
                                            key_prefix=f"prev_db_{selected_fb_id}",
                                        )

                                    # Delete action
                                    st.divider()
                                    if st.button("🗑️ Delete Feedback Record", key=f"btn_del_fb_{selected_fb_id}", type="secondary"):
                                        with closing(get_connection()) as dconn:
                                            with dconn.cursor() as dcur:
                                                dcur.execute("DELETE FROM query_feedback WHERE id = %s;", (selected_fb_id,))
                                            dconn.commit()
                                        st.success(f"Feedback #{selected_fb_id} deleted successfully.")
                                        st.rerun()
                    except Exception as e:
                        st.error(f"Error loading feedback detail: {e}")
        else:
            st.info("No feedback entries found matching your filter.")

    # -----------------------------------------------------------------------
    # Case 3: `evaluation_runs` table (Benchmark Evaluation Histories)
    # -----------------------------------------------------------------------
    elif selected_table == "evaluation_runs":
        st.subheader("Browse `evaluation_runs` Table")
        col_eval_search, col_eval_limit = st.columns([3, 1])

        with col_eval_search:
            eval_search_q = st.text_input("🔍 Search Evaluation Runs", placeholder="Search run name, dataset, notes...", key="eval_search_input")
        with col_eval_limit:
            eval_limit = st.number_input("Max rows", min_value=10, max_value=1000, value=100, step=50, key="eval_max_rows")

        eval_df = _fetch_generic_table_data("evaluation_runs", limit=eval_limit, search_text=eval_search_q)

        if not eval_df.empty:
            eval_all_cols = list(eval_df.columns)
            selected_eval_cols = st.multiselect(
                "👁️ Visible Columns",
                options=eval_all_cols,
                default=[c for c in ["id", "created_at", "run_name", "dataset_name", "avg_faithfulness", "avg_answer_relevancy", "avg_context_precision", "avg_context_recall", "generation_model"] if c in eval_all_cols] or eval_all_cols,
                key="eval_cols_picker",
            )
            st.dataframe(
                eval_df[selected_eval_cols if selected_eval_cols else eval_all_cols],
                use_container_width=True,
                hide_index=True,
            )

            # Record Inspector
            with st.expander("🔍 Inspect Single Evaluation Run Detail", expanded=False):
                eval_ids = eval_df["id"].tolist()
                selected_eval_id = st.selectbox("Select Run ID", eval_ids, key="eval_row_inspector_sel")
                if selected_eval_id:
                    try:
                        with closing(get_connection()) as conn:
                            with conn.cursor() as cur:
                                cur.execute("SELECT * FROM evaluation_runs WHERE id = %s;", (selected_eval_id,))
                                ev_row = cur.fetchone()
                                ev_cols = [d[0] for d in cur.description]
                                if ev_row:
                                    ev_data = dict(zip(ev_cols, ev_row))
                                    st.markdown(f"### Run #{ev_data.get('id')}: `{ev_data.get('run_name')}`")
                                    st.caption(f"Executed on: {ev_data.get('created_at')} | Dataset: `{ev_data.get('dataset_name')}`")

                                    # Aggregated Scores
                                    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
                                    c_m1.metric("Faithfulness", _format_metric(ev_data.get("avg_faithfulness"), "faithfulness"))
                                    c_m2.metric("Answer Relevancy", _format_metric(ev_data.get("avg_answer_relevancy"), "answer_relevancy"))
                                    c_m3.metric("Context Precision", _format_metric(ev_data.get("avg_context_precision"), "context_precision"))
                                    c_m4.metric("Context Recall", _format_metric(ev_data.get("avg_context_recall"), "context_recall"))

                                    # Configuration parameters
                                    with st.expander("⚙️ Pipeline Configuration Parameters"):
                                        st.json({
                                            "use_hybrid": ev_data.get("use_hybrid"),
                                            "use_reranker": ev_data.get("use_reranker"),
                                            "pool_size": ev_data.get("pool_size"),
                                            "final_top_k": ev_data.get("final_top_k"),
                                            "chunk_size": ev_data.get("chunk_size"),
                                            "chunk_overlap": ev_data.get("chunk_overlap"),
                                            "embedding_model": ev_data.get("embedding_model"),
                                            "rerank_model": ev_data.get("rerank_model"),
                                            "generation_model": ev_data.get("generation_model"),
                                            "notes": ev_data.get("notes"),
                                        })

                                    # Detailed results breakdown
                                    det_results = ev_data.get("detailed_results")
                                    if det_results:
                                        if isinstance(det_results, str):
                                            try:
                                                det_results = json.loads(det_results)
                                            except Exception:
                                                pass
                                        if isinstance(det_results, list) and len(det_results) > 0:
                                            st.markdown(f"**Test Cases ({len(det_results)} cases):**")
                                            det_df = pd.DataFrame(det_results)
                                            st.dataframe(det_df, use_container_width=True, hide_index=True)

                                    # Delete action
                                    st.divider()
                                    if st.button("🗑️ Delete Evaluation Run", key=f"btn_del_ev_{selected_eval_id}", type="secondary"):
                                        with closing(get_connection()) as dconn:
                                            with dconn.cursor() as dcur:
                                                dcur.execute("DELETE FROM evaluation_runs WHERE id = %s;", (selected_eval_id,))
                                            dconn.commit()
                                        st.success(f"Evaluation Run #{selected_eval_id} deleted successfully.")
                                        st.rerun()
                    except Exception as e:
                        st.error(f"Error loading evaluation run: {e}")
        else:
            st.info("No evaluation runs found.")

    # -----------------------------------------------------------------------
    # Case 4: Generic / Other Tables (`alembic_version`, etc.)
    # -----------------------------------------------------------------------
    else:
        st.subheader(f"Browse `{selected_table}` Table")
        col_gt_search, col_gt_limit = st.columns([3, 1])

        with col_gt_search:
            gt_search_q = st.text_input(f"🔍 Search in `{selected_table}`", placeholder="Search text across columns...", key=f"search_{selected_table}")

        with col_gt_limit:
            gt_row_limit = st.number_input("Max rows", min_value=10, max_value=1000, value=100, step=50, key=f"limit_{selected_table}")

        table_df = _fetch_generic_table_data(selected_table, limit=gt_row_limit, search_text=gt_search_q)

        if not table_df.empty:
            tbl_all_cols = list(table_df.columns)
            selected_tbl_cols = st.multiselect(
                "👁️ Visible Columns",
                options=tbl_all_cols,
                default=tbl_all_cols,
                key=f"tbl_cols_picker_{selected_table}",
                help="Select which table columns to display in the grid.",
            )
            st.dataframe(
                table_df[selected_tbl_cols if selected_tbl_cols else tbl_all_cols],
                use_container_width=True,
                hide_index=True,
            )

            # Record inspector
            with st.expander(f"🔍 Inspect Single Row in `{selected_table}`", expanded=False):
                first_col = table_df.columns[0]
                row_identifiers = table_df[first_col].tolist()
                selected_val = st.selectbox(f"Select by `{first_col}`", row_identifiers, key=f"row_sel_{selected_table}")
                if selected_val is not None:
                    try:
                        with closing(get_connection()) as conn:
                            with conn.cursor() as cur:
                                cur.execute(f'SELECT * FROM "{selected_table}" WHERE "{first_col}" = %s LIMIT 1;', (selected_val,))
                                row = cur.fetchone()
                                cols = [d[0] for d in cur.description]
                                if row:
                                    row_dict = {}
                                    for c, v in zip(cols, row):
                                        if isinstance(v, (bytes, memoryview)):
                                            raw_b = bytes(v) if isinstance(v, memoryview) else v
                                            row_dict[c] = f"<binary {len(raw_b)} bytes>"
                                        elif isinstance(v, datetime):
                                            row_dict[c] = v.isoformat()
                                        elif isinstance(v, (dict, list)):
                                            row_dict[c] = v
                                        else:
                                            row_dict[c] = str(v) if not isinstance(v, (int, float, bool, type(None))) else v
                                    st.json(row_dict)

                                    if "id" in cols:
                                        if st.button(f"🗑️ Delete Row from `{selected_table}`", key=f"btn_del_gen_{selected_table}_{selected_val}"):
                                            with closing(get_connection()) as dconn:
                                                with dconn.cursor() as dcur:
                                                    dcur.execute(f'DELETE FROM "{selected_table}" WHERE id = %s;', (row_dict.get("id"),))
                                                dconn.commit()
                                            st.success(f"Deleted row ID {row_dict.get('id')}.")
                                            st.rerun()
                    except Exception as e:
                        st.error(f"Error loading row: {e}")
        else:
            st.info(f"Table `{selected_table}` is currently empty or no rows match your search query.")

