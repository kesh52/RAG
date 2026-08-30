"""Database schema introspection, query helpers, and vector visualization extractors."""

import json
from contextlib import closing
from datetime import datetime
import streamlit as st
import pandas as pd
from src.db import get_connection


def _get_doc_count() -> int:
    """Return total row count from the documents table."""
    try:
        with closing(get_connection()) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM documents;")
                return cur.fetchone()[0]
    except Exception:
        return 0


def _get_doc_type_breakdown() -> dict:
    """Return document counts grouped by metadata->'type'."""
    try:
        with closing(get_connection()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COALESCE(metadata->>'type', 'unknown') AS doc_type, COUNT(*)
                    FROM documents
                    GROUP BY doc_type
                    ORDER BY COUNT(*) DESC;
                    """
                )
                return {row[0]: row[1] for row in cur.fetchall()}
    except Exception:
        return {}


def _fetch_documents(limit: int = 200, search_query: str = "", doc_type_filter: str = "All") -> pd.DataFrame:
    """Fetch documents from the DB with optional text search and type filter, returning all table columns."""
    try:
        with closing(get_connection()) as conn:
            with conn.cursor() as cur:
                conditions = []
                params: list = []

                if search_query.strip():
                    conditions.append("text_search_tsv @@ plainto_tsquery('english', %s)")
                    params.append(search_query.strip())

                if doc_type_filter and doc_type_filter != "All":
                    conditions.append("metadata->>'type' = %s")
                    params.append(doc_type_filter)

                where_clause = ""
                if conditions:
                    where_clause = "WHERE " + " AND ".join(conditions)

                params.append(limit)
                cur.execute(
                    f"""
                    SELECT id, content, metadata, 
                           COALESCE(array_length(string_to_array(embedding::text, ','), 1), 0) AS embedding_dims,
                           text_search_tsv::text
                    FROM documents
                    {where_clause}
                    ORDER BY id
                    LIMIT %s;
                    """,
                    params,
                )
                rows = cur.fetchall()
                cols = ["id", "content", "metadata", "embedding_dims", "text_search_tsv"]
                formatted_rows = []
                for row in rows:
                    formatted_row = []
                    for c, val in zip(cols, row):
                        if c == "metadata" and isinstance(val, dict):
                            formatted_row.append(json.dumps(val))
                        elif c == "embedding_dims":
                            formatted_row.append(f"vector[{val}]")
                        else:
                            formatted_row.append(val)
                    formatted_rows.append(formatted_row)
                return pd.DataFrame(formatted_rows, columns=cols)
    except Exception as e:
        st.error(f"Failed to query documents: {e}")
        return pd.DataFrame()


def _fetch_embeddings_for_viz(limit: int = 500) -> tuple[list, list, list]:
    """Fetch raw embeddings for dimensionality reduction visualization."""
    try:
        with closing(get_connection()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, embedding::text, COALESCE(metadata->>'type', 'unknown')
                    FROM documents
                    WHERE embedding IS NOT NULL
                    LIMIT %s;
                    """,
                    (limit,),
                )
                ids = []
                embeddings = []
                types_ = []
                for row in cur.fetchall():
                    ids.append(row[0])
                    # Parse the vector string "[0.1,0.2,...]" into a list of floats
                    vec_str = row[1].strip("[]")
                    if vec_str:
                        embeddings.append([float(x) for x in vec_str.split(",")])
                        types_.append(row[2])
                return ids, embeddings, types_
    except Exception:
        return [], [], []


def _list_database_tables() -> list[str]:
    """Discover all public tables in the PostgreSQL database."""
    try:
        with closing(get_connection()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                      AND table_type = 'BASE TABLE'
                    ORDER BY table_name;
                    """
                )
                tables = [r[0] for r in cur.fetchall()]
                priority = ["documents", "query_feedback", "evaluation_runs", "alembic_version"]
                ordered = [t for t in priority if t in tables] + [t for t in tables if t not in priority]
                return ordered if ordered else ["documents", "query_feedback", "evaluation_runs"]
    except Exception:
        return ["documents", "query_feedback", "evaluation_runs", "alembic_version"]


def _get_table_count(table_name: str) -> int:
    """Return total row count for a specific table."""
    try:
        with closing(get_connection()) as conn:
            with conn.cursor() as cur:
                # Sanitize table_name against schema
                cur.execute(
                    """
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = %s;
                    """,
                    (table_name,),
                )
                if not cur.fetchone():
                    return 0
                cur.execute(f'SELECT COUNT(*) FROM "{table_name}";')
                return cur.fetchone()[0]
    except Exception:
        return 0


def _fetch_table_schema(table_name: str) -> pd.DataFrame:
    """Fetch columns and data types for a table."""
    try:
        with closing(get_connection()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = %s
                    ORDER BY ordinal_position;
                    """,
                    (table_name,),
                )
                rows = cur.fetchall()
                return pd.DataFrame(rows, columns=["Column Name", "Data Type", "Nullable"])
    except Exception:
        return pd.DataFrame()


def _fetch_generic_table_data(table_name: str, limit: int = 100, search_text: str = "") -> pd.DataFrame:
    """Fetch generic table rows as a pandas DataFrame."""
    try:
        with closing(get_connection()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = %s;
                    """,
                    (table_name,),
                )
                if not cur.fetchone():
                    return pd.DataFrame()

                # Check available columns for intelligent ordering
                cur.execute(
                    """
                    SELECT column_name FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = %s;
                    """,
                    (table_name,),
                )
                cols_available = {r[0] for r in cur.fetchall()}
                order_clause = ""
                if "created_at" in cols_available:
                    order_clause = "ORDER BY created_at DESC"
                elif "id" in cols_available:
                    order_clause = "ORDER BY id DESC"

                cur.execute(f'SELECT * FROM "{table_name}" {order_clause} LIMIT %s;', (limit,))
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
                df = pd.DataFrame(formatted_rows, columns=cols)
                if search_text.strip() and not df.empty:
                    mask = df.astype(str).apply(lambda row: row.str.contains(search_text, case=False, na=False)).any(axis=1)
                    df = df[mask]
                return df
    except Exception as e:
        st.error(f"Error reading table `{table_name}`: {e}")
        return pd.DataFrame()


def run_embedding_quality_audit() -> dict:
    """
    Perform statistical, topological, and AI-powered quality audit of the vector database.
    Returns calculated metrics, outlier/redundancy diagnostics, and actionable recommendations.
    """
    import numpy as np
    from google import genai
    from google.genai import types as genai_types
    from src.utils.config import config

    # 1. Fetch document data and embeddings
    doc_records = []
    try:
        with closing(get_connection()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, content, COALESCE(metadata->>'type', 'general') AS doc_type, 
                           COALESCE(metadata->>'source_url', metadata->>'title', 'Unknown') AS source,
                           embedding::text
                    FROM documents 
                    WHERE embedding IS NOT NULL;
                    """
                )
                rows = cur.fetchall()
                for r in rows:
                    if r[4]:
                        vec = [float(x) for x in r[4].strip("[]").split(",")]
                        doc_records.append({
                            "id": r[0],
                            "content": r[1],
                            "doc_type": r[2],
                            "source": r[3],
                            "embedding": vec,
                            "length": len(r[1]),
                        })
    except Exception as e:
        return {"error": f"Database query failed: {e}"}

    if len(doc_records) < 3:
        return {"error": "Need at least 3 embedded documents in the database to run an AI audit."}

    # 2. Statistical & Topological Calculations
    embeddings = np.array([d["embedding"] for d in doc_records])
    lengths = np.array([d["length"] for d in doc_records])

    # Centroid & Distance Distribution (Outliers)
    centroid = np.mean(embeddings, axis=0)
    distances_to_center = np.linalg.norm(embeddings - centroid, axis=1)
    mean_dist = float(np.mean(distances_to_center))
    std_dist = float(np.std(distances_to_center))
    outlier_threshold = mean_dist + 2.0 * std_dist

    outliers = []
    for idx, d in enumerate(doc_records):
        if distances_to_center[idx] > outlier_threshold:
            outliers.append({
                "id": d["id"],
                "doc_type": d["doc_type"],
                "source": d["source"],
                "distance": float(distances_to_center[idx]),
                "snippet": d["content"][:250],
            })

    # Redundancy / Near-Duplicate Detection (Cosine similarity > 0.96)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / (norms + 1e-10)
    sim_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
    np.fill_diagonal(sim_matrix, 0.0)

    near_duplicates = []
    seen_pairs = set()
    for i in range(len(doc_records)):
        for j in range(i + 1, len(doc_records)):
            sim = float(sim_matrix[i, j])
            if sim > 0.95:
                pair_key = tuple(sorted([doc_records[i]["id"], doc_records[j]["id"]]))
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    near_duplicates.append({
                        "id1": doc_records[i]["id"],
                        "source1": doc_records[i]["source"],
                        "snippet1": doc_records[i]["content"][:200],
                        "id2": doc_records[j]["id"],
                        "source2": doc_records[j]["source"],
                        "snippet2": doc_records[j]["content"][:200],
                        "similarity": round(sim, 3),
                    })

    # Chunk Length Stats
    short_chunks = [d for d in doc_records if d["length"] < 100]
    long_chunks = [d for d in doc_records if d["length"] > 1800]

    # Type breakdown
    type_counts = {}
    for d in doc_records:
        type_counts[d["doc_type"]] = type_counts.get(d["doc_type"], 0) + 1

    metrics = {
        "total_documents": len(doc_records),
        "avg_char_length": int(np.mean(lengths)),
        "min_char_length": int(np.min(lengths)),
        "max_char_length": int(np.max(lengths)),
        "short_chunks_count": len(short_chunks),
        "long_chunks_count": len(long_chunks),
        "outliers_count": len(outliers),
        "near_duplicate_pairs": len(near_duplicates),
        "type_breakdown": type_counts,
        "mean_centroid_distance": round(mean_dist, 3),
    }

    # 3. AI Analysis Prompt Construction
    prompt = f"""You are a Principal AI Data Architect and Retrieval-Augmented Generation (RAG) Specialist.
Perform a thorough Quality & Topological Audit of the knowledge base embedding space based on the computed metrics and data samples below.

### Knowledge Base Metrics:
- Total Embedded Chunks: {metrics['total_documents']}
- Average Chunk Length: {metrics['avg_char_length']} characters (Range: {metrics['min_char_length']} - {metrics['max_char_length']})
- Undersized Chunks (<100 chars): {metrics['short_chunks_count']}
- Oversized Chunks (>1800 chars): {metrics['long_chunks_count']}
- Detected Outliers (>2σ distance): {metrics['outliers_count']}
- Highly Redundant / Near-Duplicate Pairs (>95% similarity): {metrics['near_duplicate_pairs']}
- Document Types: {json.dumps(type_counts)}

### Sample Outliers (Isolated/Distinct Chunks):
{json.dumps([{"id": o["id"], "type": o["doc_type"], "snippet": o["snippet"]} for o in outliers[:3]], indent=2) if outliers else "No severe outliers detected."}

### Sample Near-Duplicate Overlaps:
{json.dumps([{"pair": [d["id1"], d["id2"]], "sim": d["similarity"], "snippet": d["snippet1"]} for d in near_duplicates[:3]], indent=2) if near_duplicates else "No severe redundancies detected."}

### Chunk Samples across corpus:
{json.dumps([{"id": d["id"], "type": d["doc_type"], "snippet": d["content"][:150]} for d in doc_records[:4]], indent=2)}

---
### Your Task:
Produce a structured, executive Markdown report evaluating the knowledge base quality for hybrid RAG retrieval.
Include:
1. **📊 Health Score**: Provide an overall rating from 0 to 100 with a 1-sentence executive verdict.
2. **🔍 Semantic Topology & Dispersion**: Evaluate if the knowledge base is balanced, too fragmented, or heavily skewed toward one topic.
3. **⚠️ Quality & Chunking Diagnostics**: Critique chunk sizes, potential fragmentation, boilerplate noise, or loss of context.
4. **🔄 Redundancy & Coverage Analysis**: Analyze overlapping topics vs potential knowledge blind spots.
5. **💡 Actionable Recommendations**: Provide 3 to 5 concrete, prioritized steps for the engineering team (e.g. chunking parameter adjustments, document deduplication, runbook enrichment).
"""

    try:
        client = genai.Client(
            project=config.get("gcp.project"),
            location=config.get("gcp.location"),
        )
        response = client.models.generate_content(
            model=config.get("models.generation", "gemini-2.5-flash"),
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=1500,
            ),
        )
        ai_review = response.text.strip()
    except Exception as err:
        ai_review = f"⚠️ Could not generate AI audit: {err}"

    return {
        "metrics": metrics,
        "ai_review": ai_review,
        "outliers": outliers,
        "near_duplicates": near_duplicates,
    }

