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

