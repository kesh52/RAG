"""Persistence layer and analytics for interactive RAG query feedback."""

import os
import json
import logging
from contextlib import closing
from datetime import datetime
import psycopg

from src.db import get_connection

logger = logging.getLogger(__name__)

DATASETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "evaluation",
    "datasets",
)


def save_feedback(
    query: str,
    response: str,
    retrieved_contexts: list[str],
    sources: list[str],
    use_hybrid: bool,
    use_reranker: bool,
    pool_size: int,
    final_top_k: int,
    generation_model: str,
    latency_ms: int,
    rating: int,
    issue_tags: list[str] | None = None,
    corrected_reference: str | None = None,
    user_comment: str | None = None,
    attached_filename: str | None = None,
    attached_file_data: bytes | None = None,
    attached_file_mime: str | None = None,
    documentation_gaps: str | None = None,
) -> int:
    """Insert a new user query rating/feedback record and return its ID."""
    if issue_tags is None:
        issue_tags = []

    logger.info(f"Saving query feedback (rating={rating}) for query: '{query[:50]}...' (attached: {attached_filename})")
    with closing(get_connection()) as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    """
                    INSERT INTO query_feedback (
                        query, response, retrieved_contexts, sources,
                        use_hybrid, use_reranker, pool_size, final_top_k,
                        generation_model, latency_ms,
                        rating, issue_tags, corrected_reference, user_comment,
                        attached_filename, attached_file_data, attached_file_mime,
                        documentation_gaps
                    ) VALUES (
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s,
                        %s
                    )
                    RETURNING id;
                    """,
                    (
                        query,
                        response,
                        psycopg.types.json.Jsonb(retrieved_contexts),
                        psycopg.types.json.Jsonb(sources),
                        use_hybrid,
                        use_reranker,
                        pool_size,
                        final_top_k,
                        generation_model,
                        latency_ms,
                        rating,
                        psycopg.types.json.Jsonb(issue_tags),
                        corrected_reference,
                        user_comment,
                        attached_filename,
                        attached_file_data,
                        attached_file_mime,
                        documentation_gaps,
                    ),
                )
            except Exception:
                # Fallback if migration for newer columns has not been run yet
                conn.rollback()
                try:
                    cur.execute(
                        """
                        INSERT INTO query_feedback (
                            query, response, retrieved_contexts, sources,
                            use_hybrid, use_reranker, pool_size, final_top_k,
                            generation_model, latency_ms,
                            rating, issue_tags, corrected_reference, user_comment,
                            attached_filename, attached_file_data, attached_file_mime
                        ) VALUES (
                            %s, %s, %s, %s,
                            %s, %s, %s, %s,
                            %s, %s,
                            %s, %s, %s, %s,
                            %s, %s, %s
                        )
                        RETURNING id;
                        """,
                        (
                            query,
                            response,
                            psycopg.types.json.Jsonb(retrieved_contexts),
                            psycopg.types.json.Jsonb(sources),
                            use_hybrid,
                            use_reranker,
                            pool_size,
                            final_top_k,
                            generation_model,
                            latency_ms,
                            rating,
                            psycopg.types.json.Jsonb(issue_tags),
                            corrected_reference,
                            user_comment,
                            attached_filename,
                            attached_file_data,
                            attached_file_mime,
                        ),
                    )
                except Exception:
                    conn.rollback()
                    cur.execute(
                        """
                        INSERT INTO query_feedback (
                            query, response, retrieved_contexts, sources,
                            use_hybrid, use_reranker, pool_size, final_top_k,
                            generation_model, latency_ms,
                            rating, issue_tags, corrected_reference, user_comment
                        ) VALUES (
                            %s, %s, %s, %s,
                            %s, %s, %s, %s,
                            %s, %s,
                            %s, %s, %s, %s
                        )
                        RETURNING id;
                        """,
                        (
                            query,
                            response,
                            psycopg.types.json.Jsonb(retrieved_contexts),
                            psycopg.types.json.Jsonb(sources),
                            use_hybrid,
                            use_reranker,
                            pool_size,
                            final_top_k,
                            generation_model,
                            latency_ms,
                            rating,
                            psycopg.types.json.Jsonb(issue_tags),
                            corrected_reference,
                            user_comment,
                        ),
                    )
            feedback_id = cur.fetchone()[0]
        conn.commit()
    logger.info(f"Query feedback saved with ID={feedback_id}.")
    return feedback_id


def list_feedback(
    limit: int = 100,
    rating_filter: int | None = None,
) -> list[dict]:
    """Retrieve saved feedback entries with optional filtering."""
    try:
        with closing(get_connection()) as conn:
            with conn.cursor() as cur:
                conditions = []
                params = []

                if rating_filter is not None:
                    conditions.append("rating = %s")
                    params.append(rating_filter)

                where_clause = ""
                if conditions:
                    where_clause = "WHERE " + " AND ".join(conditions)

                params.append(limit)

                try:
                    cur.execute(
                        f"""
                        SELECT id, created_at, query, LEFT(response, 150) AS response_preview,
                                rating, issue_tags, corrected_reference, user_comment,
                                latency_ms, is_promoted_to_benchmark, attached_filename, documentation_gaps
                        FROM query_feedback
                        {where_clause}
                        ORDER BY created_at DESC
                        LIMIT %s;
                        """,
                        params,
                    )
                    columns = [
                        "id",
                        "created_at",
                        "query",
                        "response_preview",
                        "rating",
                        "issue_tags",
                        "corrected_reference",
                        "user_comment",
                        "latency_ms",
                        "is_promoted_to_benchmark",
                        "attached_filename",
                        "documentation_gaps",
                    ]
                except Exception:
                    conn.rollback()
                    try:
                        cur.execute(
                            f"""
                            SELECT id, created_at, query, LEFT(response, 150) AS response_preview,
                                    rating, issue_tags, corrected_reference, user_comment,
                                    latency_ms, is_promoted_to_benchmark, attached_filename
                            FROM query_feedback
                            {where_clause}
                            ORDER BY created_at DESC
                            LIMIT %s;
                            """,
                            params,
                        )
                        columns = [
                            "id",
                            "created_at",
                            "query",
                            "response_preview",
                            "rating",
                            "issue_tags",
                            "corrected_reference",
                            "user_comment",
                            "latency_ms",
                            "is_promoted_to_benchmark",
                            "attached_filename",
                        ]
                    except Exception:
                        conn.rollback()
                        cur.execute(
                            f"""
                            SELECT id, created_at, query, LEFT(response, 150) AS response_preview,
                                    rating, issue_tags, corrected_reference, user_comment,
                                    latency_ms, is_promoted_to_benchmark
                            FROM query_feedback
                            {where_clause}
                            ORDER BY created_at DESC
                            LIMIT %s;
                            """,
                            params,
                        )
                        columns = [
                            "id",
                            "created_at",
                            "query",
                            "response_preview",
                            "rating",
                            "issue_tags",
                            "corrected_reference",
                            "user_comment",
                            "latency_ms",
                            "is_promoted_to_benchmark",
                        ]

                rows = cur.fetchall()
                return [
                    {
                        col: (val.isoformat() if isinstance(val, datetime) else val)
                        for col, val in zip(columns, row)
                    }
                    for row in rows
                ]
    except Exception as e:
        logger.warning(f"Unable to query query_feedback table (table may not be created yet): {e}")
        return []


def get_feedback_by_id(feedback_id: int) -> dict | None:
    """Fetch complete detail for a specific feedback record."""
    try:
        with closing(get_connection()) as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        """
                        SELECT id, created_at, query, response, retrieved_contexts, sources,
                               use_hybrid, use_reranker, pool_size, final_top_k,
                               generation_model, latency_ms, rating, issue_tags,
                               corrected_reference, user_comment, is_promoted_to_benchmark,
                               attached_filename, attached_file_data, attached_file_mime,
                               documentation_gaps
                        FROM query_feedback
                        WHERE id = %s;
                        """,
                        (feedback_id,),
                    )
                    row = cur.fetchone()
                    if not row:
                        return None
                    columns = [
                        "id",
                        "created_at",
                        "query",
                        "response",
                        "retrieved_contexts",
                        "sources",
                        "use_hybrid",
                        "use_reranker",
                        "pool_size",
                        "final_top_k",
                        "generation_model",
                        "latency_ms",
                        "rating",
                        "issue_tags",
                        "corrected_reference",
                        "user_comment",
                        "is_promoted_to_benchmark",
                        "attached_filename",
                        "attached_file_data",
                        "attached_file_mime",
                        "documentation_gaps",
                    ]
                except Exception:
                    conn.rollback()
                    try:
                        cur.execute(
                            """
                            SELECT id, created_at, query, response, retrieved_contexts, sources,
                                   use_hybrid, use_reranker, pool_size, final_top_k,
                                   generation_model, latency_ms, rating, issue_tags,
                                   corrected_reference, user_comment, is_promoted_to_benchmark,
                                   attached_filename, attached_file_data, attached_file_mime
                            FROM query_feedback
                            WHERE id = %s;
                            """,
                            (feedback_id,),
                        )
                        row = cur.fetchone()
                        if not row:
                            return None
                        columns = [
                            "id",
                            "created_at",
                            "query",
                            "response",
                            "retrieved_contexts",
                            "sources",
                            "use_hybrid",
                            "use_reranker",
                            "pool_size",
                            "final_top_k",
                            "generation_model",
                            "latency_ms",
                            "rating",
                            "issue_tags",
                            "corrected_reference",
                            "user_comment",
                            "is_promoted_to_benchmark",
                            "attached_filename",
                            "attached_file_data",
                            "attached_file_mime",
                        ]
                    except Exception:
                        conn.rollback()
                        cur.execute(
                            """
                            SELECT id, created_at, query, response, retrieved_contexts, sources,
                                   use_hybrid, use_reranker, pool_size, final_top_k,
                                   generation_model, latency_ms, rating, issue_tags,
                                   corrected_reference, user_comment, is_promoted_to_benchmark
                            FROM query_feedback
                            WHERE id = %s;
                            """,
                            (feedback_id,),
                        )
                        row = cur.fetchone()
                        if not row:
                            return None
                        columns = [
                            "id",
                            "created_at",
                            "query",
                            "response",
                            "retrieved_contexts",
                            "sources",
                            "use_hybrid",
                            "use_reranker",
                            "pool_size",
                            "final_top_k",
                            "generation_model",
                            "latency_ms",
                            "rating",
                            "issue_tags",
                            "corrected_reference",
                            "user_comment",
                            "is_promoted_to_benchmark",
                        ]

                res = {}
                for col, val in zip(columns, row):
                    if isinstance(val, datetime):
                        res[col] = val.isoformat()
                    elif isinstance(val, memoryview):
                        res[col] = bytes(val)
                    else:
                        res[col] = val
                return res
    except Exception as e:
        logger.warning(f"Unable to get feedback #{feedback_id}: {e}")
        return None


def delete_feedback(feedback_id: int) -> bool:
    """Delete a feedback record by ID."""
    with closing(get_connection()) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM query_feedback WHERE id = %s;", (feedback_id,))
            deleted = cur.rowcount > 0
        conn.commit()
    return deleted


def get_feedback_attachment(feedback_id: int) -> tuple[bytes | None, str | None, str | None]:
    """Lazy-load only the binary attachment data for a feedback record on demand."""
    try:
        with closing(get_connection()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT attached_file_data, attached_filename, attached_file_mime
                    FROM query_feedback
                    WHERE id = %s;
                    """,
                    (feedback_id,),
                )
                row = cur.fetchone()
                if row:
                    data = bytes(row[0]) if isinstance(row[0], memoryview) else row[0]
                    return data, row[1], row[2]
    except Exception as e:
        logger.warning(f"Could not load attachment for feedback #{feedback_id}: {e}")
    return None, None, None


def get_feedback_analytics() -> dict:
    """Compute aggregate analytics on user ratings and issue taxonomy."""
    empty_stats = {
        "total_ratings": 0,
        "positive_count": 0,
        "negative_count": 0,
        "satisfaction_rate": 0.0,
        "avg_latency_ms": 0.0,
        "tag_breakdown": {},
        "table_exists": False,
    }
    try:
        with closing(get_connection()) as conn:
            with conn.cursor() as cur:
                # Overall counts
                cur.execute(
                    """
                    SELECT 
                        COUNT(*) AS total_count,
                        COUNT(*) FILTER (WHERE rating = 5) AS positive_count,
                        COUNT(*) FILTER (WHERE rating = 1) AS negative_count,
                        AVG(latency_ms) AS avg_latency_ms
                    FROM query_feedback;
                    """
                )
                total, pos, neg, avg_lat = cur.fetchone()
                total = total or 0
                pos = pos or 0
                neg = neg or 0
                avg_lat = float(avg_lat) if avg_lat is not None else 0.0

                # Issue tags breakdown
                cur.execute(
                    """
                    SELECT tag, COUNT(*) AS count
                    FROM query_feedback,
                         jsonb_array_elements_text(issue_tags) AS tag
                    GROUP BY tag
                    ORDER BY count DESC;
                    """
                )
                tag_counts = {row[0]: row[1] for row in cur.fetchall()}

                satisfaction_rate = (pos / total * 100) if total > 0 else 0.0

                return {
                    "total_ratings": total,
                    "positive_count": pos,
                    "negative_count": neg,
                    "satisfaction_rate": satisfaction_rate,
                    "avg_latency_ms": avg_lat,
                    "tag_breakdown": tag_counts,
                    "table_exists": True,
                }
    except Exception as e:
        logger.warning(f"query_feedback table not accessible (run migrations to create): {e}")
        return empty_stats


def update_feedback_correction(
    feedback_id: int,
    corrected_reference: str | None = None,
    user_comment: str | None = None,
) -> bool:
    """Update the corrected reference answer or comment for an existing feedback entry."""
    try:
        with closing(get_connection()) as conn:
            with conn.cursor() as cur:
                updates = []
                params = []
                if corrected_reference is not None:
                    updates.append("corrected_reference = %s")
                    params.append(corrected_reference.strip() if corrected_reference.strip() else None)
                if user_comment is not None:
                    updates.append("user_comment = %s")
                    params.append(user_comment.strip() if user_comment.strip() else None)

                if not updates:
                    return False

                params.append(feedback_id)
                query = f"UPDATE query_feedback SET {', '.join(updates)} WHERE id = %s;"
                cur.execute(query, params)
                updated = cur.rowcount > 0
            conn.commit()
        return updated
    except Exception as e:
        logger.error(f"Failed updating feedback #{feedback_id}: {e}")
        return False


def promote_to_benchmark(
    feedback_id: int,
    dataset_name: str = "default",
    override_reference: str | None = None,
) -> dict:
    """Export a rated query and its reference answer into the benchmark dataset JSON."""
    detail = get_feedback_by_id(feedback_id)
    if not detail:
        raise ValueError(f"Feedback ID {feedback_id} not found.")

    query = detail.get("query")
    # Priority: Explicit override -> Corrected reference in DB -> generated response
    if override_reference and override_reference.strip():
        reference = override_reference.strip()
        # Also persist the override back to DB
        update_feedback_correction(feedback_id, corrected_reference=reference)
    else:
        reference = detail.get("corrected_reference") or detail.get("response")

    if not query or not reference:
        raise ValueError("Both query and reference text are required for benchmark inclusion.")

    os.makedirs(DATASETS_DIR, exist_ok=True)
    dataset_path = os.path.join(DATASETS_DIR, f"{dataset_name}.json")

    cases = []
    if os.path.isfile(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            try:
                cases = json.load(f)
            except Exception:
                cases = []

    # Check if duplicate query already exists
    exists_idx = None
    for idx, c in enumerate(cases):
        if c.get("query", "").strip().lower() == query.strip().lower():
            exists_idx = idx
            break

    if exists_idx is not None:
        # Update existing case with new reference
        cases[exists_idx]["reference"] = reference.strip()
        is_new = False
    else:
        cases.append({
            "query": query.strip(),
            "reference": reference.strip(),
        })
        is_new = True

    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=4)

    # Mark as promoted in database
    try:
        with closing(get_connection()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE query_feedback SET is_promoted_to_benchmark = TRUE WHERE id = %s;",
                    (feedback_id,),
                )
            conn.commit()
    except Exception as e:
        logger.warning(f"Could not update is_promoted_to_benchmark in DB: {e}")

    return {
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "total_cases": len(cases),
        "is_new": is_new,
    }

