"""Persistence layer for saving and retrieving evaluation run results."""

import json
import logging
from contextlib import closing
from datetime import datetime

import psycopg
from src.db import get_connection

logger = logging.getLogger(__name__)


def save_evaluation_run(
    run_name: str,
    use_hybrid: bool,
    use_reranker: bool,
    pool_size: int,
    final_top_k: int,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
    rerank_model: str,
    generation_model: str,
    dataset_name: str,
    dataset_cases: list[dict],
    avg_faithfulness: float | None,
    avg_answer_relevancy: float | None,
    avg_context_precision: float | None,
    avg_context_recall: float | None,
    detailed_results: list[dict],
    notes: str = "",
) -> int:
    """Insert a new evaluation run record and return its ID."""
    logger.info(f"Saving evaluation run '{run_name}'...")
    with closing(get_connection()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO evaluation_runs (
                    run_name,
                    use_hybrid, use_reranker, pool_size, final_top_k,
                    chunk_size, chunk_overlap,
                    embedding_model, rerank_model, generation_model,
                    dataset_name, dataset_cases,
                    avg_faithfulness, avg_answer_relevancy,
                    avg_context_precision, avg_context_recall,
                    detailed_results, notes
                ) VALUES (
                    %s,
                    %s, %s, %s, %s,
                    %s, %s,
                    %s, %s, %s,
                    %s, %s,
                    %s, %s, %s, %s,
                    %s, %s
                )
                RETURNING id;
                """,
                (
                    run_name,
                    use_hybrid, use_reranker, pool_size, final_top_k,
                    chunk_size, chunk_overlap,
                    embedding_model, rerank_model, generation_model,
                    dataset_name, psycopg.types.json.Jsonb(dataset_cases),
                    avg_faithfulness, avg_answer_relevancy,
                    avg_context_precision, avg_context_recall,
                    psycopg.types.json.Jsonb(detailed_results), notes,
                ),
            )
            run_id = cur.fetchone()[0]
        conn.commit()
    logger.info(f"Evaluation run saved with ID={run_id}.")
    return run_id


def list_evaluation_runs() -> list[dict]:
    """Return all evaluation runs ordered by most recent first (summary only)."""
    try:
        with closing(get_connection()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, run_name, created_at,
                           use_hybrid, use_reranker, pool_size, final_top_k,
                           chunk_size, chunk_overlap,
                           embedding_model, rerank_model, generation_model,
                           dataset_name,
                           avg_faithfulness, avg_answer_relevancy,
                           avg_context_precision, avg_context_recall,
                           notes
                    FROM evaluation_runs
                    ORDER BY created_at DESC;
                    """
                )
                columns = [
                    "id", "run_name", "created_at",
                    "use_hybrid", "use_reranker", "pool_size", "final_top_k",
                    "chunk_size", "chunk_overlap",
                    "embedding_model", "rerank_model", "generation_model",
                    "dataset_name",
                    "avg_faithfulness", "avg_answer_relevancy",
                    "avg_context_precision", "avg_context_recall",
                    "notes",
                ]
                rows = cur.fetchall()
                return [
                    {col: (val.isoformat() if isinstance(val, datetime) else val)
                     for col, val in zip(columns, row)}
                    for row in rows
                ]
    except Exception as e:
        logger.warning(f"Could not list evaluation_runs (table may not exist yet): {e}")
        return []


def get_evaluation_run(run_id: int) -> dict | None:
    """Return full details of a single evaluation run including detailed_results."""
    try:
        with closing(get_connection()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, run_name, created_at,
                           use_hybrid, use_reranker, pool_size, final_top_k,
                           chunk_size, chunk_overlap,
                           embedding_model, rerank_model, generation_model,
                           dataset_name, dataset_cases,
                           avg_faithfulness, avg_answer_relevancy,
                           avg_context_precision, avg_context_recall,
                           detailed_results, notes
                    FROM evaluation_runs
                    WHERE id = %s;
                    """,
                    (run_id,),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                columns = [
                    "id", "run_name", "created_at",
                    "use_hybrid", "use_reranker", "pool_size", "final_top_k",
                    "chunk_size", "chunk_overlap",
                    "embedding_model", "rerank_model", "generation_model",
                    "dataset_name", "dataset_cases",
                    "avg_faithfulness", "avg_answer_relevancy",
                    "avg_context_precision", "avg_context_recall",
                    "detailed_results", "notes",
                ]
                result = {}
                for col, val in zip(columns, row):
                    if isinstance(val, datetime):
                        result[col] = val.isoformat()
                    else:
                        result[col] = val
                return result
    except Exception as e:
        logger.warning(f"Could not fetch evaluation run #{run_id}: {e}")
        return None


def delete_evaluation_run(run_id: int) -> bool:
    """Delete an evaluation run by ID. Returns True if a row was deleted."""
    with closing(get_connection()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM evaluation_runs WHERE id = %s;",
                (run_id,),
            )
            deleted = cur.rowcount > 0
        conn.commit()
    if deleted:
        logger.info(f"Deleted evaluation run ID={run_id}.")
    return deleted
