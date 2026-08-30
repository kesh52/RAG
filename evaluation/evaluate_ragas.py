import sys
import os
import json
import types
from langchain_google_vertexai import ChatVertexAI

# Create legacy alias path in memory
mod = types.ModuleType("langchain_community.chat_models.vertexai")
mod.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = mod

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from src.pipeline import get_default_pipeline, RAGPipeline

# Directory containing benchmark dataset JSON files
DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")

# --- Default Benchmark Cases (kept for backward compatibility) ---
DEFAULT_BENCHMARK_CASES = [
    {
        "query": "How does Spring Batch manage chunk-based processing?",
        "reference": "Spring Batch processes data in chunks where an ItemReader reads items, an ItemProcessor transforms them, and an ItemWriter writes them in configurable batch sizes within a transaction."
    },
    {
        "query": "What are the tradeoffs of using an HNSW index in pgvector compared to IVFFlat?",
        "reference": "HNSW provides faster query performance and higher recall than IVFFlat, but requires longer index build times and higher memory usage."
    },
    {
        "query": "How does Cloud SQL Auth Proxy secure connections to PostgreSQL?",
        "reference": "Cloud SQL Auth Proxy secures connections by establishing an encrypted local TLS tunnel over port 5432 using IAM credentials without requiring public IP whitelisting."
    },
    {
        "query": "What is the role of the Vertex AI Semantic Ranker in a RAG pipeline?",
        "reference": "The Vertex AI Semantic Ranker acts as a cross-encoder that jointly scores query-document pairs to rerank candidate chunks and filter out irrelevant search results."
    },
    {
        "query": "What are the key elements of Spring Batch according to the PDF manual?",
        "reference": "The Spring Batch manual states it is structured around Job, Step, ItemReader, and ItemWriter elements."
    },
    {
        "query": "Describe the colors and flow of the database cluster layout gradient image",
        "reference": "The image features a smooth, continuous color gradient of earthy tones (browns, olives, and golden yellows) transitioning from dark sepia in the top-left to a bright gold/olive in the top-right."
    }
]


def list_available_datasets() -> list[str]:
    """Return names of available benchmark dataset files (without .json extension)."""
    if not os.path.isdir(DATASETS_DIR):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(DATASETS_DIR)
        if f.endswith(".json")
    )


def load_dataset(name: str) -> list[dict]:
    """Load benchmark cases from a named dataset JSON file.

    Falls back to the built-in default cases if the file is not found.
    """
    filepath = os.path.join(DATASETS_DIR, f"{name}.json")
    if os.path.isfile(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    if name == "default":
        return DEFAULT_BENCHMARK_CASES
    raise FileNotFoundError(f"Dataset '{name}' not found at {filepath}")


def run_pipeline_evaluation(
    pipeline: RAGPipeline,
    use_hybrid: bool = True,
    use_reranker: bool = True,
    label: str = "Evaluation Run",
    benchmark_cases: list[dict] | None = None,
    cases: list[dict] | None = None,
    pool_size: int = 5,
    final_top_k: int = 2,
    progress_callback=None,
    prompt_template: str | None = None,
):
    """Run evaluation and return structured results.

    Args:
        pipeline: The RAGPipeline instance to evaluate.
        use_hybrid: Whether to enable hybrid search.
        use_reranker: Whether to enable semantic reranking.
        label: Human-readable label for this evaluation run.
        benchmark_cases: List of query/reference dicts. Defaults to DEFAULT_BENCHMARK_CASES.
        cases: Alias for benchmark_cases.
        pool_size: Number of Stage 1 candidates to retrieve.
        final_top_k: Number of final context chunks for the LLM.
        progress_callback: Optional callable(current, total, message) for progress updates.
        prompt_template: Optional custom prompt template or preset text.

    Returns:
        dict with keys:
            - "label": the run label
            - "dataframe": pandas DataFrame with per-question metrics
            - "aggregated": dict of averaged metric values (e.g. avg_faithfulness)
            - "aggregated_scores": dict with keys (faithfulness, answer_relevancy, etc.)
            - "detailed_results": list of per-question metric dicts
    """
    target_cases = cases if cases is not None else benchmark_cases
    if target_cases is None:
        target_cases = DEFAULT_BENCHMARK_CASES

    print(f"\n==================================================")
    print(f"Running Evaluation for: {label}")
    print(f"==================================================")

    eval_data = {
        "user_input": [],
        "retrieved_contexts": [],
        "response": [],
        "reference": [],
    }

    total = len(target_cases)
    for i, item in enumerate(target_cases):
        msg = f"Querying pipeline for: '{item['query']}'..."
        print(msg)
        if progress_callback:
            progress_callback(i, total, msg)

        pipeline_output = pipeline.retrieve_and_generate(
            query=item["query"],
            use_hybrid=use_hybrid,
            use_reranker=use_reranker,
            pool_size=pool_size,
            final_top_k=final_top_k,
            prompt_template=prompt_template,
        )

        eval_data["user_input"].append(pipeline_output["user_input"])
        eval_data["retrieved_contexts"].append(pipeline_output["retrieved_contexts"])
        eval_data["response"].append(pipeline_output["response"])
        eval_data["reference"].append(item["reference"])

    eval_dataset = Dataset.from_dict(eval_data)

    raw_judge_llm = ChatVertexAI(model_name="gemini-2.5-flash")
    raw_judge_embeddings = VertexAIEmbeddings(model_name="text-embedding-005")

    judge_llm = LangchainLLMWrapper(raw_judge_llm)
    judge_embeddings = LangchainEmbeddingsWrapper(raw_judge_embeddings)

    if progress_callback:
        progress_callback(total, total, "Running Ragas evaluation...")

    print(f"Evaluating {label} with Ragas...")
    results = evaluate(
        dataset=eval_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=judge_llm,
        embeddings=judge_embeddings
    )

    df = results.to_pandas()

    # Compute aggregated metrics
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    aggregated = {}
    aggregated_scores = {}
    for col in metric_cols:
        if col in df.columns:
            val = float(df[col].mean())
            aggregated[f"avg_{col}"] = val
            aggregated_scores[col] = val
        else:
            aggregated[f"avg_{col}"] = None
            aggregated_scores[col] = None

    detailed_results = df.to_dict(orient="records")

    return {
        "label": label,
        "dataframe": df,
        "aggregated": aggregated,
        "aggregated_scores": aggregated_scores,
        "detailed_results": detailed_results,
    }


def main():
    pipeline = get_default_pipeline()

    # 1. Evaluate baseline pipeline
    result_baseline = run_pipeline_evaluation(
        pipeline=pipeline,
        use_hybrid=False,
        use_reranker=False,
        label="Baseline Pipeline (Pure Dense Search, No Reranker)"
    )

    # 2. Evaluate advanced pipeline
    result_advanced = run_pipeline_evaluation(
        pipeline=pipeline,
        use_hybrid=True,
        use_reranker=True,
        label="Advanced Pipeline (Hybrid Search + Vertex Reranker)"
    )

    # Display final comparative summary
    print("\n\n==================================================")
    print("COMPARATIVE EVALUATION SUMMARY")
    print("==================================================")

    cols = ["user_input", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    print("\n--- BASELINE PIPELINE ---")
    print(result_baseline["dataframe"][cols].to_string(index=False))

    print("\n--- ADVANCED PIPELINE ---")
    print(result_advanced["dataframe"][cols].to_string(index=False))

if __name__ == "__main__":
    main()
