import sys
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

# --- Benchmark Cases ---
benchmark_cases = [
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

def run_pipeline_evaluation(pipeline: RAGPipeline, use_hybrid: bool, use_reranker: bool, label: str):
    print(f"\n==================================================")
    print(f"Running Evaluation for: {label}")
    print(f"==================================================")

    eval_data = {
        "user_input": [],
        "retrieved_contexts": [],
        "response": [],
        "reference": [],
    }

    for item in benchmark_cases:
        print(f"Querying pipeline for: '{item['query']}'...")
        pipeline_output = pipeline.retrieve_and_generate(
            query=item["query"],
            use_hybrid=use_hybrid,
            use_reranker=use_reranker,
            pool_size=5,
            final_top_k=2
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
    return df

def main():
    pipeline = get_default_pipeline()

    # 1. Evaluate baseline pipeline
    df_baseline = run_pipeline_evaluation(
        pipeline=pipeline,
        use_hybrid=False,
        use_reranker=False,
        label="Baseline Pipeline (Pure Dense Search, No Reranker)"
    )

    # 2. Evaluate advanced pipeline
    df_advanced = run_pipeline_evaluation(
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
    print(df_baseline[cols].to_string(index=False))
    
    print("\n--- ADVANCED PIPELINE ---")
    print(df_advanced[cols].to_string(index=False))

if __name__ == "__main__":
    main()

