import sys
import argparse
from src.pipeline import get_default_pipeline

def main():
    parser = argparse.ArgumentParser(description="Query the parameterized RAG pipeline.")
    parser.add_argument("query", type=str, help="The search query.")
    parser.add_argument("--no-hybrid", action="store_true", help="Disable sparse FTS hybrid search (uses pure dense vector instead).")
    parser.add_argument("--no-rerank", action="store_true", help="Disable semantic reranking.")
    parser.add_argument("--pool-size", type=int, default=5, help="Number of Stage 1 candidates to fetch.")
    parser.add_argument("--top-k", type=int, default=2, help="Number of final context chunks to feed into LLM.")
    
    args = parser.parse_args()

    use_hybrid = not args.no_hybrid
    use_rerank = not args.no_rerank

    print(f"\n--- Running RAG Pipeline ---")
    print(f"Query:        '{args.query}'")
    print(f"Config:       hybrid={use_hybrid}, rerank={use_rerank}, pool_size={args.pool_size}, top_k={args.top_k}")
    
    try:
        pipeline = get_default_pipeline()
        result = pipeline.retrieve_and_generate(
            query=args.query,
            use_hybrid=use_hybrid,
            use_reranker=use_rerank,
            pool_size=args.pool_size,
            final_top_k=args.top_k
        )
        
        print("\n--- Retrieved Contexts ---")
        for idx, ctx in enumerate(result["retrieved_contexts"], 1):
            print(f"  {idx}. {ctx}")
            
        print("\n--- Generated Response ---")
        print(result["response"])
        print()
    except Exception as e:
        print(f"\nError running pipeline: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

