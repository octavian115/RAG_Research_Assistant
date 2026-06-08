import argparse
import json
import os
import re
import sys
from pathlib import Path
from uuid import uuid4

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()
os.environ.setdefault("USER_AGENT", "PaperTrail/0.1 (smoke evaluation)")

PDF_PATH = "documents/Openclaw_Research_Report.pdf"
GOLDENS_FILE = Path("goldens.json")
CACHE_FILE = Path("smoke_eval_cache.json")
RESULTS_FILE = Path("smoke_eval_results.json")
STOPWORDS = {
    "about", "after", "also", "because", "before", "between", "could", "does",
    "from", "have", "into", "list", "more", "over", "than", "that", "their",
    "these", "they", "this", "through", "what", "when", "where", "which",
    "while", "with", "would",
}


def load_goldens(limit: int) -> list[dict]:
    pairs = json.loads(GOLDENS_FILE.read_text(encoding="utf-8"))
    return pairs[:limit]


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())
        if token not in STOPWORDS
    }


def run_rag_query(graph, query: str, session_id: str) -> tuple[str, list[str]]:
    config = {"configurable": {"thread_id": str(session_id)}}
    final_state = graph.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "session_id": session_id,
            "original_query": query,
            "query": query,
            "retrieval_query": None,
            "retrieved_docs": [],
            "retrieval_attempts": 0,
            "rewrite_count": 0,
            "chat_history": [],
        },
        config=config,
    )
    answer = final_state.get("answer") or ""
    retrieval_context = [doc.page_content for doc in (final_state.get("retrieved_docs") or [])]
    return answer, retrieval_context


def load_cache(refresh: bool) -> dict:
    if refresh or not CACHE_FILE.exists():
        return {}
    return json.loads(CACHE_FILE.read_text(encoding="utf-8"))


def save_cache(cache: dict) -> None:
    CACHE_FILE.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def score_case(pair: dict, answer: str, retrieval_context: list[str]) -> dict:
    expected_terms = tokenize(pair["expected_output"])
    answer_terms = tokenize(answer)
    context_terms = tokenize("\n\n".join(retrieval_context))
    answer_overlap = sorted(expected_terms & answer_terms)
    context_overlap = sorted(expected_terms & context_terms)
    return {
        "answer_non_empty": bool(answer.strip()),
        "retrieved_context_non_empty": bool(retrieval_context),
        "answer_overlap_count": len(answer_overlap),
        "context_overlap_count": len(context_overlap),
        "answer_overlap_terms": answer_overlap[:20],
        "context_overlap_terms": context_overlap[:20],
        "passed": bool(answer.strip()) and bool(retrieval_context) and len(answer_overlap) >= 3,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cheap deterministic PaperTrail smoke evals.")
    parser.add_argument("--limit", type=int, default=3, help="Number of goldens to test.")
    parser.add_argument("--refresh-cache", action="store_true", help="Rerun RAG instead of using cached outputs.")
    args = parser.parse_args()

    from backend.paper_loader import load_document
    from backend.rag_graph import build_graph
    from backend.vector_store import add_paper

    pairs = load_goldens(args.limit)
    cache = load_cache(args.refresh_cache)
    docs = None
    graph = None
    results = []

    for index, pair in enumerate(pairs, start=1):
        query = pair["input"] + " as per the report in knowledge base"
        cache_key = pair["input"]
        if cache_key in cache:
            answer = cache[cache_key]["answer"]
            retrieval_context = cache[cache_key]["retrieval_context"]
            print(f"[{index}/{len(pairs)}] Cached: {pair['input'][:90]}...", flush=True)
        else:
            if docs is None:
                docs = load_document(PDF_PATH)
            if graph is None:
                graph = build_graph(db_path="smoke_eval_checkpoints.db")
            print(f"[{index}/{len(pairs)}] Running RAG: {pair['input'][:90]}...", flush=True)
            session_id = f"smoke_eval_session_{uuid4()}"
            add_paper(docs, session_id)
            answer, retrieval_context = run_rag_query(graph, query, session_id)
            cache[cache_key] = {
                "answer": answer,
                "retrieval_context": retrieval_context,
            }
            save_cache(cache)

        scores = score_case(pair, answer, retrieval_context)
        results.append({
            "input": pair["input"],
            "actual_output": answer,
            **scores,
        })
        status = "PASS" if scores["passed"] else "FAIL"
        print(
            f"  {status}: answer_terms={scores['answer_overlap_count']} "
            f"context_terms={scores['context_overlap_count']}",
            flush=True,
        )

    RESULTS_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    passed = sum(1 for result in results if result["passed"])
    print(f"\nSmoke eval: {passed}/{len(results)} passed. Results saved to {RESULTS_FILE}.")


if __name__ == "__main__":
    main()
