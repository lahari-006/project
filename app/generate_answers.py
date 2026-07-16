"""
Runs eval/test_questions.json through the live RAG pipeline and writes
eval/eval_dataset.jsonl (question, answer, contexts, ground_truth).

This MUST run in the main project virtual environment (the one with
chromadb / anthropic / mcp installed) — NOT in ragas-env. The separate
eval/run_ragas_eval.py script then scores this jsonl inside ragas-env,
so the two environments never need to share dependencies.

    python -m app.generate_answers
"""
import json
from pathlib import Path

from app.rag_pipeline import answer_query

QUESTIONS_PATH = Path(__file__).parent.parent / "eval" / "test_questions.json"
OUT_PATH = Path(__file__).parent.parent / "eval" / "eval_dataset.jsonl"


def main():
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    print(f"Running {len(questions)} eval questions through the RAG pipeline...")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for i, item in enumerate(questions, 1):
            q = item["question"]
            result = answer_query(q, user_id="eval-harness")
            record = {
                "question": q,
                "answer": result["answer"],
                "contexts": result.get("_contexts", []),
                "ground_truth": item["ground_truth"],
                "source_page_id": item.get("source_page_id"),
                "source_status": item.get("source_status"),
                "blocked": result["blocked"],
                "latency_seconds": result["latency_seconds"],
                "cost_usd": result["cost_usd"],
            }
            f.write(json.dumps(record) + "\n")
            print(f"  [{i}/{len(questions)}] {q[:70]}...")

    print(f"\nWrote {OUT_PATH}")
    print("Now switch to ragas-env and run: python eval/run_ragas_eval.py")


if __name__ == "__main__":
    main()
