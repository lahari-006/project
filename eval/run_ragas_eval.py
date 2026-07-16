"""
Scores eval/eval_dataset.jsonl (produced by app/generate_answers.py in the
MAIN environment) using RAGAS, in your separate `ragas-env` virtual env.
 
This script has NO dependency on chromadb / mcp / the app package — it only
reads the jsonl, so ragas-env can stay minimal (see eval/requirements-eval.txt).
Because of that, it does NOT import app.config, and instead loads its own
.env directly with python-dotenv and reads AWS credentials from environment
variables.
 
Metrics (4, satisfies the "at least 3 scored metrics" requirement):
    - faithfulness        -> hallucination proxy: is the answer supported by
                              the retrieved contexts?
    - answer_relevancy     -> does the answer actually address the question?
    - context_precision    -> are the retrieved contexts relevant/ranked well?
    - context_recall       -> did retrieval surface what was needed to answer?
 
Setup (inside ragas-env):
    pip install -r eval/requirements-eval.txt
    pip install langchain-aws boto3 python-dotenv
    # .env at the project root (same file app/config.py uses) with:
    #   AWS_ACCESS_KEY_ID=...
    #   AWS_SECRET_ACCESS_KEY=...
    #   AWS_REGION=us-east-1
    python eval/run_ragas_eval.py
 
Outputs:
    eval/eval_report.csv   - per-question scores
    stdout                 - aggregate scores + a simple pass/fail summary
"""
import json
import os
from pathlib import Path
 
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
 
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_aws import ChatBedrockConverse
from langchain_huggingface import HuggingFaceEmbeddings
 
# ---------------------------------------------------------------------
# Load .env from the project root. This script deliberately doesn't import
# app/config.py (keeps ragas-env free of chromadb/mcp/etc.), so it needs
# its own dotenv load. Adjust the filename below if your .env is named
# something else (e.g. "myenv.env").
# ---------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")
 
DATASET_PATH = Path(__file__).parent / "eval_dataset.jsonl"
REPORT_PATH = Path(__file__).parent / "eval_report.csv"
 
# Minimum acceptable aggregate scores — tune these to your rubric.
THRESHOLDS = {
    "faithfulness": 0.70,
    "answer_relevancy": 0.70,
    "context_precision": 0.60,
    "context_recall": 0.60,
}
 
 
def load_dataset() -> Dataset:
    rows = []
    with open(DATASET_PATH, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            rows.append({
                "question": rec["question"],
                "answer": rec["answer"],
                "contexts": rec["contexts"] or [""],
                "ground_truth": rec["ground_truth"],
            })
    print(f"Loaded {len(rows)} scored questions from {DATASET_PATH}")
    return Dataset.from_list(rows)
 
 
def main():
    aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    aws_region = os.environ.get("AWS_REGION", "us-east-1")
    critic_model = os.environ.get("RAGAS_CRITIC_MODEL", "amazon.nova-micro-v1:0")
 
    if not (aws_access_key and aws_secret_key):
        raise SystemExit(
            "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY not found. Checked "
            f"environment variables and {ROOT_DIR / '.env'}. If your env file "
            "has a different name/path, update the load_dotenv() call at the "
            "top of this script, or `export` the two variables manually "
            "before running."
        )
 
    dataset = load_dataset()
 
    critic_llm = LangchainLLMWrapper(
        ChatBedrockConverse(
            model=critic_model,
            region_name=aws_region,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            temperature=0,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )
 
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=critic_llm,
        embeddings=embeddings,
    )
 
    df = result.to_pandas()
    df.to_csv(REPORT_PATH, index=False)
    print(f"\nPer-question report written to {REPORT_PATH}")
 
    print("\n=== Aggregate RAGAS scores ===")
    summary = {}
    for metric in THRESHOLDS:
        if metric in df.columns:
            avg = df[metric].mean()
            summary[metric] = avg
            status = "PASS" if avg >= THRESHOLDS[metric] else "BELOW THRESHOLD"
            print(f"  {metric:20s}: {avg:.3f}   (threshold {THRESHOLDS[metric]})  [{status}]")
 
    n_low_faithfulness = (df["faithfulness"] < 0.5).sum() if "faithfulness" in df else 0
    print(f"\nQuestions with faithfulness < 0.5 (likely hallucinations): {n_low_faithfulness} / {len(df)}")
 
    return summary
 
 
if __name__ == "__main__":
    main()
 
