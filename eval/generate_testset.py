"""
Generates the RAGAS-style test set (question + ground_truth pairs) directly
from YOUR real corpus, rather than hand-guessed questions — this matters
because a hallucination/eval harness is only meaningful if the ground truth
actually reflects the source documents.
 
Run in the MAIN project environment (needs boto3 + the corpus on disk,
NOT ragas):
    python -m eval.generate_testset
 
Strategy:
    - Load every chunk produced by app.ingest.ingest_corpus()
    - Stratified-sample ~35 chunks across spaces AND across status
      (current / outdated / deprecated), so the eval set deliberately
      includes traps that test whether the system prefers current info
      and avoids presenting deprecated info as fact (hallucination /
      groundedness stress-test, not just easy questions).
    - For each sampled chunk, ask the LLM (via AWS Bedrock) to write ONE
      question that chunk answers, plus the ground-truth answer *strictly
      from that chunk*.
    - Write eval/test_questions.json
 
You can and should hand-review/edit the generated file before grading —
treat it as a strong first draft, not gospel.
"""
import json
import random
from collections import defaultdict
from pathlib import Path
 
import boto3
 
from app import config
from app.ingest import ingest_corpus
 
OUT_PATH = Path(__file__).parent / "test_questions.json"
TARGET_COUNT = 8
GEN_MODEL = config.GENERATION_MODEL
 
# Same Bedrock Runtime client pattern as app/rag_pipeline.py — the
# question-generation step uses the same GENERATION_MODEL as the live app.
_bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=config.AWS_REGION,
    aws_access_key_id=config.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
)
 
QGEN_SYSTEM = """You write evaluation questions for a RAG system test set.
Given one excerpt from an internal company wiki, produce exactly one
realistic employee question that this excerpt directly answers, and the
ground-truth answer using ONLY information in the excerpt.
Respond with strict JSON: {"question": "...", "ground_truth": "..."}
No markdown, no extra keys, no commentary."""
 
 
def stratified_sample(chunks, target_count=TARGET_COUNT):
    groups = defaultdict(list)
    for c in chunks:
        # keep chunks with enough substance to ask a real question about
        if len(c["text"]) < 120:
            continue
        groups[(c["space"], c["status"])].append(c)
 
    for g in groups.values():
        random.shuffle(g)
 
    sample = []
    keys = list(groups.keys())
    i = 0
    while len(sample) < target_count and keys:
        key = keys[i % len(keys)]
        if groups[key]:
            sample.append(groups[key].pop())
        else:
            keys.remove(key)
            continue
        i += 1
        if not any(groups.values()):
            break
    return sample
 
 
def generate_question(chunk):
    prompt = (
        f"WIKI EXCERPT (page: {chunk['title']}, section: {chunk['heading_path']}, "
        f"status: {chunk['status']}):\n\n{chunk['text']}"
    )
    response = _bedrock.converse(
        modelId=GEN_MODEL,
        system=[{"text": QGEN_SYSTEM}],
        messages=[
            {"role": "user", "content": [{"text": prompt}]}
        ],
        inferenceConfig={"maxTokens": 400},
    )
    text = response["output"]["message"]["content"][0]["text"].strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # occasionally the model wraps in ```json fences despite instructions
        cleaned = text.strip("`").removeprefix("json").strip()
        data = json.loads(cleaned)
    return data["question"], data["ground_truth"]
 
 
def main():
    random.seed(7)
    print(f"Corpus dir: {config.CORPUS_DIR}")
    chunks = ingest_corpus(config.CORPUS_DIR)
    sample = stratified_sample(chunks, TARGET_COUNT)
    print(f"Sampled {len(sample)} chunks across "
          f"{len({(c['space'], c['status']) for c in sample})} (space,status) groups")
 
    test_set = []
    for i, chunk in enumerate(sample, 1):
        try:
            question, ground_truth = generate_question(chunk)
        except Exception as e:
            print(f"  [skip {i}/{len(sample)}] {chunk['chunk_id']}: {e}")
            continue
        test_set.append({
            "question": question,
            "ground_truth": ground_truth,
            "source_page_id": chunk["page_id"],
            "source_space": chunk["space"],
            "source_status": chunk["status"],
        })
        print(f"  [{i}/{len(sample)}] {chunk['chunk_id']} -> {question}")
 
    OUT_PATH.write_text(json.dumps(test_set, indent=2), encoding="utf-8")
    print(f"\nWrote {len(test_set)} questions to {OUT_PATH}")
    print("Review/edit this file by hand before using it for grading.")
 
 
if __name__ == "__main__":
    main()
 
