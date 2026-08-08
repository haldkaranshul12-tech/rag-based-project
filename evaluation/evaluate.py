"""
Evaluation script -- runs every question in golden_dataset.json through the
real retrieval + generation pipeline and reports two simple metrics.

Run from the 'bank project' folder:
    python evaluation/evaluate.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retriever import retrieve_documents
from src.llm import generate_answer

DATASET_PATH = os.path.join(os.path.dirname(__file__), "golden_dataset.json")


def keyword_hit(text, keywords):
    text_lower = (text or "").lower()
    return any(kw.lower() in text_lower for kw in keywords)


def run_evaluation():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    results = []
    retrieval_hits = 0
    answer_hits = 0

    for item in golden_set:
        question = item["question"]
        keywords = item["expected_keywords"]

        retrieval = retrieve_documents(question)
        documents = retrieval["documents"][0] if retrieval.get("documents") else []
        context = "\n".join(documents)

        retrieved_hit = keyword_hit(context, keywords)
        if retrieved_hit:
            retrieval_hits += 1

        answer = generate_answer(context, question)
        answered_hit = keyword_hit(answer, keywords)
        if answered_hit:
            answer_hits += 1

        results.append({
            "id": item["id"], "question": question,
            "retrieval_hit": retrieved_hit, "answer_hit": answered_hit, "answer": answer,
        })

        status = "✅" if answered_hit else "❌"
        print(f"{status} [{item['id']}] {question}")
        print(f"    Retrieval hit: {retrieved_hit} | Answer hit: {answered_hit}")
        print(f"    Answer: {answer[:150]}{'...' if len(answer) > 150 else ''}\n")

    total = len(golden_set)
    print("=" * 60)
    print(f"Retrieval hit-rate: {retrieval_hits}/{total} ({100 * retrieval_hits / total:.0f}%)")
    print(f"Answer keyword coverage: {answer_hits}/{total} ({100 * answer_hits / total:.0f}%)")

    return results


if __name__ == "__main__":
    run_evaluation()