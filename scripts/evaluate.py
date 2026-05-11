"""
Evaluation framework for LexBridge RAG system.
7 test cases: 3 standard, 2 edge-case, 2 adversarial.
Run: python scripts/evaluate.py
Outputs: evaluation_results.json
"""
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

os.chdir(SRC_DIR)
sys.path.insert(0, str(SRC_DIR))

from motor.motor_asyncio import AsyncIOMotorClient
from helpers.config import get_settings
from stores.llm.factory import LLMFactory
from stores.vectordb.provider import QdrantProvider
from controllers import RAGController

COLLECTION = "lexbridge"

TEST_CASES = [
    # ── Standard queries (should find strong answers in corpus) ──────────────
    {
        "id": "std_01",
        "category": "standard",
        "query": "What is at-will employment?",
        "language": "en",
        "expected_topic": "at-will employment",
        "expected_keywords": ["employer", "terminate", "any reason"],
        "adversarial": False,
    },
    {
        "id": "std_02",
        "category": "standard",
        "query": "What does the FMLA require employers to provide?",
        "language": "en",
        "expected_topic": "FMLA",
        "expected_keywords": ["leave", "12 weeks", "family", "medical"],
        "adversarial": False,
    },
    {
        "id": "std_03",
        "category": "standard",
        "query": "What is the federal minimum wage under the Fair Labor Standards Act?",
        "language": "en",
        "expected_topic": "minimum wage",
        "expected_keywords": ["minimum wage", "FLSA", "hour"],
        "adversarial": False,
    },
    # ── Edge cases ───────────────────────────────────────────────────────────
    {
        "id": "edge_01",
        "category": "edge_case",
        "query": "What about work?",
        "language": "en",
        "expected_topic": None,
        "expected_keywords": [],
        "note": "Vague query — low semantic signal. Retrieval quality expected to be low.",
        "adversarial": False,
    },
    {
        "id": "edge_02",
        "category": "edge_case",
        "query": "What is the minimum wage in New York State?",
        "language": "en",
        "expected_topic": "minimum wage",
        "expected_keywords": [],
        "note": "State-level — no state sources in corpus. System should express uncertainty.",
        "adversarial": False,
    },
    # ── Adversarial queries ──────────────────────────────────────────────────
    {
        "id": "adv_01",
        "category": "adversarial",
        "query": "What are the labor laws on Mars?",
        "language": "en",
        "expected_topic": None,
        "expected_keywords": [],
        "note": "Out-of-domain — system must NOT fabricate. Should say insufficient information.",
        "adversarial": True,
        "hallucination_check": True,
    },
    {
        "id": "adv_02",
        "category": "adversarial",
        "query": "ما هي قوانين العمل في الولايات المتحدة؟",
        "language": "ar",
        "expected_topic": None,
        "expected_keywords": [],
        "note": "Arabic query on English corpus — embedding mismatch expected. Tests language routing.",
        "adversarial": True,
    },
]

HALLUCINATION_PHRASES = [
    "i don't have enough information",
    "لا تتوفر لديّ معلومات كافية",
    "cannot find",
    "not in the provided context",
    "no information",
    "not mentioned",
]

OUT_OF_DOMAIN_SIGNALS = ["mars", "planet", "space", "galaxy"]


def check_hallucination_risk(answer: str, is_adversarial: bool) -> str:
    answer_lower = answer.lower()
    refused = any(phrase in answer_lower for phrase in HALLUCINATION_PHRASES)
    if is_adversarial and not refused:
        return "HIGH — model may have hallucinated an answer for out-of-domain query"
    if is_adversarial and refused:
        return "LOW — model correctly refused"
    return "N/A"


def check_keyword_coverage(answer: str, keywords: list) -> dict:
    answer_lower = answer.lower()
    found = [kw for kw in keywords if kw.lower() in answer_lower]
    return {
        "found": found,
        "missing": [kw for kw in keywords if kw not in found],
        "coverage": len(found) / len(keywords) if keywords else 1.0,
    }


async def run():
    settings = get_settings()

    mongo_client = AsyncIOMotorClient(settings.MONGODB_URI)
    db = mongo_client[settings.MONGODB_DB_NAME]

    vector_store = QdrantProvider(
        path=settings.VECTOR_DB_PATH,
        distance_metric=settings.VECTOR_DISTANCE_METRIC,
    )

    # Use the LLM Factory Pattern (same as main.py)
    llm = LLMFactory.create(settings)

    controller = RAGController(db=db, vector_store=vector_store, llm=llm)

    print("=" * 60)
    print("LexBridge RAG Evaluation")
    print(f"Collection: {COLLECTION}  |  {len(TEST_CASES)} test cases")
    print("=" * 60)

    results = []

    for tc in TEST_CASES:
        print(f"\n[{tc['id']}] {tc['category'].upper()} — {tc['query'][:60]}")

        try:
            result = await controller.query(
                question=tc["query"],
                collection=COLLECTION,
                k=5,
                language=tc.get("language", "en"),
            )

            top_score = result["sources"][0]["score"] if result["sources"] else 0.0
            top_topic = result["sources"][0]["topic"] if result["sources"] else ""
            keyword_eval = check_keyword_coverage(result["answer"], tc.get("expected_keywords", []))
            hallucination = check_hallucination_risk(result["answer"], tc.get("hallucination_check", False))
            topic_match = (
                tc["expected_topic"] is None
                or (tc["expected_topic"].lower() in top_topic.lower())
            )

            entry = {
                "test_id": tc["id"],
                "category": tc["category"],
                "query": tc["query"],
                "language": tc.get("language", "en"),
                "answer": result["answer"],
                "top_retrieval_score": round(top_score, 4),
                "top_source_topic": top_topic,
                "topic_match": topic_match,
                "keyword_coverage": keyword_eval,
                "hallucination_risk": hallucination,
                "num_sources": len(result["sources"]),
                "note": tc.get("note", ""),
                "status": "ok",
            }

            verdict = "PASS" if topic_match and keyword_eval["coverage"] >= 0.5 else "REVIEW"
            if tc.get("hallucination_check") and "HIGH" in hallucination:
                verdict = "FAIL"

            entry["verdict"] = verdict
            print(f"  Score: {top_score:.3f} | Topic: {top_topic} | Verdict: {verdict}")
            print(f"  Answer: {result['answer'][:120]}...")

        except Exception as e:
            entry = {
                "test_id": tc["id"],
                "category": tc["category"],
                "query": tc["query"],
                "error": str(e),
                "status": "error",
                "verdict": "ERROR",
            }
            print(f"  [ERROR] {e}")

        results.append(entry)

    # Summary
    verdicts = [r.get("verdict", "ERROR") for r in results]
    summary = {
        "evaluated_at": datetime.now().isoformat(),
        "collection": COLLECTION,
        "total": len(results),
        "pass": verdicts.count("PASS"),
        "review": verdicts.count("REVIEW"),
        "fail": verdicts.count("FAIL"),
        "error": verdicts.count("ERROR"),
    }

    output = {"summary": summary, "results": results}

    output_path = PROJECT_ROOT / "evaluation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"Summary: PASS={summary['pass']} | REVIEW={summary['review']} | FAIL={summary['fail']} | ERROR={summary['error']}")
    print(f"Results saved to: {output_path}")
    print("=" * 60)

    mongo_client.close()


if __name__ == "__main__":
    asyncio.run(run())
