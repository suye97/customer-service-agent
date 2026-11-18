"""Evaluate real Milvus vector hits and the product retrieval/filtering pipeline."""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from customer_agent import ProductAgent


CASES = Path(__file__).with_name("live_cases.json")


class RecordingEmbedding:
    def __init__(self, client):
        self.client = client
        self.embeddings = self
        self.vector = None

    def create(self, **kwargs):
        response = self.client.embeddings.create(**kwargs)
        self.vector = response.data[0].embedding
        return response


def score(expected: set[str], actual: list[str]) -> dict:
    found = set(actual)
    common = expected & found
    return {
        "precision": len(common) / len(found) if found else float(not expected),
        "recall": len(common) / len(expected) if expected else float(not found),
        "exact": found == expected,
    }


def evaluate() -> dict:
    load_dotenv()
    agent = ProductAgent()
    if agent.embedding_mock:
        raise RuntimeError("Live evaluation requires EMBEDDING_MOCK=0")
    store = agent._store()
    embedding = RecordingEmbedding(agent._embedding())
    agent._embedding_client = embedding
    data = json.loads(CASES.read_text(encoding="utf-8"))
    rows = store.query(collection_name=agent.collection, filter="", output_fields=["id", "product"], limit=16384)
    known = {str(row["id"]): row["product"] for row in rows}
    results = []
    for case in data["cases"]:
        expected = set(case["relevant_ids"])
        if not expected <= known.keys():
            raise ValueError(f"Stale IDs in label {case['id']}: {expected - known.keys()}")
        try:
            candidates = agent.filter_candidates(case["question"], agent.retrieve(case["question"], []))
            actual = [row["id"] for row in candidates]
            hits = store.search(
                collection_name=agent.collection, data=[embedding.vector], anns_field="vector",
                limit=data["top_k"], output_fields=["id"], search_params={"metric_type": "COSINE"},
            )[0]
            raw = [str(hit.get("entity", {}).get("id") or hit.get("id")) for hit in hits]
            raw_recall = len(expected & set(raw)) / len(expected) if expected else None
            first_rank = next((rank for rank, product_id in enumerate(raw, 1) if product_id in expected), None)
            result = {
                "id": case["id"], "question": case["question"], "expected_ids": sorted(expected),
                "raw_top_k_ids": raw, "raw_recall_at_k": raw_recall,
                "raw_reciprocal_rank": 1 / first_rank if first_rank else (0 if expected else None),
                "filtered_ids": actual, **score(expected, actual),
            }
        except Exception as exc:
            result = {"id": case["id"], "question": case["question"],
                      "expected_ids": sorted(expected), "error": f"{type(exc).__name__}: {exc}"}
        results.append(result)
        print(f"{len(results):02d}/{len(data['cases'])} {case['id']}: "
              f"{'ERROR' if 'error' in result else 'PASS' if result['exact'] else 'FAIL'}", flush=True)
    positive = [row for row in results if row["expected_ids"] and "error" not in row]
    completed = [row for row in results if "error" not in row]
    return {
        "scope": "real Milvus collection + real embedding; no LLM or business API",
        "collection": agent.collection, "catalogue_rows": len(rows),
        "label_snapshot": data["collection_snapshot"], "top_k": data["top_k"],
        "case_count": len(results), "completed": len(completed),
        "exact_set_passed": sum(row["exact"] for row in completed),
        "filtered_macro_precision": sum(row["precision"] for row in completed) / len(completed) if completed else None,
        "filtered_macro_recall": sum(row["recall"] for row in completed) / len(completed) if completed else None,
        "raw_vector_recall_at_k": sum(row["raw_recall_at_k"] for row in positive) / len(positive) if positive else None,
        "raw_vector_mrr": sum(row["raw_reciprocal_rank"] for row in positive) / len(positive) if positive else None,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("live_report.json"))
    args = parser.parse_args()
    report = evaluate()
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"完成 {report['completed']}/{report['case_count']}；候选集合完全匹配 "
          f"{report['exact_set_passed']}/{report['completed']}；向量 Recall@{report['top_k']}="
          f"{report['raw_vector_recall_at_k']}; MRR={report['raw_vector_mrr']}")
    return 0 if report["completed"] == report["case_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
