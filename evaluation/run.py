"""Offline, deterministic evaluation of product constraints and case coverage."""

import argparse
import json
from pathlib import Path

from customer_agent import ProductAgent


DATASET = Path(__file__).with_name("cases.json")


def load_cases(path: Path = DATASET) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 1
    products = {row["id"] for row in data["catalog"]}
    assert len(products) == len(data["catalog"]), "Duplicate catalogue ID"
    for section in ("product_cases", "target_contract_cases"):
        ids = [case["id"] for case in data[section]]
        assert len(ids) == len(set(ids)), f"Duplicate case ID in {section}"
    for case in data["product_cases"]:
        assert case["question"].strip()
        assert len(case["expected_ids"]) == len(set(case["expected_ids"]))
        assert set(case["expected_ids"]) <= products, f"Unknown expected ID: {case['id']}"
    for case in data["target_contract_cases"]:
        assert case["question"].strip() and case["expected_route"] and case["expected_action"]
    return data


def evaluate(data: dict) -> dict:
    """Score filtering against a fixed candidate pool, not live Milvus recall."""
    agent = ProductAgent()
    results = []
    for case in data["product_cases"]:
        expected = set(case["expected_ids"])
        predicted = {row["id"] for row in agent.filter_candidates(case["question"], data["catalog"])}
        true_positive = len(predicted & expected)
        precision = true_positive / len(predicted) if predicted else float(not expected)
        recall = true_positive / len(expected) if expected else float(not predicted)
        results.append({
            "id": case["id"], "passed": predicted == expected,
            "expected_ids": sorted(expected), "actual_ids": sorted(predicted),
            "precision": round(precision, 4), "recall": round(recall, 4),
        })
    count = len(results)
    return {
        "dataset_version": data["version"],
        "evaluation_scope": "offline candidate filtering with fixed catalogue; no vector search or LLM calls",
        "product_cases": count,
        "passed": sum(item["passed"] for item in results),
        "exact_set_accuracy": round(sum(item["passed"] for item in results) / count, 4),
        "macro_precision": round(sum(item["precision"] for item in results) / count, 4),
        "macro_recall": round(sum(item["recall"] for item in results) / count, 4),
        "target_contract_cases_not_run": len(data["target_contract_cases"]),
        "cases": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write a JSON report")
    args = parser.parse_args()
    report = evaluate(load_cases())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"商品约束评测: {report['passed']}/{report['product_cases']} 通过；"
          f"Precision={report['macro_precision']:.3f}, Recall={report['macro_recall']:.3f}。"
          f"业务契约待接入: {report['target_contract_cases_not_run']} 条。")
    for case in report["cases"]:
        if not case["passed"]:
            print(f"FAIL {case['id']}: expected={case['expected_ids']}, actual={case['actual_ids']}")
    return 0 if report["passed"] == report["product_cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
