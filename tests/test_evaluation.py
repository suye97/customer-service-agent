"""Check the evaluation dataset and its scoring against regressions."""

import unittest

from evaluation.run import evaluate, load_cases


class EvaluationTests(unittest.TestCase):
    def test_product_ground_truth_matches_current_filtering(self):
        report = evaluate(load_cases())
        failures = [case for case in report["cases"] if not case["passed"]]
        self.assertEqual(failures, [])
        self.assertEqual(report["product_cases"], 12)
        self.assertEqual(report["exact_set_accuracy"], 1.0)

    def test_business_contract_cases_are_not_counted_as_passed(self):
        report = evaluate(load_cases())
        self.assertEqual(report["target_contract_cases_not_run"], 18)
        self.assertEqual(report["passed"], report["product_cases"])


if __name__ == "__main__":
    unittest.main()
