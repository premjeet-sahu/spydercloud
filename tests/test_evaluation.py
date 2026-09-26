import tempfile
import unittest
from pathlib import Path

from jev_poc.evaluation import load_cases, prediction_dict, summarize
from jev_poc.providers import Prediction, RulesProvider


class EvaluationTests(unittest.TestCase):
    def test_rules_provider_returns_valid_prediction(self):
        prediction = RulesProvider().predict("The API is down in production. Fix it ASAP.")
        self.assertEqual("technical", prediction.route)
        self.assertTrue(prediction.urgent)

    def test_decision_confidence_uses_weakest_decision(self):
        prediction = Prediction(
            provider="test",
            route="billing",
            route_confidence=0.9,
            urgent_probability=0.8,
            refund_probability=0.1,
            latency_ms=1,
        )
        self.assertAlmostEqual(0.6, prediction.decision_confidence)

    def test_summary_calculates_accuracy_and_coverage(self):
        prediction = Prediction(
            provider="test",
            route="billing",
            route_confidence=0.9,
            urgent_probability=0.1,
            refund_probability=0.95,
            latency_ms=10,
        )
        records = [
            {
                "id": "one",
                "provider": "test",
                "expected": {"route": "billing", "urgent": False, "refund_requested": True},
                "prediction": prediction_dict(prediction),
            }
        ]
        summary = summarize(records, threshold=0.8)["test"]
        self.assertEqual(1.0, summary["exact_match_accuracy"])
        self.assertEqual(1.0, summary["automation_coverage"])

    def test_dataset_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text('{"id":"x"}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_cases(path)


if __name__ == "__main__":
    unittest.main()

