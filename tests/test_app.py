import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from app import binary_probability_rationale, displayed_binary_probability


class StreamlitAppTests(unittest.TestCase):
    def test_displayed_binary_probability_matches_visible_result(self):
        self.assertEqual(displayed_binary_probability(0.05, False), 0.95)
        self.assertEqual(displayed_binary_probability(0.98, True), 0.98)

    def test_binary_probability_rationale_explains_raw_and_displayed_values(self):
        rationale = binary_probability_rationale(
            0.05,
            False,
            "Jev returned 5% yes probability for urgency.",
        )

        self.assertIn("Raw yes probability: 5%", rationale)
        self.assertIn("displayed No probability: 95%", rationale)

    def test_app_renders_without_credentials(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path)).run(timeout=20)

        self.assertFalse(app.exception)
        self.assertEqual("Jev vs Claude Decision Comparator", app.title[0].value)
        self.assertEqual("Run comparison", app.button[-1].label)
        self.assertIn("I was charged twice", app.text_area[0].value)


if __name__ == "__main__":
    unittest.main()
