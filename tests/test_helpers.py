import unittest

from core import (
    clean_text,
    confidence_label,
    friendly_fallback_message,
    predicted_class_probability,
)


class HelperTests(unittest.TestCase):
    def test_clean_text_preserves_apostrophe_and_hashtag(self):
        self.assertEqual(clean_text("It's #Monday! https://example.com"), "it's #monday")

    def test_confidence_labels_are_calibrated_symmetrically(self):
        # Sarcastic side of the 0.50 threshold.
        self.assertEqual(confidence_label(0.50), "Moderate")
        self.assertEqual(confidence_label(0.64), "Moderate")
        self.assertEqual(confidence_label(0.65), "High")
        self.assertEqual(confidence_label(0.79), "High")
        self.assertEqual(confidence_label(0.80), "Very high")

        # Not-sarcastic side uses 1 - score for predicted-class confidence.
        self.assertEqual(confidence_label(0.49), "Moderate")
        self.assertEqual(confidence_label(0.35), "High")
        self.assertEqual(confidence_label(0.20), "Very high")

    def test_predicted_class_probability(self):
        self.assertAlmostEqual(predicted_class_probability(0.63), 0.63)
        self.assertAlmostEqual(predicted_class_probability(0.37), 0.63)

    def test_quota_message_is_friendly(self):
        message = friendly_fallback_message("quota_error")
        self.assertIn("quota", message.lower())
        self.assertNotIn("429", message)
        self.assertNotIn("resource_exhausted", message.lower())


if __name__ == "__main__":
    unittest.main()
