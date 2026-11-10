"""Unit tests for the Emotion Detection application."""

import unittest
from EmotionDetection.emotion_detection import emotion_detector


class TestEmotionDetector(unittest.TestCase):
    """Test emotion detection for five different emotions."""

    def test_joy(self):
        """Test detection of joy."""
        result = emotion_detector("I am glad this happened")
        self.assertEqual(result["dominant_emotion"], "joy")

    def test_anger(self):
        """Test detection of anger."""
        result = emotion_detector("I am really mad about this")
        self.assertEqual(result["dominant_emotion"], "anger")

    def test_disgust(self):
        """Test detection of disgust."""
        result = emotion_detector("I feel disgusted just hearing about this")
        self.assertEqual(result["dominant_emotion"], "disgust")

    def test_sadness(self):
        """Test detection of sadness."""
        result = emotion_detector("I am so sad about this")
        self.assertEqual(result["dominant_emotion"], "sadness")

    def test_fear(self):
        """Test detection of fear."""
        result = emotion_detector("I am really afraid that this will happen")
        self.assertEqual(result["dominant_emotion"], "fear")


if __name__ == "__main__":
    unittest.main()
