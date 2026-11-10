"""Detect emotions in text using IBM Watson NLP."""

import requests

EMOTION_URL = (
    "https://sn-watson-emotion.labs.skills.network/"
    "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
HEADERS = {
    "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
}
EMOTION_NAMES = ("anger", "disgust", "fear", "joy", "sadness")


def emotion_detector(text_to_analyze):
    """Return Watson emotion scores and the dominant emotion."""
    response = requests.post(
        EMOTION_URL,
        headers=HEADERS,
        json={"raw_document": {"text": text_to_analyze}},
        timeout=30,
    )

    if response.status_code == 400:
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }

    response.raise_for_status()
    predictions = response.json()["emotionPredictions"]
    scores = predictions[0]["emotion"]
    result = {name: scores[name] for name in EMOTION_NAMES}
    result["dominant_emotion"] = max(result, key=result.get)
    return result
