"""Emotion detection application using IBM Watson NLP."""

import json
import requests


def emotion_detector(text_to_analyze):
    """Analyze text and return formatted emotion scores."""

    url = (
        "https://sn-watson-emotion.labs.skills.network/"
        "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
    )

    headers = {
        "grpc-metadata-mm-model-id":
            "emotion_aggregated-workflow_lang_en_stock"
    }

    payload = {
        "raw_document": {
            "text": text_to_analyze
        }
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=30
    )

    response.raise_for_status()

    formatted_response = json.loads(response.text)

    emotions = formatted_response[
        "emotionPredictions"
    ][0]["emotion"]

    anger_score = emotions["anger"]
    disgust_score = emotions["disgust"]
    fear_score = emotions["fear"]
    joy_score = emotions["joy"]
    sadness_score = emotions["sadness"]

    dominant_emotion = max(emotions, key=emotions.get)

    return {
        "anger": anger_score,
        "disgust": disgust_score,
        "fear": fear_score,
        "joy": joy_score,
        "sadness": sadness_score,
        "dominant_emotion": dominant_emotion
    }
