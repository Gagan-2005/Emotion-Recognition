from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from scipy.sparse import hstack

from common import ARTIFACTS_DIR, PLOTS_DIR, PROJECT_ROOT, RESULTS_DIR, extract_speech_features, load_artifact


STATIC_DIR = PROJECT_ROOT / "web"
EMOTION_DETAILS = {
    "angry": {"label": "Angry", "tone": "#ef4444"},
    "disgust": {"label": "Disgust", "tone": "#16a34a"},
    "fear": {"label": "Fear", "tone": "#8b5cf6"},
    "happy": {"label": "Happy", "tone": "#f59e0b"},
    "neutral": {"label": "Neutral", "tone": "#64748b"},
    "pleasant_surprise": {"label": "Pleasant Surprise", "tone": "#06b6d4"},
    "sad": {"label": "Sad", "tone": "#2563eb"},
}

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return pd.read_csv(path).replace({np.nan: None}).to_dict(orient="records")


def _format_probabilities(labels: list[str], probabilities: np.ndarray | None) -> list[dict]:
    if probabilities is None:
        return []
    rows = []
    for label, probability in sorted(zip(labels, probabilities), key=lambda item: item[1], reverse=True):
        rows.append(
            {
                "emotion": label,
                "label": EMOTION_DETAILS[label]["label"],
                "probability": float(probability),
                "tone": EMOTION_DETAILS[label]["tone"],
            }
        )
    return rows


def _prediction_response(variant: str, prediction: str, labels: list[str], probabilities: np.ndarray | None) -> dict:
    return {
        "variant": variant,
        "emotion": prediction,
        "label": EMOTION_DETAILS[prediction]["label"],
        "tone": EMOTION_DETAILS[prediction]["tone"],
        "probabilities": _format_probabilities(labels, probabilities),
    }


def _predict_speech(audio_path: str) -> dict:
    artifact = load_artifact("speech_model")
    features = extract_speech_features([audio_path])
    model = artifact["model"]
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0] if hasattr(model, "predict_proba") else None
    return _prediction_response("speech_only", prediction, artifact["labels"], probabilities)


def _predict_text(transcript: str) -> dict:
    artifact = load_artifact("text_model")
    model = artifact["model"]
    prediction = model.predict([transcript])[0]
    probabilities = model.predict_proba([transcript])[0] if hasattr(model, "predict_proba") else None
    return _prediction_response("text_only", prediction, artifact["labels"], probabilities)


def _predict_fusion(audio_path: str, transcript: str) -> dict:
    artifact = load_artifact("fusion_model")
    speech_features = artifact["speech_scaler"].transform(extract_speech_features([audio_path]))
    text_features = artifact["text_vectorizer"].transform([transcript])
    fused_features = hstack([speech_features, text_features]).tocsr()
    classifier = artifact["classifier"]
    prediction = classifier.predict(fused_features)[0]
    probabilities = classifier.predict_proba(fused_features)[0] if hasattr(classifier, "predict_proba") else None
    return _prediction_response("multimodal_fusion", prediction, artifact["labels"], probabilities)


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/api/health")
def health():
    missing = [name for name in ["speech_model.joblib", "text_model.joblib", "fusion_model.joblib"] if not (ARTIFACTS_DIR / name).exists()]
    return jsonify({"ready": not missing, "missing": missing})


@app.get("/api/results")
def results():
    accuracy_rows = _read_csv(RESULTS_DIR / "accuracy_tables.csv")
    reports = {
        "speech_only": _read_csv(RESULTS_DIR / "speech_only_classification_report.csv"),
        "text_only": _read_csv(RESULTS_DIR / "text_only_classification_report.csv"),
        "multimodal_fusion": _read_csv(RESULTS_DIR / "multimodal_fusion_classification_report.csv"),
    }
    plots = {
        "speech_only": {
            "confusion": "/plots/speech_only_confusion_matrix.png",
            "clusters": "/plots/speech_only_representation_pca.png",
        },
        "text_only": {
            "confusion": "/plots/text_only_confusion_matrix.png",
            "clusters": "/plots/text_only_representation_pca.png",
        },
        "multimodal_fusion": {
            "confusion": "/plots/multimodal_fusion_confusion_matrix.png",
            "clusters": "/plots/multimodal_fusion_representation_pca.png",
        },
    }
    return jsonify({"accuracy": accuracy_rows, "reports": reports, "plots": plots, "emotions": EMOTION_DETAILS})


@app.post("/api/predict")
def predict():
    variant = request.form.get("variant", "").strip()
    transcript = request.form.get("transcript", "").strip().lower()
    audio = request.files.get("audio")

    if variant not in {"speech_only", "text_only", "multimodal_fusion"}:
        return jsonify({"error": "Choose speech-only, text-only, or multimodal fusion."}), 400
    if variant in {"speech_only", "multimodal_fusion"} and not audio:
        return jsonify({"error": "Upload a WAV audio file for this model."}), 400
    if variant in {"text_only", "multimodal_fusion"} and not transcript:
        return jsonify({"error": "Enter the transcript word for this model."}), 400
    if audio and not audio.filename.lower().endswith(".wav"):
        return jsonify({"error": "Only WAV audio files are supported."}), 400

    temp_path = None
    try:
        if audio:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                audio.save(temp_file.name)
                temp_path = temp_file.name

        if variant == "speech_only":
            response = _predict_speech(temp_path)
        elif variant == "text_only":
            response = _predict_text(transcript)
        else:
            response = _predict_fusion(temp_path, transcript)
        return jsonify(response)
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


@app.get("/plots/<path:filename>")
def plot_file(filename: str):
    return send_from_directory(PLOTS_DIR, filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
