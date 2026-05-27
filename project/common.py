from __future__ import annotations

import argparse
import json
import re
import math
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile as sf
from scipy.fftpack import dct
from scipy.signal import get_window, resample_poly
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = PROJECT_ROOT.parent / "archive" / "TESS Toronto emotional speech set data"
RESULTS_DIR = PROJECT_ROOT / "Results"
PLOTS_DIR = RESULTS_DIR / "plots"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
SPEECH_FEATURE_CACHE = ARTIFACTS_DIR / "speech_features.joblib"
RANDOM_STATE = 42
SAMPLE_RATE = 16000
_MEL_CACHE: dict[tuple[int, int, int], np.ndarray] = {}
EMOTION_NORMALIZATION = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "pleasant_surprise": "pleasant_surprise",
    "pleasant_surprised": "pleasant_surprise",
    "ps": "pleasant_surprise",
    "sad": "sad",
}


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATASET_ROOT)
    return parser.parse_args()


def parse_tess_filename(path: Path) -> tuple[str, str, str]:
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected TESS filename: {path.name}")
    speaker = parts[0]
    emotion = EMOTION_NORMALIZATION["_".join(parts[2:]).lower()]
    transcript = re.sub(r"[^a-zA-Z]+", " ", parts[1]).strip().lower()
    return speaker, transcript, emotion


def load_metadata(data_root: Path) -> pd.DataFrame:
    files = sorted(data_root.rglob("*.wav"))
    rows = []
    seen_names: set[str] = set()
    for file_path in files:
        if file_path.name in seen_names:
            continue
        seen_names.add(file_path.name)
        speaker, transcript, emotion = parse_tess_filename(file_path)
        rows.append(
            {
                "path": str(file_path),
                "file_name": file_path.name,
                "speaker": speaker,
                "transcript": transcript,
                "emotion": emotion,
            }
        )
    if not rows:
        raise FileNotFoundError(f"No .wav files found under {data_root}")
    return pd.DataFrame(rows).sort_values("file_name").reset_index(drop=True)


def split_metadata(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["emotion"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _load_audio(path: str) -> tuple[np.ndarray, int]:
    y, sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    y = y.astype(np.float32)
    if sr != SAMPLE_RATE:
        divisor = math.gcd(sr, SAMPLE_RATE)
        y = resample_poly(y, SAMPLE_RATE // divisor, sr // divisor).astype(np.float32)
        sr = SAMPLE_RATE
    peak = np.max(np.abs(y)) if y.size else 0
    if peak > 0:
        y = y / peak
    return y, sr


def _trim_silence(y: np.ndarray, top_db: float = 25.0) -> np.ndarray:
    if y.size == 0:
        return np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
    threshold = np.max(np.abs(y)) * (10 ** (-top_db / 20))
    active = np.flatnonzero(np.abs(y) > threshold)
    if active.size == 0:
        return y
    return y[active[0] : active[-1] + 1]


def _mel_filterbank(sr: int, n_fft: int, n_mels: int = 40) -> np.ndarray:
    cache_key = (sr, n_fft, n_mels)
    if cache_key in _MEL_CACHE:
        return _MEL_CACHE[cache_key]

    def hz_to_mel(hz: np.ndarray) -> np.ndarray:
        return 2595 * np.log10(1 + hz / 700)

    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700 * (10 ** (mel / 2595) - 1)

    mel_points = np.linspace(hz_to_mel(np.array([0]))[0], hz_to_mel(np.array([sr / 2]))[0], n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center > left:
            filters[m - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            filters[m - 1, center:right] = (right - np.arange(center, right)) / (right - center)
    _MEL_CACHE[cache_key] = filters
    return filters


def _frame_signal(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if y.size < frame_length:
        y = np.pad(y, (0, frame_length - y.size))
    frame_count = 1 + (len(y) - frame_length) // hop_length
    shape = (frame_count, frame_length)
    strides = (y.strides[0] * hop_length, y.strides[0])
    frames = np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)
    return frames * get_window("hann", frame_length, fftbins=True)


def _single_speech_feature(audio_path: str) -> np.ndarray:
    y, sr = _load_audio(audio_path)
    y = _trim_silence(y)
    if y.size == 0:
        y = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)

    n_fft = 512
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    frames = _frame_signal(y, frame_length, hop_length)
    spectrum = np.fft.rfft(frames, n=n_fft, axis=1)
    power = (np.abs(spectrum) ** 2).T

    mel_energy = np.dot(_mel_filterbank(sr, n_fft), power)
    mfcc = dct(np.log(mel_energy + 1e-9), type=2, axis=0, norm="ortho")[:40]
    delta = np.diff(mfcc, axis=1, prepend=mfcc[:, :1])

    freqs = np.linspace(0, sr / 2, power.shape[0])[:, None]
    energy = np.sum(power, axis=0, keepdims=True) + 1e-9
    centroid = np.sum(freqs * power, axis=0, keepdims=True) / energy
    bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * power, axis=0, keepdims=True) / energy)
    cumulative = np.cumsum(power, axis=0)
    rolloff_idx = np.argmax(cumulative >= 0.85 * cumulative[-1:, :], axis=0)
    rolloff = freqs[rolloff_idx].reshape(1, -1)

    rms = np.sqrt(np.mean(frames**2, axis=1, keepdims=True)).T
    zcr = (np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1, keepdims=True)).T
    min_frames = min(mfcc.shape[1], rms.shape[1], zcr.shape[1])
    stacked = np.vstack(
        [
            mfcc[:, :min_frames],
            delta[:, :min_frames],
            centroid[:, :min_frames],
            bandwidth[:, :min_frames],
            rolloff[:, :min_frames],
            rms[:, :min_frames],
            zcr[:, :min_frames],
        ]
    )
    return np.concatenate(
        [
            np.mean(stacked, axis=1),
            np.std(stacked, axis=1),
            np.min(stacked, axis=1),
            np.max(stacked, axis=1),
        ]
    )


def extract_speech_features(paths: list[str]) -> np.ndarray:
    ensure_dirs()
    cache = joblib.load(SPEECH_FEATURE_CACHE) if SPEECH_FEATURE_CACHE.exists() else {}
    features = []
    for audio_path in paths:
        cache_key = str(Path(audio_path).resolve())
        if cache_key not in cache:
            cache[cache_key] = _single_speech_feature(audio_path)
        features.append(cache[cache_key])
    joblib.dump(cache, SPEECH_FEATURE_CACHE)
    return np.asarray(features, dtype=np.float32)


def save_artifact(name: str, artifact: dict) -> Path:
    ensure_dirs()
    path = ARTIFACTS_DIR / f"{name}.joblib"
    joblib.dump(artifact, path)
    return path


def load_artifact(name: str) -> dict:
    return joblib.load(ARTIFACTS_DIR / f"{name}.joblib")


def evaluate_predictions(
    variant: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: list[str],
    representation: np.ndarray | None = None,
) -> dict:
    ensure_dirs()
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(RESULTS_DIR / f"{variant}_classification_report.csv")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"{variant.replace('_', ' ').title()} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{variant}_confusion_matrix.png", dpi=160)
    plt.close()

    if representation is not None and representation.shape[0] == len(y_true):
        plot_representation(variant, representation, y_true)

    summary = {"variant": variant, "accuracy": float(accuracy)}
    (RESULTS_DIR / f"{variant}_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def plot_representation(variant: str, representation: np.ndarray, labels: pd.Series) -> None:
    dense = representation.toarray() if hasattr(representation, "toarray") else np.asarray(representation)
    if dense.shape[1] > 2:
        dense = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(dense)
    plot_df = pd.DataFrame({"x": dense[:, 0], "y": dense[:, 1], "emotion": labels.to_numpy()})
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=plot_df, x="x", y="y", hue="emotion", s=35, alpha=0.8)
    plt.title(f"{variant.replace('_', ' ').title()} Representation Clusters")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{variant}_representation_pca.png", dpi=160)
    plt.close()


def update_accuracy_table() -> None:
    ensure_dirs()
    rows = []
    for metrics_file in sorted(RESULTS_DIR.glob("*_metrics.json")):
        rows.append(json.loads(metrics_file.read_text(encoding="utf-8")))
    if rows:
        pd.DataFrame(rows).sort_values("variant").to_csv(RESULTS_DIR / "accuracy_tables.csv", index=False)
