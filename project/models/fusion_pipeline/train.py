from pathlib import Path
import sys

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[2]))
from common import extract_speech_features, load_metadata, parse_args, save_artifact, split_metadata


def main() -> None:
    args = parse_args("Train the multimodal fusion emotion recognition pipeline.")
    metadata = load_metadata(args.data_root)
    train_df, test_df = split_metadata(metadata)
    labels = sorted(metadata["emotion"].unique())

    speech_scaler = StandardScaler()
    speech_features = speech_scaler.fit_transform(extract_speech_features(train_df["path"].tolist()))
    text_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))
    text_features = text_vectorizer.fit_transform(train_df["transcript"])
    fused_features = hstack([speech_features, text_features]).tocsr()

    classifier = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    classifier.fit(fused_features, train_df["emotion"])
    save_artifact(
        "fusion_model",
        {
            "speech_scaler": speech_scaler,
            "text_vectorizer": text_vectorizer,
            "classifier": classifier,
            "labels": labels,
            "train_files": train_df["file_name"].tolist(),
            "test_files": test_df["file_name"].tolist(),
        },
    )
    print("Trained multimodal fusion model.")


if __name__ == "__main__":
    main()
