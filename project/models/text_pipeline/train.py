from pathlib import Path
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

sys.path.append(str(Path(__file__).resolve().parents[2]))
from common import load_metadata, parse_args, save_artifact, split_metadata


def main() -> None:
    args = parse_args("Train the text-only emotion recognition pipeline.")
    metadata = load_metadata(args.data_root)
    train_df, test_df = split_metadata(metadata)
    labels = sorted(metadata["emotion"].unique())
    features = FeatureUnion(
        [
            ("word_tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 1))),
            ("char_tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 4))),
        ]
    )
    model = Pipeline(
        [
            ("features", features),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ]
    )
    model.fit(train_df["transcript"], train_df["emotion"])
    save_artifact(
        "text_model",
        {
            "model": model,
            "labels": labels,
            "train_files": train_df["file_name"].tolist(),
            "test_files": test_df["file_name"].tolist(),
        },
    )
    print("Trained text-only model.")


if __name__ == "__main__":
    main()
