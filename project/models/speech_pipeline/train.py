from pathlib import Path
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[2]))
from common import extract_speech_features, load_metadata, parse_args, save_artifact, split_metadata


def main() -> None:
    args = parse_args("Train the speech-only emotion recognition pipeline.")
    metadata = load_metadata(args.data_root)
    train_df, test_df = split_metadata(metadata)
    x_train = extract_speech_features(train_df["path"].tolist())
    labels = sorted(metadata["emotion"].unique())

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ]
    )
    model.fit(x_train, train_df["emotion"])
    save_artifact(
        "speech_model",
        {
            "model": model,
            "labels": labels,
            "train_files": train_df["file_name"].tolist(),
            "test_files": test_df["file_name"].tolist(),
        },
    )
    print("Trained speech-only model.")


if __name__ == "__main__":
    main()
