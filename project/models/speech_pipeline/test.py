from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from common import (
    evaluate_predictions,
    extract_speech_features,
    load_artifact,
    load_metadata,
    parse_args,
    split_metadata,
    update_accuracy_table,
)


def main() -> None:
    args = parse_args("Evaluate the speech-only emotion recognition pipeline.")
    artifact = load_artifact("speech_model")
    metadata = load_metadata(args.data_root)
    _, test_df = split_metadata(metadata)
    x_test = extract_speech_features(test_df["path"].tolist())
    y_pred = artifact["model"].predict(x_test)
    evaluate_predictions("speech_only", test_df["emotion"], y_pred, artifact["labels"], x_test)
    update_accuracy_table()
    print("Evaluated speech-only model.")


if __name__ == "__main__":
    main()
