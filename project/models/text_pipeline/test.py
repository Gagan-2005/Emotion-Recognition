from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from common import evaluate_predictions, load_artifact, load_metadata, parse_args, split_metadata, update_accuracy_table


def main() -> None:
    args = parse_args("Evaluate the text-only emotion recognition pipeline.")
    artifact = load_artifact("text_model")
    metadata = load_metadata(args.data_root)
    _, test_df = split_metadata(metadata)
    model = artifact["model"]
    y_pred = model.predict(test_df["transcript"])
    representation = model.named_steps["features"].transform(test_df["transcript"])
    evaluate_predictions("text_only", test_df["emotion"], y_pred, artifact["labels"], representation)
    update_accuracy_table()
    print("Evaluated text-only model.")


if __name__ == "__main__":
    main()
