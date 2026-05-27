from pathlib import Path
import sys

from scipy.sparse import hstack

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
    args = parse_args("Evaluate the multimodal fusion emotion recognition pipeline.")
    artifact = load_artifact("fusion_model")
    metadata = load_metadata(args.data_root)
    _, test_df = split_metadata(metadata)

    speech_features = artifact["speech_scaler"].transform(extract_speech_features(test_df["path"].tolist()))
    text_features = artifact["text_vectorizer"].transform(test_df["transcript"])
    fused_features = hstack([speech_features, text_features]).tocsr()
    y_pred = artifact["classifier"].predict(fused_features)
    evaluate_predictions("multimodal_fusion", test_df["emotion"], y_pred, artifact["labels"], fused_features)
    update_accuracy_table()
    print("Evaluated multimodal fusion model.")


if __name__ == "__main__":
    main()
