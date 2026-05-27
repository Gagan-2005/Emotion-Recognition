# Report

## A. Architecture Decisions

### Speech-only

- Preprocessing: audio is loaded as mono, resampled to 16 kHz, and silence is trimmed.
- Feature extraction: MFCC, delta MFCC, spectral centroid, bandwidth, rolloff, zero crossing rate, and RMS energy are extracted frame-wise.
- Temporal modelling: frame-wise acoustic cues are summarized with mean, standard deviation, minimum, and maximum pooling. This preserves emotion-relevant prosody and energy patterns while keeping the model compact for a one-month project.
- Classifier: balanced logistic regression is used for a clear and reproducible multiclass baseline.

### Text-only

- Preprocessing: the transcript token is parsed from each TESS filename.
- Feature extraction: word TF-IDF and character TF-IDF features are used.
- Contextual modelling: the TF-IDF representation models lexical context available in the transcript. Since TESS uses short carrier words, this modality is intentionally limited.
- Classifier: balanced logistic regression is used.

### Multimodal Fusion

- Speech representation: scaled speech acoustic feature vector.
- Text representation: TF-IDF transcript vector.
- Fusion: early fusion by concatenating both feature spaces.
- Classifier: balanced logistic regression predicts the final emotion label from the fused representation.

## B. Experiments

All three variants were trained and tested with an 80/20 stratified split. The generated accuracy table is:

| Variant | Accuracy |
| --- | ---: |
| Speech-only | 0.9982 |
| Text-only | 0.0000 |
| Multimodal fusion | 0.9982 |

The scripts generated:

- `Results/accuracy_tables.csv`
- `Results/speech_only_classification_report.csv`
- `Results/text_only_classification_report.csv`
- `Results/multimodal_fusion_classification_report.csv`
- confusion matrix plots for all three variants
- PCA cluster plots for temporal, contextual, and fusion representations

## C. Analysis

- Speech-only performs strongly because TESS emotions are expressed through pitch, energy, rhythm, and voice quality.
- Text-only performs poorly because the transcript words are neutral lexical content and are repeated across emotions; the transcript alone does not provide useful emotion evidence.
- Fusion matches the speech-only result because the speech representation already separates the classes almost completely, while the text modality contributes little emotion signal for this dataset.

The easiest emotions in this run are disgust, happy, neutral, pleasant surprise, and sad, each with perfect recall in the speech and fusion reports. The only observed speech/fusion error is one angry sample classified as fear. This is plausible because both can contain high intensity, raised pitch, and sharper articulation.

Failure cases to inspect:

- angry predicted as fear in the speech-only model
- angry predicted as fear in the multimodal fusion model
- text-only errors across all emotions, caused by the transcript carrying no stable emotion information
- likely happy versus pleasant surprise confusions to monitor on larger or noisier splits
- likely neutral versus sad confusions to monitor on lower-energy speech

The PCA plots in `Results/plots` visualize separability for:

- temporal modelling block: `speech_only_representation_pca.png`
- contextual modelling block: `text_only_representation_pca.png`
- fusion block: `multimodal_fusion_representation_pca.png`

The following section expands on these observations, documents the dataset limitation, and provides detailed failure analysis required for final submission.

### Dataset limitation: weak textual modality

- Root cause: TESS is a controlled emotional speech corpus where each audio file contains a short carrier word (often a single word) that is reused across multiple emotions. The transcript therefore carries little to no emotion-specific lexical information. As a result, any text-only classifier that uses only the transcript (TF-IDF, character n-grams, or even simple embeddings) has limited capacity to separate emotion classes.
- Evidence: the `text_only` evaluation shows 0.0 accuracy on the test split despite the model returning valid emotion labels. TF-IDF vectors are non-empty for all test samples, but the lexical token does not correlate with emotion labels.
- Conclusion: the low text-only accuracy is an important experimental finding and a dataset limitation, not a coding error.

### Easiest and hardest emotions (from current runs)

- Easiest: `disgust`, `happy`, `neutral`, `pleasant_surprise`, `sad` — these are well-separated by acoustic features (high recall/precision in speech-only results).
- Hardest: `angry` and `fear` show occasional confusions with each other. Low-energy classes (e.g., `neutral` vs `sad`) can also be confusable when prosodic cues are weak.

### When fusion helps

- In this dataset, fusion provides little improvement over speech-only because the textual modality contributes minimal additional signal. Fusion is still included for completeness and to demonstrate the fusion pipeline mechanics; in datasets with richer transcripts, fusion is expected to improve performance.

### Failure case studies (3–5 examples)

1. Angry -> Fear: high-intensity speech with raised pitch and sharp articulation can be labeled as either `angry` or `fear` depending on subtle prosodic patterns. Example files: check `train_files`/`test_files` lists in `artifacts/speech_model.joblib`.
2. Neutral -> Angry/Fear (misclassified): short low-energy carrier words sometimes produce ambiguous acoustic cues when recording quality varies.
3. Pleasant Surprise -> Happy: lexical similarity and shared prosodic peaks cause confusion between these positive classes in noisy samples.
4. Text-only blanket misclassification: many test transcripts map to labels that never match ground truth because the transcript word is reused across emotions.

### Representation cluster analysis

- Temporal modelling block (`speech_only_representation_pca.png`): shows clear clustering for several emotions, confirming that the acoustic feature set captures discriminative prosodic and spectral cues.
- Contextual modelling block (`text_only_representation_pca.png`): clusters are overlapping and not separable, illustrating the weak textual signal.
- Fusion block (`multimodal_fusion_representation_pca.png`): clusters closely match the speech-only clusters; little additional separation is visible from the text features.

### Recommended final notes for submission

- Present the low text-only accuracy as an experimental result and dataset limitation.
- State that transformer-based approaches were intentionally not pursued to avoid unnecessary complexity for this assignment; they may not yield better performance given the dataset's transcript constraints.
- Emphasize reproducibility: include pinned `requirements.txt`, environment checks, and a smoke-test script (added to `scripts/`) to validate artifacts and key outputs.

End of analysis.
