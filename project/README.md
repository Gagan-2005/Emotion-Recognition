# Multimodal Emotion Recognition

This project follows the supplied PDF brief and builds three non-UI model variants for TESS emotion recognition:

- Speech-only
- Text-only
- Multimodal fusion

The dataset is expected at:

```text
../archive/TESS Toronto emotional speech set data
```

The TESS filename provides the transcript word and emotion label. Duplicate nested copies are ignored by filename.

## Setup

```bash
pip install -r requirements.txt
```

## Run

From the `project` folder:

```bash
python models/speech_pipeline/train.py
python models/speech_pipeline/test.py
python models/text_pipeline/train.py
python models/text_pipeline/test.py
python models/fusion_pipeline/train.py
python models/fusion_pipeline/test.py
```

Each script also accepts:

```bash
--data-root "path/to/TESS Toronto emotional speech set data"
```

## Website

Start the local website from the `project` folder:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

The website provides speech-only, text-only, and multimodal fusion prediction forms, the generated accuracy table, classification reports, confusion matrices, and representation cluster plots.

## Outputs

- `Results/accuracy_tables.csv`
- `Results/*_classification_report.csv`
- `Results/*_metrics.json`
- `Results/plots/*_confusion_matrix.png`
- `Results/plots/*_representation_pca.png`
- `artifacts/*.joblib`

## Architecture

Speech preprocessing resamples audio to 16 kHz, converts to mono, and trims silence. Speech features use MFCC, delta MFCC, spectral centroid, bandwidth, rolloff, zero crossing rate, and RMS statistics. The temporal modelling representation is a pooled summary over frame-level acoustic features, followed by logistic regression.

Text preprocessing uses the transcript word from each filename. Text features use TF-IDF word and character n-grams. The contextual modelling representation is the TF-IDF feature vector, followed by logistic regression.

Fusion concatenates scaled speech features and text TF-IDF features, then trains a logistic regression classifier.

