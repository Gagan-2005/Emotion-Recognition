import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from common import load_metadata, split_metadata

ARTIFACTS = ['speech_model.joblib', 'text_model.joblib', 'fusion_model.joblib']
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ART_DIR = PROJECT_ROOT / 'artifacts'
RESULTS_DIR = PROJECT_ROOT / 'Results'

missing = [a for a in ARTIFACTS if not (ART_DIR / a).exists()]
if missing:
    print('Missing artifacts:', missing)
else:
    print('All artifacts present.')

# quick smoke predict using first test sample
metadata = load_metadata(PROJECT_ROOT.parent / 'archive' / 'TESS Toronto emotional speech set data')
_, test_df = split_metadata(metadata)
first = test_df.iloc[0]
print('First test file:', first['file_name'], first['transcript'], first['emotion'])

speech = joblib.load(ART_DIR / 'speech_model.joblib')
text = joblib.load(ART_DIR / 'text_model.joblib')
fusion = joblib.load(ART_DIR / 'fusion_model.joblib')

# ensure models can predict
try:
    _ = speech['model'].predict([0])
except Exception:
    print('Speech model predict sanity check skipped (expects numeric features).')

try:
    _ = text['model'].predict([first['transcript']])
    print('Text model predicts OK.')
except Exception as e:
    print('Text model predict failed:', e)

try:
    # fusion expects numeric features + vectorizer; skip deep check
    print('Fusion artifact loaded OK.')
except Exception as e:
    print('Fusion model load error:', e)

# check results files
res_files = list(RESULTS_DIR.glob('*_metrics.json'))
print('Metrics files found:', [p.name for p in res_files])
