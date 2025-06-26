import sys
import json
import pandas as pd
import joblib
from data_pipeline.utils import load_config, get_logger

def predict_signals(config_path: str):
    cfg = load_config(config_path)
    log = get_logger("predict_file")

    # Load features
    df = pd.read_csv(cfg['data']['features_csv'], parse_dates=['time'])
    log.info(f"Loaded features for prediction: {len(df)} rows")

    # Load trained classifier
    clf = joblib.load(cfg['models']['label_classes'])
    classes = clf.classes_     # should be [0,1,2]
    log.info(f"Loaded model; classes = {classes}")

    # Predict probabilities
    X = df[[c for c in df.columns if c not in ('time','mss')]]
    probs = clf.predict_proba(X)
    thresh = cfg['training']['predict_threshold']

    # Build signals list
    out = []
    for t, p in zip(df['time'], probs):
        idx  = p.argmax()
        prob = p[idx]
        mapped = classes[idx]            # 0/1/2
        sig = (mapped - 1) if prob >= thresh else 0  # map back: 0→-1,1→0,2→1
        out.append({
            "time": pd.to_datetime(t).isoformat(),
            "signal": int(sig),
            "prob": float(prob)
        })

    # Write JSON
    path = cfg['data']['signals_json']
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    log.info(f"Saved {len(out)} signals to {path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m data_pipeline.predict_file config.yaml")
        sys.exit(1)
    predict_signals(sys.argv[1])
