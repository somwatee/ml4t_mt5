# data_pipeline/ensemble.py
import sys
import joblib
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score
from data_pipeline.utils import load_config, get_logger

def run_ensemble(config_path: str):
    cfg = load_config(config_path)
    log = get_logger("ensemble")

    df = pd.read_csv(cfg['data']['features_csv'])
    feature_cols = [c for c in df.columns if c not in ('time','mss','fvg','ob','lv','bb')]
    X = df[feature_cols]
    y = df['mss'].astype(int)

    estimators = []
    for name, path in cfg['ensemble']['estimators']:
        estimators.append((name, joblib.load(path)))

    ens = VotingClassifier(
        estimators=estimators,
        voting=cfg['ensemble']['voting'],
        weights=cfg['ensemble']['weights']
    )
    ens.fit(X, y)

    preds = ens.predict(X)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average='macro')
    log.info(f"ผล Ensemble: accuracy={acc:.4f}, f1={f1:.4f}")

    joblib.dump(ens, "models/ensemble.pkl")
    log.info("บันทึกโมเดล ensemble ที่ models/ensemble.pkl")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m data_pipeline.ensemble <config.yaml>")
        sys.exit(1)
    run_ensemble(sys.argv[1])
