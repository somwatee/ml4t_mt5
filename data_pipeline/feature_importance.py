# data_pipeline/feature_importance.py
import sys
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from data_pipeline.utils import load_config, get_logger

def plot_importance(config_path: str):
    cfg = load_config(config_path)
    log = get_logger("feature_importance")

    model = joblib.load(cfg['models']['xgb_model'])
    df = pd.read_csv(cfg['data']['features_csv'])
    feature_cols = [c for c in df.columns if c not in ('time','mss','fvg','ob','lv','bb')]

    importances = model.feature_importances_
    imp_sr = pd.Series(importances, index=feature_cols).sort_values(ascending=False)

    plt.figure(figsize=(8,6))
    imp_sr.head(20).plot.bar()
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    log.info("บันทึกกราฟ feature_importance.png")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m data_pipeline.feature_importance <config.yaml>")
        sys.exit(1)
    plot_importance(sys.argv[1])
