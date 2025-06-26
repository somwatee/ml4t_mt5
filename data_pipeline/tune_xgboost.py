# data_pipeline/tune_xgboost.py
import sys
import json
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import xgboost as xgb
import joblib
from data_pipeline.utils import load_config, get_logger

def tune_model(config_path: str):
    cfg = load_config(config_path)
    log = get_logger("tune_xgboost")

    df = pd.read_csv(cfg['data']['features_csv'])
    feature_cols = [c for c in df.columns if c not in ('mss','fvg','ob','lv','bb')]
    X = df[feature_cols]
    y = df['mss'].astype(int)

    tss = TimeSeriesSplit(n_splits=cfg['tuning']['cv_folds'])
    grid = GridSearchCV(
        xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        cfg['tuning']['param_grid'],
        cv=tss,
        scoring='f1_macro',
        n_jobs=-1
    )
    grid.fit(X, y)
    log.info(f"พารามิเตอร์ที่ดีที่สุด: {grid.best_params_}")
    with open('best_params.json', 'w') as f:
        json.dump(grid.best_params_, f, indent=2)

    joblib.dump(grid.best_estimator_, cfg['models']['xgb_model'])
    log.info(f"บันทึกโมเดลที่ tuned แล้วที่ {cfg['models']['xgb_model']}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m data_pipeline.tune_xgboost <config.yaml>")
        sys.exit(1)
    tune_model(sys.argv[1])
