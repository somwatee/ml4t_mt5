# data_pipeline/tune_xgboost.py
import sys
import os
import json
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import xgboost as xgb
import joblib
from data_pipeline.utils import load_config, get_logger

def tune_model(config_path: str):
    cfg = load_config(config_path)
    log = get_logger("tune_xgboost")

    # อ่านฟีเจอร์
    df = pd.read_csv(cfg['data']['features_csv'])
    df['mss_mapped'] = df['mss'].map({-1: 0, 0: 1, 1: 2}).astype(int)

    feature_cols = [
        c for c in df.columns
        if c not in ('time','mss','mss_mapped','fvg','ob','lv','bb')
    ]
    X = df[feature_cols]
    y = df['mss_mapped']

    # กำหนด TimeSeriesSplit ตาม config
    tss = TimeSeriesSplit(n_splits=cfg['tuning']['cv_folds'])
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    grid = GridSearchCV(
        estimator=model,
        param_grid=cfg['tuning']['param_grid'],
        cv=tss,
        scoring='f1_macro',
        n_jobs=-1,
        error_score='raise'
    )

    log.info("เริ่มทำ Hyper-parameter tuning ด้วย GridSearchCV...")
    grid.fit(X, y)
    log.info(f"พารามิเตอร์ที่ดีที่สุด: {grid.best_params_}")

    # บันทึกพารามิเตอร์ที่ดีที่สุด
    with open('best_params.json', 'w', encoding='utf-8') as f:
        json.dump(grid.best_params_, f, ensure_ascii=False, indent=2)
    log.info("บันทึกไฟล์ best_params.json เรียบร้อย")

    # สร้างโฟลเดอร์ models ถ้ายังไม่มี
    model_path = cfg['models']['xgb_model']
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # บันทึกโมเดล tuned
    joblib.dump(grid.best_estimator_, model_path)
    log.info(f"บันทึกโมเดล tuned ที่ {model_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m data_pipeline.tune_xgboost <config.yaml>")
        sys.exit(1)
    tune_model(sys.argv[1])
