# data_pipeline/train_xgboost.py
import sys
import os
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from data_pipeline.utils import load_config, get_logger

def train_walk_forward(config_path: str):
    cfg = load_config(config_path)
    log = get_logger("train_xgboost")

    # โหลดฟีเจอร์
    df = pd.read_csv(cfg['data']['features_csv'])
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    # แม็ป label MSS จาก -1,0,1 → 0,1,2
    df['mss_mapped'] = df['mss'].map({-1: 0, 0: 1, 1: 2}).astype(int)

    # เตรียม X และ y
    feature_cols = [
        c for c in df.columns
        if c not in ('mss','mss_mapped','fvg','ob','lv','bb')
    ]
    X = df[feature_cols]
    y = df['mss_mapped']

    # สร้างลิสต์เดือนสำหรับ walk-forward
    dates = X.index.to_series().dt.to_period('M')
    months = sorted(dates.unique())

    window = cfg['cv']['window_size']
    test_m = cfg['cv']['test_size']
    step   = cfg['cv']['step_size']

    metrics = []
    for start in range(0, len(months) - window - test_m + 1, step):
        train_m = months[start:start+window]
        test_mo = months[start+window:start+window+test_m]
        idx_tr  = dates.isin(train_m)
        idx_te  = dates.isin(test_mo)

        X_tr, X_te = X[idx_tr], X[idx_te]
        y_tr, y_te = y[idx_tr], y[idx_te]

        model = xgb.XGBClassifier(**cfg['training']['xgb_params'])
        model.fit(X_tr, y_tr)

        preds = model.predict(X_te)
        metrics.append({
            'fold':     f"{train_m[0]}→{test_mo[-1]}",
            'accuracy': accuracy_score(y_te, preds),
            'f1':       f1_score(y_te, preds, average='macro')
        })

    # สรุป metric ถ้ามีข้อมูล
    if metrics:
        dfm = pd.DataFrame(metrics)
        desc = dfm.describe()[['accuracy','f1']]
        log.info(f"\nWalk-forward results:\n{desc}")
    else:
        log.warning("ไม่มี fold ให้ประมวลผล: ตรวจสอบช่วงเวลาใน config.cv")

    # เทรนใหม่ทั้งชุดข้อมูล
    final = xgb.XGBClassifier(**cfg['training']['xgb_params'])
    final.fit(X, y)

    # สร้างโฟลเดอร์ models ถ้ายังไม่มี
    model_path = cfg['models']['xgb_model']
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(final, model_path)
    log.info(f"บันทึกโมเดลเต็มชุดที่ {model_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m data_pipeline.train_xgboost <config.yaml>")
        sys.exit(1)
    train_walk_forward(sys.argv[1])
