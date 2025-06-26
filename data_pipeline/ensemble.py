# data_pipeline/ensemble.py
import sys
import os
import joblib
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score
from data_pipeline.utils import load_config, get_logger

def run_ensemble(config_path: str):
    cfg = load_config(config_path)
    log = get_logger("ensemble")

    # โหลดฟีเจอร์
    df = pd.read_csv(cfg['data']['features_csv'])
    feature_cols = [c for c in df.columns if c not in ('time','mss','fvg','ob','lv','bb')]
    X = df[feature_cols]
    y = df['mss'].map({-1: 0, 0: 1, 1: 2}).astype(int)

    # เตรียม estimators
    estimators = []
    for name, path in cfg['ensemble']['estimators']:
        if not os.path.isfile(path):
            log.warning(f"ไม่พบโมเดล `{name}` ที่ {path} — ข้ามโมเดลนี้")
            continue
        model = joblib.load(path)
        estimators.append((name, model))
        log.info(f"โหลดโมเดล `{name}` จาก {path}")

    if not estimators:
        log.error("ไม่พบโมเดลใด ๆ สำหรับ ensemble. ตรวจสอบ config.yaml")
        sys.exit(1)

    # ตรวจสอบ weights ให้ตรงจำนวน estimators
    weights = cfg['ensemble'].get('weights')
    if weights and len(weights) == len(estimators):
        use_weights = weights
    else:
        if weights:
            log.warning(f"จำนวน weights ({len(weights)}) ไม่ตรงกับ estimators ({len(estimators)}). จะไม่ใช้ weights")
        use_weights = None

    # สร้าง VotingClassifier
    ens = VotingClassifier(
        estimators=estimators,
        voting=cfg['ensemble']['voting'],
        weights=use_weights
    )
    log.info(f"เริ่มฝึก ensemble จาก {len(estimators)} โมเดล...")
    ens.fit(X, y)

    # ประเมินผลบนชุดเดียวกัน
    preds = ens.predict(X)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average='macro')
    log.info(f"ผล Ensemble: accuracy={acc:.4f}, f1={f1:.4f}")

    # บันทึก ensemble model
    out_path = cfg['ensemble'].get('output_model', 'models/ensemble.pkl')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(ens, out_path)
    log.info(f"บันทึกโมเดล ensemble ที่ {out_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m data_pipeline.ensemble <config.yaml>")
        sys.exit(1)
    run_ensemble(sys.argv[1])
