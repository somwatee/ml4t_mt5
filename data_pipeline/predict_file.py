# data_pipeline/predict_file.py
import sys
import pandas as pd
import joblib
from data_pipeline.utils import load_config, get_logger

def predict_signals(config_path: str):
    cfg = load_config(config_path)
    log = get_logger("predict_file")

    df = pd.read_csv(cfg['data']['features_csv'], parse_dates=['time'])
    log.info(f"โหลดฟีเจอร์สำหรับพรีดิกต์: {len(df)} เรคคอร์ด")

    clf = joblib.load(cfg['models']['label_classes'])
    classes = clf.classes_
    log.info(f"โหลดโมเดลเสร็จ คลาส = {classes}")

    probs = clf.predict_proba(df.drop(columns=['time']))
    signals = [classes[p.argmax()] for p in probs]

    out = pd.DataFrame({'time': df['time'], 'signal': signals})
    out.to_json(cfg['data']['signals_json'], orient='records', date_format='iso')
    log.info(f"บันทึกสัญญาณไปที่ {cfg['data']['signals_json']}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m data_pipeline.predict_file <config.yaml>")
        sys.exit(1)
    predict_signals(sys.argv[1])
