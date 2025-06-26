# data_pipeline/backtest.py
import sys
import traceback
import pandas as pd
import joblib
from datetime import datetime
from data_pipeline.utils import load_config, get_logger

def run_backtest(config_path: str):
    cfg = load_config(config_path)
    log = get_logger("backtest")

    # โหลดข้อมูลฟีเจอร์
    df = pd.read_csv(cfg['data']['features_csv'], parse_dates=['time'])
    df.set_index('time', inplace=True)

    # โหลดโมเดล XGBoost
    model = joblib.load(cfg['models']['xgb_model'])
    log.info(f"โหลดโมเดลจาก {cfg['models']['xgb_model']}")

    # สัญญาณพรีดิกต์บนทุกแถว
    X = df.drop(columns=['mss', 'fvg', 'ob', 'lv', 'bb'])
    preds = model.predict(X)
    df['signal'] = preds

    # สร้างคอลัมน์ผลตอบแทน (สมมติ Long-only เมื่อ signal==1)
    df['return'] = df['close'].pct_change().shift(-1)  # ผลตอบแทนแท่งถัดไป
    df['strategy_ret'] = df['signal'] * df['return']

    # คำนวณผลรวมและดึง drawdown
    df['equity_curve'] = (1 + df['strategy_ret']).cumprod()
    peak = df['equity_curve'].cummax()
    df['drawdown'] = df['equity_curve'] / peak - 1

    # สรุปผล
    total_return = df['equity_curve'].iloc[-2] - 1
    max_dd       = df['drawdown'].min()
    days         = (df.index[-1] - df.index[0]).days
    annual_ret   = (1 + total_return) ** (365/days) - 1

    log.info(f"ผล Backtest:")
    log.info(f"  Total Return: {total_return:.2%}")
    log.info(f"  Annualized Return: {annual_ret:.2%}")
    log.info(f"  Max Drawdown: {max_dd:.2%}")

    # บันทึกรายงานเป็น CSV
    report = pd.DataFrame({
        'equity_curve': df['equity_curve'],
        'drawdown': df['drawdown']
    })
    report.to_csv("backtest_report.csv")
    log.info("บันทึก equity curve และ drawdown เป็น backtest_report.csv")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m data_pipeline.backtest <config.yaml>")
        sys.exit(1)
    try:
        run_backtest(sys.argv[1])
    except Exception:
        traceback.print_exc()
        sys.exit(2)
