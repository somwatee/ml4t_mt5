# data_pipeline/data_collection.py
import MetaTrader5 as mt5
import pandas as pd
import sys
from data_pipeline.utils import load_config, get_logger

def collect_historical(config_path: str):
    cfg = load_config(config_path)
    log = get_logger("data_collection")

    # เชื่อมต่อ MT5
    term = cfg['mt5'].get('terminal_path')
    ok = mt5.initialize(path=term) if term else mt5.initialize()
    if not ok:
        log.error(f"MT5 initialize ล้มเหลว: {mt5.last_error()}")
        sys.exit(1)

    # ดึงข้อมูลราคา
    symbol = cfg['mt5']['symbol']
    tf     = cfg['mt5']['timeframe']
    bars   = cfg['mt5']['bars_to_fetch']
    rates  = mt5.copy_rates_from(symbol, tf, 0, bars)
    df     = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # บันทึกเป็น CSV
    df.to_csv(cfg['data']['historical_csv'], index=False)
    log.info(f"บันทึก {len(df)} แท่ง ไปที่ {cfg['data']['historical_csv']}")
    mt5.shutdown()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m data_pipeline.data_collection <config.yaml>")
        sys.exit(1)
    collect_historical(sys.argv[1])
