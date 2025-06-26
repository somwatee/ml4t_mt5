# data_pipeline/data_collection.py

import MetaTrader5 as mt5
import pandas as pd
from data_pipeline.utils import load_config, get_logger

def collect_historical(config_path: str):
    # โหลด config & logger
    cfg = load_config(config_path)
    log = get_logger("data_collection")

    # เริ่มต้น MT5 (ถ้ามีระบุ terminal_path จึงใส่พารามิเตอร์
    term = cfg['mt5'].get('terminal_path')
    if term:
        ok = mt5.initialize(path=term)
    else:
        ok = mt5.initialize()
    if not ok:
        log.error(f"MT5 initialize failed: {mt5.last_error()}")
        return

    # ถ้ามี login ให้ล็อกอิน
    if cfg['mt5'].get('login'):
        ok = mt5.login(
            cfg['mt5']['login'],
            password=cfg['mt5']['password'],
            server=cfg['mt5']['server']
        )
        if not ok:
            log.error(f"MT5 login failed: {mt5.last_error()}")
            mt5.shutdown()
            return

    symbol    = cfg['mt5']['symbol']
    timeframe = getattr(mt5, cfg['mt5']['timeframe'])
    n_bars    = cfg['mt5']['bars_to_fetch']

    log.info(f"Fetching {n_bars} bars for {symbol} @ {cfg['mt5']['timeframe']}")
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_bars)
    if rates is None or len(rates) == 0:
        log.error("No bars returned by MT5 API")
        mt5.shutdown()
        return

    # สร้าง DataFrame
    df = pd.DataFrame(rates)
    # ถ้าไม่มี column 'time' แต่มีชื่ออื่น เช่น 'time_msc' ให้เปลี่ยนชื่อ
    if 'time' not in df.columns and 'time_msc' in df.columns:
        df.rename(columns={'time_msc': 'time'}, inplace=True)

    # แปลงเป็น datetime (วินาที)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # เขียนไฟล์
    out_path = cfg['data']['historical_csv']
    df.to_csv(out_path, index=False)
    log.info(f"Saved historical data to {out_path}")

    mt5.shutdown()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python -m data_pipeline.data_collection config.yaml")
        sys.exit(1)
    collect_historical(sys.argv[1])
