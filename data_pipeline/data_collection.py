# data_pipeline/data_collection.py
import MetaTrader5 as mt5
import pandas as pd
import sys
from data_pipeline.utils import load_config, get_logger

def collect_historical(config_path: str):
    cfg = load_config(config_path)
    log = get_logger("data_collection")

    # 1. เชื่อมต่อกับ MT5
    term = cfg['mt5'].get('terminal_path')
    ok = mt5.initialize(path=term) if term else mt5.initialize()
    if not ok:
        log.error(f"MT5 initialize ล้มเหลว: {mt5.last_error()}")
        sys.exit(1)

    # 2. ล็อกอิน MT5 (ถ้ามี)
    login = cfg['mt5'].get('login')
    pwd   = cfg['mt5'].get('password')
    srv   = cfg['mt5'].get('server')
    if login and pwd:
        if not mt5.login(login, pwd, server=srv):
            log.error(f"MT5 login ล้มเหลว: {mt5.last_error()}")
            mt5.shutdown()
            sys.exit(1)
    else:
        log.warning("ไม่พบ mt5.login/mt5.password ใน config — อาจ fetch ไม่สำเร็จ")

    # 3. แปลง timeframe (รองรับ int หรือ str เช่น 'M1')
    tf_cfg = cfg['mt5']['timeframe']
    if isinstance(tf_cfg, str):
        tf = getattr(mt5, f"TIMEFRAME_{tf_cfg}", None)
        if tf is None:
            log.error(f"Invalid timeframe: {tf_cfg}")
            mt5.shutdown()
            sys.exit(1)
    else:
        tf = tf_cfg

    symbol = cfg['mt5']['symbol']
    bars   = cfg['mt5']['bars_to_fetch']

    # 4. ตรวจสอบสัญลักษณ์ใน Market Watch
    if mt5.symbol_info(symbol) is None:
        log.error(f"สัญลักษณ์ {symbol} ไม่ถูกต้อง หรือไม่ได้เพิ่มใน Market Watch")
        mt5.shutdown()
        sys.exit(1)
    else:
        log.info(f"Symbol {symbol} พร้อมใช้งาน")

    # 5. ดึงข้อมูลย้อนหลังด้วย positional API
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
    if rates is None or len(rates) == 0:
        log.warning("ไม่พบข้อมูลราคาย้อนหลัง หรือ fetch ไม่สำเร็จ")
        mt5.shutdown()
        return

    # 6. แปลงเป็น DataFrame
    df = pd.DataFrame(rates)

    # 7. รองรับชื่อคอลัมน์เวลาเป็น time หรือ time_msc
    if 'time' not in df.columns:
        if 'time_msc' in df.columns:
            df.rename(columns={'time_msc': 'time'}, inplace=True)
        else:
            log.error("ไม่มีคอลัมน์ 'time' หรือ 'time_msc' จาก MT5 API")
            mt5.shutdown()
            sys.exit(1)

    # 8. แปลง timestamp → datetime และบันทึก CSV
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.to_csv(cfg['data']['historical_csv'], index=False)
    log.info(f"บันทึก {len(df)} แท่ง ไปที่ {cfg['data']['historical_csv']}")

    # 9. ปิดการเชื่อมต่อ MT5
    mt5.shutdown()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m data_pipeline.data_collection <config.yaml>")
        sys.exit(1)
    collect_historical(sys.argv[1])
