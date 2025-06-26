# data_pipeline/features.py
import sys
import pandas as pd
import numpy as np
from data_pipeline.utils import load_config, get_logger
from data_pipeline.features_ict import compute_mss, compute_fvg
from data_pipeline.features_smc import (
    compute_order_block,
    compute_liquidity_void,
    compute_breaker_block
)
from ta.trend import EMAIndicator

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """คำนวณ RSI จาก series ปิด"""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """คำนวณ ATR จาก high, low, close"""
    high_low = df['high'] - df['low']
    high_pc  = (df['high'] - df['close'].shift()).abs()
    low_pc   = (df['low']  - df['close'].shift()).abs()
    tr       = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()

def generate_features(config_path: str):
    cfg = load_config(config_path)
    log = get_logger("features")

    # โหลดข้อมูลราคาย้อนหลัง
    df = pd.read_csv(cfg['data']['historical_csv'], parse_dates=['time'])
    df.set_index('time', inplace=True)

    # 1. พื้นฐาน: ATR, RSI
    df['atr'] = compute_atr(df, cfg['features']['ict']['atr_period'])
    df['rsi'] = compute_rsi(df['close'], cfg['features']['ict']['rsi_period'])

    # 2. คำนวณ trend_h1 ด้วย EMA (ใช้ค่า h1_ema_period จาก config)
    ema = EMAIndicator(df['close'], cfg['features']['trend_filter']['h1_ema_period']).ema_indicator()
    df['trend_h1'] = np.where(df['close'] > ema, 1, -1)
    log.info("คำนวณ trend_h1 ด้วย EMA เสร็จแล้ว")

    # 3. คำนวณฟีเจอร์ ICT/SMC
    df['mss'] = compute_mss(df, cfg['features']['ict']['mss_lookback'])
    df['fvg'] = compute_fvg(df, cfg['features']['ict']['fvg_threshold'])
    df['ob']  = compute_order_block(df, cfg['features']['smc']['order_block_size'])
    df['lv']  = compute_liquidity_void(df, cfg['features']['smc']['liquidity_void_depth'])
    df['bb']  = compute_breaker_block(df, cfg['features']['smc']['breaker_block_lookback'])

    # 4. บันทึกผลลัพธ์เป็น CSV
    df.reset_index().to_csv(cfg['data']['features_csv'], index=False)
    log.info(f"บันทึกฟีเจอร์ไปที่ {cfg['data']['features_csv']}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m data_pipeline.features <config.yaml>")
        sys.exit(1)
    generate_features(sys.argv[1])
