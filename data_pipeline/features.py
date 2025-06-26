import sys
import pandas as pd
from data_pipeline.utils import load_config, get_logger
from data_pipeline.features_ict import compute_mss, compute_fvg
from data_pipeline.features_smc import (
    compute_order_block,
    compute_liquidity_void,
    compute_breaker_block
)

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df['high'] - df['low']
    hp = (df['high'] - df['close'].shift()).abs()
    lp = (df['low']  - df['close'].shift()).abs()
    tr = pd.concat([hl, hp, lp], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()

def compute_features(config_path: str):
    cfg = load_config(config_path)
    log = get_logger("features")

    # 1) load raw
    df = pd.read_csv(cfg['data']['historical_csv'], parse_dates=['time'])
    log.info("Loaded historical data")

    # 2) basic RSI/ATR
    df['rsi'] = compute_rsi(df['close'], period=cfg['training']['rsi_period'])
    df['atr'] = compute_atr(df,      period=cfg['training']['atr_period'])
    log.info("Calculated RSI and ATR")

    # 3) dropna for RSI/ATR
    df = df.dropna().reset_index(drop=True)

    # 4) trend H1‐equivalent
    if cfg['features'].get('use_trend', False):
        span_bars      = cfg['features']['trend_ema_h1'] * 60
        df['ema_h1']   = df['close'].ewm(span=span_bars, min_periods=span_bars).mean()
        df['trend_h1'] = (df['close'] > df['ema_h1']).astype(int)*2 - 1
        log.info(f"Calculated EMA{cfg['features']['trend_ema_h1']}H1 trend flag")
    else:
        df['trend_h1'] = 0

    # 5) ICT features
    if cfg['features'].get('use_ict', True):
        ict = cfg['features']['ict']
        df['mss'] = compute_mss(df, ict['mss_lookback'])
        df['fvg'] = compute_fvg(df, ict['fvg_threshold'])
        log.info("Calculated ICT features")
    else:
        df['mss'] = 0
        df['fvg'] = 0

    # 6) SMC features (trend‐filtered)
    smc = cfg['features']['smc']
    df['order_block']    = compute_order_block(df, smc['order_block_size'])
    df['liquidity_void'] = compute_liquidity_void(df, smc['liquidity_void_depth'])
    df['breaker_block']  = compute_breaker_block(df, smc['breaker_block_lookback'])
    log.info("Calculated trend‐filtered SMC features")

    # 7) lag features
    for col in ['close','rsi','atr','mss','fvg','order_block','liquidity_void','breaker_block']:
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_lag2"] = df[col].shift(2)
    log.info("Added lag features")

    # 8) save
    df.to_csv(cfg['data']['features_csv'], index=False)
    log.info(f"Saved features to {cfg['data']['features_csv']}")

if __name__ == "__main__":
    if len(sys.argv)!=2:
        print("Usage: python -m data_pipeline.features config.yaml")
        sys.exit(1)
    compute_features(sys.argv[1])
