# data_pipeline/features_smc.py
import pandas as pd

def compute_order_block(df: pd.DataFrame, size: int) -> pd.Series:
    """Order Block: สัญญาณตามแนวโน้ม H1 และราคาทะลุกรอบ"""
    flags = []
    for i in range(len(df)):
        if i < size:
            flags.append(0)
            continue
        # อ่าน trend_h1 ด้วย .iat สำหรับตำแหน่ง i
        tr = df['trend_h1'].iat[i]
        if tr < 0:
            window_highs = df['high'].iloc[i-size:i]
            curr_high = df['high'].iat[i]
            flags.append(1 if curr_high <= window_highs.max() else 0)
        else:
            window_lows = df['low'].iloc[i-size:i]
            curr_low = df['low'].iat[i]
            flags.append(-1 if curr_low >= window_lows.min() else 0)
    return pd.Series(flags, index=df.index)

def compute_liquidity_void(df: pd.DataFrame, depth: int) -> pd.Series:
    """Liquidity Void: แท่งเล็กกว่าระดับ threshold"""
    flags = []
    for i in range(len(df)):
        if i < depth:
            flags.append(0)
            continue
        tr = df['trend_h1'].iat[i]
        window = df.iloc[i-depth:i]
        rng = window['high'] - window['low']
        curr_range = df['high'].iat[i] - df['low'].iat[i]
        threshold = rng.max() * 0.2
        if curr_range < threshold:
            flags.append(1 if tr < 0 else -1)
        else:
            flags.append(0)
    return pd.Series(flags, index=df.index)

def compute_breaker_block(df: pd.DataFrame, lookback: int) -> pd.Series:
    """Breaker Block: แท่งย้อนหลังกำหนดตำแหน่ง high/low"""
    flags = []
    for i in range(len(df)):
        if i < lookback + 1:
            flags.append(0)
            continue
        prev_window = df.iloc[i-lookback-1:i-1]
        prev_high = prev_window['high'].max()
        prev_low = prev_window['low'].min()
        open_i = df['open'].iat[i]
        close_i = df['close'].iat[i]
        if close_i < open_i and open_i > prev_high:
            flags.append(-1)
        elif close_i > open_i and open_i < prev_low:
            flags.append(1)
        else:
            flags.append(0)
    return pd.Series(flags, index=df.index)
