# data_pipeline/features_smc.py
import pandas as pd

def compute_order_block(df: pd.DataFrame, size: int) -> pd.Series:
    """Order Block: สัญญาณตามแนวโน้ม H1 และราคาทะลุกรอบ"""
    flags = []
    for i in range(len(df)):
        if i < size:
            flags.append(0)
            continue
        tr = df.at[i, 'trend_h1']
        if tr < 0:
            win = df['high'].iloc[i-size:i]
            flags.append(1 if df['high'].iat[i] <= win.max() else 0)
        else:
            win = df['low'].iloc[i-size:i]
            flags.append(-1 if df['low'].iat[i] >= win.min() else 0)
    return pd.Series(flags, index=df.index)

def compute_liquidity_void(df: pd.DataFrame, depth: int) -> pd.Series:
    """Liquidity Void: แท่งเล็กกว่าระดับ threshold"""
    flags = []
    for i in range(len(df)):
        if i < depth:
            flags.append(0)
            continue
        tr = df.at[i, 'trend_h1']
        rng = df['high'].iloc[i-depth:i] - df['low'].iloc[i-depth:i]
        curr = df.at[i, 'high'] - df.at[i, 'low']
        thr = rng.max() * 0.2
        if curr < thr:
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
        prev_high = df['high'].iloc[i-lookback-1:i-1].max()
        prev_low  = df['low'].iloc[i-lookback-1:i-1].min()
        o,c = df.at[i, 'open'], df.at[i, 'close']
        if c < o and o > prev_high:
            flags.append(-1)
        elif c > o and o < prev_low:
            flags.append(1)
        else:
            flags.append(0)
    return pd.Series(flags, index=df.index)
