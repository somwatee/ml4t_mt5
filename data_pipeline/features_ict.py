import pandas as pd

def compute_mss(df: pd.DataFrame, lookback: int) -> pd.Series:
    highs = df['high'].rolling(window=lookback+1).max().shift(1)
    lows  = df['low'].rolling (window=lookback+1).min().shift(1)
    mss = pd.Series(0, index=df.index)
    mss[df['high'] > highs] = 1
    mss[df['low']  < lows ] = -1
    return mss.fillna(0).astype(int)

def compute_fvg(df: pd.DataFrame, threshold: float) -> pd.Series:
    # ตัวอย่างใช้ absolute gap
    gap = (df['open'] - df['close'].shift()).abs()
    # flag เมื่อ gap > threshold × ATR (ATR ต้องคำนวณก่อนใน features.py)
    # สมมติว่า df มีคอลัมน์ 'atr'
    fvg = (gap > threshold * df['atr']).astype(int)
    return fvg.fillna(0).astype(int)
