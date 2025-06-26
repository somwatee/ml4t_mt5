# data_pipeline/features_ict.py
import pandas as pd

def compute_mss(df: pd.DataFrame, lookback: int) -> pd.Series:
    """Market Structure Shift: 1=สูงทะลุ, -1=ต่ำทะลุ, 0=ไม่เปลี่ยน"""
    highs = df['high'].rolling(window=lookback+1).max().shift(1)
    lows  = df['low'].rolling (window=lookback+1).min().shift(1)
    mss = pd.Series(0, index=df.index)
    mss[df['high'] > highs] = 1
    mss[df['low']  < lows ] = -1
    return mss.fillna(0).astype(int)

def compute_fvg(df: pd.DataFrame, threshold: float) -> pd.Series:
    """Fair Value Gap: |open–close_prev| > threshold×ATR"""
    gap = (df['open'] - df['close'].shift()).abs()
    fvg = (gap > threshold * df['atr']).astype(int)
    return fvg.fillna(0).astype(int)
