import pandas as pd

def compute_order_block(df: pd.DataFrame, size: int) -> pd.Series:
    flags=[]
    for i in range(len(df)):
        if i<size:
            flags.append(0); continue
        tr = df.at[i,'trend_h1']
        if tr<0:
            win = df['high'].iloc[i-size:i]
            flags.append(1 if df['high'].iat[i] <= win.max() else 0)
        else:
            win = df['low'].iloc[i-size:i]
            flags.append(-1 if df['low'].iat[i] >= win.min() else 0)
    return pd.Series(flags, index=df.index)

def compute_liquidity_void(df: pd.DataFrame, depth: int) -> pd.Series:
    flags=[]
    for i in range(len(df)):
        if i<depth:
            flags.append(0); continue
        tr  = df.at[i,'trend_h1']
        rng = df['high'].iat[i]-df['low'].iat[i]
        hist = df['high'].iloc[i-depth:i].max() - df['low'].iloc[i-depth:i].min()
        if rng<0.2*hist:
            flags.append(1 if tr<0 else -1)
        else:
            flags.append(0)
    return pd.Series(flags, index=df.index)

def compute_breaker_block(df: pd.DataFrame, lookback: int) -> pd.Series:
    flags=[]
    for i in range(len(df)):
        if i<lookback:
            flags.append(0); continue
        tr= df.at[i,'trend_h1']
        lowp  = df['low'].iloc[i-lookback:i].min()
        highp = df['high'].iloc[i-lookback:i].max()
        if tr<0 and df['low'].iat[i]<lowp and df['close'].iat[i]<df['open'].iat[i]:
            flags.append(1)
        elif tr>0 and df['high'].iat[i]>highp and df['close'].iat[i]>df['open'].iat[i]:
            flags.append(-1)
        else:
            flags.append(0)
    return pd.Series(flags, index=df.index)
