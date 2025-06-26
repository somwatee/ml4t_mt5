# data_pipeline/trade_executor.py
import sys
import os
import time
import joblib
import csv
from datetime import datetime, timedelta

import pandas as pd
import MetaTrader5 as mt5
from ta.trend import EMAIndicator

from data_pipeline.utils import load_config, get_logger
from data_pipeline.features import compute_rsi, compute_atr
from data_pipeline.features_ict import compute_mss, compute_fvg
from data_pipeline.features_smc import (
    compute_order_block,
    compute_liquidity_void,
    compute_breaker_block
)

def init_trade_log(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.isfile(path):
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp_request','timestamp_filled','ticket',
                'side','price','lot','sl','tp','event','pnl'
            ])

def send_order(req: dict, cfg: dict, log):
    order_type = mt5.ORDER_TYPE_BUY if req['side']=='buy' else mt5.ORDER_TYPE_SELL
    request = {
        'action': mt5.TRADE_ACTION_DEAL,
        'symbol': cfg['mt5']['symbol'],
        'volume': req['lot'],
        'type': order_type,
        'price': req['price'],
        'sl': req['sl'],
        'tp': req['tp'],
        'deviation': 10,
        'magic': cfg['live'].get('magic_number', 0),
        'type_time': mt5.ORDER_TIME_GTC,
        'type_filling': mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error(f"Order send failed: {result.retcode}")
    return result

def execute_trading(config_path: str):
    cfg = load_config(config_path)
    log = get_logger("trade_executor")
    model = joblib.load(cfg['models']['xgb_model'])
    init_trade_log(cfg['data'].get('trade_log_csv','data/trade_log.csv'))

    # MT5 initialize & login
    term = cfg['mt5'].get('terminal_path')
    mt5.initialize(path=term) if term else mt5.initialize()
    if cfg['mt5'].get('login'):
        if not mt5.login(cfg['mt5']['login'], cfg['mt5']['password'], server=cfg['mt5']['server']):
            log.error(f"MT5 login failed: {mt5.last_error()}")
            mt5.shutdown()
            sys.exit(1)

    symbol = cfg['mt5']['symbol']
    if not mt5.symbol_select(symbol, True):
        mt5.symbol_add(symbol)

    # map timeframe
    tf_cfg = cfg['mt5']['timeframe']
    if isinstance(tf_cfg, str):
        tf = getattr(mt5, f"TIMEFRAME_{tf_cfg.upper()}", None)
        if tf is None:
            log.error(f"Invalid timeframe: {tf_cfg}")
            mt5.shutdown()
            sys.exit(1)
    else:
        tf = tf_cfg

    last_run = datetime.utcnow() - timedelta(minutes=cfg['live']['throttle_mins'])
    trades_today = 0
    open_positions = []

    required_lookback = max(
        cfg['features']['ict']['mss_lookback'],
        cfg['features']['smc']['order_block_size'],
        cfg['features']['smc']['liquidity_void_depth'] + 1,
        cfg['features']['smc']['breaker_block_lookback'] + 1
    )

    while True:
        now = datetime.utcnow()
        if (now - last_run).total_seconds() < cfg['live']['throttle_mins']*60:
            time.sleep(1)
            continue
        last_run = now

        rates = mt5.copy_rates_from(symbol, tf, 0, cfg['mt5']['bars_to_fetch'])
        df = pd.DataFrame(rates)
        log.info(f"Columns from MT5: {list(df.columns)}; rows={len(df)}")
        if df.empty or len(df) < required_lookback or 'time' not in df.columns:
            log.warning("ไม่พอข้อมูลสำหรับคำนวณฟีเจอร์ รอรอบหน้า...")
            continue

        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        # normalize OHLC
        col_map = {c:c.lower() for c in df.columns if c.lower() in ['open','high','low','close']}
        df.rename(columns=col_map, inplace=True)

        # calculate features
        df['atr'] = compute_atr(df, cfg['features']['ict']['atr_period'])
        df['rsi'] = compute_rsi(df['close'], cfg['features']['ict']['rsi_period'])
        ema = EMAIndicator(df['close'], cfg['features']['trend_filter']['h1_ema_period']).ema_indicator()
        df['trend_h1'] = (df['close'] > ema).astype(int).replace({0:-1})
        df['mss'] = compute_mss(df, cfg['features']['ict']['mss_lookback'])
        df['fvg'] = compute_fvg(df, cfg['features']['ict']['fvg_threshold'])
        df['ob']  = compute_order_block(df, cfg['features']['smc']['order_block_size'])
        df['lv']  = compute_liquidity_void(df, cfg['features']['smc']['liquidity_void_depth'])
        df['bb']  = compute_breaker_block(df, cfg['features']['smc']['breaker_block_lookback'])

        # prepare X
        drop_cols = ['open','high','low','close','tick_volume','spread','real_volume']
        X = df.iloc[[-1]].drop(columns=drop_cols, errors='ignore')

        expected = ['atr','rsi','trend_h1','mss','fvg','ob','lv','bb']
        if set(X.columns) != set(expected):
            log.error(f"Feature mismatch: expected {expected}, got {list(X.columns)}")
            continue

        pred = model.predict(X)[0]
        proba = model.predict_proba(X).max()
        log.info(f"Signal={pred} (p={proba:.2f})")

        # compute order size & SL/TP
        atr = df['atr'].iat[-1]
        bal = mt5.account_info().balance
        risk_amt = bal * cfg['live']['max_risk_pct']
        sl_dist = cfg['live']['sl_multiplier'] * atr
        lot = risk_amt / (sl_dist * cfg['mt5']['contract_size'])
        sl = df['close'].iat[-1] - sl_dist if pred==1 else df['close'].iat[-1] + sl_dist
        tp_dist = cfg['live']['tp2_multiplier'] * atr
        tp = df['close'].iat[-1] + tp_dist if pred==1 else df['close'].iat[-1] - tp_dist

        if trades_today < cfg['live']['max_trades_per_day'] and proba >= cfg['training']['predict_threshold']:
            req = {
                'side': 'buy' if pred==1 else 'sell',
                'price': df['close'].iat[-1],
                'lot': round(lot,2),
                'sl': sl,
                'tp': tp
            }
            res = send_order(req, cfg, log)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                trades_today += 1
                open_positions.append({
                    'ticket': res.order,
                    'open_time': datetime.utcnow(),
                    'side': req['side'],
                    'sl': sl,
                    'tp': tp
                })
                log.info(f"Opened order {req}")

        # manage open positions...
        for pos in open_positions.copy():
            age = datetime.utcnow() - pos['open_time']
            ticket = pos['ticket']
            if age > timedelta(minutes=cfg['live']['time_stop_mins']):
                vol = mt5.positions_get(ticket=ticket)[0].volume
                mt5.order_close(ticket, vol, df['close'].iat[-1], 10)
                open_positions.remove(pos)
                log.info(f"Closed {ticket} by time-stop")
            else:
                if pos['side']=='buy':
                    new_sl = max(pos['sl'], df['close'].iat[-1] - cfg['live']['trailing_multiplier']*atr)
                else:
                    new_sl = min(pos['sl'], df['close'].iat[-1] + cfg['live']['trailing_multiplier']*atr)
                if new_sl != pos['sl']:
                    mt5.order_modify(ticket, df['close'].iat[-1], new_sl, pos['tp'], 10)
                    pos['sl'] = new_sl
                    log.info(f"Updated SL for {ticket} → {new_sl:.5f}")

    mt5.shutdown()

if __name__=="__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    execute_trading(args.config)
