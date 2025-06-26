# data_pipeline/trade_executor.py
import sys
import time
import joblib
import csv
import os
from datetime import datetime, timedelta
from collections import deque

import pandas as pd
import MetaTrader5 as mt5
from ta.trend import ADXIndicator, EMAIndicator

from data_pipeline.utils import load_config, get_logger
from data_pipeline.features import compute_rsi, compute_atr
from data_pipeline.features_ict import compute_mss, compute_fvg
from data_pipeline.features_smc import (
    compute_order_block,
    compute_liquidity_void,
    compute_breaker_block
)

def send_order(request: dict, cfg: dict, log):
    """ฟังก์ชันส่งคำสั่งเปิดออร์เดอร์ (implement ตาม logic ของคุณ)"""
    # สร้างคำสั่ง mt5.order_send(...)
    return {}

def execute_trading(config_path: str):
    cfg = load_config(config_path)
    log = get_logger("trade_executor")
    model = joblib.load(cfg['models']['xgb_model'])

    # เริ่มต้น MT5
    term = cfg['mt5'].get('terminal_path')
    ok = mt5.initialize(path=term) if term else mt5.initialize()
    if not ok:
        log.error(f"MT5 init ล้มเหลว: {mt5.last_error()}")
        sys.exit(1)

    trades_today = 0
    last_run = datetime.utcnow() - timedelta(minutes=cfg['live']['throttle_mins'])

    while True:
        now = datetime.utcnow()
        if (now - last_run).total_seconds() < cfg['live']['throttle_mins']*60:
            time.sleep(1)
            continue
        last_run = now

        # ดึงราคาย้อนหลัง
        rates = mt5.copy_rates_from(
            cfg['mt5']['symbol'],
            cfg['mt5']['timeframe'], 0,
            cfg['mt5']['bars_to_fetch']
        )
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # คำนวณฟีเจอร์
        df['atr'] = compute_atr(df, cfg['features']['ict']['atr_period'])
        df['rsi'] = compute_rsi(df['close'], cfg['features']['ict']['rsi_period'])
        df['mss'] = compute_mss(df, cfg['features']['ict']['mss_lookback'])
        df['fvg'] = compute_fvg(df, cfg['features']['ict']['fvg_threshold'])
        df['ob']  = compute_order_block(df, cfg['features']['smc']['order_block_size'])
        df['lv']  = compute_liquidity_void(df, cfg['features']['smc']['liquidity_void_depth'])
        df['bb']  = compute_breaker_block(df, cfg['features']['smc']['breaker_block_lookback'])

        # พรีดิกต์แถวสุดท้าย
        X = df.drop(columns=['time']).iloc[-1:].values
        pred = model.predict(X)[0]
        proba = model.predict_proba(X).max()

        if trades_today < cfg['live']['max_trades_per_day'] and proba >= cfg['training']['predict_threshold']:
            req = {'side': 'buy' if pred==1 else 'sell'}
            fill = send_order(req, cfg, log)
            trades_today += 1

        # TODO: จัดการ time_stop & trailing

    mt5.shutdown()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    execute_trading(args.config)
