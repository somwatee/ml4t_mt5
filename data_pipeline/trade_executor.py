import sys
import time
import joblib
import csv
import os
from datetime import datetime, timedelta
from collections import deque

import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from ta.trend import ADXIndicator, EMAIndicator

from data_pipeline.utils import load_config, get_logger
from data_pipeline.features import compute_rsi, compute_atr
from data_pipeline.features_ict import compute_mss, compute_fvg
from data_pipeline.features_smc import compute_order_block, compute_liquidity_void, compute_breaker_block

# trade_log path
DATA_DIR = 'data'
LOG_FILE = os.path.join(DATA_DIR, 'trade_log.csv')

# Initialize log file with header if not exists
def init_log():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.isfile(LOG_FILE):
        header = ['timestamp_request','timestamp_filled','ticket','side','price','lot','sl','tp','event','pnl']
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

# Append a row dict to trade_log
def append_log(row: dict):
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)

# Calculate lot size with rounding
def calculate_lot(equity, atr, max_risk_pct, sl_mult, contract_size, decimals):
    risk_amount = equity * max_risk_pct
    sl_distance = atr * sl_mult
    raw_lot = risk_amount / (sl_distance * contract_size) if sl_distance > 0 else 0
    return round(raw_lot, decimals)

# Calculate SL and TP levels
def calculate_levels(price, atr, sl_mult, tp_mult, direction):
    sl = price - direction * atr * sl_mult
    tp = price + direction * atr * tp_mult
    return sl, tp

# Calculate trailing stop (based on ATR multiplier)
def calculate_trailing_stop(current_price, entry_price, current_sl, atr, trail_mult, direction):
    if direction == 1:
        new_sl = current_price - atr * trail_mult
        return max(new_sl, current_sl)
    else:
        new_sl = current_price + atr * trail_mult
        return min(new_sl, current_sl)

# Main executor
def execute_trades(config_path: str):
    cfg = load_config(config_path)
    log = get_logger('executor')
    dry_run = cfg.get('dry_run', False)
    log.info(f"✅ Dry run = {dry_run}")

    init_log()
    if not mt5.initialize():
        log.error(f"⛔ MT5 initialize failed: {mt5.last_error()}")
        return
    log.info("✅ MT5 initialized")

    # Load config values
    symbol = cfg['mt5']['symbol']
    timeframe = getattr(mt5, cfg['mt5']['timeframe'])
    bars = cfg['mt5']['bars_to_fetch']
    contract_size = cfg['mt5']['contract_size']
    max_risk_pct = cfg['risk']['max_risk_pct']
    sl_mult = cfg['risk']['sl_multiplier']
    tp_mult = cfg['risk']['tp2_multiplier']
    time_stop_mins = cfg['risk']['time_stop_mins']
    adx_window = cfg['risk']['adx']['window']
    adx_min = cfg['risk']['adx']['min_trend']
    adx_side = cfg['risk']['adx']['sideway']
    max_trades = cfg['risk']['max_trades_per_day']
    throttle_mins = cfg['execution']['throttle_mins']
    lot_decimals = cfg['rounding']['lot_decimals']
    predict_threshold = cfg['training']['predict_threshold']
    ema_period = cfg['features']['trend_filter']['h1_ema_period']

    # Load model
    mdl = joblib.load(cfg['models']['label_classes'])
    if isinstance(mdl, tuple):
        model, classes = mdl
    else:
        model, classes = mdl, mdl.classes_
    log.info(f"✅ Loaded model; classes={classes}, threshold={predict_threshold}")

    # State trackers
    last_predict = datetime.now() - timedelta(minutes=throttle_mins)
    signal_buffer = deque(maxlen=3)
    trade_times = deque()
    open_times = {}

    try:
        while True:
            now = datetime.now()
            #log.info("Loop iteration start")

            # Prediction throttle
            if (now - last_predict) < timedelta(minutes=throttle_mins):
                time.sleep(5)
                continue
            last_predict = now

            # Daily limit sliding window
            while trade_times and (now - trade_times[0]) > timedelta(hours=24):
                trade_times.popleft()
            if len(trade_times) >= max_trades:
                time.sleep(60)
                continue

            # Fetch M1 bars
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
            if rates is None or len(rates) == 0:
                time.sleep(5)
                continue
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            curr_time = df['time'].iat[-1]
            curr_price = df['close'].iat[-1]
            log.info(f"Bar time={curr_time}, Price={curr_price}")
            # Account & Position Summary
            account = mt5.account_info()
            if account:
                balance = account.balance
                equity = account.equity
                log.info(f"💲 Balance={balance:.2f}, Equity={equity:.2f}")
            positions = mt5.positions_get(symbol=symbol) or []
            total_unreal = sum(p.profit for p in positions)
            log.info(f"💲 Unrealized P/L={total_unreal:.2f}, Open positions={len(positions)}")
            # Daily trade count and next prediction
            log.info(f"Trades in last 24h={len(trade_times)}")
            next_run = last_predict + timedelta(minutes=throttle_mins)
            log.info(f"Next prediction at {next_run.strftime('%Y-%m-%d %H:%M:%S')}")

            # EMA H1 trend filter
            rates_h1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 250)
            df_h1 = pd.DataFrame(rates_h1)
            ema_h1 = EMAIndicator(close=df_h1['close'], window=ema_period).ema_indicator()
            curr_ema = ema_h1.iat[-1]
            log.info(f"EMA_H1 (period={ema_period}): {curr_ema:.5f}")  # Log EMA200 H1 value
            # Log price relation to EMA200 H1
            if curr_price >= curr_ema:
                log.info("Price is เหนือ EMA_H1")
            else:
                log.info("Price is ล่าง EMA_H1")
            df['trend_h1'] = np.where(df['close'] >= curr_ema, 1, -1)

            # Indicators
            df['rsi'] = compute_rsi(df['close'], cfg['training']['rsi_period'])
            df['atr'] = compute_atr(df, cfg['training']['atr_period'])
            rsi_val = df['rsi'].iat[-1]
            atr_val = df['atr'].iat[-1]
            log.info(f"RSI={rsi_val:.2f}, ATR={atr_val:.2f}")

            # ICT/SMC features
            df['mss'] = compute_mss(df, cfg['features']['ict']['mss_lookback'])
            df['fvg'] = compute_fvg(df, cfg['features']['ict']['fvg_threshold'])
            smc = cfg['features']['smc']
            df['order_block'] = compute_order_block(df, smc['order_block_size'])
            df['liquidity_void'] = compute_liquidity_void(df, smc['liquidity_void_depth'])
            df['breaker_block'] = compute_breaker_block(df, smc['breaker_block_lookback'])
            log.info(f"MSS={df['mss'].iat[-1]}, FVG={df['fvg'].iat[-1]}, OB={df['order_block'].iat[-1]}")

            # ADX filter
            adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=adx_window).adx()
            curr_adx = adx.iat[-1]
            log.info(f"ADX={curr_adx:.2f}")
            if curr_adx < adx_side or curr_adx < adx_min:
                time.sleep(5)
                continue

            # Lag features
            for feat in ['close','rsi','atr','mss','fvg','order_block','liquidity_void','breaker_block']:
                df[f"{feat}_lag1"] = df[feat].shift(1)
                df[f"{feat}_lag2"] = df[feat].shift(2)
            df.fillna(0, inplace=True)

            # Model prediction
            row = df.iloc[[-1]]
            X = row[model.get_booster().feature_names]
            probs = model.predict_proba(X)[0]
            sig = int(classes[np.argmax(probs)]) if probs.max() >= predict_threshold else 0
            log.info(f"🔔 Probs={probs.tolist()}, sig={sig}")

            # Signal persistence
            signal_buffer.append(sig)
            persistence_count = signal_buffer.count(sig)
            log.info(f"🔔 Persistence count for signal {sig}: {persistence_count}/{signal_buffer.maxlen}")
            if sig != 0 and persistence_count < signal_buffer.maxlen:
                time.sleep(5)
                continue

            # Trend direction check
            if (sig == 1 and curr_price < curr_ema) or (sig == 2 and curr_price > curr_ema):
                time.sleep(5)
                continue

            # Existing position check
            positions = mt5.positions_get(symbol=symbol) or []
            if sig == 0 or positions:
                time.sleep(5)
                continue

            # Place order
            direction = 1 if sig == 1 else -1
            order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
            lot = calculate_lot(mt5.account_info().balance, atr_val, max_risk_pct, sl_mult, contract_size, lot_decimals)
            sl, tp = calculate_levels(curr_price, atr_val, sl_mult, tp_mult, direction)
            log.info(f"🔔 Placing {'BUY' if direction==1 else 'SELL'} lot={lot} SL={sl} TP={tp}")
            if not dry_run:
                tick = mt5.symbol_info_tick(symbol)
                exec_price = tick.ask if direction == 1 else tick.bid
                req = {'action': mt5.TRADE_ACTION_DEAL, 'symbol': symbol, 'volume': lot,
                       'type': order_type, 'price': exec_price, 'sl': sl, 'tp': tp,
                       'deviation': 10, 'type_time': mt5.ORDER_TIME_GTC,
                       'type_filling': mt5.ORDER_FILLING_IOC}
                res = mt5.order_send(req)
                log.info(f"Order send retcode={res.retcode}")
                if res.retcode == mt5.TRADE_RETCODE_DONE:
                    ticket = res.order
                    trade_times.append(now)
                    open_times[ticket] = now
                    append_log({
                        'timestamp_request': now.strftime('%Y-%m-%d %H:%M:%S'),
                        'timestamp_filled': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'ticket': ticket,
                        'side': 'BUY' if direction == 1 else 'SELL',
                        'price': exec_price,
                        'lot': lot,
                        'sl': sl,
                        'tp': tp,
                        'event': 'OPEN',
                        'pnl': ''
                    })
                    log.info(f"🟢 Order placed: ticket={ticket}")

                        # Trailing Stop: adjust SL when profit >= min_profit_atr
            for pos in mt5.positions_get(symbol=symbol) or []:
                entry_price = pos.price_open
                direction = 1 if pos.type == mt5.ORDER_TYPE_BUY else -1
                profit_atr = (curr_price - entry_price) * direction / atr_val if atr_val > 0 else 0
                if profit_atr >= cfg['risk']['min_profit_atr']:
                    new_sl = calculate_trailing_stop(curr_price, entry_price, pos.sl,
                                                     atr_val, cfg['risk']['trailing_multiplier'], direction)
                    if new_sl != pos.sl:
                        req_trail = {
                            'action': mt5.TRADE_ACTION_SLTP,
                            'symbol': symbol,
                            'position': pos.ticket,
                            'sl': new_sl,
                            'tp': pos.tp
                        }
                        res_trail = mt5.order_send(req_trail)
                        if res_trail.retcode == mt5.TRADE_RETCODE_DONE:
                            log.info(f"Trailing stop updated {pos.ticket}: {pos.sl:.5f} -> {new_sl:.5f}")
                        else:
                            log.warning(f"Failed updating trailing stop {pos.ticket}: {res_trail.comment}")
            # Time-stop close
            for pos in positions:
                if (now - open_times.get(pos.ticket, now)) > timedelta(minutes=time_stop_mins):
                    close_req = {'action': mt5.TRADE_ACTION_DEAL, 'position': pos.ticket,
                                 'symbol': symbol, 'volume': pos.volume,
                                 'type': mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                                 'price': curr_price}
                    mt5.order_send(close_req)
                    log.info(f"Time-stop closed {pos.ticket}")
            time.sleep(5)

    except KeyboardInterrupt:
        log.info("🚧 Stopped by user")
    finally:
        mt5.shutdown()
        log.info("⛔ MT5 shutdown")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python -m data_pipeline.trade_executor config.yaml")
        sys.exit(1)
    execute_trades(sys.argv[1])
