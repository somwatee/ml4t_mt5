# data_pipeline/backtest.py

import sys
import traceback
import backtrader as bt
import pandas as pd
import json
from data_pipeline.utils import load_config, get_logger

class HybridStrategy(bt.Strategy):
    params = (
        ('signals',            []),
        ('max_risk_pct',       0.05),
        ('sl_multiplier',      2.0),
        ('tp2_mult',           6.0),
        ('time_stop',          480),
        ('atr_period',         14),
        ('contract_size',      100),
        ('trailing_multiplier', 1.5),  # ATR multiple for trailing stop
        ('min_profit_atr',     1.0),   # minimum ATR profit before enabling trailing
    )

    def __init__(self):
        # map datetime → signal
        self.sig_map        = {pd.to_datetime(s['time']): s['signal'] for s in self.p.signals}
        # main ATR
        self.atr            = bt.indicators.ATR(self.data, period=self.p.atr_period)
        # track entry state
        self.entry_pr       = None
        self.entry_bar      = None
        # trailing stop control
        self.trailing_set   = False
        self.trailing_order = None
        # performance counters
        self.pnl_list       = []
        self.entry_sz       = []
        self.equity_curve   = []
        self.buy_cnt        = 0
        self.sell_cnt       = 0
        self.buy_wins       = 0
        self.sell_wins      = 0
        self.tp2_cnt        = 0
        self.sl_cnt         = 0
        # map order.ref → direction
        self._open_dirs     = {}

    def next(self):
        # record current equity
        self.equity_curve.append(self.broker.getvalue())

        dt    = self.data.datetime.datetime(0)
        sig   = self.sig_map.get(dt, 0)
        price = self.data.close[0]
        atr   = float(self.atr[0])

        # ENTRY: only if flat
        if not self.position and sig != 0:
            self.entry_pr     = price
            self.entry_bar    = len(self)
            self.trailing_set = False
            self.trailing_order = None

            equity      = self.broker.getvalue()
            risk_amount = equity * self.p.max_risk_pct
            stop_dist   = atr * self.p.sl_multiplier
            stop_value  = stop_dist * self.p.contract_size
            size        = round(risk_amount / stop_value, 4) if stop_value > 0 else 0
            if size <= 0:
                return

            self.entry_sz.append(size)
            if sig > 0:
                self.buy_cnt += 1
                order = self.buy(size=size)
            else:
                self.sell_cnt += 1
                order = self.sell(size=size)
            self._open_dirs[order.ref] = 1 if sig > 0 else -1
            return

        # MANAGEMENT: only if in a trade
        if self.position and self.entry_bar is not None:
            held = len(self) - self.entry_bar

            # time stop
            if held > self.p.time_stop:
                self.close()
                self.sl_cnt += 1
                return

            direction = 1 if self.position.size > 0 else -1
            profit_atr = (price - self.entry_pr) * direction / atr

            # set up trailing stop once profit exceeds threshold
            if not self.trailing_set and profit_atr >= self.p.min_profit_atr:
                if direction == 1:
                    self.trailing_order = self.sell(
                        exectype=bt.Order.StopTrail,
                        trailamount=atr * self.p.trailing_multiplier
                    )
                else:
                    self.trailing_order = self.buy(
                        exectype=bt.Order.StopTrail,
                        trailamount=atr * self.p.trailing_multiplier
                    )
                self.trailing_set = True
                return

            # fixed single TP (formerly TP2)
            target = self.entry_pr + atr * self.p.tp2_mult * direction
            if (direction == 1 and price >= target) or (direction == -1 and price <= target):
                self.close()
                self.tp2_cnt += 1
                return

    def notify_order(self, order):
        # when fully closed, reset entry/trailing state
        if order.status in (order.Completed, order.Canceled, order.Margin, order.Rejected) and not self.position:
            self.entry_pr       = None
            self.entry_bar      = None
            self.trailing_set   = False
            self.trailing_order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            pnl       = trade.pnl
            direction = self._open_dirs.get(trade.ref, 0)
            self.pnl_list.append({'pnl': pnl, 'dir': direction})
            if pnl > 0:
                if direction == 1:
                    self.buy_wins += 1
                elif direction == -1:
                    self.sell_wins += 1

    def stop(self):
        self._open_dirs.clear()

def run_backtest(config_path: str):
    try:
        cfg     = load_config(config_path)
        log     = get_logger("backtest")

        # load data & signals
        hist    = pd.read_csv(cfg['data']['historical_csv'], parse_dates=['time']).set_index('time')
        feed    = bt.feeds.PandasData(dataname=hist)
        signals = json.load(open(cfg['data']['signals_json'], 'r', encoding='utf-8'))

        # Cerebro setup
        cerebro = bt.Cerebro()
        cerebro.adddata(feed)
        cerebro.broker.setcash(200.0)
        cerebro.broker.setcommission(0.0)
        cerebro.addstrategy(
            HybridStrategy,
            signals            = signals,
            max_risk_pct       = cfg['risk']['max_risk_pct'],
            sl_multiplier      = cfg['risk']['sl_multiplier'],
            tp2_mult           = cfg['risk']['tp2_multiplier'],
            time_stop          = cfg['risk']['time_stop_mins'],
            atr_period         = cfg['training']['atr_period'],
            contract_size      = cfg['mt5']['contract_size'],
            trailing_multiplier= cfg['risk']['trailing_multiplier'],
            min_profit_atr     = cfg['risk']['min_profit_atr'],
            
        )
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')

        log.info(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
        strat = cerebro.run()[0]
        final = cerebro.broker.getvalue()
        dd    = strat.analyzers.dd.get_analysis()

        # compute stats
        pnl        = [t['pnl'] for t in strat.pnl_list]
        total      = len(pnl)
        wins       = [x for x in pnl if x > 0]
        losses     = [x for x in pnl if x <= 0]
        win_rate   = len(wins)/total*100 if total else 0.0
        avg_win    = sum(wins)/len(wins)   if wins  else 0.0
        avg_loss   = sum(losses)/len(losses) if losses else 0.0
        pf         = sum(wins)/abs(sum(losses)) if losses else float('nan')
        expectancy = win_rate/100*avg_win + (1-win_rate/100)*avg_loss
        max_dd     = dd.get('max', {}).get('drawdown', 0.0)
        highest    = max(strat.equity_curve) if strat.equity_curve else final
        sizes      = strat.entry_sz
        min_lot    = min(sizes) if sizes else 0.0
        max_lot    = max(sizes) if sizes else 0.0

        # output
        print("\n=== Backtest Results ===")
        print(f"Final Value         : {final:.2f}")
        print(f"Highest Balance     : {highest:.2f}")
        print(f"Total Trades        : {total}")
        print(f"Win Rate            : {win_rate:.2f}%")
        print(f"Avg Win / Avg Loss  : {avg_win:.4f} / {avg_loss:.4f}")
        print(f"Expectancy          : {expectancy:.4f}")
        print(f"Profit Factor       : {pf:.2f}")
        print(f"Max Drawdown        : {max_dd:.2f}%")
        print(f"Min Lot / Max Lot   : {min_lot:.4f} / {max_lot:.4f}")
        print(f"Buy Count / Sell Count : {strat.buy_cnt} / {strat.sell_cnt}")
        print(f"Buy Wins   / Sell Wins  : {strat.buy_wins} / {strat.sell_wins}")
        print(f"TP Hits              : {strat.tp2_cnt}")
        print(f"SL Hits              : {strat.sl_cnt}")

    except Exception:
        traceback.print_exc()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python -m data_pipeline.backtest config.yaml")
        sys.exit(1)
    run_backtest(sys.argv[1])
