import binance_client
import trade_manager
import strategy

import numpy as np
from itertools import product
from strategy import SignalGenerator, Backtester
from binance_client import BinanceFuturesClient

# Constants
BASE_CURRENCY = "XRP"
QUOTE_CURRENCY = "USDT"
SYMBOL = f"{BASE_CURRENCY}/{QUOTE_CURRENCY}"
TIMEFRAME = "5m"
CANDLESTICK_LIMIT = 500
DYNAMIC_LEVELS = False
MODE = 1 # 1: Close on TP/SL, 2: Close on opposite signal
TAKE_PROFIT = 0.02
STOP_LOSS = 0.01
ATR_TAKE_PROFIT = 2
ATR_STOP_LOSS = 1
BALANCE_ALLOCATION = 0.05
SIGNAL_THRESHOLD = 0.4

ema = 0.5
rsi = 0.1
macd = 0.1
bb = 0.1
atr = 0.1

def optimize_strategy(symbol='XRP/USDT', initial_balance=1000):
    weights = np.arange(0.1, 1.1, 0.2)
    best_config = {'score': -np.inf}
    client = BinanceFuturesClient(symbol=symbol)

    for ema, rsi, macd, bb, atr in product(weights, repeat=5):
        if ema + rsi + macd + bb + atr > 1.01:
            continue

        signal_generator = SignalGenerator(client, ema_weight=ema, rsi_weight=rsi,
                                           macd_weight=macd, bb_weight=bb, atr_weight=atr)
        backtester = Backtester(signal_generator, initial_balance)
        backtester.run_backtest()
        metrics = backtester.calculate_performance_metrics()

        score = (metrics['final_balance'] / initial_balance) + (metrics['win_rate'] / 100)
        if score > best_config['score']:
            best_config = {
                'score': score,
                'ema': ema, 'rsi': rsi, 'macd': macd, 'bb': bb, 'atr': atr,
                'metrics': metrics
            }

        print(f"Weights: EMA:{ema}, RSI:{rsi}, MACD:{macd}, BB:{bb}, ATR:{atr} => Score: {score:.2f}")

    print("Best Configuration:", best_config)
    return best_config

def main():
    client = binance_client.BinanceFuturesClient(symbol = SYMBOL, timeframe = TIMEFRAME,
                                                 candlestick_limit = CANDLESTICK_LIMIT)

    signal_generator = strategy.SignalGenerator(client = client, signal_threshold = SIGNAL_THRESHOLD,
                                                ema_weight=ema, rsi_weight=rsi, macd_weight=macd,
                                               bb_weight=bb, atr_weight=atr)

    backtester = strategy.Backtester(signal_generator = signal_generator, initial_balance = 1000,
                                     take_profit = TAKE_PROFIT, stop_loss = STOP_LOSS,
                                     mode = MODE, dynamic_levels = DYNAMIC_LEVELS,
                                     atr_tp_mult = ATR_TAKE_PROFIT, atr_sl_mult = ATR_STOP_LOSS)

    trader = trade_manager.TradeManager(client = client, signal_generator = signal_generator,
                                         balance_allocation = BALANCE_ALLOCATION,
                                         take_profit = TAKE_PROFIT, stop_loss = STOP_LOSS,
                                         mode = MODE, dynamic_levels = DYNAMIC_LEVELS,
                                         atr_tp_mult = ATR_TAKE_PROFIT, atr_sl_mult = ATR_STOP_LOSS)

    # trader.execute_trade(signal=-1)


    # optimize_strategy()

    # backtester.run_backtest()
    # backtester.plot_trades()
    # metrics = backtester.calculate_performance_metrics()
    # for key, value in metrics.items():
    #     print(f"{key}: {value}")

if __name__ == "__main__":
    main()
