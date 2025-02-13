import binance_client
import trade_manager
import strategy

# Constants
BASE_CURRENCY = "XRP"
QUOTE_CURRENCY = "USDT"
SYMBOL = f"{BASE_CURRENCY}/{QUOTE_CURRENCY}"
TIMEFRAME = "5m"
CANDLESTICK_LIMIT = 500
DYNAMIC_LEVELS = True
MODE = 2 # 1: Close on TP/SL, 2: Close on opposite signal
TAKE_PROFIT = 0.02
STOP_LOSS = 0.01
ATR_TAKE_PROFIT = 2
ATR_STOP_LOSS = 1
BALANCE_ALLOCATION = 0.05
SIGNAL_THRESHOLD = 0.5
ema = 0.3
rsi = 0.3
macd = 0.1
bb = 0.1
atr = 0.1

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

    backtester.run_backtest()
    backtester.plot_trades()
    metrics = backtester.calculate_performance_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value}")

    import numpy as np
    from itertools import product
    from strategy import SignalGenerator, Backtester
    from binance_client import BinanceFuturesClient

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

    # optimize_strategy()
    # trader.execute_trade(signal=1)

    # print("Balance:", client.fetch_balance())
    # print("Market Data:", client.fetch_market_data())
    # print("Open Positions:", client.fetch_open_positions())
    # print("Latest Candles:", client.fetch_latest_candles())
    # print("Closing Position:", trading.close_position())
    # print("Fetching Trades:", client.fetch_historical_trades())
    # print("Fetching Order Book:", client.fetch_order_book())
    # print("Fetching Open Orders:", client.fetch_open_orders())
    # print("Fetching Order Status:", client.fetch_order_status("86162481850"))
    # print("Fetching Funding Rate:", client.fetch_funding_rate())
    # print("Setting Leverage:", client.set_leverage(leverage=75))

if __name__ == "__main__":
    main()
