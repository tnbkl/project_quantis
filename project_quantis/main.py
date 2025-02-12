import binance_client
import trade_manager
import strategy

# Constants
BASE_CURRENCY = "XRP"
QUOTE_CURRENCY = "USDT"
SYMBOL = f"{BASE_CURRENCY}/{QUOTE_CURRENCY}"
TIMEFRAME = "15m"
CANDLESTICK_LIMIT = 250
DYNAMIC_LEVELS = True
MODE = 1 # 1: Close on TP/SL, 2: Close on opposite signal
TAKE_PROFIT = 0.02
STOP_LOSS = 0.01
ATR_TAKE_PROFIT = 2
ATR_STOP_LOSS = 1
BALANCE_ALLOCATION = 0.05
SIGNAL_THRESHOLD = 0

def main():
    client = binance_client.BinanceFuturesClient(symbol = SYMBOL, timeframe = TIMEFRAME,
                                                 candlestick_limit = CANDLESTICK_LIMIT)

    signal_generator = strategy.SignalGenerator(client = client, signal_threshold = SIGNAL_THRESHOLD)
    backtester = strategy.Backtester(signal_generator = signal_generator, initial_balance = 1000,
                                     take_profit = TAKE_PROFIT, stop_loss = STOP_LOSS,
                                     mode = MODE, dynamic_levels = DYNAMIC_LEVELS,
                                     atr_tp_mult = ATR_TAKE_PROFIT, atr_sl_mult = ATR_STOP_LOSS)
    trader = trade_manager.TradeManager(client = client, signal_generator = signal_generator,
                                         balance_allocation = BALANCE_ALLOCATION,
                                         take_profit = TAKE_PROFIT, stop_loss = STOP_LOSS,
                                         mode = MODE, dynamic_levels = DYNAMIC_LEVELS,
                                         atr_tp_mult = ATR_TAKE_PROFIT, atr_sl_mult = ATR_STOP_LOSS)



    # signal_generator.generate_signal()
    # backtester.run_backtest()
    # backtester.plot_trades()
    # metrics = backtester.calculate_performance_metrics()
    #
    # # Print Metrics
    # for key, value in metrics.items():
    #     print(f"{key}: {value}")

    # trader.execute_trade(signal=1)

    # client.place_order("market", "buy", 10)

    # # Example trade execution
    # if signal != 0:
    #     tm.execute_trade('BUY' if signal == 1 else 'SELL')


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
