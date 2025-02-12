import binance_client
import trade_manager
import strategy

# Constants
BASE_CURRENCY = "XRP"
QUOTE_CURRENCY = "USDT"
SYMBOL = f"{BASE_CURRENCY}/{QUOTE_CURRENCY}"
TIMEFRAME = "5m"
CANDLESTICK_LIMIT = 250
DYNAMIC_LEVELS = False
TAKE_PROFIT = 0.02
STOP_LOSS = 0.01
BALANCE_ALLOCATION = 0.5
SIGNAL_THRESHOLD = 0.5

def main():
    client = binance_client.BinanceFuturesClient(symbol = SYMBOL, timeframe = TIMEFRAME,
                                  take_profit = TAKE_PROFIT, stop_loss = STOP_LOSS,
                                  balance_allocation = BALANCE_ALLOCATION, candlestick_limit = CANDLESTICK_LIMIT)
    signal_generator = strategy.SignalGenerator(client = client, signal_threshold = SIGNAL_THRESHOLD)
    trading = trade_manager.TradeManager(client = client)
    backtester = strategy.Backtester(signal_generator = signal_generator, initial_balance = 1000)


    # signal_generator.generate_signal()
    backtester.run_backtest()
    backtester.plot_trades()
    metrics = backtester.calculate_performance_metrics()

    # Print Metrics
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # print("Balance:", client.fetch_balance())
    # print("Market Data:", client.fetch_market_data())
    # print("Open Positions:", client.fetch_open_positions())
    # print("Latest Candles:", client.fetch_latest_candles())
    # client.place_order("market", "buy", 10)
    # # Example trade execution
    # if signal != 0:
    #     tm.execute_trade('BUY' if signal == 1 else 'SELL')
    # print("Closing Position:", trading.close_position())
    # print("Fetching Trades:", client.fetch_historical_trades())
    # print("Fetching Order Book:", client.fetch_order_book())
    # print("Fetching Open Orders:", client.fetch_open_orders())
    # print("Fetching Order Status:", client.fetch_order_status("86162481850"))
    # print("Fetching Funding Rate:", client.fetch_funding_rate())
    # print("Setting Leverage:", client.set_leverage(leverage=75))

if __name__ == "__main__":
    main()
