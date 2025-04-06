import binance_client
import trade_manager
import strategy

# Constants
BASE_CURRENCY = "FUN"
QUOTE_CURRENCY = "USDT"
SYMBOL = f"{BASE_CURRENCY}/{QUOTE_CURRENCY}"
TIMEFRAME = "1m"
DYNAMIC_LEVELS = False
MODE = 1 # 1: Close on TP/SL, 2: Close on opposite signal
TAKE_PROFIT = 0.02
STOP_LOSS = 0.01
ATR_TAKE_PROFIT = 2
ATR_STOP_LOSS = 1
BALANCE_ALLOCATION = 0.05
SIGNAL_THRESHOLD = 0.2

ema = 0.2
rsi = 0.2
macd = 0.2
bb = 0.35
atr = 0.05

def main():
    client = binance_client.BinanceFuturesClient(symbol = SYMBOL, timeframe = TIMEFRAME)

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
    metrics = backtester.calculate_performance_metrics()
    print("\nPerformance Metrics:")
    print("-" * 40)
    for key, value in metrics.items():
        print(f"{key.replace('_', ' ').title():20} : {value}")

    # SIGNAL GENERATOR TAMAM SADECE ATR STRATEJİSİ DÜZENLE EN SON STRATEJİLERE TEKRAR BAK
    # BACKTESTER CLASS İNCELE DÜZGÜN ÇALIŞTIĞINDAN EMİN OL
    # TRADE MANAGER CLASS BİTİR

if __name__ == "__main__":
    main()
