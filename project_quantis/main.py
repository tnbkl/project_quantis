import binance_client
import trade_manager
import strategy

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

import pprint

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

    #pprint.pprint(client.fetch_open_orders())

    trader.close_position()

    ###pprint.pprint(client.exchange.create_order(symbol=SYMBOL, type='limit', side='buy', amount=5, price=2.15)) # amount in XRP
    ###pprint.pprint(client.exchange.create_order(symbol=SYMBOL, type='take_profit_market', side='sell', amount=None, price=None, params={'closePosition': True, 'stopPrice': 2.18, 'timeInForce': 'GTE_GTC', 'workingType': 'MARK_PRICE'}))
    ###pprint.pprint(client.exchange.create_order(symbol=SYMBOL, type='stop_market', side='sell', amount=None, price=None, params={'closePosition': True, 'stopPrice': 2.12, 'timeInForce': 'GTE_GTC', 'workingType': 'MARK_PRICE'}))

if __name__ == "__main__":
    main()
