import logging
import numpy as np
from binance_client import BinanceFuturesClient

class TradeManager:
    def __init__(self, client, signal_generator, balance_allocation=0.1,
                 mode=1, take_profit=0.02, stop_loss=0.01,
                 dynamic_levels=False, atr_sl_mult=1.5, atr_tp_mult=3):
        """
        :param signal_generator: Instance of SignalGenerator for strategy data
        :param risk_per_trade: % of balance to risk per trade (e.g., 0.01 = 1%)
        :param atr_multiplier: Multiplier for ATR-based position sizing
        """
        self.client = client
        self.signal_generator = signal_generator
        self.balance_allocation = balance_allocation
        self.mode = mode
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.dynamic_levels = dynamic_levels
        self.atr_tp_mult = atr_tp_mult
        self.atr_sl_mult = atr_sl_mult

    def execute_trade(self):
        order = self.client.place_order(side = 'buy', type = 'market', amount = 3)
        return order

    def close_position(self):
        order = self.client.place_order(side = 'sell', type = 'market', price = None, amount = 200,
                                        reduce_only = True)
        return order

    def create_stop_loss_order(self):
        order = self.client.place_stop_order(side = 'sell', type = 'stop_market', amount = None, price = None,
                                        close_position = True, stop_price = 2.12, time_in_force = 'GTE_GTC', working_type = 'contract_price')
        return order

    def create_take_profit_order(self):
        order = self.client.place_stop_order(side = 'sell', type = 'take_profit_market', amount = None,  price = None,
                                        close_position = True, stop_price = 2.135, time_in_force = 'GTE_GTC', working_type = 'mark_price')
        return order
























'''
IGNORE BELOW CODE
    def calculate_position_size(self, price):
        """Calculate position size using ATR-based volatility"""
        if self.signal_generator:
            # Fetch latest ATR value from strategy data
            df = self.signal_generator.generate_signal(historical=False)
            current_atr = df['atr'].iloc[-1] if 'atr' in df.columns else 1
        else:
            current_atr = 1  # Fallback if no strategy data

        balance = self.client.fetch_balance().get('total', {}).get('USDT', 0)
        risk_amount = balance * self.risk_per_trade
        return (risk_amount * self.atr_multiplier) / current_atr

    def execute_trade(self, signal, order_type='market'):
        """Execute trade based on a buy/sell signal"""
        if signal not in [1, -1]:
            logging.error("Invalid signal. Use 1 (BUY) or -1 (SELL).")
            return None

        balance = self.client.fetch_balance()
        if balance is None:
            logging.error("Failed to fetch balance. Cannot execute trade.")
            return None

        usdt_balance = balance.get('total', {}).get('USDT', 0)
        trade_amount = (usdt_balance * self.balance_allocation) / self.client.fetch_market_data()['last']

        if trade_amount <= 0:
            logging.error("Insufficient balance to place trade.")
            return None

        side = 'buy' if signal == 1 else 'sell'
        order_type = 'market'
        order = self.client.place_order(order_type, side, trade_amount)

        if order:
            logging.info(f"Executed {order_type} {side} order for {trade_amount} {self.client.symbol}.")
        else:
            logging.error("Trade execution failed.")

        return order

    def close_position(self):
        """Close an open position on Binance Futures"""
        positions = self.client.fetch_open_positions()
        if positions:
            for position in positions:
                if position['info']['symbol'] == self.client.symbol.replace("/", "") and float(position['contracts']) > 0:
                    side = 'sell' if position['side'] == 'long' else 'buy'
                    amount = abs(float(position['contracts']))
                    return self.client.safe_api_call(self.client.exchange.create_order, self.client.symbol, 'market',
                                                     side, amount)
        logging.info(f"No open position found for {self.client.symbol}.")
        return None
'''