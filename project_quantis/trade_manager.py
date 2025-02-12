import logging
from binance_client import BinanceFuturesClient

class TradeManager:
    def __init__(self, client):
        self.client = client

    def execute_trade(self, signal, order_type='market'):
        """Execute trade based on a buy/sell signal"""
        balance = self.client.fetch_balance()
        if balance is None:
            logging.error("Failed to fetch balance. Cannot execute trade.")
            return None

        usdt_balance = balance.get('total', {}).get('USDT', 0)
        trade_amount = (usdt_balance * self.client.balance_allocation) / self.client.fetch_market_data()['last']

        if trade_amount <= 0:
            logging.error("Insufficient balance to place trade.")
            return None

        side = 'buy' if signal == 'BUY' else 'sell'
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