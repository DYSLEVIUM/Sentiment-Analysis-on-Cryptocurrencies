from .coin import Coin
from datetime import datetime


class Bitcoin(Coin):
    def __init__(self):
        super().__init__("BTC", datetime(2009, 1, 3))
