from .coin import Coin
from datetime import datetime


class Ethereum(Coin):
    def __init__(self):
        super().__init__("ETH", datetime(2015, 7, 30))
