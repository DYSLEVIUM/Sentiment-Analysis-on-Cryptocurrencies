from abc import ABC
from datetime import datetime
from dataclasses import dataclass

@dataclass
class Coin(ABC):
    ticker: str
    start_date: datetime
