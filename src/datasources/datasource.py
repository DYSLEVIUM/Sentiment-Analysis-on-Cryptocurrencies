from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd

@dataclass
class DataSource(ABC):
    df: pd.DataFrame