from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass

class Tokenizer(ABC):
    @abstractmethod
    def encode(self, text: str):
        pass
