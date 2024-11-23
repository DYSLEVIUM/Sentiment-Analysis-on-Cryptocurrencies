from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass

from utils.settings import DEVICE


@dataclass
class Model(ABC):
    model: Any = None

    @abstractmethod
    def predict(self, text: str):
        pass


class PretrainedModel(Model):
    def __init__(self, model: Any):
        model.to(DEVICE)
        super().__init__(model)

    @abstractmethod
    def predict(self, text: str):
        pass


class CustomModel(Model):
    def __init__(self, model: Any):
        super().__init__(model)

    @abstractmethod
    def train(self, data):
        pass

    @abstractmethod
    def predict(self, text: str):
        pass
