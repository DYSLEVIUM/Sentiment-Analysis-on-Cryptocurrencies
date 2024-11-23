import torch
from typing_extensions import override
from src.models.model import PretrainedModel
from src.tokenizers.tokenizer import Tokenizer
from transformers import AutoModelForSequenceClassification
from dataclasses import dataclass
from src.utils.settings import DEVICE


@dataclass
class TransformersModel(PretrainedModel):
    model: AutoModelForSequenceClassification = None

    def __init__(self, model: AutoModelForSequenceClassification, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        super().__init__(model)

    @override
    def predict(self, text: str):
        tokens = self.tokenizer.encode(text=text)
        tokens = tokens.to(DEVICE)
        with torch.no_grad():
            result = self.model(tokens).logits
        return int(torch.argmax(result)) + 1
