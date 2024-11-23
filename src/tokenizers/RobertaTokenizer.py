from transformers import AutoTokenizer

from .tokenizer import Tokenizer
from typing_extensions import override


class RobertaTokenizer(Tokenizer):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment"
        )

    @override
    def encode(self, text):
        return self.tokenizer.encode(text, return_tensors="pt")
