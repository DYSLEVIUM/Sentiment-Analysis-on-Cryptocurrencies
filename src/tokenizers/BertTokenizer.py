from transformers import AutoTokenizer

from .tokenizer import Tokenizer
from typing_extensions import override


class BertTokenizer(Tokenizer):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "nlptown/bert-base-multilingual-uncased-sentiment",
        )

    @override
    def encode(self, text):
        return self.tokenizer.encode(text, return_tensors="pt")
