from transformers import AutoModelForSequenceClassification

from .transformers import TransformersModel
from src.tokenizers import BertTokenizer


class BertModel(TransformersModel):
    def __init__(
        self,
    ):
        model = AutoModelForSequenceClassification.from_pretrained(
            "nlptown/bert-base-multilingual-uncased-sentiment"
        )
        tokenizer = BertTokenizer()

        super().__init__(model, tokenizer)
