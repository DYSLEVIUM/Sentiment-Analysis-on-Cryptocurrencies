from transformers import AutoModelForSequenceClassification

from .transformers import TransformersModel
from src.tokenizers import RobertaTokenizer


class RobertaModel(TransformersModel):
    def __init__(
        self,
    ):
        model = AutoModelForSequenceClassification.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment"
        )
        tokenizer = RobertaTokenizer()

        super().__init__(model, tokenizer)
