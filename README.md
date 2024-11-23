# Sentiment Analysis



## Setup

## Data
1. Get the ethereum tweets data from [here](https://www.kaggle.com/datasets/mathurinache/ethereum-tweets) and put it in `data/raw/eth` and name it `tweets.csv`.
1. Get the bitcoin tweets data from [here](https://www.kaggle.com/datasets/kaushiksuresh147/bitcoin-tweets) and put it in `data/raw/btc` and name it `tweets.csv`.

### Install dependencies

```
source .venv/bin/activate
pip install -r requirements.txt

```

### Install package

```
pip install -e .
```

## Run

```
python3 -m src.main
```
