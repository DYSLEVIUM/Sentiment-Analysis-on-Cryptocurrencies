import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from src.utils.logger import get_logger
from models.transformers.BertModel import BertModel

from models.transformers.RobertaModel import RobertaModel
from src.utils.settings import reset

logger = get_logger(__name__)

import pandas as pd

from coins.ethereum import Ethereum
from coins.bitcoin import Bitcoin
from coins.coin import Coin

from models.model import Model

from datasources.posts.tweets.tweetsCSVDatasource import TweetsCSVDataSource
from datasources.coinDatasource import CoinDatasource

from visualizations.visualization import Visualization
from visualizations.correlation import CorrelationPlot
from visualizations.boxplot import BoxPlot

from tqdm import tqdm

pd.set_option("display.max_colwidth", None)

if __name__ == "__main__":
    reset()

    coins: Coin = [Ethereum(), Bitcoin()]
    models: Model = [BertModel(), RobertaModel()]

    # Data sources
    coins_data = [CoinDatasource(coin) for coin in coins]
    tweets_csv_data = [
        TweetsCSVDataSource(f"data/raw/{coin.ticker.lower()}/posts/tweets.csv")
        for coin in coins
    ]
    logger.info("data sources loaded")

    # #! TODO: Remove this, only for testing
    # tweets_csv_data = [
    #     (lambda ds: setattr(ds, "df", ds.df[:1000]) or ds)(tweet_csv_data)
    #     for tweet_csv_data in tweets_csv_data
    # ]

    # Preprocessing
    tweets_data = [
        (
            lambda ds: setattr(
                ds,
                "df",
                ds.df.assign(
                    date=lambda x: pd.to_datetime(
                        x["date"], format="mixed", errors="coerce"
                    )
                    .dt.tz_localize(None)
                    .dt.date
                )
                .sort_values(by="date")
                .dropna(subset=["date"]),
            )
            or ds
        )(tweet_ds)
        for tweet_ds in tweets_csv_data
    ]

    coins_data = [
        (
            lambda ds: setattr(
                ds,
                "df",
                ds.df.assign(
                    date=lambda x: pd.to_datetime(
                        x["date"], format="mixed", errors="coerce"
                    )
                    .dt.tz_localize(None)
                    .dt.date
                )
                .sort_values(by="date")
                .dropna(subset=["date"]),
            )
            or ds
        )(coin_ds)
        for coin_ds in coins_data
    ]

    logger.info("preprocessing done")
    
    total_tweets = sum([len(ds.df) for ds in tweets_data])
    logger.info(f"total tweets: {total_tweets}")

    # Sentiment Prediction
    sentiments: dict[str, dict[str, list[TweetsCSVDataSource]]] = {}
    for model in models:
        model_name = model.__class__.__name__
        logger.info(f"processing tweets for {model_name}")

        processed_tweets = 0
        with tqdm(total=total_tweets, desc=f"{model_name} Processing", 
                 miniters=max(1, int(total_tweets/100))) as pbar:
            
            sentiment_ds_list = []
            for tweet_ds in tweets_data:
                df_copy = tweet_ds.df.copy()
                for idx, row in df_copy.iterrows():
                    df_copy.at[idx, "sentiment"] = model.predict(row["text"])
                    processed_tweets += 1
                    pbar.update(1)

                tweet_ds.df = df_copy
                sentiment_ds_list.append(tweet_ds)

            sentiments[model_name] = {
                "sentiment_ds": sentiment_ds_list
            }

    logger.info("sentiment prediction done")

    # Analysis
    for model in sentiments:
        all_sentiments = pd.concat(
            [ds.df[["date", "sentiment"]] for ds in sentiments[model]["sentiment_ds"]]
        )

        sentiment_date_group = all_sentiments.groupby("date")["sentiment"]

        sentiments[model]["sentiment_mean"] = (
            sentiment_date_group.mean().to_frame("sentiment").reset_index()
        )

        sentiments[model]["sentiment_sma_7"] = (
            sentiment_date_group.rolling(window=7)
            .mean()
            .to_frame("sentiment")
            .reset_index()
        )

        sentiments[model]["sentiment_ema_7"] = (
            sentiment_date_group.ewm(span=7, adjust=False)
            .mean()
            .to_frame("sentiment")
            .reset_index()
        )

        sentiments[model]["sentiment_variance"] = (
            sentiment_date_group.var().to_frame("sentiment").reset_index()
        )

    for coin in coins_data:
        coin.df["price_change"] = coin.df["price"].pct_change()
        coin.df["price_volatility"] = coin.df["price_change"].rolling(window=2).std()
        coin.df["price_sma_7"] = coin.df["price"].rolling(window=7).mean()
        coin.df["price_ema_7"] = coin.df["price"].ewm(span=7, adjust=False).mean()

    analysis_df = pd.DataFrame()
    for model in sentiments:
        sentiment_mean_data = sentiments[model]["sentiment_mean"]
        sentiment_var_data = sentiments[model]["sentiment_variance"]
        sentiment_sma_data = sentiments[model]["sentiment_sma_7"]
        sentiment_ema_data = sentiments[model]["sentiment_ema_7"]

        daily_sentiment_mean = (
            sentiment_mean_data.groupby("date")["sentiment"].mean().reset_index()
        )
        daily_sentiment_var = (
            sentiment_var_data.groupby("date")["sentiment"].mean().reset_index()
        )

        daily_sentiment_sma = (
            sentiment_sma_data.groupby("date")["sentiment"].mean().reset_index()
        )
        daily_sentiment_ema = (
            sentiment_ema_data.groupby("date")["sentiment"].mean().reset_index()
        )

        for coin in coins_data:
            sentiment_mean_col = f"{model}_sentiment"
            sentiment_var_col = f"{model}_sentiment_variance"
            sentiment_sma_col = f"{model}_sentiment_sma_7"
            sentiment_ema_col = f"{model}_sentiment_ema_7"

            price_change_col = f"{coin.coin.ticker}_price_change"
            price_vol_col = f"{coin.coin.ticker}_price_volatility"
            price_sma_col = f"{coin.coin.ticker}_price_sma_7"
            price_ema_col = f"{coin.coin.ticker}_price_ema_7"

            if analysis_df.empty:
                analysis_df["date"] = daily_sentiment_mean["date"]
                analysis_df.set_index("date", inplace=True)

            analysis_df[sentiment_mean_col] = daily_sentiment_mean.set_index("date")[
                "sentiment"
            ]
            analysis_df[sentiment_var_col] = daily_sentiment_var.set_index("date")[
                "sentiment"
            ]
            analysis_df[sentiment_sma_col] = daily_sentiment_sma.set_index("date")[
                "sentiment"
            ]
            analysis_df[sentiment_ema_col] = daily_sentiment_ema.set_index("date")[
                "sentiment"
            ]

            price_data = coin.df.copy()
            price_data = price_data.groupby("date").agg(
                {
                    "price_change": "mean",
                    "price_volatility": "mean",
                    "price_sma_7": "mean",
                    "price_ema_7": "mean",
                }
            )

            analysis_df[price_change_col] = price_data["price_change"]
            analysis_df[price_vol_col] = price_data["price_volatility"]
            analysis_df[price_sma_col] = price_data["price_sma_7"]
            analysis_df[price_ema_col] = price_data["price_ema_7"]

    analysis_df = analysis_df.dropna()
    logger.info("analysis done")

    # visualizations
    visualizations: list[Visualization] = []

    # Correlation Plots
    for model in sentiments:
        model_name = model

        for coin in coins:
            coin_ticker = coin.ticker

            # Sentiment vs. Price Change Correlation
            sent_price_corr = analysis_df[
                [f"{model_name}_sentiment", f"{coin_ticker}_price_change"]
            ].corr()
            visualizations.append(
                CorrelationPlot(
                    sent_price_corr,
                    f"{model_name} Sentiment vs. {coin_ticker} Price Change Correlation",
                    f"{model_name}_sentiment_price_change_correlation",
                )
            )

            # Price Volatility vs. Sentiment Variance Correlation
            vol_sent_corr = analysis_df[
                [f"{coin_ticker}_price_volatility", f"{model_name}_sentiment_variance"]
            ].corr()
            visualizations.append(
                CorrelationPlot(
                    vol_sent_corr,
                    f"{coin_ticker} Price Volatility vs. {model_name} Sentiment Variance Correlation",
                    f"{coin_ticker}_price_volatility_{model_name}_sentiment_variance_correlation",
                )
            )

            # Sentiment SMA vs. Price SMA Correlation
            sent_price_sma_corr = analysis_df[
                [f"{model_name}_sentiment_sma_7", f"{coin_ticker}_price_sma_7"]
            ].corr()

            visualizations.append(
                CorrelationPlot(
                    sent_price_sma_corr,
                    f"{model_name} Sentiment SMA vs. {coin_ticker} Price SMA Correlation",
                    f"{model_name}_sentiment_sma_7_price_sma_7_correlation",
                )
            )

            # Sentiment EMA vs. Price EMA Correlation
            sent_price_ema_corr = analysis_df[
                [f"{model_name}_sentiment_ema_7", f"{coin_ticker}_price_ema_7"]
            ].corr()

            visualizations.append(
                CorrelationPlot(
                    sent_price_ema_corr,
                    f"{model_name} Sentiment EMA vs. {coin_ticker} Price EMA Correlation",
                    f"{model_name}_sentiment_ema_7_price_ema_7_correlation",
                )
            )

    # Sentiment Distribution by Model
    sentiment_distribution_by_model_data = pd.concat(
        [analysis_df[f"{model}_sentiment"] for model in sentiments], axis=1
    ).melt(var_name="Model", value_name="Sentiment")
    visualizations.append(
        BoxPlot(
            sentiment_distribution_by_model_data,
            x="Model",
            y="Sentiment",
            title="Sentiment Distribution by Model",
            filename="sentiment_distribution_by_model",
        )
    )

    for visualization in visualizations:
        visualization.save()

    logger.info("visualizations saved")

    pd.DataFrame(sentiments).to_csv(
        "data/out/sentiment.csv", sep="\t", encoding="utf-8", index=False, header=True
    )

    analysis_df.to_csv(
        "data/out/analysis.csv", sep="\t", encoding="utf-8", index=False, header=True
    )

    logger.info("saved to csv")