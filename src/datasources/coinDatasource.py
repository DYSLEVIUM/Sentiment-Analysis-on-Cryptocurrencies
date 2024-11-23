from .datasource import DataSource
from src.coins.coin import Coin
from datetime import datetime
import pandas as pd
from utils.csvReader import CSVReader
import cryptocompare
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CoinDatasource(DataSource):
    _cache_dir = "data/out"

    def __init__(self, coin: Coin):
        self.coin = coin
        self._cache_file = f"{self._cache_dir}/{self.coin.ticker.lower()}/price.csv"
        os.makedirs(self._cache_dir, exist_ok=True)

        data = self._load_data()
        super().__init__(data)

    def _load_data(self):
        coin_start_timestamp = int(self.coin.start_date.timestamp())
        current_timestamp = int(datetime.now().timestamp())

        df = pd.DataFrame()
        earliest_cached_timestamp = coin_start_timestamp

        # check if cached data exists
        if os.path.exists(self._cache_file):
            try:
                df = CSVReader.read(self._cache_file)
                earliest_cached_timestamp = int(df.iloc[-1]["date"])
            except Exception as e:
                logger.error(f"Error reading cache file: {e}")

        # fetch new data
        new_df = pd.DataFrame()
        while earliest_cached_timestamp < current_timestamp:
            try:
                batch = cryptocompare.get_historical_price_hour(
                    self.coin.ticker, currency="USD", limit=2000, toTs=current_timestamp
                )
                if not batch:
                    break

                valid_data = [
                    entry
                    for entry in batch
                    if not (
                        entry.get("close") == 0
                        and entry.get("high") == 0
                        and entry.get("low") == 0
                        and entry.get("open") == 0
                        and entry.get("volumefrom") == 0
                        and entry.get("volumeto") == 0
                    )
                ]

                new_df = pd.concat([new_df, pd.DataFrame(valid_data)])
                current_timestamp = batch[0]["time"]

            except Exception as e:
                logger.error(f"Error fetching data: {e}")
                break

        new_df = new_df.rename(
            columns={
                "time": "date",
                "close": "price",
                "high": "high",
                "low": "low",
                "open": "open",
                "volumefrom": "volume_from",
                "volumeto": "volume_to",
            }
        )
        new_df.sort_values(by="date", inplace=True)

        mode = "a" if os.path.exists(self._cache_file) else "w"
        header = not os.path.exists(self._cache_file)

        new_df.to_csv(self._cache_file, mode=mode, header=header, index=False)

        df = pd.concat([df, new_df])

        df["date"] = pd.to_datetime(df["date"], unit="s")

        return df
