from datetime import datetime

import cryptocompare
import pandas as pd


def get_all_eth_historical_data():
    all_data = []
    current_timestamp = int(datetime.now().timestamp())

    eth_start_timestamp = int(datetime(2015, 7, 30).timestamp())

    while True:
        # Get batch of historical data
        batch = cryptocompare.get_historical_price_day(
            "ETH", currency="USD", limit=2000, toTs=current_timestamp
        )

        if not batch:
            break

        all_data.extend(batch)

        current_timestamp = batch[0]["time"]

        if eth_start_timestamp >= current_timestamp:
            break

    return all_data


def get_historical_data(coin, currency, start_date, end_date):
    return cryptocompare.get_historical_price_day(coin, currency, start_date, end_date)


def get_eth_prices_for_dates(dates):
    unique_dates = sorted(set(dates.dt.date))

    eth_prices = {}
    for i in range(
        0, len(unique_dates), 30
    ):  # Batch by 30 days (or whatever works within rate limits)
        batch_dates = unique_dates[i : i + 30]
        # Get the start and end dates for the batch
        start_date = batch_dates[0]
        end_date = batch_dates[-1]

        # Get the historical prices for the range of dates
        historical_data = cryptocompare.get_historical_price_day(
            "ETH",
            currency="USD",
            limit=len(batch_dates) - 1,
            toTs=int(datetime.combine(end_date, datetime.min.time()).timestamp()),
        )

        # Map prices to dates
        for data in historical_data:
            date_str = datetime.fromtimestamp(data["time"]).date()
            eth_prices[date_str] = data["close"]

    # Map prices back to the original dates list
    return [eth_prices.get(pd.to_datetime(date).date()) for date in dates]
