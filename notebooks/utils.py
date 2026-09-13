from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "return_1d",
    "return_3d",
    "return_5d",
    "ma5_distance",
    "ma20_distance",
    "volatility_5d",
    "volatility_20d",
    "volume_change",
]


def load_stock_data(file_path):
    """
    Load raw stock data from CSV.
    """
    df = pd.read_csv(file_path)

    return df


def clean_stock_data(df):
    """
    Rename columns, convert date, and sort chronologically.
    """
    df = df.copy()

    df = df.rename(
        columns={
            "日期": "date",
            "股票代码": "ticker",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "涨跌幅": "pct_change",
            "涨跌额": "price_change",
            "换手率": "turnover",
        }
    )

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date").reset_index(drop=True)

    return df


def create_features(df):
    """
    Create technical features from historical stock data.
    """
    df = df.copy()

    # Price returns
    df["return_1d"] = df["close"].pct_change(1)
    df["return_3d"] = df["close"].pct_change(3)
    df["return_5d"] = df["close"].pct_change(5)

    # Moving averages
    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_10"] = df["close"].rolling(10).mean()
    df["ma_20"] = df["close"].rolling(20).mean()

    # Distance from moving averages
    df["ma5_distance"] = df["close"] / df["ma_5"] - 1

    df["ma20_distance"] = df["close"] / df["ma_20"] - 1

    # Volatility
    df["volatility_5d"] = df["return_1d"].rolling(5).std()

    df["volatility_20d"] = df["return_1d"].rolling(20).std()

    # Volume change
    df["volume_change"] = df["volume"].pct_change()

    return df


def create_prediction_target(df):
    """
    Create the supervised-learning target.

    target = 1: next trading day goes up
    target = 0: next trading day does not go up
    """
    df = df.copy()
    ## Calculate the next trading day's return

    df["next_day_return"] = df["close"].shift(-1) / df["close"] - 1

    # Create the binary target:
    # 1 = next trading day goes up
    # 0 = next trading day does not go up
    df["target"] = (df["next_day_return"] > 0).astype(int)

    return df


def prepare_prediction_data(df):
    """
    Prepare the final DataFrame for supervised learning.
    """
    model_df = df[["date"] + FEATURE_COLUMNS + ["next_day_return", "target"]].copy()
    # Replace infinite values with NaN
    model_df = model_df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Remove rows containing missing values(NaN)
    model_df = model_df.dropna().reset_index(drop=True)

    return model_df
