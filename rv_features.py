"""
Compute daily Realized Volatility features from Nifty spot OHLC data.

Features:
  1. RV_today  – Yang-Zhang single-day estimator, annualized (* sqrt(252))
     RV_today = SQRT( LN(O/C_prev)^2 + LN(H/O)*(LN(H/O)-LN(C/O)) + LN(L/O)*(LN(L/O)-LN(C/O)) ) * SQRT(252)

  2. RV_3d_avg – Simple mean of previous 3 trading days' annualized RV (excluding today)
     RV_3d_avg = MEAN(RV_today[t-1], RV_today[t-2], RV_today[t-3])

  3. RV_ratio
     RV_ratio = RV_today / RV_3d_avg
"""

import numpy as np
import pandas as pd
from pathlib import Path

SPOT_PATH = Path(__file__).resolve().parent / "nifty_spot_daily.parquet"
OUTPUT_PATH = Path(__file__).resolve().parent / "rv_daily.parquet"


def load_spot_ohlc(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def compute_rv_features(df: pd.DataFrame) -> pd.DataFrame:

    ro = np.log(df['open'] / df['close'].shift(1))
    rc = np.log(df['close'] / df['open'])
    rh = np.log(df['high'] / df['open'])
    rl = np.log(df['low'] / df['open'])

    RS = rh * (rh - rc) + rl * (rl - rc)
    RS = np.maximum(RS,0)  # Ensure non-negative to avoid sqrt of negative numbers
    df['RV_today'] = np.sqrt(ro**2 + RS)     * np.sqrt(252) * 100  # Annualize by multiplying with sqrt(252)

    # RV_3d_avg: mean of previous 3 trading days (excluding today)
    df["RV_3d_avg"] = df["RV_today"].shift(1).rolling(window=3, min_periods=3).mean()

    # RV_ratio
    df["RV_ratio"] = df["RV_today"] / df["RV_3d_avg"]

    return df


def main():
    df = load_spot_ohlc(SPOT_PATH)
    df = compute_rv_features(df)

    df = df[["timestamp", "open", "high", "low", "close", "RV_today", "RV_3d_avg", "RV_ratio"]]

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
    print(df.dropna().head(10).to_string(index=False))


if __name__ == "__main__":
    main()
