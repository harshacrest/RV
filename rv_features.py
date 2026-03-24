"""
Compute daily Realized Volatility features from Nifty spot OHLC data.

Features:
  1. RV_today    – Yang-Zhang single-day RV estimator
  2. RV_3d_avg   – Mean of RV_today over the previous 3 trading days (excluding today)
  3. RV_ratio    – RV_today / RV_3d_avg
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
    """
    Yang-Zhang single-day RV estimator (per-bar):
        RV_today = sqrt(
            ln(O/C_prev)^2
          + ln(H/O) * (ln(H/O) - ln(C/O))
          + ln(L/O) * (ln(L/O) - ln(C/O))
        )

    RV_3d_avg = mean of RV_today over [t-1, t-2, t-3]  (previous 3 trading days, excluding today)
    RV_ratio  = RV_today / RV_3d_avg
    """
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values

    prev_c = np.empty_like(c)
    prev_c[0] = np.nan
    prev_c[1:] = c[:-1]

    ln_o_pc = np.log(o / prev_c)            # ln(Open / Prev_Close)
    ln_h_o = np.log(h / o)                  # ln(High / Open)
    ln_l_o = np.log(l / o)                  # ln(Low  / Open)
    ln_c_o = np.log(c / o)                  # ln(Close / Open)

    rv_sq = (
        ln_o_pc ** 2
        + ln_h_o * (ln_h_o - ln_c_o)
        + ln_l_o * (ln_l_o - ln_c_o)
    )
    # Clamp to zero before sqrt to handle floating-point noise
    rv_today = np.sqrt(np.maximum(rv_sq, 0.0))

    df["RV_today"] = rv_today

    # RV_3d_avg: rolling mean of previous 3 days (shift by 1 to exclude today)
    df["RV_3d_avg"] = df["RV_today"].shift(1).rolling(window=3, min_periods=3).mean()

    # RV_ratio
    df["RV_ratio"] = df["RV_today"] / df["RV_3d_avg"]

    return df


def main():
    df = load_spot_ohlc(SPOT_PATH)
    df = compute_rv_features(df)

    # Keep only the required columns
    df = df[["timestamp", "open", "high", "low", "close", "RV_today", "RV_3d_avg", "RV_ratio"]]

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
    print(df.dropna().head(10).to_string(index=False))


if __name__ == "__main__":
    main()
