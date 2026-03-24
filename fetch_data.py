"""
Fetch 1-second Nifty spot tick data from the parent DATA folder
and resample to Daily OHLC (one bar per trading day).

Source: /Users/harsha/Desktop/Research/DATA/NSE/NIFTY/{date}/Index/Cleaned_Spot.parquet
Each file has columns: datetime (UTC), ltp (last traded price)
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

DATA_ROOT = Path(__file__).resolve().parent.parent / "DATA" / "NSE" / "NIFTY"
OUTPUT_PATH = Path(__file__).resolve().parent / "nifty_spot_daily.parquet"


def load_tick_data() -> pd.DataFrame:
    """Load all Cleaned_Spot.parquet files and concatenate."""
    date_dirs = sorted(DATA_ROOT.iterdir())
    frames = []
    skipped = 0

    for d in date_dirs:
        if not d.is_dir():
            continue
        spot_file = d / "Index" / "Cleaned_Spot.parquet"
        if not spot_file.exists():
            skipped += 1
            continue
        df = pd.read_parquet(spot_file)
        df["trade_date"] = d.name  # YYYY-MM-DD folder name
        frames.append(df)

    print(f"Loaded {len(frames)} trading days ({skipped} skipped)")
    return pd.concat(frames, ignore_index=True)


def resample_to_daily(ticks: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 1-second tick data to daily OHLC bars.
    One bar per trading day using the folder date as the grouping key.
    """
    ticks["trade_date"] = pd.to_datetime(ticks["trade_date"])

    daily = ticks.groupby("trade_date").agg(
        open=("ltp", "first"),
        high=("ltp", "max"),
        low=("ltp", "min"),
        close=("ltp", "last"),
    ).reset_index()

    daily = daily.rename(columns={"trade_date": "timestamp"})
    daily = daily.sort_values("timestamp").reset_index(drop=True)

    # Ensure correct dtypes
    for col in ["open", "high", "low", "close"]:
        daily[col] = daily[col].astype(float)

    return daily


def main():
    print("Loading 1-second tick data from DATA folder...")
    ticks = load_tick_data()

    print("Resampling to daily OHLC...")
    daily = resample_to_daily(ticks)

    daily.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(daily)} daily bars to {OUTPUT_PATH}")
    print(f"Date range: {daily['timestamp'].min().date()} → {daily['timestamp'].max().date()}")
    print(daily.head(5).to_string(index=False))
    print("...")
    print(daily.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
