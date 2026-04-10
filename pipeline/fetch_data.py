"""
Fetch Nifty spot data via NSQA and resample to Daily OHLC.

Uses NSQA LocalGateway to read Cleaned_Spot.parquet files from:
  DATA/NSE/NIFTY/{date}/Index/Cleaned_Spot.parquet

Output: features/nifty_spot_daily.parquet
"""

from pathlib import Path
from pipeline.nsqa_data import fetch_daily_ohlc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "features" / "nifty_spot_daily.parquet"


def main():
    print("Fetching daily OHLC via NSQA...")
    daily = fetch_daily_ohlc()

    daily.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(daily)} daily bars to {OUTPUT_PATH}")
    print(f"Date range: {daily['timestamp'].min().date()} → {daily['timestamp'].max().date()}")
    print(daily.head(5).to_string(index=False))
    print("...")
    print(daily.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
