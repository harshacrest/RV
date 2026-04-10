"""
Compute daily Realized Volatility features from NSQA data.

All data is fetched via NSQA LocalGateway:
  - Daily OHLC from Cleaned_Spot.parquet
  - IV_7d from option chain data (ATM IV M1, interpolated across two expiries)

Output: features/rv_daily.parquet (cached snapshot for downstream consumers)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from pipeline.nsqa_data import fetch_rv_daily

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "features" / "rv_daily.parquet"


def compute_rv_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Yang-Zhang RV to a DataFrame with OHLC columns."""
    ro = np.log(df['open'] / df['close'].shift(1))
    rc = np.log(df['close'] / df['open'])
    rh = np.log(df['high'] / df['open'])
    rl = np.log(df['low'] / df['open'])

    RS = rh * (rh - rc) + rl * (rl - rc)
    RS = np.maximum(RS, 0)
    df['RV_today'] = np.sqrt(ro**2 + RS) * np.sqrt(252) * 100

    return df


def main():
    """Build rv_daily.parquet entirely from NSQA data."""
    print("Building RV features from NSQA...")
    df = fetch_rv_daily()

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
    print(df.dropna().head(5).to_string(index=False))


if __name__ == "__main__":
    main()
