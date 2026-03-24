"""
Compute daily Realized Volatility features from Nifty spot OHLC data,
and merge IV 7-day constant-maturity forward from the IV:HV pipeline.

Features:
  1. RV_today    – Yang-Zhang single-day estimator, annualized (* sqrt(252))
  2. RV_3d_avg   – Simple mean of previous 3 trading days' RV (excluding today)
  3. RV_ratio    – RV_today / RV_3d_avg
  4. RV_7d_avg   – Simple mean of previous 7 trading days' RV (excluding today)
  5. RV_7d_ratio – RV_today / RV_7d_avg
  6. RV_pctrank_30d – Percentile rank of RV_today over trailing 30-day window (0–100%)
  7. IV_7d       – 7-day constant-maturity forward IV (close value, %)
                   Linear interpolation between nearest & next expiry ATM IVs:
                   w1 = (d2 - 7) / (d2 - d1),  w2 = (7 - d1) / (d2 - d1)
                   IV_7d = w1 * atm_iv_e0 + w2 * atm_iv_e1
  8. IV_change_1d – IV_7d_today − IV_7d_yesterday (daily change in 7d forward IV)
  9. VRP_today   – IV_7d − RV_today (volatility risk premium)
"""

import numpy as np
import pandas as pd
from pathlib import Path

SPOT_PATH = Path(__file__).resolve().parent / "nifty_spot_daily.parquet"
OUTPUT_PATH = Path(__file__).resolve().parent / "rv_daily.parquet"
IV_FEATURES_DIR = Path(__file__).resolve().parent.parent / "IV:HV" / "Macro" / "data" / "features"


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

    # RV_7d_avg: mean of previous 7 trading days (excluding today)
    df["RV_7d_avg"] = df["RV_today"].shift(1).rolling(window=7, min_periods=7).mean()

    # RV_7d_ratio
    df["RV_7d_ratio"] = df["RV_today"] / df["RV_7d_avg"]

    # RV_pctrank_30d: percentile rank of RV_today (1-day) over trailing 30-day window
    df["RV_pctrank_30d"] = df["RV_today"].rolling(window=30, min_periods=30).apply(
        lambda w: (w.iloc[-1] > w.iloc[:-1]).sum() / (len(w) - 1) * 100, raw=False
    )

    return df


def load_iv_7d_daily(features_dir: Path) -> pd.DataFrame:
    """
    Extract daily close IV_7d from per-second features parquets.

    Each date folder contains a features.parquet with ~22,501 rows (1-sec ticks).
    We take the last row (market close) iv_7d value for each date.
    """
    records = []
    if not features_dir.exists():
        print(f"WARNING: IV features dir not found: {features_dir}")
        return pd.DataFrame(columns=["timestamp", "IV_7d"])

    for date_dir in sorted(features_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        pq = date_dir / "features.parquet"
        if not pq.exists():
            continue
        try:
            feat = pd.read_parquet(pq, columns=["iv_7d"])
            if len(feat) == 0 or "iv_7d" not in feat.columns:
                continue
            close_iv = feat["iv_7d"].iloc[-1]
            if pd.notna(close_iv):
                records.append({"timestamp": pd.Timestamp(date_dir.name), "IV_7d": round(float(close_iv), 2)})
        except Exception:
            continue

    if not records:
        return pd.DataFrame(columns=["timestamp", "IV_7d"])

    return pd.DataFrame(records)


def main():
    df = load_spot_ohlc(SPOT_PATH)
    df = compute_rv_features(df)

    # Merge IV_7d from IV:HV pipeline
    iv_daily = load_iv_7d_daily(IV_FEATURES_DIR)
    if len(iv_daily) > 0:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.normalize()
        iv_daily["timestamp"] = pd.to_datetime(iv_daily["timestamp"]).dt.normalize()
        df = df.merge(iv_daily, on="timestamp", how="left")
        matched = df["IV_7d"].notna().sum()
        print(f"Merged IV_7d: {matched}/{len(df)} days matched")
    else:
        df["IV_7d"] = np.nan
        print("WARNING: No IV_7d data loaded")

    # IV_change_1d: daily change in 7d forward IV
    df["IV_change_1d"] = df["IV_7d"] - df["IV_7d"].shift(1)

    # VRP_today: volatility risk premium
    df["VRP_today"] = df["IV_7d"] - df["RV_today"]

    df = df[["timestamp", "open", "high", "low", "close",
             "RV_today", "RV_3d_avg", "RV_ratio", "RV_7d_avg", "RV_7d_ratio",
             "RV_pctrank_30d", "IV_7d", "IV_change_1d", "VRP_today"]]

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
    print(df.dropna().head(10).to_string(index=False))


if __name__ == "__main__":
    main()
