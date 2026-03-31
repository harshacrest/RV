"""
Compute daily Realized Volatility features from Nifty spot OHLC data,
and merge IV 7-day constant-maturity forward from the IV:HV pipeline.

Features:
  1. RV_today           – Yang-Zhang single-day estimator, annualized (* sqrt(252) * 100)
  2. IV_7d              – 7-day constant-maturity forward IV (close value, %)
  3. IV_change_1d       – IV_7d_today − IV_7d_yesterday (daily change in 7d forward IV)
  4. VRP_today          – IV_7d − RV_today (volatility risk premium)
  5. IV_intraday_change – IV_7d at market open − IV_7d at market close (intraday IV move)
"""

import numpy as np
import pandas as pd
from pathlib import Path

SPOT_PATH = Path(__file__).resolve().parent / "nifty_spot_daily.parquet"
OUTPUT_PATH = Path(__file__).resolve().parent / "rv_daily.parquet"
IV_FEATURES_DIR = Path(__file__).resolve().parent.parent / "Volatilities" / "Macro" / "data" / "features"


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
    df['RV_today'] = np.sqrt(ro**2 + RS) * np.sqrt(252) * 100  # Annualize by multiplying with sqrt(252)

    return df


def load_iv_7d_daily(features_dir: Path) -> pd.DataFrame:
    """
    Extract IV_7d at 4 intraday timestamps from per-second features parquets:
      09:15 (market open), 09:16, 15:29, 15:30 (market close).

    Each date folder contains a features.parquet with ~22,501 rows (1-sec ticks).
    """
    from datetime import time as dtime

    SNAP_TIMES = {
        "0915": dtime(9, 15),
        "0916": dtime(9, 16),
        "1529": dtime(15, 29),
        "1530": dtime(15, 30),
    }

    records = []
    if not features_dir.exists():
        print(f"WARNING: IV features dir not found: {features_dir}")
        return pd.DataFrame()

    for date_dir in sorted(features_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        pq = date_dir / "features.parquet"
        if not pq.exists():
            continue
        try:
            feat = pd.read_parquet(pq, columns=["datetime", "iv_7d"])
            if len(feat) == 0 or "iv_7d" not in feat.columns:
                continue

            times = feat["datetime"].dt.time
            rec = {"timestamp": pd.Timestamp(date_dir.name)}
            for suffix, t in SNAP_TIMES.items():
                mask = times == t
                if mask.any():
                    val = feat.loc[mask, "iv_7d"].iloc[0]
                    rec[f"IV_7d_{suffix}"] = round(float(val), 2) if pd.notna(val) else np.nan
                else:
                    rec[f"IV_7d_{suffix}"] = np.nan

            # Only add if at least the close value exists
            if pd.notna(rec.get("IV_7d_1530")):
                records.append(rec)
        except Exception:
            continue

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


def main():
    df = load_spot_ohlc(SPOT_PATH)
    df = compute_rv_features(df)

    # Merge IV_7d snapshots from IV:HV pipeline
    iv_daily = load_iv_7d_daily(IV_FEATURES_DIR)
    if len(iv_daily) > 0:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.normalize()
        iv_daily["timestamp"] = pd.to_datetime(iv_daily["timestamp"]).dt.normalize()
        df = df.merge(iv_daily, on="timestamp", how="left")
        matched = df["IV_7d_1530"].notna().sum()
        print(f"Merged IV_7d snapshots: {matched}/{len(df)} days matched")
    else:
        for s in ["0915", "0916", "1529", "1530"]:
            df[f"IV_7d_{s}"] = np.nan
        print("WARNING: No IV_7d data loaded")

    # Backward-compatible default columns (3:30 PM snapshot)
    df["IV_7d"] = df["IV_7d_1530"]
    df["IV_change_1d"] = df["IV_7d"] - df["IV_7d"].shift(1)
    df["VRP_today"] = df["IV_7d"] - df["RV_today"]
    df["IV_intraday_change"] = df["IV_7d_0915"] - df["IV_7d_1530"]

    out_cols = ["timestamp", "open", "high", "low", "close", "RV_today",
                "IV_7d_0915", "IV_7d_0916", "IV_7d_1529", "IV_7d_1530",
                "IV_7d", "IV_change_1d", "VRP_today", "IV_intraday_change"]
    df = df[out_cols]

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
    print(df.dropna().head(5).to_string(index=False))


if __name__ == "__main__":
    main()
