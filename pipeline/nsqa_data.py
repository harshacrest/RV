"""
nsqa_data.py — RV pipeline data layer via NSQA ProtosAdapter.

All market data is fetched through ProtosAdapter which reads local
parquet files from DATA/NSE/ via NSQA's MarketDataReader.

Requires:
  - pip install -e /path/to/nsqa_codebase
  - .env file in nsqa_codebase root (placeholder OK for local mode)
  - Data root: /Users/harsha/Desktop/Research/DATA/NSE
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, time

from data_management.market_reader_api.protos_adapter import ProtosAdapter

# ── Constants ──

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "DATA" / "NSE"
_FMT = "%Y-%m-%d %H:%M:%S"

SNAP_TIMES = {
    "0915": time(9, 15),
    "0916": time(9, 16),
    "1529": time(15, 29),
    "1530": time(15, 30),
}

# ── Adapter singleton ──

_adapter: ProtosAdapter | None = None


def get_adapter(ticker: str = "NIFTY") -> ProtosAdapter:
    global _adapter
    if _adapter is None:
        _adapter = ProtosAdapter(ticker=ticker, data_root=DATA_ROOT)
    return _adapter


def get_trading_dates(ticker: str = "NIFTY") -> list[date]:
    return get_adapter(ticker)._reader.gateway.list_dates(ticker)


# ── Index / Spot data ──

def fetch_daily_ohlc(start_date: date | None = None,
                     end_date: date | None = None,
                     ticker: str = "NIFTY") -> pd.DataFrame:
    """
    Fetch cleaned index ticks via ProtosAdapter and resample to daily OHLC.
    Returns DataFrame with columns: timestamp, open, high, low, close.
    """
    pa = get_adapter(ticker)
    dates = get_trading_dates(ticker)
    if start_date:
        dates = [d for d in dates if d >= start_date]
    if end_date:
        dates = [d for d in dates if d <= end_date]

    records = []
    for d in dates:
        df = pa.get_index_price(
            start_time=f"{d} 09:15:00", end_time=f"{d} 15:30:00",
            columns=["datetime", "ltp"],
        )
        if df.empty:
            continue
        ltp = df["ltp"]
        records.append({
            "timestamp": pd.Timestamp(d),
            "open": float(ltp.iloc[0]),
            "high": float(ltp.max()),
            "low": float(ltp.min()),
            "close": float(ltp.iloc[-1]),
        })

    if not records:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])

    daily = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    for col in ["open", "high", "low", "close"]:
        daily[col] = daily[col].astype(float)
    print(f"NSQA: Built {len(daily)} daily OHLC bars")
    return daily


# ── IV data ──

def _get_atm_iv_m1_snapshot(pa: ProtosAdapter, trading_date: date,
                            expiry_date: date, snap_time: time) -> float | None:
    """Extract ATM IV (M1, CE side) at a specific snapshot time for a given expiry."""
    t = f"{trading_date} {snap_time.strftime('%H:%M:%S')}"
    t1 = f"{trading_date} {snap_time.strftime('%H:%M')}:01"

    # Get ATM strike
    atm_strike = pa.get_strike_price(t, "CE", money_ness="ATM")
    if atm_strike == 0:
        return None

    # Get option chain at ATM strike for CE
    chain = pa.get_option_chain(
        start_time=t, end_time=t1,
        option_type="CE", strikes=[atm_strike],
        expiry_date=str(expiry_date),
    )
    if chain.empty or "iv_M1" not in chain.columns:
        return None

    val = chain["iv_M1"].iloc[0]
    return float(val) if pd.notna(val) and val != 0 else None


def _compute_iv_7d_for_date(pa: ProtosAdapter, trading_date: date,
                            snap_time: time) -> float | None:
    """7-day constant-maturity forward IV via linear interpolation of two expiry ATM IVs."""
    t = f"{trading_date} {snap_time.strftime('%H:%M:%S')}"
    expiries = pa.get_option_expiry_dates_list(t)
    if len(expiries) < 2:
        return None

    future = [e for e in expiries if e >= trading_date]
    if len(future) >= 2:
        e0, e1 = future[0], future[1]
    elif len(future) == 1:
        e0 = future[0]
        others = [e for e in expiries if e != e0]
        e1 = others[-1] if others else e0
    else:
        e0, e1 = expiries[-1], expiries[-2] if len(expiries) > 1 else expiries[-1]

    iv_e0 = _get_atm_iv_m1_snapshot(pa, trading_date, e0, snap_time)
    iv_e1 = _get_atm_iv_m1_snapshot(pa, trading_date, e1, snap_time)

    d1 = (e0 - trading_date).days
    d2 = (e1 - trading_date).days

    # On expiry day (d1==0): w1=0, so iv_e0 is not needed — use iv_e1 directly.
    if iv_e0 is None:
        if d1 == 0 and iv_e1 is not None:
            return round(iv_e1 * 100, 2)
        return None

    iv_e0_pct = iv_e0 * 100
    if iv_e1 is None:
        return round(iv_e0_pct, 2)
    iv_e1_pct = iv_e1 * 100

    if d2 != d1 and d2 > 0:
        w1 = (d2 - 7) / (d2 - d1)
        w2 = (7 - d1) / (d2 - d1)
        return round(w1 * iv_e0_pct + w2 * iv_e1_pct, 2)
    return round(iv_e0_pct, 2)


def fetch_iv_7d_daily(start_date: date | None = None,
                      end_date: date | None = None,
                      ticker: str = "NIFTY") -> pd.DataFrame:
    """
    Compute IV_7d at 4 intraday snapshots for each trading date.
    Returns DataFrame: timestamp, IV_7d_0915, IV_7d_0916, IV_7d_1529, IV_7d_1530
    """
    pa = get_adapter(ticker)
    dates = get_trading_dates(ticker)
    if start_date:
        dates = [d for d in dates if d >= start_date]
    if end_date:
        dates = [d for d in dates if d <= end_date]

    records = []
    for d in dates:
        if not (DATA_ROOT / ticker / str(d) / "Options").exists():
            continue
        rec = {"timestamp": pd.Timestamp(d)}
        for suffix, snap_t in SNAP_TIMES.items():
            try:
                val = _compute_iv_7d_for_date(pa, d, snap_t)
                rec[f"IV_7d_{suffix}"] = val if val is not None else np.nan
            except Exception:
                rec[f"IV_7d_{suffix}"] = np.nan
        if pd.notna(rec.get("IV_7d_1530")):
            records.append(rec)

    if not records:
        return pd.DataFrame()
    result = pd.DataFrame(records)
    print(f"NSQA: Computed IV_7d for {len(result)} trading days")
    return result


# ── Combined RV daily features ──

def fetch_rv_daily(start_date: date | None = None,
                   end_date: date | None = None,
                   ticker: str = "NIFTY",
                   cache_path: Path | None = None) -> pd.DataFrame:
    """
    Build full rv_daily DataFrame from NSQA: OHLC + RV + IV_7d.

    Args:
        cache_path: If provided and exists, read from cache (fast startup).
                    Pass None to compute fresh from NSQA.
    """
    if cache_path is not None and cache_path.exists():
        print(f"NSQA: Loading from cache {cache_path.name}")
        return pd.read_parquet(cache_path)

    df = fetch_daily_ohlc(start_date=start_date, end_date=end_date, ticker=ticker)
    if len(df) == 0:
        return df

    # Yang-Zhang RV
    ro = np.log(df['open'] / df['close'].shift(1))
    rc = np.log(df['close'] / df['open'])
    rh = np.log(df['high'] / df['open'])
    rl = np.log(df['low'] / df['open'])
    RS = rh * (rh - rc) + rl * (rl - rc)
    RS = np.maximum(RS, 0)
    df['RV_today'] = np.sqrt(ro**2 + RS) * np.sqrt(252) * 100

    # IV_7d
    iv_daily = fetch_iv_7d_daily(start_date=start_date, end_date=end_date, ticker=ticker)
    if len(iv_daily) > 0:
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.normalize()
        iv_daily["timestamp"] = pd.to_datetime(iv_daily["timestamp"]).dt.normalize()
        df = df.merge(iv_daily, on="timestamp", how="left")
        print(f"NSQA: Merged IV_7d for {df['IV_7d_1530'].notna().sum()}/{len(df)} days")
    else:
        for s in ["0915", "0916", "1529", "1530"]:
            df[f"IV_7d_{s}"] = np.nan

    df["IV_7d"] = df["IV_7d_1530"]
    df["IV_change_1d"] = df["IV_7d"] - df["IV_7d"].shift(1)
    df["VRP_today"] = df["IV_7d"] - df["RV_today"]
    df["IV_intraday_change"] = df["IV_7d_0915"] - df["IV_7d_1530"]

    out_cols = ["timestamp", "open", "high", "low", "close", "RV_today",
                "IV_7d_0915", "IV_7d_0916", "IV_7d_1529", "IV_7d_1530",
                "IV_7d", "IV_change_1d", "VRP_today", "IV_intraday_change"]
    return df[out_cols]


# ── DTE computation ──

def compute_dte_data(ticker: str = "NIFTY",
                     cache_path: Path | None = None) -> pd.DataFrame:
    """Compute DTE (days to nearest expiry) for each trading date."""
    if cache_path is not None and cache_path.exists():
        print(f"NSQA: Loading DTE from cache {cache_path.name}")
        df = pd.read_csv(cache_path, usecols=["t_date", "DTE"])
        df["t_date"] = pd.to_datetime(df["t_date"]).dt.date
        df["DTE"] = pd.to_numeric(df["DTE"], errors="coerce").fillna(0).astype(int)
        return df

    pa = get_adapter(ticker)
    dates = get_trading_dates(ticker)
    records = []
    for d in dates:
        expiries = pa.get_option_expiry_dates_list(f"{d} 09:15:00")
        future = [e for e in expiries if e >= d]
        dte = (future[0] - d).days if future else 0
        records.append({"t_date": d, "DTE": dte})
    return pd.DataFrame(records)
