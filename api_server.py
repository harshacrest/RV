"""
RV Dashboard API Server
Serves pre-computed strategy returns bucketed by RV features.
"""

import math
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from scipy import stats as scipy_stats
from statsmodels.tsa.stattools import adfuller, acf as sm_acf

app = FastAPI(title="RV Dashboard API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE = Path(__file__).resolve().parent

STRATEGY_FILES = {
    "dm": BASE / "strategy_returns_DM_per_trade_both_max_100.xlsx",
    "wc": BASE / "strategy_returns_90_0_both_itm.xlsx",
    "orion": BASE / "strategy_returns_orion_index_kd_60_40_sl10_max90_min20.xlsx",
    "dmo": BASE / "strategy_returns_DMO.xlsx",
}

FEATURES = ["RV_today", "IV_7d", "IV_change_1d", "VRP_today", "IV_intraday_change"]

RISK_FREE_PCT = 5.5  # annual risk-free rate in % (same units as Net_Daily_PnL_PerCent)

# ── Regime Classification Constants ──
# IV Level boundaries (from Step 4 of the framework)
IV_L1_UPPER = 12
IV_L2_UPPER = 17
# PK/IV ratio thresholds — computed as medians at startup for equal splits
PKIV_L1_THRESHOLD = 0.63  # fallback; overwritten by _init_thresholds()
PKIV_L2_THRESHOLD = 0.65
PKIV_L3_THRESHOLD = 0.67

REGIME_STATES = [
    "L1 Safe", "L1 Exposed",
    "L2 Safe", "L2 Caution-A", "L2 Caution-B", "L2 Risky",
    "L3 Safe", "L3 Exposed",
]

REGIME_COLORS = {
    "L1 Safe": "#00e676",
    "L1 Exposed": "#ffab40",
    "L2 Safe": "#00e676",
    "L2 Caution-A": "#ffab40",
    "L2 Caution-B": "#ff9100",
    "L2 Risky": "#ff5252",
    "L3 Safe": "#64ffda",
    "L3 Exposed": "#ff1744",
}

REGIME_DESCRIPTIONS = {
    "L1 Safe": "Low IV, deep premium cushion. PK/IV ≤ 0.63",
    "L1 Exposed": "Low IV, thin cushion. PK/IV > 0.63",
    "L2 Safe": "Moderate IV, low PK/IV, IV falling. Best regime.",
    "L2 Caution-A": "Moderate IV, high PK/IV, IV falling. DM/WC ok, Orion weak.",
    "L2 Caution-B": "Moderate IV, low PK/IV, IV rising. DM weak, Orion strong.",
    "L2 Risky": "Moderate IV, high PK/IV, IV rising. Mixed signals.",
    "L3 Safe": "High IV, wide cushion. PK/IV ≤ 0.67. DM standout.",
    "L3 Exposed": "High IV, thin cushion. PK/IV > 0.67. Most dangerous — 21% AL.",
}


def _clean(v):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    return v


TRADING_DATES_PATH = BASE.parent / "DATA" / "NSE" / "trading_dates.csv"


def _load():
    rv = pd.read_parquet(BASE / "rv_daily.parquet")
    rv["timestamp"] = pd.to_datetime(rv["timestamp"])
    rv["date"] = rv["timestamp"].dt.date

    strats = {}
    for key, path in STRATEGY_FILES.items():
        df = pd.read_excel(path, sheet_name="returns")
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        strats[key] = df

    # Load DTE from trading_dates.csv
    dte_df = pd.read_csv(TRADING_DATES_PATH, usecols=["t_date", "DTE"])
    dte_df["t_date"] = pd.to_datetime(dte_df["t_date"]).dt.date
    dte_df["DTE"] = pd.to_numeric(dte_df["DTE"], errors="coerce")
    dte_df = dte_df.dropna(subset=["DTE"])
    dte_df["DTE"] = dte_df["DTE"].astype(int)

    return rv, strats, dte_df


RV_DATA, STRAT_DATA, DTE_DATA = _load()


def _init_pkiv_thresholds():
    """Compute PK/IV median per level from default snapshot for equal day splits."""
    global PKIV_L1_THRESHOLD, PKIV_L2_THRESHOLD, PKIV_L3_THRESHOLD
    try:
        df = RV_DATA.copy()
        df["PK_today"] = _compute_parkinson_vol(df["high"], df["low"])
        iv_col = "IV_7d_1530"
        if iv_col not in df.columns:
            iv_col = "IV_7d"
        df["_iv"] = df[iv_col]
        df["IV_5d"] = df["_iv"].shift(1).rolling(5, min_periods=3).mean()
        df["PK_5d"] = df["PK_today"].shift(1).rolling(5, min_periods=3).mean()
        df["iv_lag"] = df["_iv"].shift(1)
        df["PK_IV_ratio"] = np.where(df["IV_5d"] > 0, df["PK_5d"] / df["IV_5d"], np.nan)
        clean = df.dropna(subset=["iv_lag", "PK_IV_ratio"])
        l1 = clean[clean["iv_lag"] < IV_L1_UPPER]["PK_IV_ratio"]
        l2 = clean[(clean["iv_lag"] >= IV_L1_UPPER) & (clean["iv_lag"] < IV_L2_UPPER)]["PK_IV_ratio"]
        l3 = clean[clean["iv_lag"] >= IV_L2_UPPER]["PK_IV_ratio"]
        if len(l1) > 10:
            PKIV_L1_THRESHOLD = round(float(l1.median()), 4)
        if len(l2) > 10:
            PKIV_L2_THRESHOLD = round(float(l2.median()), 4)
        if len(l3) > 10:
            PKIV_L3_THRESHOLD = round(float(l3.median()), 4)
        print(f"PK/IV thresholds (median): L1={PKIV_L1_THRESHOLD}, L2={PKIV_L2_THRESHOLD}, L3={PKIV_L3_THRESHOLD}")
    except Exception as e:
        print(f"Warning: could not compute PK/IV medians, using fallbacks. Error: {e}")


def _compute_parkinson_vol(high: pd.Series, low: pd.Series) -> pd.Series:
    """Compute daily Parkinson volatility from High/Low prices (annualized, in %)."""
    log_hl = np.log(high / low)
    pk_daily = np.sqrt(log_hl ** 2 / (4 * np.log(2)))
    return pk_daily * np.sqrt(252) * 100


# Compute median-based thresholds now that _compute_parkinson_vol is defined
_init_pkiv_thresholds()


def _compute_regime_features(rv_df: pd.DataFrame, snapshot: str = "1530") -> pd.DataFrame:
    """Compute the 3 regime inputs: IV_5d, PK_5d, IV_chg_5d, plus iv_lag and PK/IV ratio."""
    df = rv_df.copy()

    # Compute Parkinson Vol from OHLC
    df["PK_today"] = _compute_parkinson_vol(df["high"], df["low"])

    # Use snapshot-specific IV
    iv_col = f"IV_7d_{snapshot}"
    if iv_col in df.columns:
        df["_iv"] = df[iv_col]
    else:
        df["_iv"] = df["IV_7d"]

    # IV daily change
    df["_iv_change"] = df["_iv"] - df["_iv"].shift(1)

    # 5-day lagged averages (days t-5 to t-1)
    df["IV_5d"] = df["_iv"].shift(1).rolling(5, min_periods=3).mean()
    df["PK_5d"] = df["PK_today"].shift(1).rolling(5, min_periods=3).mean()
    df["IV_chg_5d"] = df["_iv_change"].shift(1).rolling(5, min_periods=3).mean()

    # Lagged IV for level classification
    df["iv_lag"] = df["_iv"].shift(1)

    # PK/IV ratio
    df["PK_IV_ratio"] = np.where(df["IV_5d"] > 0, df["PK_5d"] / df["IV_5d"], np.nan)

    # Clean up temp columns
    df.drop(columns=["_iv", "_iv_change"], inplace=True, errors="ignore")

    return df


def _compute_pkiv_medians(df: pd.DataFrame) -> tuple[float, float, float]:
    """Compute per-level PK/IV medians from the given data for equal day splits."""
    clean = df.dropna(subset=["iv_lag", "PK_IV_ratio"])
    l1 = clean[clean["iv_lag"] < IV_L1_UPPER]["PK_IV_ratio"]
    l2 = clean[(clean["iv_lag"] >= IV_L1_UPPER) & (clean["iv_lag"] < IV_L2_UPPER)]["PK_IV_ratio"]
    l3 = clean[clean["iv_lag"] >= IV_L2_UPPER]["PK_IV_ratio"]
    th_l1 = round(float(l1.median()), 4) if len(l1) > 10 else PKIV_L1_THRESHOLD
    th_l2 = round(float(l2.median()), 4) if len(l2) > 10 else PKIV_L2_THRESHOLD
    th_l3 = round(float(l3.median()), 4) if len(l3) > 10 else PKIV_L3_THRESHOLD
    return th_l1, th_l2, th_l3


def _classify_regime_with_thresholds(row: pd.Series, th_l1: float, th_l2: float, th_l3: float) -> str | None:
    """Classify a single day into one of 8 regime states using given thresholds."""
    iv_lag = row.get("iv_lag")
    pk_iv = row.get("PK_IV_ratio")
    iv_chg = row.get("IV_chg_5d")

    if pd.isna(iv_lag) or pd.isna(pk_iv):
        return None

    if iv_lag < IV_L1_UPPER:
        return "L1 Safe" if pk_iv <= th_l1 else "L1 Exposed"
    elif iv_lag < IV_L2_UPPER:
        iv_falling = iv_chg <= 0 if not pd.isna(iv_chg) else True
        if pk_iv <= th_l2:
            return "L2 Safe" if iv_falling else "L2 Caution-B"
        else:
            return "L2 Caution-A" if iv_falling else "L2 Risky"
    else:
        return "L3 Safe" if pk_iv <= th_l3 else "L3 Exposed"


def _add_regime_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add regime_state column using per-level PK/IV medians for equal day splits."""
    df = df.copy()
    th_l1, th_l2, th_l3 = _compute_pkiv_medians(df)
    # Store thresholds on the dataframe for downstream access
    df.attrs["pkiv_l1"] = th_l1
    df.attrs["pkiv_l2"] = th_l2
    df.attrs["pkiv_l3"] = th_l3
    df["regime_state"] = df.apply(lambda r: _classify_regime_with_thresholds(r, th_l1, th_l2, th_l3), axis=1)
    return df


def _filter_dates(df: pd.DataFrame, date_col: str, start: str | None, end: str | None) -> pd.DataFrame:
    if start:
        df = df[df[date_col] >= pd.Timestamp(start).date()]
    if end:
        df = df[df[date_col] <= pd.Timestamp(end).date()]
    return df


VALID_SNAPSHOTS = {"0915", "0916", "1529", "1530"}

# Mapping: snapshot → which pair to use for IV_intraday_change
# 0915/1530 (edge times) → full day 0915-1530
# 0916/1529 (inner times) → inner day 0916-1529
INTRADAY_PAIRS = {
    "0915": ("0915", "1530"),
    "0916": ("0916", "1529"),
    "1529": ("0916", "1529"),
    "1530": ("0915", "1530"),
}


def _merge(strategy: str, start: str | None = None, end: str | None = None,
           snapshot: str = "1530") -> pd.DataFrame:
    rv = RV_DATA.copy()
    st = STRAT_DATA[strategy].copy()
    merged = rv.merge(st, left_on="date", right_on="Date", how="inner")
    merged = merged.merge(DTE_DATA, left_on="date", right_on="t_date", how="left")
    merged = _filter_dates(merged, "date", start, end)

    # Recompute IV-dependent features based on snapshot
    if snapshot in VALID_SNAPSHOTS:
        iv_col = f"IV_7d_{snapshot}"
        if iv_col in merged.columns:
            merged["IV_7d"] = merged[iv_col]
            merged["IV_change_1d"] = merged["IV_7d"] - merged["IV_7d"].shift(1)
            merged["VRP_today"] = merged["IV_7d"] - merged["RV_today"]

        # IV_intraday_change: use the appropriate pair
        open_snap, close_snap = INTRADAY_PAIRS[snapshot]
        open_col = f"IV_7d_{open_snap}"
        close_col = f"IV_7d_{close_snap}"
        if open_col in merged.columns and close_col in merged.columns:
            merged["IV_intraday_change"] = merged[open_col] - merged[close_col]

    return merged


def _summary(df: pd.DataFrame) -> dict:
    pnl = df["Net_Daily_PnL_PerCent"]
    total_pct = df["Net_Equity_Curve"].iloc[-1] if len(df) > 0 else 0
    pos = (pnl > 0).sum()
    neg = (pnl < 0).sum()
    flat = (pnl == 0).sum()
    std = pnl.std()
    mean = pnl.mean()
    ann_return = mean * 252
    ann_vol = float(std * math.sqrt(252)) if std > 0 else None
    sharpe = ((ann_return - RISK_FREE_PCT) / ann_vol) if ann_vol and ann_vol > 0 else None

    # Cumulative & Drawdown
    cum = pnl.cumsum()
    running_max = cum.cummax()
    dd = cum - running_max
    max_dd = dd.min()

    # CAGR (compound)
    n_years = len(df) / 252 if len(df) > 0 else 1
    equity_final = 1 + total_pct / 100
    cagr = (equity_final ** (1 / n_years) - 1) * 100 if equity_final > 0 and n_years > 0 else None

    # Sortino (proper downside deviation: sqrt(mean(min(r - target, 0)^2)) over ALL obs)
    daily_rf = RISK_FREE_PCT / 252
    downside = np.minimum(pnl - daily_rf, 0)
    downside_dev = float(np.sqrt(np.mean(downside**2)))
    downside_ann = downside_dev * math.sqrt(252) if downside_dev > 0 else None
    sortino = ((ann_return - RISK_FREE_PCT) / downside_ann) if downside_ann and downside_ann > 0 else None

    # Calmar (CAGR / |max_dd|)
    calmar = (cagr / abs(max_dd)) if cagr is not None and max_dd < 0 else None

    # Profit Factor
    gross_profit = float(pnl[pnl > 0].sum()) if pos > 0 else 0
    gross_loss = float(abs(pnl[pnl < 0].sum())) if neg > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None

    # Avg Win / Avg Loss / Payoff Ratio / Expectancy
    avg_win = float(pnl[pnl > 0].mean()) if pos > 0 else 0
    avg_loss = float(pnl[pnl < 0].mean()) if neg > 0 else 0
    payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else None
    win_rate_f = float(pos / max(pos + neg, 1))
    expectancy = win_rate_f * avg_win + (1 - win_rate_f) * avg_loss

    # Skewness & Kurtosis
    skew = float(pnl.skew()) if len(pnl) > 2 else None
    kurt = float(pnl.kurtosis()) if len(pnl) > 3 else None

    # Max Drawdown Duration
    dd_duration = 0
    max_dd_duration = 0
    for v in dd.values:
        if v < 0:
            dd_duration += 1
            max_dd_duration = max(max_dd_duration, dd_duration)
        else:
            dd_duration = 0

    # Recovery Factor (total return / |max_dd|)
    recovery_factor = (total_pct / abs(max_dd)) if max_dd < 0 else None

    # Consecutive wins/losses
    max_consec_wins = 0
    max_consec_losses = 0
    cw = 0
    cl = 0
    for v in pnl.values:
        if v > 0:
            cw += 1
            cl = 0
        elif v < 0:
            cl += 1
            cw = 0
        else:
            cw = 0
            cl = 0
        max_consec_wins = max(max_consec_wins, cw)
        max_consec_losses = max(max_consec_losses, cl)

    # Tail ratios
    p95 = float(np.percentile(pnl, 95)) if len(pnl) > 0 else None
    p5 = float(np.percentile(pnl, 5)) if len(pnl) > 0 else None
    tail_ratio = abs(p95 / p5) if p5 and p5 != 0 else None

    # Daily VaR (5%)
    var_95 = float(np.percentile(pnl, 5)) if len(pnl) > 0 else None

    # Drawdown series for chart
    dd_series = [{"date": str(d), "dd_pct": round(float(v), 4)} for d, v in zip(df["date"], dd)]

    return {
        "total_pct": round(float(total_pct), 2),
        "mean_daily_pct": round(float(mean), 4),
        "median_daily_pct": round(float(pnl.median()), 4),
        "std_daily_pct": round(float(std), 4),
        "win_rate": round(win_rate_f, 4),
        "sharpe": round(float(sharpe), 2) if sharpe else None,
        "max_win_pct": round(float(pnl.max()), 4),
        "max_loss_pct": round(float(pnl.min()), 4),
        "max_drawdown_pct": round(float(max_dd), 4),
        "total_days": int(len(df)),
        "positive_days": int(pos),
        "negative_days": int(neg),
        "flat_days": int(flat),
        # Extended metrics
        "cagr_pct": round(float(cagr), 2) if cagr is not None else None,
        "ann_return_pct": round(float(ann_return), 2),
        "ann_vol_pct": round(float(ann_vol), 2) if ann_vol else None,
        "sortino": round(float(sortino), 2) if sortino else None,
        "calmar": round(float(calmar), 2) if calmar else None,
        "profit_factor": round(float(profit_factor), 2) if profit_factor else None,
        "avg_win_pct": round(avg_win, 4),
        "avg_loss_pct": round(avg_loss, 4),
        "payoff_ratio": round(float(payoff_ratio), 2) if payoff_ratio else None,
        "expectancy_pct": round(float(expectancy), 4),
        "skewness": round(float(skew), 3) if skew is not None else None,
        "kurtosis": round(float(kurt), 3) if kurt is not None else None,
        "max_dd_duration_days": int(max_dd_duration),
        "recovery_factor": round(float(recovery_factor), 2) if recovery_factor else None,
        "max_consec_wins": int(max_consec_wins),
        "max_consec_losses": int(max_consec_losses),
        "var_95_pct": round(float(var_95), 4) if var_95 is not None else None,
        "p95_pct": round(float(p95), 4) if p95 is not None else None,
        "p5_pct": round(float(p5), 4) if p5 is not None else None,
        "tail_ratio": round(float(tail_ratio), 2) if tail_ratio else None,
        "gross_profit_pct": round(gross_profit, 2),
        "gross_loss_pct": round(gross_loss, 2),
        "dd_series": dd_series,
    }


def _bucket_metrics(sub: pd.DataFrame, label: str, rng: list, feature: str = None) -> dict:
    pnl = sub["Net_Daily_PnL_PerCent"]
    days = len(sub)
    if days == 0:
        return {
            "label": label, "range": rng, "trading_days": 0,
            "total_pct": 0, "avg_daily_pct": 0, "win_rate": 0, "loss_rate": 0,
            "sharpe": None, "sharpe_pct": None, "ann_vol_pct": None,
            "max_win_pct": 0, "max_loss_pct": 0, "max_drawdown_pct": 0,
            "feat_mean": None, "feat_max": None, "feat_min": None,
            "streak_mean": None, "streak_median": None, "streak_min": None, "streak_max": None,
        }
    pos = (pnl > 0).sum()
    neg = (pnl < 0).sum()
    std = pnl.std()
    mean = pnl.mean()
    sharpe = ((mean * 252 - RISK_FREE_PCT) / (std * math.sqrt(252))) if std > 0 else None
    ann_vol = float(std * math.sqrt(252)) if std > 0 else None
    win_rate = float(pos / max(pos + neg, 1))
    cum = pnl.cumsum()
    dd = cum - cum.cummax()
    return {
        "label": label,
        "range": [_clean(rng[0]), _clean(rng[1])],
        "trading_days": int(days),
        "total_pct": round(float(pnl.sum()), 4),
        "avg_daily_pct": round(float(mean), 4),
        "win_rate": round(win_rate, 4),
        "loss_rate": round(1 - win_rate, 4),
        "sharpe": round(float(sharpe), 2) if sharpe else None,
        "sharpe_pct": round(float(sharpe), 2) if sharpe else None,
        "ann_vol_pct": round(ann_vol, 4) if ann_vol else None,
        "max_win_pct": round(float(pnl.max()), 4) if days > 0 else 0,
        "max_loss_pct": round(float(pnl.min()), 4) if days > 0 else 0,
        "max_drawdown_pct": round(float(dd.min()), 4) if days > 0 else 0,
        "feat_mean": round(float(sub[feature].mean()), 6) if feature and feature in sub.columns and days > 0 else None,
        "feat_max": round(float(sub[feature].max()), 6) if feature and feature in sub.columns and days > 0 else None,
        "feat_min": round(float(sub[feature].min()), 6) if feature and feature in sub.columns and days > 0 else None,
        "streak_mean": None, "streak_median": None, "streak_min": None, "streak_max": None,
    }


def _compute_streaks(bucket_labels: np.ndarray) -> dict[int, list[int]]:
    """Given an array of bucket indices (one per day, time-ordered), compute consecutive streaks per bucket."""
    streaks: dict[int, list[int]] = {}
    if len(bucket_labels) == 0:
        return streaks
    cur_label = bucket_labels[0]
    cur_len = 1
    for j in range(1, len(bucket_labels)):
        if bucket_labels[j] == cur_label:
            cur_len += 1
        else:
            streaks.setdefault(cur_label, []).append(cur_len)
            cur_label = bucket_labels[j]
            cur_len = 1
    streaks.setdefault(cur_label, []).append(cur_len)
    return streaks


def _inject_streaks(buckets: list[dict], streaks: dict[int, list[int]]):
    for i, b in enumerate(buckets):
        s = streaks.get(i, [])
        if s:
            b["streak_mean"] = round(float(np.mean(s)), 1)
            b["streak_median"] = round(float(np.median(s)), 1)
            b["streak_min"] = int(np.min(s))
            b["streak_max"] = int(np.max(s))


def _compute_buckets(df: pd.DataFrame, feature: str) -> list[dict]:
    valid = df.dropna(subset=[feature]).sort_values("date")
    if len(valid) == 0:
        return []

    # 3 equal-count (tercile) buckets
    edges = np.unique(np.quantile(valid[feature].values, [0, 1/3, 2/3, 1.0])).tolist()

    buckets = []
    bucket_labels = np.full(len(valid), -1, dtype=int)
    vals = valid[feature].values
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            mask = (vals >= lo) & (vals <= hi)
        else:
            mask = (vals >= lo) & (vals < hi)
        label = f"{lo:.2f} – {hi:.2f}"
        bucket_labels[mask] = i
        sub = valid[mask]
        buckets.append(_bucket_metrics(sub, label, [_clean(float(lo)), _clean(float(hi))], feature))

    streaks = _compute_streaks(bucket_labels)
    _inject_streaks(buckets, streaks)
    return buckets


def _compute_percentile_buckets(df: pd.DataFrame, feature: str) -> list[dict]:
    valid = df.dropna(subset=[feature]).sort_values("date")
    if len(valid) == 0:
        return []
    valid = valid.copy()
    valid["_pct"] = valid[feature].rank(pct=True)
    labels = ["P0–P33", "P33–P67", "P67–P100"]
    p_edges = [0, 1/3, 2/3, 1.01]
    buckets = []
    bucket_labels = np.full(len(valid), -1, dtype=int)
    pct_vals = valid["_pct"].values
    for i in range(3):
        mask = (pct_vals >= p_edges[i]) & (pct_vals < p_edges[i + 1])
        bucket_labels[mask] = i
        sub = valid[mask]
        buckets.append(_bucket_metrics(sub, labels[i], [p_edges[i], p_edges[i + 1]], feature))

    streaks = _compute_streaks(bucket_labels)
    _inject_streaks(buckets, streaks)
    return buckets


def _compute_dte_cross(df: pd.DataFrame, feature: str) -> dict:
    """Cross-tab: feature buckets (rows) x DTE values (columns)."""
    valid = df.dropna(subset=[feature, "DTE"]).sort_values("date")
    if len(valid) == 0:
        return {"dte_labels": [], "feature_labels": [], "grid": []}

    dte_values = sorted(valid["DTE"].dropna().unique().astype(int).tolist())
    dte_labels = [str(d) for d in dte_values]

    edges = np.unique(np.quantile(valid[feature].values, [0, 1/3, 2/3, 1.0])).tolist()

    vals = valid[feature].values
    feature_labels = []
    grid = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        fmt = ".2f" if any(e != int(e) for e in edges if e != float("inf")) else ".0f"
        if hi == float("inf"):
            fmask = vals >= lo
            label = f"≥ {lo:{fmt}}"
        else:
            fmask = (vals >= lo) & (vals < hi)
            label = f"{lo:{fmt}} – {hi:{fmt}}"
        feature_labels.append(label)

        row = []
        for dte_val in dte_values:
            dmask = valid["DTE"].values == dte_val
            sub = valid[fmask & dmask]
            row.append(_bucket_metrics(sub, f"{label} DTE={dte_val}", [_clean(float(lo)), _clean(float(hi))], feature))
        grid.append(row)

    return {"dte_labels": dte_labels, "feature_labels": feature_labels, "grid": grid}


def _compute_composite_dte_cross(df: pd.DataFrame, row_feature: str, col_feature: str) -> dict:
    """For each of the 9 (row×col) combos, break down by DTE."""
    valid = df.dropna(subset=[row_feature, col_feature, "DTE"]).sort_values("date")
    if len(valid) == 0:
        return {"combo_labels": [], "dte_labels": [], "grid": []}

    terciles = [0, 1/3, 2/3, 1.0]
    row_edges = np.unique(np.quantile(valid[row_feature].values, terciles))
    col_edges = np.unique(np.quantile(valid[col_feature].values, terciles))
    dte_values = sorted(valid["DTE"].dropna().unique().astype(int).tolist())
    dte_labels = [str(d) for d in dte_values]

    combo_labels = []
    grid = []  # each row = one combo, each col = one DTE

    for i in range(len(row_edges) - 1):
        rlo, rhi = row_edges[i], row_edges[i + 1]
        if i == len(row_edges) - 2:
            rmask = (valid[row_feature] >= rlo) & (valid[row_feature] <= rhi)
        else:
            rmask = (valid[row_feature] >= rlo) & (valid[row_feature] < rhi)
        r_label = f"{rlo:.2f}–{rhi:.2f}"

        for j in range(len(col_edges) - 1):
            clo, chi = col_edges[j], col_edges[j + 1]
            if j == len(col_edges) - 2:
                cmask = (valid[col_feature] >= clo) & (valid[col_feature] <= chi)
            else:
                cmask = (valid[col_feature] >= clo) & (valid[col_feature] < chi)
            c_label = f"{clo:.2f}–{chi:.2f}"

            combo_mask = rmask & cmask
            combo_label = f"{r_label} × {c_label}"
            combo_labels.append({"row_label": r_label, "col_label": c_label, "combo_label": combo_label})

            row = []
            for dte_val in dte_values:
                dmask = valid["DTE"].values == dte_val
                sub = valid[combo_mask & dmask]
                row.append(_bucket_metrics(sub, f"{combo_label} DTE={dte_val}", [float(rlo), float(rhi)]))
            grid.append(row)

    return {"combo_labels": combo_labels, "dte_labels": dte_labels, "grid": grid}


def _compute_cross(df: pd.DataFrame, row_feature: str, col_feature: str) -> dict:
    valid = df.dropna(subset=[row_feature, col_feature])
    if len(valid) == 0:
        return {"feature_labels": [], "static_labels": [], "grid": [], "pct_feature_labels": [], "pct_grid": []}

    # 3 equal-count (tercile) buckets for both axes
    terciles = [0, 1/3, 2/3, 1.0]
    row_edges = np.unique(np.quantile(valid[row_feature].values, terciles))
    col_edges = np.unique(np.quantile(valid[col_feature].values, terciles))

    def make_grid(r_edges, c_edges, feat_r, feat_c):
        f_labels = []
        s_labels = []
        for i in range(len(c_edges) - 1):
            s_labels.append(f"{c_edges[i]:.2f}–{c_edges[i+1]:.2f}")
        grid = []
        for i in range(len(r_edges) - 1):
            rlo, rhi = r_edges[i], r_edges[i + 1]
            f_labels.append(f"{rlo:.2f}–{rhi:.2f}")
            row = []
            for j in range(len(c_edges) - 1):
                clo, chi = c_edges[j], c_edges[j + 1]
                if i == len(r_edges) - 2:
                    rmask = (valid[feat_r] >= rlo) & (valid[feat_r] <= rhi)
                else:
                    rmask = (valid[feat_r] >= rlo) & (valid[feat_r] < rhi)
                if j == len(c_edges) - 2:
                    cmask = (valid[feat_c] >= clo) & (valid[feat_c] <= chi)
                else:
                    cmask = (valid[feat_c] >= clo) & (valid[feat_c] < chi)
                sub = valid[rmask & cmask]
                row.append(_bucket_metrics(sub, f"{f_labels[-1]} × {s_labels[j]}", [float(rlo), float(rhi)]))
            grid.append(row)
        return f_labels, s_labels, grid

    f_labels, s_labels, grid = make_grid(row_edges, col_edges, row_feature, col_feature)

    # Percentile grid
    valid_pct = valid.copy()
    valid_pct["_rpct"] = valid_pct[row_feature].rank(pct=True)
    pct_labels = ["P0–P33", "P33–P67", "P67–P100"]
    pct_edges = [0, 1/3, 2/3, 1.01]
    pct_grid = []
    for i in range(3):
        row = []
        rmask = (valid_pct["_rpct"] >= pct_edges[i]) & (valid_pct["_rpct"] < pct_edges[i + 1])
        for j in range(len(col_edges) - 1):
            clo, chi = col_edges[j], col_edges[j + 1]
            if j == len(col_edges) - 2:
                cmask = (valid_pct[col_feature] >= clo) & (valid_pct[col_feature] <= chi)
            else:
                cmask = (valid_pct[col_feature] >= clo) & (valid_pct[col_feature] < chi)
            sub = valid_pct[rmask & cmask]
            row.append(_bucket_metrics(sub, f"{pct_labels[i]} × {s_labels[j] if j < len(s_labels) else ''}", [pct_edges[i], pct_edges[i + 1]]))
        pct_grid.append(row)

    return {
        "feature_labels": f_labels,
        "static_labels": s_labels,
        "grid": grid,
        "pct_feature_labels": pct_labels,
        "pct_grid": pct_grid,
    }


# ── Endpoints ──

@app.get("/api/strategies")
def get_strategies():
    return [
        {"key": "dm", "name": "DM Strategy", "accent": "#ffd740", "accentRgb": "255,215,64"},
        {"key": "wc", "name": "WC Strategy", "accent": "#448aff", "accentRgb": "68,138,255"},
        {"key": "orion", "name": "Orion Strategy", "accent": "#64ffda", "accentRgb": "100,255,218"},
        {"key": "dmo", "name": "DMO Strategy", "accent": "#ff80ab", "accentRgb": "255,128,171"},
    ]


@app.get("/api/features")
def get_features():
    return [
        {"key": "RV_today", "label": "RV Today (Yang-Zhang)"},
        {"key": "IV_7d", "label": "IV 7d Forward"},
        {"key": "IV_change_1d", "label": "IV Change 1d"},
        {"key": "VRP_today", "label": "VRP (IV−RV)"},
        {"key": "IV_intraday_change", "label": "IV Intraday Change (Open−Close)"},
    ]


@app.get("/api/plain-returns/{strategy}")
def get_plain_returns(strategy: str, start_date: str | None = None, end_date: str | None = None, snapshot: str = "1530"):
    merged = _merge(strategy, start_date, end_date, snapshot)
    merged = merged.sort_values("date")
    s = _summary(merged)
    pnl = merged["Net_Daily_PnL_PerCent"].values
    cum = np.cumsum(pnl)

    daily = [
        {"date": str(r["date"]), "pnl_pct": round(float(r["Net_Daily_PnL_PerCent"]), 4), "cumulative_pct": round(float(c), 4)}
        for (_, r), c in zip(merged.iterrows(), cum)
    ]

    # Yearly
    merged["year"] = pd.to_datetime(merged["date"]).dt.year if not pd.api.types.is_datetime64_any_dtype(merged["date"]) else merged["date"].apply(lambda d: d.year)
    yearly = []
    for yr, grp in merged.groupby("year"):
        pnl_yr = grp["Net_Daily_PnL_PerCent"]
        std_yr = pnl_yr.std()
        mean_yr = pnl_yr.mean()
        sh = ((mean_yr * 252 - RISK_FREE_PCT) / (std_yr * math.sqrt(252))) if std_yr > 0 else None
        yearly.append({
            "year": int(yr),
            "total_pct": round(float(pnl_yr.sum()), 2),
            "mean_daily_pct": round(float(mean_yr), 4),
            "std_daily_pct": round(float(std_yr), 4),
            "win_rate": round(float((pnl_yr > 0).sum() / max(len(pnl_yr), 1)), 4),
            "sharpe_pct": round(float(sh), 2) if sh else None,
            "max_win_pct": round(float(pnl_yr.max()), 4),
            "max_loss_pct": round(float(pnl_yr.min()), 4),
            "trading_days": int(len(grp)),
        })

    # Monthly
    merged["month"] = pd.to_datetime(merged["date"]).dt.month if not pd.api.types.is_datetime64_any_dtype(merged["date"]) else merged["date"].apply(lambda d: d.month)
    monthly = []
    for (yr, mo), grp in merged.groupby(["year", "month"]):
        pnl_m = grp["Net_Daily_PnL_PerCent"]
        monthly.append({
            "year": int(yr), "month": int(mo),
            "total_pct": round(float(pnl_m.sum()), 2),
            "mean_daily_pct": round(float(pnl_m.mean()), 4),
            "trading_days": int(len(grp)),
            "win_rate": round(float((pnl_m > 0).sum() / max(len(pnl_m), 1)), 4),
        })

    dates = [str(d) for d in merged["date"]]
    return {
        "strategy": strategy,
        "date_range": [dates[0], dates[-1]] if dates else ["", ""],
        "summary": s,
        "yearly": yearly,
        "monthly": monthly,
        "daily_timeseries": daily,
    }


@app.get("/api/feature-buckets/{strategy}/{feature}")
def get_feature_buckets(strategy: str, feature: str, start_date: str | None = None, end_date: str | None = None, snapshot: str = "1530"):
    merged = _merge(strategy, start_date, end_date, snapshot)
    raw = _compute_buckets(merged, feature)
    pct = _compute_percentile_buckets(merged, feature)
    dte_cross = _compute_dte_cross(merged, feature)
    return {
        "strategy": strategy,
        "feature": feature,
        "raw_buckets": raw,
        "percentile_buckets": pct,
        "dte_cross": dte_cross,
    }


@app.get("/api/composite/{strategy}")
def get_composite(
    strategy: str,
    row_feature: str = Query(...),
    col_feature: str = Query(...),
    start_date: str | None = None,
    end_date: str | None = None,
    snapshot: str = "1530",
):
    merged = _merge(strategy, start_date, end_date, snapshot)
    cross = _compute_cross(merged, row_feature, col_feature)
    composite_dte = _compute_composite_dte_cross(merged, row_feature, col_feature)
    return {
        "strategy": strategy,
        "row_feature": row_feature,
        "col_feature": col_feature,
        **cross,
        "composite_dte_cross": composite_dte,
    }


@app.get("/api/rv-timeseries")
def get_rv_timeseries(start_date: str | None = None, end_date: str | None = None):
    rv = RV_DATA.copy()
    rv = _filter_dates(rv, "date", start_date, end_date)
    rv = rv.dropna(subset=["RV_today"])
    records = []
    for _, r in rv.iterrows():
        records.append({
            "date": str(r["date"]),
            "open": round(float(r["open"]), 2),
            "high": round(float(r["high"]), 2),
            "low": round(float(r["low"]), 2),
            "close": round(float(r["close"]), 2),
            "RV_today": _clean(round(float(r["RV_today"]), 6)),
            "IV_7d": _clean(round(float(r["IV_7d"]), 2)) if pd.notna(r.get("IV_7d")) else None,
            "IV_change_1d": _clean(round(float(r["IV_change_1d"]), 2)) if pd.notna(r.get("IV_change_1d")) else None,
            "VRP_today": _clean(round(float(r["VRP_today"]), 2)) if pd.notna(r.get("VRP_today")) else None,
        })
    return records


def _regime_merge(strategy: str | None, start: str | None = None, end: str | None = None,
                   snapshot: str = "1530") -> pd.DataFrame:
    """Merge RV data with regime features and optionally a strategy's returns."""
    rv = _compute_regime_features(RV_DATA, snapshot)
    rv = _add_regime_column(rv)

    if strategy:
        st = STRAT_DATA[strategy].copy()
        merged = rv.merge(st, left_on="date", right_on="Date", how="inner")
    else:
        merged = rv.copy()

    merged = _filter_dates(merged, "date", start, end)
    return merged.sort_values("date")


def _regime_merge_all_strategies(start: str | None = None, end: str | None = None,
                                  snapshot: str = "1530") -> pd.DataFrame:
    """Merge RV+regime with ALL strategies for combined portfolio analysis."""
    rv = _compute_regime_features(RV_DATA, snapshot)
    rv = _add_regime_column(rv)
    rv = _filter_dates(rv, "date", start, end).sort_values("date")

    result = rv.copy()
    for skey, sdf in STRAT_DATA.items():
        st = sdf[["Date", "Net_Daily_PnL_PerCent"]].copy()
        st.columns = ["Date", f"pnl_{skey}"]
        result = result.merge(st, left_on="date", right_on="Date", how="left")
        result.drop(columns=["Date"], inplace=True, errors="ignore")

    # Combined portfolio PnL (equal-weight of available strategies: dm, wc, orion)
    pnl_cols = [f"pnl_{s}" for s in ["dm", "wc", "orion"] if f"pnl_{s}" in result.columns]
    if pnl_cols:
        result["pnl_combined"] = result[pnl_cols].mean(axis=1)
        # All-lose: all 3 strategies negative on the same day
        result["all_lose"] = (result[pnl_cols] < 0).all(axis=1)
        # All-win: all 3 strategies positive on the same day
        result["all_win"] = (result[pnl_cols] > 0).all(axis=1)

    # Merge DTE from trading_dates
    result = result.merge(DTE_DATA, left_on="date", right_on="t_date", how="left")
    result.drop(columns=["t_date"], inplace=True, errors="ignore")

    return result


def _regime_state_metrics(df: pd.DataFrame, state: str, pnl_col: str = "pnl_combined") -> dict:
    """Compute performance metrics for a single regime state."""
    sub = df[df["regime_state"] == state]
    days = len(sub)
    total = len(df[df["regime_state"].notna()])

    if days == 0:
        return {
            "state": state, "color": REGIME_COLORS.get(state, "#666"),
            "description": REGIME_DESCRIPTIONS.get(state, ""),
            "days": 0, "pct_of_total": 0,
            "al_pct": None, "aw_pct": None,
            "port_avg": None, "sharpe": None,
            "iv_lag_mean": None, "pk_iv_mean": None, "iv_chg_5d_mean": None,
        }

    pnl = sub[pnl_col].dropna()
    al_count = sub["all_lose"].sum() if "all_lose" in sub.columns else 0
    aw_count = sub["all_win"].sum() if "all_win" in sub.columns else 0
    mean_pnl = float(pnl.mean()) if len(pnl) > 0 else 0
    std_pnl = float(pnl.std()) if len(pnl) > 1 else 0
    sharpe = ((mean_pnl * 252 - RISK_FREE_PCT) / (std_pnl * math.sqrt(252))) if std_pnl > 0 else None

    return {
        "state": state,
        "color": REGIME_COLORS.get(state, "#666"),
        "description": REGIME_DESCRIPTIONS.get(state, ""),
        "days": days,
        "pct_of_total": round(days / total * 100, 1) if total > 0 else 0,
        "al_pct": round(float(al_count / days * 100), 1) if days > 0 else None,
        "aw_pct": round(float(aw_count / days * 100), 1) if days > 0 else None,
        "port_avg": round(mean_pnl, 4),
        "sharpe": round(float(sharpe), 2) if sharpe is not None else None,
        "iv_lag_mean": round(float(sub["iv_lag"].mean()), 2) if "iv_lag" in sub.columns else None,
        "pk_iv_mean": round(float(sub["PK_IV_ratio"].mean()), 3) if "PK_IV_ratio" in sub.columns else None,
        "iv_chg_5d_mean": round(float(sub["IV_chg_5d"].mean()), 3) if "IV_chg_5d" in sub.columns else None,
    }


# ── Regime Endpoints ──

@app.get("/api/regime/states")
def get_regime_states(
    start_date: str | None = None, end_date: str | None = None,
    snapshot: str = "1530",
):
    """Return the complete 8-state regime table with portfolio metrics."""
    merged = _regime_merge_all_strategies(start_date, end_date, snapshot)

    states = []
    for state in REGIME_STATES:
        metrics = _regime_state_metrics(merged, state)
        states.append(metrics)

    # Overall metrics
    valid = merged[merged["regime_state"].notna()]
    pnl = valid["pnl_combined"].dropna()
    overall_mean = float(pnl.mean()) if len(pnl) > 0 else 0
    overall_std = float(pnl.std()) if len(pnl) > 1 else 0
    overall_sharpe = ((overall_mean * 252 - RISK_FREE_PCT) / (overall_std * math.sqrt(252))) if overall_std > 0 else None
    al_total = valid["all_lose"].sum() if "all_lose" in valid.columns else 0
    aw_total = valid["all_win"].sum() if "all_win" in valid.columns else 0

    # Current regime (latest day)
    latest = merged.dropna(subset=["regime_state"]).iloc[-1] if len(merged.dropna(subset=["regime_state"])) > 0 else None
    current = {
        "state": latest["regime_state"] if latest is not None else None,
        "date": str(latest["date"]) if latest is not None else None,
        "iv_lag": round(float(latest["iv_lag"]), 2) if latest is not None and pd.notna(latest.get("iv_lag")) else None,
        "pk_iv_ratio": round(float(latest["PK_IV_ratio"]), 3) if latest is not None and pd.notna(latest.get("PK_IV_ratio")) else None,
        "iv_chg_5d": round(float(latest["IV_chg_5d"]), 3) if latest is not None and pd.notna(latest.get("IV_chg_5d")) else None,
        "pk_5d": round(float(latest["PK_5d"]), 2) if latest is not None and pd.notna(latest.get("PK_5d")) else None,
        "iv_5d": round(float(latest["IV_5d"]), 2) if latest is not None and pd.notna(latest.get("IV_5d")) else None,
        "color": REGIME_COLORS.get(latest["regime_state"], "#666") if latest is not None else None,
        "description": REGIME_DESCRIPTIONS.get(latest["regime_state"], "") if latest is not None else None,
    }

    return {
        "states": states,
        "current": current,
        "overall": {
            "days": int(len(valid)),
            "port_avg": round(overall_mean, 4),
            "sharpe": round(float(overall_sharpe), 2) if overall_sharpe is not None else None,
            "al_pct": round(float(al_total / len(valid) * 100), 1) if len(valid) > 0 else None,
            "aw_pct": round(float(aw_total / len(valid) * 100), 1) if len(valid) > 0 else None,
        },
    }


@app.get("/api/regime/timeseries")
def get_regime_timeseries(
    start_date: str | None = None, end_date: str | None = None,
    snapshot: str = "1530",
):
    """Return daily regime state timeseries for timeline visualization."""
    merged = _regime_merge_all_strategies(start_date, end_date, snapshot)
    valid = merged.dropna(subset=["regime_state"])

    records = []
    for _, r in valid.iterrows():
        rec = {
            "date": str(r["date"]),
            "regime_state": r["regime_state"],
            "color": REGIME_COLORS.get(r["regime_state"], "#666"),
            "iv_lag": _clean(round(float(r["iv_lag"]), 2)) if pd.notna(r.get("iv_lag")) else None,
            "pk_iv_ratio": _clean(round(float(r["PK_IV_ratio"]), 3)) if pd.notna(r.get("PK_IV_ratio")) else None,
            "iv_chg_5d": _clean(round(float(r["IV_chg_5d"]), 3)) if pd.notna(r.get("IV_chg_5d")) else None,
            "pk_5d": _clean(round(float(r["PK_5d"]), 2)) if pd.notna(r.get("PK_5d")) else None,
            "iv_5d": _clean(round(float(r["IV_5d"]), 2)) if pd.notna(r.get("IV_5d")) else None,
            "pk_today": _clean(round(float(r["PK_today"]), 2)) if pd.notna(r.get("PK_today")) else None,
            "close": round(float(r["close"]), 2),
        }
        # Add per-strategy PnL
        for skey in ["dm", "wc", "orion", "dmo"]:
            col = f"pnl_{skey}"
            rec[col] = _clean(round(float(r[col]), 4)) if col in r.index and pd.notna(r.get(col)) else None
        rec["pnl_combined"] = _clean(round(float(r["pnl_combined"]), 4)) if "pnl_combined" in r.index and pd.notna(r.get("pnl_combined")) else None
        records.append(rec)

    return records


@app.get("/api/regime/strategy/{strategy}")
def get_regime_strategy(
    strategy: str,
    start_date: str | None = None, end_date: str | None = None,
    snapshot: str = "1530",
):
    """Per-strategy breakdown by regime state."""
    merged = _regime_merge(strategy, start_date, end_date, snapshot)
    merged = merged.dropna(subset=["regime_state", "Net_Daily_PnL_PerCent"])

    results = []
    for state in REGIME_STATES:
        sub = merged[merged["regime_state"] == state]
        days = len(sub)
        if days == 0:
            results.append({
                "state": state, "color": REGIME_COLORS.get(state, "#666"),
                "days": 0, "avg_pnl": None, "sharpe": None, "win_rate": None,
                "total_pct": None, "max_win": None, "max_loss": None,
            })
            continue

        pnl = sub["Net_Daily_PnL_PerCent"]
        mean = float(pnl.mean())
        std = float(pnl.std()) if len(pnl) > 1 else 0
        sharpe = ((mean * 252 - RISK_FREE_PCT) / (std * math.sqrt(252))) if std > 0 else None
        pos = (pnl > 0).sum()
        results.append({
            "state": state,
            "color": REGIME_COLORS.get(state, "#666"),
            "days": days,
            "avg_pnl": round(mean, 4),
            "sharpe": round(float(sharpe), 2) if sharpe is not None else None,
            "win_rate": round(float(pos / days), 4) if days > 0 else None,
            "total_pct": round(float(pnl.sum()), 2),
            "max_win": round(float(pnl.max()), 4),
            "max_loss": round(float(pnl.min()), 4),
        })

    return {"strategy": strategy, "states": results}


@app.get("/api/regime/transitions")
def get_regime_transitions(
    start_date: str | None = None, end_date: str | None = None,
    snapshot: str = "1530",
):
    """Regime transition matrix and streak analysis."""
    merged = _regime_merge_all_strategies(start_date, end_date, snapshot)
    valid = merged.dropna(subset=["regime_state"]).sort_values("date")
    states_series = valid["regime_state"].values

    # Transition counts
    transitions: dict[str, dict[str, int]] = {s: {s2: 0 for s2 in REGIME_STATES} for s in REGIME_STATES}
    for i in range(1, len(states_series)):
        prev, curr = states_series[i - 1], states_series[i]
        if prev in transitions and curr in transitions[prev]:
            transitions[prev][curr] += 1

    # Convert to probabilities
    transition_probs: dict[str, dict[str, float | None]] = {}
    for s in REGIME_STATES:
        row_total = sum(transitions[s].values())
        transition_probs[s] = {}
        for s2 in REGIME_STATES:
            transition_probs[s][s2] = round(transitions[s][s2] / row_total, 3) if row_total > 0 else None

    # Streak analysis per state
    streaks_by_state: dict[str, list[int]] = {s: [] for s in REGIME_STATES}
    if len(states_series) > 0:
        cur_state = states_series[0]
        cur_len = 1
        for i in range(1, len(states_series)):
            if states_series[i] == cur_state:
                cur_len += 1
            else:
                streaks_by_state[cur_state].append(cur_len)
                cur_state = states_series[i]
                cur_len = 1
        streaks_by_state[cur_state].append(cur_len)

    streak_stats = {}
    for s in REGIME_STATES:
        ss = streaks_by_state[s]
        if ss:
            streak_stats[s] = {
                "mean": round(float(np.mean(ss)), 1),
                "median": round(float(np.median(ss)), 1),
                "min": int(np.min(ss)),
                "max": int(np.max(ss)),
                "count": len(ss),
            }
        else:
            streak_stats[s] = {"mean": None, "median": None, "min": None, "max": None, "count": 0}

    # Self-transition rate (probability of staying in same state)
    self_trans = {}
    for s in REGIME_STATES:
        row_total = sum(transitions[s].values())
        self_trans[s] = round(transitions[s][s] / row_total, 3) if row_total > 0 else None

    return {
        "states": REGIME_STATES,
        "transition_counts": transitions,
        "transition_probs": transition_probs,
        "streak_stats": streak_stats,
        "self_transition_rate": self_trans,
    }


@app.get("/api/regime/feature-inputs")
def get_regime_feature_inputs(
    start_date: str | None = None, end_date: str | None = None,
    snapshot: str = "1530",
):
    """Return the raw regime input features timeseries for detailed analysis."""
    rv = _compute_regime_features(RV_DATA, snapshot)
    rv = _filter_dates(rv, "date", start_date, end_date)
    rv = rv.dropna(subset=["iv_lag"])

    records = []
    for _, r in rv.iterrows():
        records.append({
            "date": str(r["date"]),
            "iv_lag": _clean(round(float(r["iv_lag"]), 2)),
            "IV_5d": _clean(round(float(r["IV_5d"]), 2)) if pd.notna(r.get("IV_5d")) else None,
            "PK_5d": _clean(round(float(r["PK_5d"]), 2)) if pd.notna(r.get("PK_5d")) else None,
            "PK_today": _clean(round(float(r["PK_today"]), 2)) if pd.notna(r.get("PK_today")) else None,
            "IV_chg_5d": _clean(round(float(r["IV_chg_5d"]), 3)) if pd.notna(r.get("IV_chg_5d")) else None,
            "PK_IV_ratio": _clean(round(float(r["PK_IV_ratio"]), 3)) if pd.notna(r.get("PK_IV_ratio")) else None,
            "iv_level": "L1" if r["iv_lag"] < IV_L1_UPPER else ("L2" if r["iv_lag"] < IV_L2_UPPER else "L3"),
        })
    return records


@app.get("/api/regime/all-lose")
def get_regime_all_lose(
    start_date: str | None = None, end_date: str | None = None,
    snapshot: str = "1530",
):
    """Return all-lose day spot movement analysis grouped by regime state."""
    merged = _regime_merge_all_strategies(start_date, end_date, snapshot)
    valid = merged.dropna(subset=["regime_state"]).copy()

    # Compute spot % change (close-to-close)
    valid = valid.sort_values("date")
    valid["prev_close"] = valid["close"].shift(1)
    valid["spot_chg_pct"] = ((valid["close"] - valid["prev_close"]) / valid["prev_close"] * 100)
    valid["intraday_range_pct"] = ((valid["high"] - valid["low"]) / valid["open"] * 100)
    valid["gap_pct"] = ((valid["open"] - valid["prev_close"]) / valid["prev_close"] * 100)

    al_days = valid[valid.get("all_lose", pd.Series(dtype=bool)) == True].copy()

    # Per-state summary for all-lose days
    state_summaries = []
    for state in REGIME_STATES:
        sub_all = valid[valid["regime_state"] == state]
        sub_al = al_days[al_days["regime_state"] == state]
        total_days = len(sub_all)
        al_count = len(sub_al)

        if al_count == 0:
            state_summaries.append({
                "state": state,
                "color": REGIME_COLORS.get(state, "#666"),
                "total_days": total_days,
                "al_days": 0,
                "al_pct": 0,
                "spot_chg_mean": None,
                "spot_chg_median": None,
                "spot_chg_std": None,
                "spot_chg_min": None,
                "spot_chg_max": None,
                "spot_chg_p25": None,
                "spot_chg_p75": None,
                "intraday_range_mean": None,
                "gap_mean": None,
                "pnl_combined_mean": None,
                "pnl_dm_mean": None,
                "pnl_wc_mean": None,
                "pnl_orion_mean": None,
            })
            continue

        spot = sub_al["spot_chg_pct"].dropna()
        rng = sub_al["intraday_range_pct"].dropna()
        gap = sub_al["gap_pct"].dropna()

        state_summaries.append({
            "state": state,
            "color": REGIME_COLORS.get(state, "#666"),
            "total_days": total_days,
            "al_days": al_count,
            "al_pct": round(al_count / total_days * 100, 1) if total_days > 0 else 0,
            "spot_chg_mean": _clean(round(float(spot.mean()), 4)) if len(spot) > 0 else None,
            "spot_chg_median": _clean(round(float(spot.median()), 4)) if len(spot) > 0 else None,
            "spot_chg_std": _clean(round(float(spot.std()), 4)) if len(spot) > 1 else None,
            "spot_chg_min": _clean(round(float(spot.min()), 4)) if len(spot) > 0 else None,
            "spot_chg_max": _clean(round(float(spot.max()), 4)) if len(spot) > 0 else None,
            "spot_chg_p25": _clean(round(float(spot.quantile(0.25)), 4)) if len(spot) > 0 else None,
            "spot_chg_p75": _clean(round(float(spot.quantile(0.75)), 4)) if len(spot) > 0 else None,
            "intraday_range_mean": _clean(round(float(rng.mean()), 4)) if len(rng) > 0 else None,
            "gap_mean": _clean(round(float(gap.mean()), 4)) if len(gap) > 0 else None,
            "pnl_combined_mean": _clean(round(float(sub_al["pnl_combined"].dropna().mean()), 4)) if "pnl_combined" in sub_al.columns else None,
            "pnl_dm_mean": _clean(round(float(sub_al["pnl_dm"].dropna().mean()), 4)) if "pnl_dm" in sub_al.columns else None,
            "pnl_wc_mean": _clean(round(float(sub_al["pnl_wc"].dropna().mean()), 4)) if "pnl_wc" in sub_al.columns else None,
            "pnl_orion_mean": _clean(round(float(sub_al["pnl_orion"].dropna().mean()), 4)) if "pnl_orion" in sub_al.columns else None,
        })

    # Individual all-lose day records for scatter/detail view
    day_records = []
    for _, r in al_days.iterrows():
        if pd.isna(r.get("spot_chg_pct")):
            continue
        day_records.append({
            "date": str(r["date"]),
            "regime_state": r["regime_state"],
            "color": REGIME_COLORS.get(r["regime_state"], "#666"),
            "close": round(float(r["close"]), 2),
            "spot_chg_pct": _clean(round(float(r["spot_chg_pct"]), 4)),
            "intraday_range_pct": _clean(round(float(r["intraday_range_pct"]), 4)) if pd.notna(r.get("intraday_range_pct")) else None,
            "gap_pct": _clean(round(float(r["gap_pct"]), 4)) if pd.notna(r.get("gap_pct")) else None,
            "pnl_combined": _clean(round(float(r["pnl_combined"]), 4)) if pd.notna(r.get("pnl_combined")) else None,
            "pnl_dm": _clean(round(float(r["pnl_dm"]), 4)) if pd.notna(r.get("pnl_dm")) else None,
            "pnl_wc": _clean(round(float(r["pnl_wc"]), 4)) if pd.notna(r.get("pnl_wc")) else None,
            "pnl_orion": _clean(round(float(r["pnl_orion"]), 4)) if pd.notna(r.get("pnl_orion")) else None,
            "iv_lag": _clean(round(float(r["iv_lag"]), 2)) if pd.notna(r.get("iv_lag")) else None,
            "pk_iv_ratio": _clean(round(float(r["PK_IV_ratio"]), 3)) if pd.notna(r.get("PK_IV_ratio")) else None,
        })

    # Distribution buckets for histogram
    all_spot = al_days["spot_chg_pct"].dropna()
    distribution = []
    if len(all_spot) > 0:
        bins = [-999, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 999]
        labels = ["<-3%", "-3 to -2%", "-2 to -1.5%", "-1.5 to -1%", "-1 to -0.5%",
                  "-0.5 to 0%", "0 to 0.5%", "0.5 to 1%", "1 to 1.5%", "1.5 to 2%", "2 to 3%", ">3%"]
        counts, _ = np.histogram(all_spot.values, bins=bins)
        for i, label in enumerate(labels):
            distribution.append({"bucket": label, "count": int(counts[i])})

    # Overall stats
    overall = {
        "total_al_days": len(al_days),
        "total_trading_days": len(valid),
        "al_pct": round(len(al_days) / len(valid) * 100, 1) if len(valid) > 0 else 0,
        "spot_chg_mean": _clean(round(float(all_spot.mean()), 4)) if len(all_spot) > 0 else None,
        "spot_chg_median": _clean(round(float(all_spot.median()), 4)) if len(all_spot) > 0 else None,
        "spot_chg_std": _clean(round(float(all_spot.std()), 4)) if len(all_spot) > 1 else None,
        "spot_down_pct": round(float((all_spot < 0).sum() / len(all_spot) * 100), 1) if len(all_spot) > 0 else None,
        "spot_up_pct": round(float((all_spot > 0).sum() / len(all_spot) * 100), 1) if len(all_spot) > 0 else None,
    }

    return {
        "states": state_summaries,
        "days": day_records,
        "distribution": distribution,
        "overall": overall,
    }


# ── Data Exploration ──

# All explorable features: RV features + regime features
EXPLORATION_FEATURES = {
    # From _merge / RV_DATA
    "RV_today": {"label": "RV Today (Yang-Zhang)", "source": "rv"},
    "IV_7d": {"label": "IV 7d Forward", "source": "rv"},
    "IV_change_1d": {"label": "IV Change 1d", "source": "rv"},
    "VRP_today": {"label": "VRP (IV−RV)", "source": "rv"},
    "IV_intraday_change": {"label": "IV Intraday Chg (Open−Close)", "source": "rv"},
    # From _compute_regime_features
    "PK_today": {"label": "Parkinson Vol (today)", "source": "regime"},
    "iv_lag": {"label": "IV Lag (t-1)", "source": "regime"},
    "IV_5d": {"label": "IV 5d Avg", "source": "regime"},
    "PK_5d": {"label": "PK 5d Avg", "source": "regime"},
    "IV_chg_5d": {"label": "IV Change 5d Avg", "source": "regime"},
    "PK_IV_ratio": {"label": "PK/IV Ratio", "source": "regime"},
}


@app.get("/api/data-exploration/features")
def get_exploration_features():
    """Return available features for data exploration."""
    return [{"key": k, "label": v["label"], "source": v["source"]} for k, v in EXPLORATION_FEATURES.items()]


@app.get("/api/data-exploration/{feature}")
def get_data_exploration(
    feature: str,
    start_date: str | None = None, end_date: str | None = None,
    snapshot: str = "1530",
):
    """Full EDA pipeline for a single feature."""
    if feature not in EXPLORATION_FEATURES:
        return {"error": f"Unknown feature: {feature}"}

    feat_info = EXPLORATION_FEATURES[feature]

    # Build merged dataset with feature + all strategy PnLs + regime state
    rv = _compute_regime_features(RV_DATA, snapshot)
    rv = _add_regime_column(rv)

    # Ensure RV-source features are recomputed for snapshot
    if feat_info["source"] == "rv" and snapshot in VALID_SNAPSHOTS:
        iv_col = f"IV_7d_{snapshot}"
        if iv_col in rv.columns:
            rv["IV_7d"] = rv[iv_col]
            rv["IV_change_1d"] = rv["IV_7d"] - rv["IV_7d"].shift(1)
            rv["VRP_today"] = rv["IV_7d"] - rv["RV_today"]
        open_snap, close_snap = INTRADAY_PAIRS[snapshot]
        open_col = f"IV_7d_{open_snap}"
        close_col = f"IV_7d_{close_snap}"
        if open_col in rv.columns and close_col in rv.columns:
            rv["IV_intraday_change"] = rv[open_col] - rv[close_col]

    rv = _filter_dates(rv, "date", start_date, end_date).sort_values("date")

    # Merge strategy PnLs
    result = rv.copy()
    for skey, sdf in STRAT_DATA.items():
        st = sdf[["Date", "Net_Daily_PnL_PerCent"]].copy()
        st.columns = ["Date", f"pnl_{skey}"]
        result = result.merge(st, left_on="date", right_on="Date", how="left")
        result.drop(columns=["Date"], inplace=True, errors="ignore")

    pnl_cols = [f"pnl_{s}" for s in ["dm", "wc", "orion"] if f"pnl_{s}" in result.columns]
    if pnl_cols:
        result["pnl_combined"] = result[pnl_cols].mean(axis=1)

    # Merge DTE
    result = result.merge(DTE_DATA, left_on="date", right_on="t_date", how="left")
    result.drop(columns=["t_date"], inplace=True, errors="ignore")

    # Extract feature series (drop NaN)
    if feature not in result.columns:
        return {"error": f"Feature '{feature}' not found in computed data"}

    valid = result.dropna(subset=[feature]).copy()
    vals = valid[feature].astype(float)

    if len(vals) < 5:
        return {"error": f"Not enough data for feature '{feature}' ({len(vals)} points)"}

    # ═══ 1. Descriptive Statistics ═══
    desc = {
        "count": int(len(vals)),
        "mean": round(float(vals.mean()), 6),
        "median": round(float(vals.median()), 6),
        "std": round(float(vals.std()), 6),
        "min": round(float(vals.min()), 6),
        "max": round(float(vals.max()), 6),
        "skew": round(float(vals.skew()), 4),
        "kurtosis": round(float(vals.kurtosis()), 4),
        "p5": round(float(vals.quantile(0.05)), 6),
        "p10": round(float(vals.quantile(0.10)), 6),
        "p25": round(float(vals.quantile(0.25)), 6),
        "p75": round(float(vals.quantile(0.75)), 6),
        "p90": round(float(vals.quantile(0.90)), 6),
        "p95": round(float(vals.quantile(0.95)), 6),
        "iqr": round(float(vals.quantile(0.75) - vals.quantile(0.25)), 6),
    }

    # Normality test (Jarque-Bera)
    if len(vals) >= 20:
        jb_stat, jb_p = scipy_stats.jarque_bera(vals.values)
        desc["jb_stat"] = round(float(jb_stat), 4)
        desc["jb_pvalue"] = round(float(jb_p), 6)
        desc["is_normal"] = bool(float(jb_p) > 0.05)

    # ═══ 2. Histogram / Distribution ═══
    n_bins = min(40, max(15, int(len(vals) ** 0.5)))
    hist_counts, hist_edges = np.histogram(vals.values, bins=n_bins)
    histogram = []
    for i in range(len(hist_counts)):
        histogram.append({
            "bin_start": round(float(hist_edges[i]), 6),
            "bin_end": round(float(hist_edges[i + 1]), 6),
            "count": int(hist_counts[i]),
            "bin_label": f"{hist_edges[i]:.2f}",
        })

    # ═══ 3. Time Series + Rolling Stats ═══
    valid_sorted = valid.sort_values("date")
    feat_series = valid_sorted[feature].astype(float)
    roll_20_mean = feat_series.rolling(20, min_periods=10).mean()
    roll_20_std = feat_series.rolling(20, min_periods=10).std()
    roll_50_mean = feat_series.rolling(50, min_periods=25).mean()

    timeseries = []
    for idx, (_, r) in enumerate(valid_sorted.iterrows()):
        timeseries.append({
            "date": str(r["date"]),
            "value": round(float(r[feature]), 6),
            "roll_20_mean": _clean(round(float(roll_20_mean.iloc[idx]), 6)) if pd.notna(roll_20_mean.iloc[idx]) else None,
            "roll_20_std": _clean(round(float(roll_20_std.iloc[idx]), 6)) if pd.notna(roll_20_std.iloc[idx]) else None,
            "roll_50_mean": _clean(round(float(roll_50_mean.iloc[idx]), 6)) if pd.notna(roll_50_mean.iloc[idx]) else None,
        })

    # ═══ 4. Autocorrelation ═══
    max_lag = min(30, len(vals) // 3)
    try:
        acf_vals = sm_acf(vals.values, nlags=max_lag, fft=True)
        conf_bound = 1.96 / np.sqrt(len(vals))
        autocorrelation = [
            {"lag": i, "acf": round(float(acf_vals[i]), 4), "significant": bool(abs(float(acf_vals[i])) > conf_bound)}
            for i in range(1, len(acf_vals))
        ]
    except Exception:
        autocorrelation = []
        conf_bound = 0

    # ═══ 5. Stationarity (ADF Test) ═══
    try:
        adf_result = adfuller(vals.values, autolag='AIC')
        stationarity = {
            "adf_statistic": round(float(adf_result[0]), 4),
            "p_value": round(float(adf_result[1]), 6),
            "lags_used": int(adf_result[2]),
            "n_obs": int(adf_result[3]),
            "critical_1pct": round(float(adf_result[4]["1%"]), 4),
            "critical_5pct": round(float(adf_result[4]["5%"]), 4),
            "critical_10pct": round(float(adf_result[4]["10%"]), 4),
            "is_stationary_5pct": bool(float(adf_result[1]) < 0.05),
        }
    except Exception:
        stationarity = None

    # ═══ 6. Feature vs PnL (per strategy + combined) ═══
    feature_vs_pnl = {}
    strat_keys = ["dm", "wc", "orion", "dmo", "combined"]
    for skey in strat_keys:
        pcol = f"pnl_{skey}"
        if pcol not in valid.columns:
            continue
        pair = valid[[feature, pcol]].dropna()
        if len(pair) < 10:
            continue

        fv = pair[feature].astype(float)
        pv = pair[pcol].astype(float)

        corr, corr_p = scipy_stats.pearsonr(fv.values, pv.values)
        spearman, spearman_p = scipy_stats.spearmanr(fv.values, pv.values)

        # Scatter data (subsample if > 500 points)
        scatter_df = pair.copy()
        if len(scatter_df) > 500:
            scatter_df = scatter_df.sample(500, random_state=42)

        scatter = [
            {"x": round(float(row[feature]), 6), "y": round(float(row[pcol]), 6)}
            for _, row in scatter_df.iterrows()
        ]

        feature_vs_pnl[skey] = {
            "pearson_r": round(float(corr), 4),
            "pearson_p": round(float(corr_p), 6),
            "spearman_r": round(float(spearman), 4),
            "spearman_p": round(float(spearman_p), 6),
            "scatter": scatter,
        }

    # ═══ 6b. DTE-filtered Feature vs PnL correlations ═══
    dte_feature_vs_pnl = {}
    if "DTE" in valid.columns:
        dte_vals_corr = sorted(valid["DTE"].dropna().unique().astype(int).tolist())
        for dte_val in dte_vals_corr:
            dte_label = str(dte_val)
            dte_sub = valid[valid["DTE"] == dte_val]
            if len(dte_sub) < 10:
                continue
            dte_corr_dict = {}
            for skey in strat_keys:
                pcol = f"pnl_{skey}"
                if pcol not in dte_sub.columns:
                    continue
                pair = dte_sub[[feature, pcol]].dropna()
                if len(pair) < 10:
                    continue
                fv = pair[feature].astype(float)
                pv = pair[pcol].astype(float)
                try:
                    corr_val, corr_p_val = scipy_stats.pearsonr(fv.values, pv.values)
                    sp_val, sp_p_val = scipy_stats.spearmanr(fv.values, pv.values)
                except Exception:
                    continue
                dte_corr_dict[skey] = {
                    "pearson_r": round(float(corr_val), 4),
                    "pearson_p": round(float(corr_p_val), 6),
                    "spearman_r": round(float(sp_val), 4),
                    "spearman_p": round(float(sp_p_val), 6),
                    "scatter": [],  # skip scatter for DTE subsets to keep response small
                }
            if dte_corr_dict:
                dte_feature_vs_pnl[dte_label] = dte_corr_dict

    # ═══ 7. Quintile Bucket Analysis ═══
    # Compute quintile bins ONCE based on feature values only (consistent across all strategies)
    quintile_analysis = {}
    q_labels = ["Q1 (Low)", "Q2", "Q3", "Q4", "Q5 (High)"]
    try:
        valid["_quintile"] = pd.qcut(vals, 5, labels=q_labels, duplicates="drop")
    except (ValueError, TypeError):
        valid["_quintile"] = pd.NA

    # Pre-compute all_lose per quintile bucket (needed for AL% metric)
    if "all_lose" not in valid.columns:
        al_cols = [f"pnl_{s}" for s in ["dm", "wc", "orion"] if f"pnl_{s}" in valid.columns]
        if al_cols:
            valid["all_lose"] = (valid[al_cols] < 0).all(axis=1)
        else:
            valid["all_lose"] = False

    if valid["_quintile"].notna().sum() >= 25:
        for skey in strat_keys:
            pcol = f"pnl_{skey}"
            if pcol not in valid.columns:
                continue

            buckets = []
            n_with_pnl = 0
            for q in q_labels:
                sub = valid[valid["_quintile"] == q]
                pnl_sub = sub[pcol].dropna()
                # AL% for this quintile bucket
                al_pct_val = round(float(sub["all_lose"].sum()) / len(sub) * 100, 1) if len(sub) > 0 else 0.0
                if len(pnl_sub) == 0:
                    buckets.append({
                        "quintile": q,
                        "count": int(len(sub)),
                        "count_with_pnl": 0,
                        "feature_mean": round(float(sub[feature].mean()), 6),
                        "feature_range": f"{sub[feature].min():.4f} – {sub[feature].max():.4f}",
                        "pnl_mean": None, "pnl_median": None, "pnl_std": None,
                        "win_rate": None, "pnl_sum": None, "sharpe": None,
                        "al_pct": al_pct_val,
                    })
                    continue
                n_with_pnl += len(pnl_sub)
                pos = (pnl_sub > 0).sum()
                buckets.append({
                    "quintile": q,
                    "count": int(len(sub)),
                    "count_with_pnl": int(len(pnl_sub)),
                    "feature_mean": round(float(sub[feature].mean()), 6),
                    "feature_range": f"{sub[feature].min():.4f} – {sub[feature].max():.4f}",
                    "pnl_mean": round(float(pnl_sub.mean()), 4),
                    "pnl_median": round(float(pnl_sub.median()), 4),
                    "pnl_std": round(float(pnl_sub.std()), 4) if len(pnl_sub) > 1 else None,
                    "win_rate": round(float(pos / len(pnl_sub)), 4),
                    "pnl_sum": round(float(pnl_sub.sum()), 4),
                    "sharpe": round(float((pnl_sub.mean() * 252 - RISK_FREE_PCT) / (pnl_sub.std() * math.sqrt(252))), 2) if pnl_sub.std() > 0 else None,
                    "al_pct": al_pct_val,
                })
            if buckets:
                quintile_analysis[skey] = {
                    "buckets": buckets,
                    "n_total": int(len(vals)),
                    "n_used": n_with_pnl,
                    "n_dropped_missing_pnl": int(len(vals)) - n_with_pnl,
                }

        # Clean up temp column
        valid.drop(columns=["_quintile"], inplace=True, errors="ignore")

    # ═══ DTE Quintile Analysis (each DTE value separately) ═══
    dte_quintile_analysis = {}
    if "DTE" in valid.columns:
        dte_values_available = sorted(valid["DTE"].dropna().unique().astype(int).tolist())
        for dte_val in dte_values_available:
            dte_label = str(dte_val)
            dte_sub = valid[valid["DTE"] == dte_val]
            if len(dte_sub) < 20:
                continue
            dte_feat_vals = dte_sub[feature].dropna()
            if len(dte_feat_vals) < 20:
                continue
            try:
                dte_sub = dte_sub.copy()
                dte_sub["_dte_q"] = pd.qcut(dte_sub[feature], 5, labels=q_labels, duplicates="drop")
            except (ValueError, TypeError):
                continue
            if dte_sub["_dte_q"].notna().sum() < 25:
                continue
            # Ensure all_lose exists in dte_sub
            if "all_lose" not in dte_sub.columns:
                al_cols = [f"pnl_{s}" for s in ["dm", "wc", "orion"] if f"pnl_{s}" in dte_sub.columns]
                if al_cols:
                    dte_sub["all_lose"] = (dte_sub[al_cols] < 0).all(axis=1)
                else:
                    dte_sub["all_lose"] = False
            dte_bucket_analysis = {}
            for skey in strat_keys:
                pcol = f"pnl_{skey}"
                if pcol not in dte_sub.columns:
                    continue
                buckets = []
                n_with_pnl = 0
                for q in q_labels:
                    sub = dte_sub[dte_sub["_dte_q"] == q]
                    pnl_sub = sub[pcol].dropna()
                    al_pct_val = round(float(sub["all_lose"].sum()) / len(sub) * 100, 1) if len(sub) > 0 else 0.0
                    if len(pnl_sub) == 0:
                        buckets.append({
                            "quintile": q,
                            "count": int(len(sub)),
                            "count_with_pnl": 0,
                            "feature_mean": round(float(sub[feature].mean()), 6) if len(sub) > 0 else None,
                            "feature_range": f"{sub[feature].min():.4f} – {sub[feature].max():.4f}" if len(sub) > 0 else None,
                            "pnl_mean": None, "pnl_median": None, "pnl_std": None,
                            "win_rate": None, "pnl_sum": None, "sharpe": None,
                            "al_pct": al_pct_val,
                        })
                        continue
                    n_with_pnl += len(pnl_sub)
                    pos = (pnl_sub > 0).sum()
                    buckets.append({
                        "quintile": q,
                        "count": int(len(sub)),
                        "count_with_pnl": int(len(pnl_sub)),
                        "feature_mean": round(float(sub[feature].mean()), 6),
                        "feature_range": f"{sub[feature].min():.4f} – {sub[feature].max():.4f}",
                        "pnl_mean": round(float(pnl_sub.mean()), 4),
                        "pnl_median": round(float(pnl_sub.median()), 4),
                        "pnl_std": round(float(pnl_sub.std()), 4) if len(pnl_sub) > 1 else None,
                        "win_rate": round(float(pos / len(pnl_sub)), 4),
                        "pnl_sum": round(float(pnl_sub.sum()), 4),
                        "sharpe": round(float((pnl_sub.mean() * 252 - RISK_FREE_PCT) / (pnl_sub.std() * math.sqrt(252))), 2) if pnl_sub.std() > 0 else None,
                        "al_pct": al_pct_val,
                    })
                if buckets:
                    dte_bucket_analysis[skey] = {
                        "buckets": buckets,
                        "n_total": int(len(dte_feat_vals)),
                        "n_used": n_with_pnl,
                        "n_dropped_missing_pnl": int(len(dte_feat_vals)) - n_with_pnl,
                    }
            if dte_bucket_analysis:
                dte_quintile_analysis[dte_label] = dte_bucket_analysis

    # ═══ 8. Regime-Conditional Distributions ═══
    regime_distributions = []
    for state in REGIME_STATES:
        sub = valid[valid["regime_state"] == state]
        feat_sub = sub[feature].dropna()
        if len(feat_sub) < 3:
            regime_distributions.append({
                "state": state, "color": REGIME_COLORS.get(state, "#666"),
                "count": int(len(feat_sub)),
                "mean": None, "median": None, "std": None, "min": None, "max": None,
            })
            continue

        regime_distributions.append({
            "state": state,
            "color": REGIME_COLORS.get(state, "#666"),
            "count": int(len(feat_sub)),
            "mean": round(float(feat_sub.mean()), 6),
            "median": round(float(feat_sub.median()), 6),
            "std": round(float(feat_sub.std()), 6),
            "min": round(float(feat_sub.min()), 6),
            "max": round(float(feat_sub.max()), 6),
            "p25": round(float(feat_sub.quantile(0.25)), 6),
            "p75": round(float(feat_sub.quantile(0.75)), 6),
        })

    # ═══ 9. Outlier Detection (beyond 2.5σ + IQR method) ═══
    mean_v = float(vals.mean())
    std_v = float(vals.std())
    q1 = float(vals.quantile(0.25))
    q3 = float(vals.quantile(0.75))
    iqr = q3 - q1
    sigma_lower = mean_v - 2.5 * std_v
    sigma_upper = mean_v + 2.5 * std_v
    iqr_lower = q1 - 1.5 * iqr
    iqr_upper = q3 + 1.5 * iqr

    outlier_mask = (vals < sigma_lower) | (vals > sigma_upper)
    iqr_outlier_mask = (vals < iqr_lower) | (vals > iqr_upper)

    outlier_days = valid_sorted[outlier_mask | iqr_outlier_mask].head(100)
    outliers = {
        "sigma_threshold": 2.5,
        "sigma_count": int(outlier_mask.sum()),
        "iqr_count": int(iqr_outlier_mask.sum()),
        "sigma_bounds": {"lower": round(sigma_lower, 6), "upper": round(sigma_upper, 6)},
        "iqr_bounds": {"lower": round(iqr_lower, 6), "upper": round(iqr_upper, 6)},
        "days": [
            {
                "date": str(r["date"]),
                "value": round(float(r[feature]), 6),
                "regime_state": r.get("regime_state") if pd.notna(r.get("regime_state")) else None,
                "z_score": round(float((r[feature] - mean_v) / std_v), 2) if std_v > 0 else None,
            }
            for _, r in outlier_days.iterrows()
        ],
    }

    # ═══ 10. Rolling Statistics (20d, 50d) ═══
    rolling_stats = {
        "current_20d_mean": _clean(round(float(roll_20_mean.iloc[-1]), 6)) if len(roll_20_mean) > 0 and pd.notna(roll_20_mean.iloc[-1]) else None,
        "current_20d_std": _clean(round(float(roll_20_std.iloc[-1]), 6)) if len(roll_20_std) > 0 and pd.notna(roll_20_std.iloc[-1]) else None,
        "current_50d_mean": _clean(round(float(roll_50_mean.iloc[-1]), 6)) if len(roll_50_mean) > 0 and pd.notna(roll_50_mean.iloc[-1]) else None,
        "current_z_20d": _clean(round(float((vals.iloc[-1] - roll_20_mean.iloc[-1]) / roll_20_std.iloc[-1]), 2)) if (
            len(roll_20_mean) > 0 and pd.notna(roll_20_mean.iloc[-1]) and pd.notna(roll_20_std.iloc[-1]) and float(roll_20_std.iloc[-1]) > 0
        ) else None,
        "percentile_rank": round(float(scipy_stats.percentileofscore(vals.values, vals.iloc[-1])), 1),
    }

    return {
        "feature": feature,
        "label": feat_info["label"],
        "source": feat_info["source"],
        "descriptive_stats": desc,
        "histogram": histogram,
        "timeseries": timeseries,
        "autocorrelation": autocorrelation,
        "acf_confidence_bound": round(float(conf_bound), 4) if conf_bound else None,
        "stationarity": stationarity,
        "feature_vs_pnl": feature_vs_pnl,
        "dte_feature_vs_pnl": dte_feature_vs_pnl,
        "quintile_analysis": quintile_analysis,
        "dte_quintile_analysis": dte_quintile_analysis,
        "regime_distributions": regime_distributions,
        "outliers": outliers,
        "rolling_stats": rolling_stats,
    }


# ════════════════════════════════════════════════════════════════════════════
# Regime Classification Dashboard — Step 2, 3, 4 endpoints
# ════════════════════════════════════════════════════════════════════════════


# DTE values analysed individually (0 = expiry day, 5 = 5 days to expiry)


def _sharpe(pnl_series):
    """Compute annualised Sharpe from a daily PnL% series."""
    vals = pnl_series.dropna()
    if len(vals) < 2:
        return None
    m = float(vals.mean())
    s = float(vals.std())
    if s == 0:
        return None
    return (m * 252 - RISK_FREE_PCT) / (s * math.sqrt(252))


@app.get("/api/regime/feature-ranking")
def get_feature_ranking(
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    snapshot: str = Query("1530"),
):
    """Step 2: Quintile analysis of 6 candidate features, ranked by AL spread."""
    merged = _regime_merge_all_strategies(start_date, end_date, snapshot)

    # Compute efficiency
    merged["efficiency"] = (
        (merged["close"] - merged["open"]).abs()
        / (merged["high"] - merged["low"]).replace(0, np.nan)
    )

    feature_defs = [
        ("PK_today", "Parkinson Vol (PK)"),
        ("VRP_today", "VRP (IV - RV)"),
        ("IV_7d", "IV Level (7d ATM)"),
        ("IV_change_1d", "IV Daily Change"),
        ("efficiency", "|Close-Open|/Range"),
        ("RV_today", "RV (Yang-Zhang)"),
    ]

    features_out = []
    for fkey, flabel in feature_defs:
        sub = merged.dropna(subset=[fkey, "pnl_combined", "all_lose", "all_win"])
        if len(sub) < 20:
            continue

        # Quintile assignment
        try:
            sub = sub.copy()
            sub["_q"] = pd.qcut(sub[fkey], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")
        except ValueError:
            continue

        quintiles = []
        al_by_q = {}
        for q_label in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
            qs = sub[sub["_q"] == q_label]
            if len(qs) == 0:
                continue
            n = len(qs)
            al_pct = float(qs["all_lose"].sum()) / n * 100 if n > 0 else 0
            aw_pct = float(qs["all_win"].sum()) / n * 100 if n > 0 else 0
            al_by_q[q_label] = al_pct
            fmin = float(qs[fkey].min())
            fmax = float(qs[fkey].max())
            quintiles.append({
                "quintile": q_label,
                "range": f"{fmin:.2f}-{fmax:.2f}",
                "n": n,
                "dm_avg": _clean(round(float(qs["pnl_dm"].mean()), 4)) if "pnl_dm" in qs.columns else None,
                "dm_wr": _clean(round(float((qs["pnl_dm"] > 0).mean()), 2)) if "pnl_dm" in qs.columns else None,
                "wc_avg": _clean(round(float(qs["pnl_wc"].mean()), 4)) if "pnl_wc" in qs.columns else None,
                "wc_wr": _clean(round(float((qs["pnl_wc"] > 0).mean()), 2)) if "pnl_wc" in qs.columns else None,
                "orion_avg": _clean(round(float(qs["pnl_orion"].mean()), 4)) if "pnl_orion" in qs.columns else None,
                "orion_wr": _clean(round(float((qs["pnl_orion"] > 0).mean()), 2)) if "pnl_orion" in qs.columns else None,
                "combined_avg": _clean(round(float(qs["pnl_combined"].mean()), 4)),
            })
            # Compute per-strategy Sharpe (separately to avoid walrus scoping issues)
            _last = quintiles[-1]
            for _sk, _sc in [("dm", "pnl_dm"), ("wc", "pnl_wc"), ("orion", "pnl_orion"), ("combined", "pnl_combined")]:
                _sh = _sharpe(qs[_sc]) if _sc in qs.columns else None
                _last[f"{_sk}_sharpe"] = _clean(round(float(_sh), 2)) if _sh is not None else None
            _last.update({
                "al_pct": round(al_pct, 1),
                "aw_pct": round(aw_pct, 1),
            })

        # ── DTE quintiles (each DTE value separately) ──
        dte_quintiles = {}
        dte_al_spread = {}
        if "DTE" in sub.columns:
            dte_vals_available = sorted(sub["DTE"].dropna().unique().astype(int).tolist())
            for dte_val in dte_vals_available:
                dte_label = str(dte_val)
                dte_sub = sub[sub["DTE"] == dte_val]
                if len(dte_sub) < 20:
                    continue
                try:
                    dte_sub = dte_sub.copy()
                    dte_sub["_dq"] = pd.qcut(dte_sub[fkey], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")
                except ValueError:
                    continue
                dte_q_list = []
                dte_al_by_q = {}
                for q_label in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
                    qs = dte_sub[dte_sub["_dq"] == q_label]
                    if len(qs) == 0:
                        continue
                    n = len(qs)
                    al_pct = float(qs["all_lose"].sum()) / n * 100 if n > 0 else 0
                    aw_pct = float(qs["all_win"].sum()) / n * 100 if n > 0 else 0
                    dte_al_by_q[q_label] = al_pct
                    fmin = float(qs[fkey].min())
                    fmax = float(qs[fkey].max())
                    q_dict = {
                        "quintile": q_label,
                        "range": f"{fmin:.2f}-{fmax:.2f}",
                        "n": n,
                        "dm_avg": _clean(round(float(qs["pnl_dm"].mean()), 4)) if "pnl_dm" in qs.columns else None,
                        "dm_wr": _clean(round(float((qs["pnl_dm"] > 0).mean()), 2)) if "pnl_dm" in qs.columns else None,
                        "wc_avg": _clean(round(float(qs["pnl_wc"].mean()), 4)) if "pnl_wc" in qs.columns else None,
                        "wc_wr": _clean(round(float((qs["pnl_wc"] > 0).mean()), 2)) if "pnl_wc" in qs.columns else None,
                        "orion_avg": _clean(round(float(qs["pnl_orion"].mean()), 4)) if "pnl_orion" in qs.columns else None,
                        "orion_wr": _clean(round(float((qs["pnl_orion"] > 0).mean()), 2)) if "pnl_orion" in qs.columns else None,
                        "combined_avg": _clean(round(float(qs["pnl_combined"].mean()), 4)),
                    }
                    for _sk, _sc in [("dm", "pnl_dm"), ("wc", "pnl_wc"), ("orion", "pnl_orion"), ("combined", "pnl_combined")]:
                        _sh = _sharpe(qs[_sc]) if _sc in qs.columns else None
                        q_dict[f"{_sk}_sharpe"] = _clean(round(float(_sh), 2)) if _sh is not None else None
                    q_dict.update({
                        "al_pct": round(al_pct, 1),
                        "aw_pct": round(aw_pct, 1),
                    })
                    dte_q_list.append(q_dict)
                if dte_q_list:
                    dte_quintiles[dte_label] = dte_q_list
                    d_q1_al = dte_al_by_q.get("Q1", 0)
                    d_q5_al = dte_al_by_q.get("Q5", 0)
                    dte_al_spread[dte_label] = round(d_q5_al - d_q1_al, 1)

        # AL spread: Q5 AL% minus Q1 AL% (portfolio-level discrimination)
        q1_al = al_by_q.get("Q1", 0)
        q5_al = al_by_q.get("Q5", 0)
        al_spread = q5_al - q1_al
        best_q = min(al_by_q, key=al_by_q.get) if al_by_q else "Q1"
        worst_q = max(al_by_q, key=al_by_q.get) if al_by_q else "Q1"

        # Strategy correlations (must dropna per pair to avoid NaN from pearsonr)
        dm_corr = wc_corr = orion_corr = al_corr = 0.0
        for col_name, attr in [("pnl_dm", "dm_corr"), ("pnl_wc", "wc_corr"), ("pnl_orion", "orion_corr")]:
            try:
                pair = sub[[fkey, col_name]].dropna()
                if len(pair) >= 10:
                    r = float(scipy_stats.pearsonr(pair[fkey], pair[col_name])[0])
                    if not math.isnan(r):
                        if attr == "dm_corr": dm_corr = r
                        elif attr == "wc_corr": wc_corr = r
                        elif attr == "orion_corr": orion_corr = r
            except Exception:
                pass
        try:
            pair = sub[[fkey, "all_lose"]].dropna()
            pair["_al"] = pair["all_lose"].astype(float)
            if len(pair) >= 10:
                r = float(scipy_stats.pearsonr(pair[fkey], pair["_al"])[0])
                if not math.isnan(r):
                    al_corr = r
        except Exception:
            pass

        # Correlation gap: |corr_DM - corr_Orion|
        dm_orion_gap = abs(dm_corr - orion_corr)

        # Verdict logic
        if fkey == "IV_7d":
            verdict, reason = "KEEP", "regime backbone"
        elif fkey == "PK_today":
            verdict, reason = "KEEP", "DM/Orion tilt signal"
        elif fkey == "VRP_today":
            verdict, reason = "KEEP", "strategy tilt signal"
        elif fkey == "IV_change_1d":
            verdict, reason = "KEEP", "L2 direction signal"
        elif fkey == "efficiency":
            try:
                eff_ac = float(sm_acf(sub[fkey].dropna().values, nlags=1, fft=True)[1])
            except Exception:
                eff_ac = 0.0
            verdict, reason = "DROP", f"Unpredictable (AC = {eff_ac:.2f})"
        elif fkey == "RV_today":
            try:
                rv_pk_pair = sub[["RV_today", "PK_today"]].dropna()
                rv_pk_corr = float(scipy_stats.pearsonr(rv_pk_pair["RV_today"], rv_pk_pair["PK_today"])[0])
            except Exception:
                rv_pk_corr = 0.0
            if math.isnan(rv_pk_corr): rv_pk_corr = 0.0
            verdict, reason = "DROP", f"Redundant with PK (r={rv_pk_corr:.2f})"
        else:
            verdict, reason = "KEEP", ""

        features_out.append({
            "feature_key": fkey,
            "label": flabel,
            "dm_corr": _clean(round(dm_corr, 4)),
            "wc_corr": _clean(round(wc_corr, 4)),
            "orion_corr": _clean(round(orion_corr, 4)),
            "al_corr": _clean(round(al_corr, 4)),
            "al_spread": round(al_spread, 1),
            "best_q": f"{best_q}: {al_by_q.get(best_q, 0):.0f}%",
            "worst_q": f"{worst_q}: {al_by_q.get(worst_q, 0):.0f}%",
            "dm_orion_gap": _clean(round(dm_orion_gap, 4)),
            "verdict": verdict,
            "verdict_reason": reason,
            "quintiles": quintiles,
            "dte_quintiles": dte_quintiles,
            "dte_al_spread": dte_al_spread,
        })

    # Sort by abs(al_spread) descending
    features_out.sort(key=lambda x: abs(x["al_spread"]), reverse=True)
    for i, f in enumerate(features_out):
        f["rank"] = i + 1

    # Portfolio stats
    psub = merged.dropna(subset=["pnl_combined"])
    combined_sharpe = _sharpe(psub["pnl_combined"])
    n_days = len(psub)
    al_pct = float(psub["all_lose"].sum()) / n_days * 100 if n_days > 0 else 0
    aw_pct = float(psub["all_win"].sum()) / n_days * 100 if n_days > 0 else 0

    dm_orion_corr = dm_wc_corr = wc_orion_corr = 0.0
    try:
        pair = psub[["pnl_dm", "pnl_orion"]].dropna()
        dm_orion_corr = float(pair["pnl_dm"].corr(pair["pnl_orion"]))
    except Exception:
        pass
    try:
        pair = psub[["pnl_dm", "pnl_wc"]].dropna()
        dm_wc_corr = float(pair["pnl_dm"].corr(pair["pnl_wc"]))
    except Exception:
        pass
    try:
        pair = psub[["pnl_wc", "pnl_orion"]].dropna()
        wc_orion_corr = float(pair["pnl_wc"].corr(pair["pnl_orion"]))
    except Exception:
        pass

    portfolio_stats = {
        "days": n_days,
        "combined_sharpe": _clean(round(float(combined_sharpe), 2)) if combined_sharpe is not None else None,
        "al_pct": round(al_pct, 1),
        "aw_pct": round(aw_pct, 1),
        "dm_orion_corr": _clean(round(dm_orion_corr, 2)),
        "dm_wc_corr": _clean(round(dm_wc_corr, 2)),
        "wc_orion_corr": _clean(round(wc_orion_corr, 2)),
    }

    # Two-jobs framework
    # Find the features for each job
    job1_features = [f["feature_key"] for f in features_out if f["feature_key"] in ("IV_7d", "efficiency")]
    job2_features = [f["feature_key"] for f in features_out if f["feature_key"] in ("PK_today", "VRP_today")]

    # Build descriptions from computed data
    iv_feat = next((f for f in features_out if f["feature_key"] == "IV_7d"), None)
    eff_feat = next((f for f in features_out if f["feature_key"] == "efficiency"), None)
    pk_feat = next((f for f in features_out if f["feature_key"] == "PK_today"), None)
    vrp_feat = next((f for f in features_out if f["feature_key"] == "VRP_today"), None)

    job1_desc_parts = []
    if iv_feat:
        job1_desc_parts.append(f"IV level ({iv_feat['al_spread']:+.0f}pp)")
    if eff_feat:
        job1_desc_parts.append(f"Efficiency ({eff_feat['al_spread']:+.0f}pp)")
    job1_desc = ", ".join(job1_desc_parts) + ". Predict whether ALL 3 lose together."

    job2_desc_parts = []
    if pk_feat:
        dm_c = pk_feat['dm_corr'] or 0
        or_c = pk_feat['orion_corr'] or 0
        job2_desc_parts.append(f"PK (DM {dm_c:+.2f}, Orion {or_c:+.2f})")
    if vrp_feat:
        dm_c = vrp_feat['dm_corr'] or 0
        or_c = vrp_feat['orion_corr'] or 0
        job2_desc_parts.append(f"VRP (DM {dm_c:+.2f}, Orion {or_c:+.2f})")
    job2_desc = ", ".join(job2_desc_parts) + ". Combined portfolio positive in ALL quintiles."

    two_jobs = {
        "job1": {
            "title": "Should I trade today?",
            "subtitle": "Portfolio risk / sizing",
            "description": job1_desc,
            "features": job1_features,
        },
        "job2": {
            "title": "Which strategy to overweight?",
            "subtitle": "Strategy tilt",
            "description": job2_desc,
            "features": job2_features,
        },
    }

    # Discarded features
    discarded = [
        {"feature": f["feature_key"], "label": f["label"], "reason": f["verdict_reason"]}
        for f in features_out if f["verdict"] == "DROP"
    ]

    return {
        "features": features_out,
        "portfolio_stats": portfolio_stats,
        "two_jobs": two_jobs,
        "discarded": discarded,
    }


@app.get("/api/regime/feature-selection")
def get_feature_selection(
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    snapshot: str = Query("1530"),
):
    """Step 3: Autocorrelation analysis — raw vs averaged features."""
    merged = _regime_merge_all_strategies(start_date, end_date, snapshot)

    # Compute efficiency
    merged["efficiency"] = (
        (merged["close"] - merged["open"]).abs()
        / (merged["high"] - merged["low"]).replace(0, np.nan)
    )

    # Also need raw IV change for the direction persistence table
    iv_col = f"IV_7d_{snapshot}" if f"IV_7d_{snapshot}" in merged.columns else "IV_7d"
    if iv_col in merged.columns and iv_col != "IV_7d":
        merged["_iv_raw"] = merged[iv_col]
    else:
        merged["_iv_raw"] = merged["IV_7d"]
    merged["_iv_change_raw"] = merged["_iv_raw"] - merged["_iv_raw"].shift(1)

    feature_defs = [
        ("IV_7d", "IV level"),
        ("IV_change_1d", "IV change"),
        ("PK_today", "Parkinson vol"),
        ("VRP_today", "VRP"),
        ("efficiency", "Efficiency"),
        ("RV_today", "RV (Yang-Zhang)"),
    ]

    windows = [3, 5, 7, 10]

    ac_table = []
    for fkey, flabel in feature_defs:
        series = merged[fkey].dropna()
        if len(series) < 10:
            continue

        # Raw lag-1 AC
        try:
            raw_ac = float(sm_acf(series.values, nlags=1, fft=True)[1])
        except Exception:
            raw_ac = None

        row = {
            "feature_key": fkey,
            "label": flabel,
            "raw_1d": _clean(round(raw_ac, 2)) if raw_ac is not None else None,
        }

        # For IV_7d, don't compute averaged versions (already persistent)
        if fkey == "IV_7d":
            for w in windows:
                row[f"avg_{w}d"] = None
            row["verdict"] = "Strongest raw — backbone"
        else:
            for w in windows:
                rolled = merged[fkey].rolling(w, min_periods=max(2, w - 1)).mean().dropna()
                if len(rolled) < 10:
                    row[f"avg_{w}d"] = None
                    continue
                try:
                    ac_val = float(sm_acf(rolled.values, nlags=1, fft=True)[1])
                    row[f"avg_{w}d"] = _clean(round(ac_val, 2))
                except Exception:
                    row[f"avg_{w}d"] = None

            # Assign verdicts
            if fkey == "IV_change_1d":
                row["verdict"] = "Mean-reverts. 5d+ recovers."
            elif fkey == "PK_today":
                row["verdict"] = "5d avg BEST persistence"
            elif fkey == "VRP_today":
                row["verdict"] = "5d avg strong persistence"
            elif fkey == "efficiency":
                row["verdict"] = "Unpredictable at all windows"
            elif fkey == "RV_today":
                row["verdict"] = "Redundant with PK"
            else:
                row["verdict"] = ""

        ac_table.append(row)

    # IV Direction Persistence by Level
    iv_direction_rows = []
    window_configs = [
        ("1d raw", None),
        ("3d avg", 3),
        ("5d avg", 5),
        ("7d avg", 7),
        ("10d avg", 10),
    ]

    for wlabel, wsize in window_configs:
        if wsize is None:
            iv_change_series = merged["_iv_change_raw"]
        else:
            iv_change_series = merged["_iv_change_raw"].rolling(wsize, min_periods=max(2, wsize - 1)).mean()

        merged["_iv_chg_work"] = iv_change_series

        level_acs = {}
        level_filters = {
            "l1": merged["iv_lag"] < IV_L1_UPPER,
            "l2": (merged["iv_lag"] >= IV_L1_UPPER) & (merged["iv_lag"] < IV_L2_UPPER),
            "l3": merged["iv_lag"] >= IV_L2_UPPER,
            "all": pd.Series(True, index=merged.index),
        }

        for lkey, lmask in level_filters.items():
            subset_vals = merged.loc[lmask, "_iv_chg_work"].dropna()
            if len(subset_vals) < 10:
                level_acs[lkey] = None
                continue
            try:
                ac_val = float(sm_acf(subset_vals.values, nlags=1, fft=True)[1])
                level_acs[lkey] = _clean(round(ac_val, 2))
            except Exception:
                level_acs[lkey] = None

        # Notes based on window
        if wsize is None:
            note = "Near zero everywhere"
        elif wsize == 3:
            note = "Only L3 persists"
        elif wsize == 5:
            note = "L2 usable, L1 still weak"
        elif wsize == 7:
            note = "Moderate everywhere"
        elif wsize == 10:
            note = "Best but still weak at L1"
        else:
            note = ""

        iv_direction_rows.append({
            "window": wlabel,
            "l1": level_acs.get("l1"),
            "l2": level_acs.get("l2"),
            "l3": level_acs.get("l3"),
            "all": level_acs.get("all"),
            "note": note,
        })

    merged.drop(columns=["_iv_chg_work", "_iv_raw", "_iv_change_raw"], inplace=True, errors="ignore")

    # Selected features (final picks from the analysis)
    selected_features = [
        {"feature": "iv_lag", "formula": "IV_7d[t-1]", "role": "Level classification (L1/L2/L3)"},
        {"feature": "IV_5d", "formula": "mean(IV_7d, t-5 to t-1)", "role": "PK/IV denominator"},
        {"feature": "PK_5d", "formula": "mean(PK, t-5 to t-1)", "role": "PK/IV numerator"},
        {"feature": "IV_chg_5d", "formula": "mean(IV daily change, t-5 to t-1)", "role": "L2 direction only"},
    ]

    # Key finding — compute actual values
    pk_row = next((r for r in ac_table if r["feature_key"] == "PK_today"), None)
    iv_row = next((r for r in ac_table if r["feature_key"] == "IV_7d"), None)
    pk_5d_ac = pk_row["avg_5d"] if pk_row else "N/A"
    iv_raw_ac = iv_row["raw_1d"] if iv_row else "N/A"
    key_finding = f"5-day averaging makes PK and VRP MORE persistent than IV level itself. PK_5d AC = {pk_5d_ac} vs IV level AC = {iv_raw_ac}."

    return {
        "ac_table": ac_table,
        "iv_direction_by_level": iv_direction_rows,
        "selected_features": selected_features,
        "key_finding": key_finding,
    }


@app.get("/api/regime/regime-construction")
def get_regime_construction(
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    snapshot: str = Query("1530"),
):
    """Step 4: Build the 8-state regime classification from scratch."""
    merged = _regime_merge_all_strategies(start_date, end_date, snapshot)
    merged = merged.dropna(subset=["pnl_combined", "iv_lag"])

    # Compute per-level PK/IV medians from this filtered data for equal day splits
    pkiv_l1, pkiv_l2, pkiv_l3 = _compute_pkiv_medians(merged)
    # Re-classify regime states using these data-specific medians
    merged["regime_state"] = merged.apply(
        lambda r: _classify_regime_with_thresholds(r, pkiv_l1, pkiv_l2, pkiv_l3), axis=1
    )

    # ── Part 1: IV Boundary Configs ──
    boundary_configs_input = [
        ([12], "2 levels: 12"),
        ([15], "2 levels: 15"),
        ([12, 17], "3 levels: 12/17"),
        ([13, 17], "3 levels: 13/17"),
        ([10, 13, 17], "4 levels: 10/13/17"),
    ]
    selected_boundaries = [12, 17]

    boundary_configs = []
    for bounds, label in boundary_configs_input:
        is_selected = bool(bounds == selected_boundaries)
        # Classify into levels
        edges = [-np.inf] + bounds + [np.inf]
        levels_out = []
        all_al_pcts = []
        level_days_map = {}
        total = len(merged)

        for i in range(len(edges) - 1):
            low_e = edges[i]
            high_e = edges[i + 1]
            lname = f"L{i + 1}"
            if low_e == -np.inf:
                mask = merged["iv_lag"] < high_e
                rule = f"IV < {high_e}"
            elif high_e == np.inf:
                mask = merged["iv_lag"] >= low_e
                rule = f"IV >= {low_e}"
            else:
                mask = (merged["iv_lag"] >= low_e) & (merged["iv_lag"] < high_e)
                rule = f"IV {low_e}-{high_e}"

            lsub = merged[mask]
            ldays = len(lsub)
            lpct = ldays / total * 100 if total > 0 else 0
            lal = float(lsub["all_lose"].sum()) / ldays * 100 if ldays > 0 else 0
            lsharpe = _sharpe(lsub["pnl_combined"])
            all_al_pcts.append(lal)
            level_days_map[lname] = lsub

            levels_out.append({
                "name": lname,
                "rule": rule,
                "days": ldays,
                "pct": round(lpct, 1),
                "al_pct": round(lal, 1),
                "sharpe": _clean(round(float(lsharpe), 2)) if lsharpe is not None else None,
            })

        spread = max(all_al_pcts) - min(all_al_pcts) if all_al_pcts else 0

        # Average streak length and self-transition rate
        level_labels = []
        for _, row in merged.iterrows():
            iv = row["iv_lag"]
            if pd.isna(iv):
                level_labels.append(None)
                continue
            assigned = None
            for i in range(len(edges) - 1):
                if edges[i] == -np.inf:
                    if iv < edges[i + 1]:
                        assigned = f"L{i + 1}"
                        break
                elif edges[i + 1] == np.inf:
                    if iv >= edges[i]:
                        assigned = f"L{i + 1}"
                        break
                else:
                    if edges[i] <= iv < edges[i + 1]:
                        assigned = f"L{i + 1}"
                        break
            level_labels.append(assigned)

        level_ser = pd.Series(level_labels)
        # Self-transition rate
        same_next = (level_ser == level_ser.shift(1))
        valid_pairs = level_ser.notna() & level_ser.shift(1).notna()
        self_trans = float(same_next[valid_pairs].sum()) / float(valid_pairs.sum()) * 100 if valid_pairs.sum() > 0 else 0

        # Average streak
        streaks = []
        current_streak = 1
        for i in range(1, len(level_ser)):
            if level_ser.iloc[i] == level_ser.iloc[i - 1] and level_ser.iloc[i] is not None:
                current_streak += 1
            else:
                if level_ser.iloc[i - 1] is not None:
                    streaks.append(current_streak)
                current_streak = 1
        if level_ser.iloc[-1] is not None:
            streaks.append(current_streak)
        avg_streak = float(np.mean(streaks)) if streaks else 0

        boundary_configs.append({
            "label": label,
            "boundaries": bounds,
            "is_selected": is_selected,
            "levels": levels_out,
            "spread": round(spread, 1),
            "avg_streak": round(avg_streak, 1),
            "self_trans_pct": round(self_trans, 1),
        })

    # ── Part 2: Per-Level Analysis ──
    l1_mask = merged["iv_lag"] < IV_L1_UPPER
    l2_mask = (merged["iv_lag"] >= IV_L1_UPPER) & (merged["iv_lag"] < IV_L2_UPPER)
    l3_mask = merged["iv_lag"] >= IV_L2_UPPER

    def _pk_iv_quintile_analysis(subset, threshold):
        """Compute PK/IV quintiles and 2-state split for L1/L3."""
        sub = subset.dropna(subset=["PK_IV_ratio"]).copy()
        quintiles = []
        try:
            sub["_pkq"] = pd.qcut(sub["PK_IV_ratio"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")
        except ValueError:
            sub["_pkq"] = "Q1"

        for ql in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
            qs = sub[sub["_pkq"] == ql]
            if len(qs) == 0:
                continue
            n = len(qs)
            al_pct = float(qs["all_lose"].sum()) / n * 100 if n > 0 else 0
            sh = _sharpe(qs["pnl_combined"])
            quintiles.append({
                "quintile": ql,
                "pk_iv_range": f"{float(qs['PK_IV_ratio'].min()):.2f}-{float(qs['PK_IV_ratio'].max()):.2f}",
                "days": n,
                "al_pct": round(al_pct, 1),
                "sharpe": _clean(round(float(sh), 2)) if sh is not None else None,
            })

        # 2-state split
        safe_mask = sub["PK_IV_ratio"] <= threshold
        final_states = []
        for state_label, smask in [("Safe", safe_mask), ("Exposed", ~safe_mask)]:
            ss = sub[smask]
            n = len(ss)
            total_sub = len(sub)
            if n == 0:
                continue
            al_pct = float(ss["all_lose"].sum()) / n * 100
            aw_pct = float(ss["all_win"].sum()) / n * 100
            port_avg = float(ss["pnl_combined"].mean())
            sh = _sharpe(ss["pnl_combined"])
            final_states.append({
                "state": state_label,
                "rule": f"PK/IV <= {threshold}" if state_label == "Safe" else f"PK/IV > {threshold}",
                "days": n,
                "pct": round(n / total_sub * 100, 1) if total_sub > 0 else 0,
                "al_pct": round(al_pct, 1),
                "aw_pct": round(aw_pct, 1),
                "port_avg": _clean(round(port_avg, 4)),
                "sharpe": _clean(round(float(sh), 2)) if sh is not None else None,
            })

        # Threshold AC
        th_ac = None
        try:
            binary = (sub["PK_IV_ratio"] > threshold).astype(float)
            if len(binary.dropna()) >= 10:
                th_ac = float(sm_acf(binary.values, nlags=1, fft=True)[1])
        except Exception:
            pass

        return quintiles, final_states, _clean(round(th_ac, 2)) if th_ac is not None else None

    # L1 analysis
    l1_data = merged[l1_mask]
    l1_quintiles, l1_final, l1_th_ac = _pk_iv_quintile_analysis(l1_data, pkiv_l1)
    l1_final_named = []
    for fs in l1_final:
        fs_copy = dict(fs)
        fs_copy["state"] = f"L1 {fs['state']}"
        l1_final_named.append(fs_copy)

    # L1 strategy profiles per state
    l1_strat_profiles = []
    for fs in l1_final_named:
        sname = fs["state"]
        if "Safe" in sname:
            ss = l1_data[l1_data["PK_IV_ratio"] <= pkiv_l1]
        else:
            ss = l1_data[l1_data["PK_IV_ratio"] > pkiv_l1]
        sp = {"state": sname}
        for skey in ["dm", "wc", "orion"]:
            col = f"pnl_{skey}"
            if col in ss.columns and ss[col].notna().sum() > 0:
                sp[f"{skey}_avg"] = _clean(round(float(ss[col].mean()), 4))
                shr = _sharpe(ss[col])
                sp[f"{skey}_sharpe"] = _clean(round(float(shr), 2)) if shr is not None else None
            else:
                sp[f"{skey}_avg"] = None
                sp[f"{skey}_sharpe"] = None
        l1_strat_profiles.append(sp)

    per_level_l1 = {
        "description": "IV < 12: PK/IV Ratio is the Signal",
        "pk_iv_quintiles": l1_quintiles,
        "final_states": l1_final_named,
        "threshold": pkiv_l1,
        "threshold_ac": l1_th_ac,
        "strategy_profiles": l1_strat_profiles,
    }

    # L2 analysis — 2x2 matrix
    l2_data = merged[l2_mask].dropna(subset=["PK_IV_ratio", "IV_chg_5d"]).copy()
    pk_iv_hi = l2_data["PK_IV_ratio"] > pkiv_l2
    iv_rising = l2_data["IV_chg_5d"] > 0

    l2_state_defs = [
        ("L2 Safe", ~pk_iv_hi & ~iv_rising, f"<={pkiv_l2}", "Fall"),
        ("L2 Caution-A", pk_iv_hi & ~iv_rising, f">{pkiv_l2}", "Fall"),
        ("L2 Caution-B", ~pk_iv_hi & iv_rising, f"<={pkiv_l2}", "Rise"),
        ("L2 Risky", pk_iv_hi & iv_rising, f">{pkiv_l2}", "Rise"),
    ]

    l2_final = []
    l2_strat_profiles = []
    total_l2 = len(l2_data)
    for sname, smask, pk_label, iv_dir in l2_state_defs:
        ss = l2_data[smask]
        n = len(ss)
        if n == 0:
            continue
        al_pct = float(ss["all_lose"].sum()) / n * 100
        aw_pct = float(ss["all_win"].sum()) / n * 100
        port_avg = float(ss["pnl_combined"].mean())
        sh = _sharpe(ss["pnl_combined"])

        l2_final.append({
            "state": sname,
            "pk_iv": pk_label,
            "iv_dir": iv_dir,
            "days": n,
            "pct": round(n / total_l2 * 100, 1) if total_l2 > 0 else 0,
            "al_pct": round(al_pct, 1),
            "aw_pct": round(aw_pct, 1),
            "port_avg": _clean(round(port_avg, 4)),
            "sharpe": _clean(round(float(sh), 2)) if sh is not None else None,
        })

        # Strategy profiles
        sp = {"state": sname}
        for skey in ["dm", "wc", "orion"]:
            col = f"pnl_{skey}"
            if col in ss.columns:
                sp[f"{skey}_avg"] = _clean(round(float(ss[col].mean()), 4))
                shr = _sharpe(ss[col])
                sp[f"{skey}_sharpe"] = _clean(round(float(shr), 2)) if shr is not None else None
            else:
                sp[f"{skey}_avg"] = None
                sp[f"{skey}_sharpe"] = None
        l2_strat_profiles.append(sp)

    per_level_l2 = {
        "description": "IV 12-17: IV Direction + PK/IV Ratio",
        "final_states": l2_final,
        "strategy_profiles": l2_strat_profiles,
        "pk_iv_threshold": pkiv_l2,
    }

    # L3 analysis
    l3_data = merged[l3_mask]
    l3_quintiles, l3_final, l3_th_ac = _pk_iv_quintile_analysis(l3_data, pkiv_l3)
    l3_final_named = []
    for fs in l3_final:
        fs_copy = dict(fs)
        fs_copy["state"] = f"L3 {fs['state']}"
        l3_final_named.append(fs_copy)

    # L3 strategy profiles per state
    l3_strat_profiles = []
    for fs in l3_final_named:
        sname = fs["state"]
        if "Safe" in sname:
            ss = l3_data[l3_data["PK_IV_ratio"] <= pkiv_l3]
        else:
            ss = l3_data[l3_data["PK_IV_ratio"] > pkiv_l3]
        sp = {"state": sname}
        for skey in ["dm", "wc", "orion"]:
            col = f"pnl_{skey}"
            if col in ss.columns and ss[col].notna().sum() > 0:
                sp[f"{skey}_avg"] = _clean(round(float(ss[col].mean()), 4))
                shr = _sharpe(ss[col])
                sp[f"{skey}_sharpe"] = _clean(round(float(shr), 2)) if shr is not None else None
            else:
                sp[f"{skey}_avg"] = None
                sp[f"{skey}_sharpe"] = None
        l3_strat_profiles.append(sp)

    per_level_l3 = {
        "description": "IV > 17: PK/IV Ratio — Higher Stakes",
        "pk_iv_quintiles": l3_quintiles,
        "final_states": l3_final_named,
        "threshold": pkiv_l3,
        "strategy_profiles": l3_strat_profiles,
    }

    per_level = {"L1": per_level_l1, "L2": per_level_l2, "L3": per_level_l3}

    # ── Part 3: Complete Regime Table ──
    complete_table = []
    regime_rules = {
        "L1 Safe": f"IV<{IV_L1_UPPER}, PK/IV<={pkiv_l1}",
        "L1 Exposed": f"IV<{IV_L1_UPPER}, PK/IV>{pkiv_l1}",
        "L2 Safe": f"IV {IV_L1_UPPER}-{IV_L2_UPPER}, PK/IV<={pkiv_l2}, IV falling",
        "L2 Caution-A": f"IV {IV_L1_UPPER}-{IV_L2_UPPER}, PK/IV>{pkiv_l2}, IV falling",
        "L2 Caution-B": f"IV {IV_L1_UPPER}-{IV_L2_UPPER}, PK/IV<={pkiv_l2}, IV rising",
        "L2 Risky": f"IV {IV_L1_UPPER}-{IV_L2_UPPER}, PK/IV>{pkiv_l2}, IV rising",
        "L3 Safe": f"IV>={IV_L2_UPPER}, PK/IV<={pkiv_l3}",
        "L3 Exposed": f"IV>={IV_L2_UPPER}, PK/IV>{pkiv_l3}",
    }

    total_days = len(merged[merged["regime_state"].notna()])
    for state in REGIME_STATES:
        sub = merged[merged["regime_state"] == state]
        n = len(sub)
        if n == 0:
            row_out = {
                "state": state,
                "color": REGIME_COLORS.get(state, "#666"),
                "rule": regime_rules.get(state, ""),
                "days": 0, "pct": 0, "al_pct": None, "aw_pct": None,
                "port_avg": None, "sharpe": None,
            }
            for skey in ["dm", "wc", "orion"]:
                row_out[f"{skey}_avg"] = None
                row_out[f"{skey}_sharpe"] = None
            complete_table.append(row_out)
            continue

        al_pct = float(sub["all_lose"].sum()) / n * 100
        aw_pct = float(sub["all_win"].sum()) / n * 100
        port_avg = float(sub["pnl_combined"].mean())
        sh = _sharpe(sub["pnl_combined"])

        row_out = {
            "state": state,
            "color": REGIME_COLORS.get(state, "#666"),
            "rule": regime_rules.get(state, ""),
            "days": n,
            "pct": round(n / total_days * 100, 1) if total_days > 0 else 0,
            "al_pct": round(al_pct, 1),
            "aw_pct": round(aw_pct, 1),
            "port_avg": _clean(round(port_avg, 4)),
            "sharpe": _clean(round(float(sh), 2)) if sh is not None else None,
        }
        for skey in ["dm", "wc", "orion"]:
            col = f"pnl_{skey}"
            if col in sub.columns:
                row_out[f"{skey}_avg"] = _clean(round(float(sub[col].mean()), 4))
                strat_sh = _sharpe(sub[col])
                row_out[f"{skey}_sharpe"] = _clean(round(float(strat_sh), 2)) if strat_sh is not None else None
            else:
                row_out[f"{skey}_avg"] = None
                row_out[f"{skey}_sharpe"] = None

        complete_table.append(row_out)

    # Overall row
    total_n = len(merged.dropna(subset=["pnl_combined"]))
    overall_pnl = merged["pnl_combined"].dropna()
    overall_al = float(merged["all_lose"].sum()) / total_n * 100 if total_n > 0 else 0
    overall_aw = float(merged["all_win"].sum()) / total_n * 100 if total_n > 0 else 0
    overall_avg = float(overall_pnl.mean()) if len(overall_pnl) > 0 else 0
    overall_sh = _sharpe(overall_pnl)

    overall = {
        "days": total_n,
        "al_pct": round(overall_al, 1),
        "aw_pct": round(overall_aw, 1),
        "port_avg": _clean(round(overall_avg, 4)),
        "sharpe": _clean(round(float(overall_sh), 2)) if overall_sh is not None else None,
    }

    # ── DTE Breakdown per Regime State ──
    dte_breakdown = {}
    if "DTE" in merged.columns:
        for state in REGIME_STATES:
            state_sub = merged[merged["regime_state"] == state]
            if len(state_sub) == 0:
                continue
            state_dte_rows = []
            dte_vals_avail = sorted(state_sub["DTE"].dropna().unique().astype(int).tolist())
            for dte_val in dte_vals_avail:
                dte_label = str(dte_val)
                dsub = state_sub[state_sub["DTE"] == dte_val]
                dn = len(dsub)
                if dn == 0:
                    continue
                d_al_pct = float(dsub["all_lose"].sum()) / dn * 100
                d_aw_pct = float(dsub["all_win"].sum()) / dn * 100
                d_port_avg = float(dsub["pnl_combined"].mean()) if "pnl_combined" in dsub.columns else 0
                d_sh = _sharpe(dsub["pnl_combined"]) if "pnl_combined" in dsub.columns else None
                dte_row = {
                    "dte": int(dte_val),
                    "days": dn,
                    "al_pct": round(d_al_pct, 1),
                    "aw_pct": round(d_aw_pct, 1),
                    "port_avg": _clean(round(d_port_avg, 4)),
                    "sharpe": _clean(round(float(d_sh), 2)) if d_sh is not None else None,
                }
                for skey in ["dm", "wc", "orion"]:
                    col = f"pnl_{skey}"
                    if col in dsub.columns and dsub[col].notna().sum() > 0:
                        dte_row[f"{skey}_avg"] = _clean(round(float(dsub[col].mean()), 4))
                        s_sh = _sharpe(dsub[col])
                        dte_row[f"{skey}_sharpe"] = _clean(round(float(s_sh), 2)) if s_sh is not None else None
                    else:
                        dte_row[f"{skey}_avg"] = None
                        dte_row[f"{skey}_sharpe"] = None
                state_dte_rows.append(dte_row)
            if state_dte_rows:
                dte_breakdown[state] = state_dte_rows

    # ── VRP per regime state ──
    vrp_by_state = {}
    if "VRP_today" in merged.columns:
        for state in REGIME_STATES:
            sub = merged[merged["regime_state"] == state]
            vrp_vals = sub["VRP_today"].dropna()
            if len(vrp_vals) > 0:
                vrp_by_state[state] = _clean(round(float(vrp_vals.mean()), 2))
            else:
                vrp_by_state[state] = None

    # ── Tested and failed per level (computed, not hardcoded) ──
    def _test_alternative_splitter(level_data, feature_col, label):
        """Try splitting by median of feature_col, report AL spread."""
        if feature_col not in level_data.columns:
            return None
        clean = level_data.dropna(subset=[feature_col, "pnl_combined"])
        if len(clean) < 20:
            return None
        median_val = clean[feature_col].median()
        lo = clean[clean[feature_col] <= median_val]
        hi = clean[clean[feature_col] > median_val]
        if len(lo) < 5 or len(hi) < 5:
            return None
        al_lo = float(lo["all_lose"].sum()) / len(lo) * 100
        al_hi = float(hi["all_lose"].sum()) / len(hi) * 100
        spread = round(al_hi - al_lo, 1)
        # Autocorrelation
        try:
            binary = (clean[feature_col] > median_val).astype(float)
            ac_val = float(sm_acf(binary.values, nlags=1, fft=True)[1])
            ac_str = f"AC={ac_val:.2f}"
        except Exception:
            ac_str = "AC=n/a"
        return {"approach": label, "result": f"AL spread {spread:+.1f}pp, {ac_str}. {'Weak' if abs(spread) < 4 else 'Decent'} discrimination."}

    l1_tested = []
    for col, lbl in [("IV_chg_5d", "IV direction (avg_chg_5d)"), ("VRP_today", "VRP alone")]:
        r = _test_alternative_splitter(l1_data, col, lbl)
        if r:
            l1_tested.append(r)

    # Additional L1 test: 1d IV change (raw, not averaged)
    if "IV_change_1d" in l1_data.columns:
        r_1d = _test_alternative_splitter(l1_data, "IV_change_1d", "1d IV change (raw)")
        if r_1d:
            r_1d["result"] += " Doesn't persist day-to-day. Combined with PK/IV flips at 2-3d window."
            l1_tested.append(r_1d)

    # Additional L1 test: IV mean reversion (direction flip from previous day)
    l1_clean_mr = l1_data.dropna(subset=["IV_chg_5d", "pnl_combined"]).copy()
    if len(l1_clean_mr) >= 20:
        l1_clean_mr["_iv_reverting"] = l1_clean_mr["IV_chg_5d"].shift(1) * l1_clean_mr["IV_chg_5d"] < 0
        rev = l1_clean_mr[l1_clean_mr["_iv_reverting"] == True]
        cont = l1_clean_mr[l1_clean_mr["_iv_reverting"] == False]
        if len(rev) >= 5 and len(cont) >= 5:
            al_rev = float(rev["all_lose"].sum()) / len(rev) * 100
            al_cont = float(cont["all_lose"].sum()) / len(cont) * 100
            mr_spread = round(al_cont - al_rev, 1)
            l1_tested.append({
                "approach": "IV mean reversion",
                "result": f"Reverting AL={al_rev:.1f}%, continuing AL={al_cont:.1f}%, spread {mr_spread:+.1f}pp. Exists but doesn't separate risk cleanly."
            })

    per_level_l1["tested_and_failed"] = l1_tested

    l2_tested = []
    # L2: test PK/IV alone without IV direction
    l2_clean = l2_data.dropna(subset=["PK_IV_ratio", "pnl_combined"])
    if len(l2_clean) >= 20:
        pk_med = l2_clean["PK_IV_ratio"].median()
        lo = l2_clean[l2_clean["PK_IV_ratio"] <= pk_med]
        hi = l2_clean[l2_clean["PK_IV_ratio"] > pk_med]
        if len(lo) >= 5 and len(hi) >= 5:
            al_lo = float(lo["all_lose"].sum()) / len(lo) * 100
            al_hi = float(hi["all_lose"].sum()) / len(hi) * 100
            l2_pk_only = round(al_hi - al_lo, 1)
            # Compare with 4-state gap
            l2_with_dir = l2_data.dropna(subset=["PK_IV_ratio", "IV_chg_5d", "pnl_combined"])
            if len(l2_with_dir) >= 20:
                pk_med_4 = l2_with_dir["PK_IV_ratio"].median()
                safe_4 = l2_with_dir[(l2_with_dir["PK_IV_ratio"] <= pk_med_4) & (l2_with_dir["IV_chg_5d"] <= 0)]
                risky_4 = l2_with_dir[(l2_with_dir["PK_IV_ratio"] > pk_med_4) & (l2_with_dir["IV_chg_5d"] > 0)]
                if len(safe_4) >= 5 and len(risky_4) >= 5:
                    al_safe_4 = float(safe_4["all_lose"].sum()) / len(safe_4) * 100
                    al_risky_4 = float(risky_4["all_lose"].sum()) / len(risky_4) * 100
                    l2_full_gap = round(al_risky_4 - al_safe_4, 1)
                    l2_tested.append({"approach": "PK/IV alone (L1 rules on L2)",
                                      "result": f"AL spread {l2_pk_only:+.1f}pp vs {l2_full_gap:+.1f}pp with IV direction. Missing {round(l2_full_gap - l2_pk_only, 1)}pp of separation."})
                else:
                    l2_tested.append({"approach": "PK/IV alone (no IV direction)", "result": f"AL spread {l2_pk_only:+.1f}pp — decent but misses IV direction signal."})
            else:
                l2_tested.append({"approach": "PK/IV alone (no IV direction)", "result": f"AL spread {l2_pk_only:+.1f}pp — decent but misses IV direction signal."})

    # Cross-level: L2 rules (4-state) applied to L1
    l1_cross = l1_data.dropna(subset=["PK_IV_ratio", "IV_chg_5d", "pnl_combined"])
    if len(l1_cross) >= 20:
        l1_pk_med = l1_cross["PK_IV_ratio"].median()
        # L1 with 2 states (current)
        l1_safe_2 = l1_cross[l1_cross["PK_IV_ratio"] <= l1_pk_med]
        al_l1_safe_2 = float(l1_safe_2["all_lose"].sum()) / len(l1_safe_2) * 100 if len(l1_safe_2) > 0 else 0
        # L1 with 4 states (L2 rules)
        l1_safe_4 = l1_cross[(l1_cross["PK_IV_ratio"] <= l1_pk_med) & (l1_cross["IV_chg_5d"] <= 0)]
        if len(l1_safe_4) >= 5:
            al_l1_safe_4 = float(l1_safe_4["all_lose"].sum()) / len(l1_safe_4) * 100
            direction = "worse" if al_l1_safe_4 > al_l1_safe_2 else "better"
            l2_tested.append({
                "approach": "L2 rules on L1 (adding IV direction to L1)",
                "result": f"L1 Safe goes from {al_l1_safe_2:.1f}% to {al_l1_safe_4:.1f}% AL — makes it {direction}. IV direction is noise at low IV."
            })

    per_level_l2["tested_and_failed"] = l2_tested

    l3_tested = []
    for col, lbl in [("IV_chg_5d", "IV direction at L3")]:
        r = _test_alternative_splitter(l3_data, col, lbl)
        if r:
            # Note: at L3, IV is almost always rising — direction can't split
            if col == "IV_chg_5d" and col in l3_data.columns:
                rising_pct = (l3_data[col].dropna() > 0).mean() * 100
                r["result"] += f" But {rising_pct:.0f}% of L3 days have rising IV — direction can't split."
            l3_tested.append(r)
    per_level_l3["tested_and_failed"] = l3_tested

    return {
        "boundary_configs": boundary_configs,
        "per_level": per_level,
        "complete_table": complete_table,
        "overall": overall,
        "dte_breakdown": dte_breakdown,
        "vrp_by_state": vrp_by_state,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5501)
