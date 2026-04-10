"""
prepare_rv_meta.py — Parameterized data prep + evaluation harness for meta-harness.

This is a modified version of prepare_rv.py that reads scoring weights,
normalization constants, train/val periods, and feature pipeline config
from meta_config.json (if present). When no meta_config.json exists,
it behaves identically to the original prepare_rv.py.

The inner loop (autoresearch) imports from prepare_rv.py, which delegates
to this module when meta_config.json is detected.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure RV project root is on sys.path for pipeline imports
_RV_ROOT = Path(__file__).resolve().parent.parent
if str(_RV_ROOT) not in sys.path:
    sys.path.insert(0, str(_RV_ROOT))

from meta_config import MetaConfig, load_active_config
from pipeline.nsqa_data import fetch_rv_daily

# ── Fixed Constants ──
BASE_DIR = Path(__file__).parent.parent
DATA_STRATEGIES = BASE_DIR.parent / "DATA" / "Strategies"
STRATEGY_FILES = {
    "dm": DATA_STRATEGIES / "DM" / "DM_merged.xlsx",
    "wc": DATA_STRATEGIES / "WC" / "WC_merged.xlsx",
    "orion": DATA_STRATEGIES / "Orion" / "Orion_merged.xlsx",
}

# Load active meta config (or None for defaults)
_CFG = load_active_config()

# Periods — parameterized by meta config
if _CFG:
    TRAIN_PERIOD = (_CFG.train_start, _CFG.train_end)
    VAL_PERIOD = (_CFG.val_start, _CFG.val_end)
else:
    TRAIN_PERIOD = ("2023-02-01", "2025-06-30")
    VAL_PERIOD = ("2025-07-01", "2026-01-30")

# OOS periods are NEVER modified
OOS_PERIOD_1 = ("2021-01-01", "2023-01-31")
OOS_PERIOD_2 = ("2026-02-01", "2026-03-23")

RISK_FREE_PCT = 5.5
ANNUALIZATION = 252

REGIME_STATES_8 = [
    "L1 Safe", "L1 Exposed",
    "L2 Safe", "L2 Caution-A", "L2 Caution-B", "L2 Risky",
    "L3 Safe", "L3 Exposed",
]


# ── Extra Feature Registry ──
# Each entry: feature_name -> function(rv_df) that adds the column in-place

def _feat_iv_20d(rv):
    rv["IV_20d"] = rv["_iv"].shift(1).rolling(20, min_periods=10).mean()

def _feat_pk_20d(rv):
    rv["PK_20d"] = rv["PK_today"].shift(1).rolling(20, min_periods=10).mean()

def _feat_iv_momentum_5d(rv):
    # Rate of change of IV over 5 days (%)
    iv_lag = rv["_iv"].shift(1)
    iv_lag_5 = rv["_iv"].shift(6)
    rv["IV_momentum_5d"] = np.where(iv_lag_5 > 0, (iv_lag - iv_lag_5) / iv_lag_5 * 100, np.nan)

def _feat_vrp_5d(rv):
    if "VRP_today" in rv.columns:
        rv["VRP_5d"] = rv["VRP_today"].rolling(5, min_periods=3).mean()

def _feat_iv_range_10d(rv):
    iv_shifted = rv["_iv"].shift(1)
    rv["IV_range_10d"] = iv_shifted.rolling(10, min_periods=5).max() - iv_shifted.rolling(10, min_periods=5).min()

def _feat_rv_iv_gap(rv):
    if "RV_today" in rv.columns:
        rv["RV_IV_gap"] = rv["RV_today"] - rv["_iv"].shift(1)

def _feat_pk_iv_zscore_60d(rv):
    pk_iv = rv.get("PK_IV_ratio")
    if pk_iv is not None:
        rv["PK_IV_zscore_60d"] = pk_iv.rolling(60, min_periods=20).apply(
            lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0, raw=False
        )

def _feat_iv_vol_of_vol(rv):
    iv_chg = rv["_iv"] - rv["_iv"].shift(1)
    rv["IV_vol_of_vol_20d"] = iv_chg.shift(1).rolling(20, min_periods=10).std()


FEATURE_REGISTRY = {
    "IV_20d": _feat_iv_20d,
    "PK_20d": _feat_pk_20d,
    "IV_momentum_5d": _feat_iv_momentum_5d,
    "VRP_5d": _feat_vrp_5d,
    "IV_range_10d": _feat_iv_range_10d,
    "RV_IV_gap": _feat_rv_iv_gap,
    "PK_IV_zscore_60d": _feat_pk_iv_zscore_60d,
    "IV_vol_of_vol_20d": _feat_iv_vol_of_vol,
}


def _compute_parkinson_vol(high: pd.Series, low: pd.Series) -> pd.Series:
    """Parkinson volatility from High/Low (annualized, in %)."""
    log_hl = np.log(high / low)
    return np.sqrt(log_hl ** 2 / (4 * np.log(2))) * np.sqrt(252) * 100


def load_data(snapshot: str = "1530") -> pd.DataFrame:
    """Load and merge all data via NSQA. Returns DataFrame with features + strategy PnLs."""
    rv = fetch_rv_daily()
    rv["date"] = pd.to_datetime(rv["timestamp"]).dt.date
    rv.sort_values("date", inplace=True)
    rv.reset_index(drop=True, inplace=True)

    # Compute Parkinson Vol
    rv["PK_today"] = _compute_parkinson_vol(rv["high"], rv["low"])

    # Snapshot-specific IV
    iv_col = f"IV_7d_{snapshot}"
    if iv_col in rv.columns:
        if snapshot in ("0915", "0916"):
            rv["_iv"] = rv[iv_col].shift(-1)
        else:
            rv["_iv"] = rv[iv_col]
    else:
        rv["_iv"] = rv["IV_7d"]

    # Derived features (baseline — same as original prepare_rv.py)
    rv["_iv_change"] = rv["_iv"] - rv["_iv"].shift(1)
    rv["IV_5d"] = rv["_iv"].shift(1).rolling(5, min_periods=3).mean()
    rv["PK_5d"] = rv["PK_today"].shift(1).rolling(5, min_periods=3).mean()
    rv["IV_chg_5d"] = rv["_iv_change"].shift(1).rolling(5, min_periods=3).mean()
    rv["iv_lag"] = rv["_iv"].shift(1)
    rv["PK_IV_ratio"] = np.where(rv["IV_5d"] > 0, rv["PK_5d"] / rv["IV_5d"], np.nan)

    rv["IV_10d"] = rv["_iv"].shift(1).rolling(10, min_periods=5).mean()
    rv["PK_10d"] = rv["PK_today"].shift(1).rolling(10, min_periods=5).mean()
    rv["PK_IV_10d"] = np.where(rv["IV_10d"] > 0, rv["PK_10d"] / rv["IV_10d"], np.nan)
    rv["IV_chg_1d"] = rv["_iv_change"].shift(1)
    rv["IV_percentile_60d"] = rv["_iv"].shift(1).rolling(60, min_periods=20).apply(
        lambda x: (x.iloc[-1] <= x).mean() * 100 if len(x) > 0 else np.nan, raw=False
    )
    rv["PK_IV_zscore_30d"] = rv["PK_IV_ratio"].rolling(30, min_periods=10).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0, raw=False
    )

    # ── Extra features from meta config ──
    if _CFG and _CFG.extra_features:
        for feat_name in _CFG.extra_features:
            if feat_name in FEATURE_REGISTRY:
                FEATURE_REGISTRY[feat_name](rv)

    rv.drop(columns=["_iv", "_iv_change"], inplace=True, errors="ignore")

    # Merge strategy PnLs
    for skey, fpath in STRATEGY_FILES.items():
        if fpath.exists():
            sdf = pd.read_excel(fpath, sheet_name="returns")
            sdf["Date"] = pd.to_datetime(sdf["Date"]).dt.date
            sdf = sdf[["Date", "Net_Daily_PnL_PerCent"]].rename(
                columns={"Date": "date", "Net_Daily_PnL_PerCent": f"pnl_{skey}"}
            )
            rv = rv.merge(sdf, on="date", how="left")

    # Portfolio
    pnl_cols = [f"pnl_{s}" for s in ["dm", "wc", "orion"] if f"pnl_{s}" in rv.columns]
    if pnl_cols:
        rv["pnl_combined"] = rv[pnl_cols].mean(axis=1)
        _pnl = rv[pnl_cols]
        _has_all = _pnl.notna().all(axis=1)
        rv["all_lose"] = _has_all & (_pnl < 0).all(axis=1)
        rv["all_win"] = _has_all & (_pnl > 0).all(axis=1)

    return rv


def _sharpe(pnl: pd.Series) -> float | None:
    """Compute annualized Sharpe from daily % returns."""
    pnl = pnl.dropna()
    if len(pnl) < 10:
        return None
    m = float(pnl.mean())
    s = float(pnl.std())
    if s == 0:
        return None
    return round((m * ANNUALIZATION - RISK_FREE_PCT) / (s * np.sqrt(ANNUALIZATION)), 4)


def _al_pct(df: pd.DataFrame) -> float | None:
    """All-lose percentage."""
    if "all_lose" not in df.columns or len(df) == 0:
        return None
    return round(float(df["all_lose"].sum() / len(df) * 100), 2)


def evaluate(df: pd.DataFrame, regime_col: str = "regime_state") -> dict:
    """
    Evaluate a regime classification — parameterized by MetaConfig.

    When meta_config.json is present, uses its weights and normalization.
    Otherwise behaves identically to the original prepare_rv.evaluate().
    """
    from datetime import date as dt_date
    from scipy.stats import spearmanr

    # Use meta config weights or defaults
    if _CFG:
        w_sharpe = _CFG.w_sharpe
        w_safe_sep = _CFG.w_safe_sep
        w_rank = _CFG.w_rank_corr
        w_cov = _CFG.w_coverage
        s_norm = _CFG.sharpe_norm
        s_cap = _CFG.sharpe_cap
        ss_norm = _CFG.safe_sep_norm
        ss_cap = _CFG.safe_sep_cap
        min_sd = _CFG.min_state_days
        min_su = _CFG.min_states_used
    else:
        w_sharpe, w_safe_sep, w_rank, w_cov = 0.40, 0.25, 0.20, 0.15
        s_norm, s_cap = 5.0, 1.5
        ss_norm, ss_cap = 10.0, 1.0
        min_sd, min_su = 5, 6

    train_start, train_end = [dt_date(*map(int, d.split("-"))) for d in TRAIN_PERIOD]
    val_start, val_end = [dt_date(*map(int, d.split("-"))) for d in VAL_PERIOD]

    clean = df.dropna(subset=[regime_col, "pnl_combined"]).copy()
    train = clean[(clean["date"] >= train_start) & (clean["date"] <= train_end)]
    val = clean[(clean["date"] >= val_start) & (clean["date"] <= val_end)]

    if len(val) < 20:
        return {"composite_score": -999, "error": "Too few validation days"}

    # 1. Validation Sharpe
    val_sharpe = _sharpe(val["pnl_combined"]) or 0

    # 2. Safe vs Exposed separation
    def _level_gap(subset, prefix, exposed_name=None):
        safe_name = f"{prefix} Safe"
        exp_name = exposed_name or f"{prefix} Exposed"
        safe = subset[subset[regime_col] == safe_name]
        exposed = subset[subset[regime_col] == exp_name]
        safe_al = _al_pct(safe)
        exp_al = _al_pct(exposed)
        if safe_al is not None and exp_al is not None:
            return exp_al - safe_al
        return 0

    gaps = []
    for prefix in ["L1", "L3"]:
        g = _level_gap(val, prefix)
        gaps.append(g)
    g_l2 = _level_gap(val, "L2", exposed_name="L2 Risky")
    gaps.append(g_l2)
    safe_separation = np.mean(gaps) if gaps else 0

    # 3. Rank stability
    train_ranks = []
    val_ranks = []
    for s in REGIME_STATES_8:
        t_sub = train[train[regime_col] == s]["pnl_combined"]
        v_sub = val[val[regime_col] == s]["pnl_combined"]
        train_ranks.append(float(t_sub.mean()) if len(t_sub) > 0 else 0)
        val_ranks.append(float(v_sub.mean()) if len(v_sub) > 0 else 0)

    try:
        rank_corr = float(spearmanr(train_ranks, val_ranks).correlation)
        if np.isnan(rank_corr):
            rank_corr = 0
    except Exception:
        rank_corr = 0

    # 4. State coverage penalty
    states_in_val = val[regime_col].value_counts()
    n_states_used = len(states_in_val)
    min_state_days_val = int(states_in_val.min()) if len(states_in_val) > 0 else 0
    coverage = min(1.0, min_state_days_val / float(min_sd)) * min(1.0, n_states_used / float(min_su))

    # Composite score (parameterized)
    composite = (
        w_sharpe * min(val_sharpe / s_norm, s_cap) +
        w_safe_sep * min(safe_separation / ss_norm, ss_cap) +
        w_rank * max(rank_corr, 0) +
        w_cov * coverage
    )

    # Per-state breakdown
    state_metrics = {}
    for s in REGIME_STATES_8:
        v_sub = val[val[regime_col] == s]
        state_metrics[s] = {
            "days": len(v_sub),
            "sharpe": _sharpe(v_sub["pnl_combined"]),
            "al_pct": _al_pct(v_sub),
            "port_avg": round(float(v_sub["pnl_combined"].mean()), 4) if len(v_sub) > 0 else None,
        }

    return {
        "composite_score": round(composite, 6),
        "val_sharpe": round(val_sharpe, 4),
        "safe_separation": round(safe_separation, 2),
        "rank_stability": round(rank_corr, 4),
        "state_coverage": round(coverage, 4),
        "val_days": len(val),
        "train_days": len(train),
        "n_states_used": n_states_used,
        "min_state_days": min_state_days_val,
        "state_metrics": state_metrics,
    }


def evaluate_oos(df: pd.DataFrame, regime_col: str = "regime_state", period: str = "oos1") -> dict:
    """Evaluate on OOS periods (for meta-harness evaluation only, NEVER for inner loop optimization)."""
    from datetime import date as dt_date

    periods = {
        "oos1": OOS_PERIOD_1,
        "oos2": OOS_PERIOD_2,
    }
    start_str, end_str = periods.get(period, OOS_PERIOD_1)
    start = dt_date(*map(int, start_str.split("-")))
    end = dt_date(*map(int, end_str.split("-")))

    clean = df.dropna(subset=[regime_col, "pnl_combined"]).copy()
    oos = clean[(clean["date"] >= start) & (clean["date"] <= end)]

    if len(oos) < 10:
        return {"error": "Too few OOS days", "days": len(oos)}

    sharpe = _sharpe(oos["pnl_combined"]) or 0
    al = _al_pct(oos)

    # Per-state breakdown on OOS
    state_metrics = {}
    for s in REGIME_STATES_8:
        s_sub = oos[oos[regime_col] == s]
        state_metrics[s] = {
            "days": len(s_sub),
            "sharpe": _sharpe(s_sub["pnl_combined"]),
            "al_pct": _al_pct(s_sub),
        }

    return {
        "period": period,
        "days": len(oos),
        "sharpe": round(sharpe, 4),
        "al_pct": al,
        "state_metrics": state_metrics,
    }


if __name__ == "__main__":
    cfg = _CFG
    if cfg:
        print(f"Meta config active: {cfg.summary()}")
    else:
        print("No meta config — using defaults")

    print("Loading data...")
    df = load_data("1530")
    print(f"Loaded {len(df)} rows")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Train period: {TRAIN_PERIOD}")
    print(f"Val period: {VAL_PERIOD}")
    print(f"Columns: {len(df.columns)}")

    # Show extra features if any
    if cfg and cfg.extra_features:
        print(f"\nExtra features computed: {cfg.extra_features}")
        for f in cfg.extra_features:
            if f in df.columns:
                vals = df[f].dropna()
                print(f"  {f:25s}: {len(vals):4d} vals, mean={vals.mean():.3f}, std={vals.std():.3f}")
