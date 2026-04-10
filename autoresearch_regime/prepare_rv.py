"""
prepare_rv.py — Data prep + evaluation harness for regime research.

When meta_config.json is present in this directory, delegates to
prepare_rv_meta.py for parameterized scoring/features/periods.
Otherwise uses original hardcoded defaults (backward compatible).
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

from pipeline.nsqa_data import fetch_rv_daily

# ── Meta-harness delegation ──
# If meta_config.json exists, re-export everything from prepare_rv_meta
# and skip the original hardcoded implementation entirely.
_META_CONFIG_FILE = Path(__file__).parent / "meta_config.json"
_META_ACTIVE = _META_CONFIG_FILE.exists()

if _META_ACTIVE:
    from prepare_rv_meta import (  # noqa: F401
        load_data, evaluate, evaluate_oos,
        REGIME_STATES_8, TRAIN_PERIOD, VAL_PERIOD,
        OOS_PERIOD_1, OOS_PERIOD_2,
        RISK_FREE_PCT, ANNUALIZATION,
    )

# ═══════════════════════════════════════════════════════════════
# ORIGINAL IMPLEMENTATION (only used when no meta_config.json)
# ═══════════════════════════════════════════════════════════════

if not _META_ACTIVE:

    # ── Fixed Constants ──
    BASE_DIR = Path(__file__).parent.parent
    DATA_STRATEGIES = BASE_DIR.parent / "DATA" / "Strategies"
    STRATEGY_FILES = {
        "dm": DATA_STRATEGIES / "DM" / "DM_merged.xlsx",
        "wc": DATA_STRATEGIES / "WC" / "WC_merged.xlsx",
        "orion": DATA_STRATEGIES / "Orion" / "Orion_merged.xlsx",
    }

    # Evaluation periods (fixed, no peeking)
    TRAIN_PERIOD = ("2023-02-01", "2025-06-30")  # ~600 days
    VAL_PERIOD = ("2025-07-01", "2026-01-30")    # ~150 days (held out)
    OOS_PERIOD_1 = ("2021-01-01", "2023-01-31")  # 515 days (early, high-IV)
    OOS_PERIOD_2 = ("2026-02-01", "2026-03-23")  # recent (Feb-Mar 2026)

    RISK_FREE_PCT = 5.5
    ANNUALIZATION = 252

    REGIME_STATES_8 = [
        "L1 Safe", "L1 Exposed",
        "L2 Safe", "L2 Caution-A", "L2 Caution-B", "L2 Risky",
        "L3 Safe", "L3 Exposed",
    ]

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

        rv["PK_today"] = _compute_parkinson_vol(rv["high"], rv["low"])

        iv_col = f"IV_7d_{snapshot}"
        if iv_col in rv.columns:
            if snapshot in ("0915", "0916"):
                rv["_iv"] = rv[iv_col].shift(-1)
            else:
                rv["_iv"] = rv[iv_col]
        else:
            rv["_iv"] = rv["IV_7d"]

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

        rv.drop(columns=["_iv", "_iv_change"], inplace=True, errors="ignore")

        for skey, fpath in STRATEGY_FILES.items():
            if fpath.exists():
                sdf = pd.read_excel(fpath, sheet_name="returns")
                sdf["Date"] = pd.to_datetime(sdf["Date"]).dt.date
                sdf = sdf[["Date", "Net_Daily_PnL_PerCent"]].rename(
                    columns={"Date": "date", "Net_Daily_PnL_PerCent": f"pnl_{skey}"}
                )
                rv = rv.merge(sdf, on="date", how="left")

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
        Evaluate a regime classification. This is the GROUND TRUTH scoring function.
        """
        from datetime import date as dt_date
        from scipy.stats import spearmanr

        train_start, train_end = [dt_date(*map(int, d.split("-"))) for d in TRAIN_PERIOD]
        val_start, val_end = [dt_date(*map(int, d.split("-"))) for d in VAL_PERIOD]

        clean = df.dropna(subset=[regime_col, "pnl_combined"]).copy()
        train = clean[(clean["date"] >= train_start) & (clean["date"] <= train_end)]
        val = clean[(clean["date"] >= val_start) & (clean["date"] <= val_end)]

        if len(val) < 20:
            return {"composite_score": -999, "error": "Too few validation days"}

        val_sharpe = _sharpe(val["pnl_combined"]) or 0

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

        states_in_val = val[regime_col].value_counts()
        n_states_used = len(states_in_val)
        min_state_days = int(states_in_val.min()) if len(states_in_val) > 0 else 0
        coverage = min(1.0, min_state_days / 5.0) * min(1.0, n_states_used / 6.0)

        composite = (
            0.40 * min(val_sharpe / 5.0, 1.5) +
            0.25 * min(safe_separation / 10.0, 1.0) +
            0.20 * max(rank_corr, 0) +
            0.15 * coverage
        )

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
            "min_state_days": min_state_days,
            "state_metrics": state_metrics,
        }

    def evaluate_oos(df: pd.DataFrame, regime_col: str = "regime_state", period: str = "oos1") -> dict:
        """Evaluate on OOS periods (for final reporting only, NOT for optimization)."""
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

        return {
            "period": period,
            "days": len(oos),
            "sharpe": round(sharpe, 4),
            "al_pct": al,
        }


if __name__ == "__main__":
    print(f"Meta-harness active: {_META_ACTIVE}")
    print("Loading data...")
    df = load_data("1530")
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"PnL available: {df['pnl_combined'].notna().sum()} days")
    print(f"\nAvailable features for regime classification:")
    feature_cols = ["iv_lag", "PK_IV_ratio", "IV_chg_5d", "IV_5d", "PK_5d",
                    "IV_10d", "PK_10d", "PK_IV_10d", "IV_chg_1d",
                    "IV_percentile_60d", "PK_IV_zscore_30d", "RV_today", "VRP_today"]
    for f in feature_cols:
        if f in df.columns:
            vals = df[f].dropna()
            print(f"  {f:25s}: {len(vals):4d} vals, mean={vals.mean():.3f}, std={vals.std():.3f}")
    print("\nReady for experiments.")
