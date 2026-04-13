"""
diagnostics.py — Read-only analysis of the current regime framework.

Produces insights that inform config fixes, prompt improvements, and
experiment prioritization. Does NOT modify any files or state.

Usage:
    python diagnostics.py                    # Run all diagnostics
    python diagnostics.py --only regime      # Run one diagnostic
    python diagnostics.py --save             # Save to diagnostics_report.json
"""

import sys
import json
import numpy as np
import pandas as pd
from datetime import date as dt_date
from pathlib import Path
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).parent.parent  # autoresearch root
CORE_DIR = SCRIPT_DIR / "core"
sys.path.insert(0, str(CORE_DIR))

# Periods (hardcoded to match prepare_rv.py — read-only, no imports needed)
TRAIN_PERIOD = ("2023-02-01", "2025-06-30")
VAL_PERIOD = ("2025-07-01", "2026-01-30")
OOS_PERIOD_1 = ("2021-01-01", "2023-01-31")
OOS_PERIOD_2 = ("2026-02-01", "2026-03-23")

REGIME_STATES_8 = [
    "L1 Safe", "L1 Exposed",
    "L2 Safe", "L2 Caution-A", "L2 Caution-B", "L2 Risky",
    "L3 Safe", "L3 Exposed",
]

RISK_FREE_PCT = 5.5
ANNUALIZATION = 252


def _parse_date(s):
    return dt_date(*map(int, s.split("-")))


def _period_mask(df, period):
    start, end = _parse_date(period[0]), _parse_date(period[1])
    return (df["date"] >= start) & (df["date"] <= end)


def _sharpe(pnl):
    pnl = pnl.dropna()
    if len(pnl) < 10:
        return None
    m, s = float(pnl.mean()), float(pnl.std())
    if s == 0:
        return None
    return round((m * ANNUALIZATION - RISK_FREE_PCT) / (s * np.sqrt(ANNUALIZATION)), 4)


def _load_data_standalone(snapshot="1530"):
    """Load data directly from cached parquet + strategy Excel files.

    Replicates prepare_rv.load_data() without requiring NSQA imports.
    """
    BASE_DIR = SCRIPT_DIR.parent
    rv_cache = BASE_DIR / "features" / "rv_daily.parquet"

    if not rv_cache.exists():
        raise FileNotFoundError(f"Cached data not found at {rv_cache}. Run pipeline first.")

    rv = pd.read_parquet(rv_cache)
    rv["date"] = pd.to_datetime(rv["timestamp"]).dt.date
    rv.sort_values("date", inplace=True)
    rv.reset_index(drop=True, inplace=True)

    # Parkinson Vol
    log_hl = np.log(rv["high"] / rv["low"])
    rv["PK_today"] = np.sqrt(log_hl ** 2 / (4 * np.log(2))) * np.sqrt(252) * 100

    # Snapshot-specific IV
    iv_col = f"IV_7d_{snapshot}"
    if iv_col in rv.columns:
        if snapshot in ("0915", "0916"):
            rv["_iv"] = rv[iv_col].shift(-1)
        else:
            rv["_iv"] = rv[iv_col]
    else:
        rv["_iv"] = rv["IV_7d"]

    # Derived features
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

    # Strategy PnLs
    DATA_STRATEGIES = BASE_DIR.parent / "DATA" / "Strategies"
    strat_files = {
        "dm": DATA_STRATEGIES / "DM" / "DM_merged.xlsx",
        "wc": DATA_STRATEGIES / "WC" / "WC_merged.xlsx",
        "orion": DATA_STRATEGIES / "Orion" / "Orion_merged.xlsx",
    }
    for skey, fpath in strat_files.items():
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


def _evaluate_standalone(df, regime_col="regime_state"):
    """Replicate prepare_rv.evaluate() without requiring NSQA imports."""

    clean = df.dropna(subset=[regime_col, "pnl_combined"]).copy()
    train_start, train_end = _parse_date(TRAIN_PERIOD[0]), _parse_date(TRAIN_PERIOD[1])
    val_start, val_end = _parse_date(VAL_PERIOD[0]), _parse_date(VAL_PERIOD[1])

    train = clean[(clean["date"] >= train_start) & (clean["date"] <= train_end)]
    val = clean[(clean["date"] >= val_start) & (clean["date"] <= val_end)]

    if len(val) < 20:
        return {"composite_score": -999, "error": "Too few validation days"}

    val_sharpe = _sharpe(val["pnl_combined"]) or 0

    def _al_pct(sub):
        if "all_lose" not in sub.columns or len(sub) == 0:
            return None
        return round(float(sub["all_lose"].sum() / len(sub) * 100), 2)

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
        gaps.append(_level_gap(val, prefix))
    gaps.append(_level_gap(val, "L2", exposed_name="L2 Risky"))
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
    min_state_days_val = int(states_in_val.min()) if len(states_in_val) > 0 else 0
    coverage = min(1.0, min_state_days_val / 5.0) * min(1.0, n_states_used / 6.0)

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
        "min_state_days": min_state_days_val,
        "state_metrics": state_metrics,
    }


def _load_classified_data():
    """Load data and run current baseline classification + evaluation."""
    df = _load_data_standalone("1530")

    # Mock the NSQA module to avoid import errors
    import types
    mock_pipeline = types.ModuleType("pipeline")
    mock_nsqa = types.ModuleType("pipeline.nsqa_data")
    mock_nsqa.fetch_rv_daily = lambda **kw: None  # never called
    sys.modules["pipeline"] = mock_pipeline
    sys.modules["pipeline.nsqa_data"] = mock_nsqa

    # Also mock data_management if needed
    if "data_management" not in sys.modules:
        mock_dm = types.ModuleType("data_management")
        sys.modules["data_management"] = mock_dm
        sys.modules["data_management.market_reader_api"] = types.ModuleType("data_management.market_reader_api")
        sys.modules["data_management.market_reader_api.protos_adapter"] = types.ModuleType("data_management.market_reader_api.protos_adapter")

    # Clear regime_experiment from cache to force fresh import
    for mod in list(sys.modules.keys()):
        if mod in ("regime_experiment", "prepare_rv", "prepare_rv_meta", "meta_config"):
            del sys.modules[mod]

    # Remove meta_config.json temporarily to use original scoring
    meta_json = SCRIPT_DIR / "meta_config.json"
    had_meta = meta_json.exists()
    if had_meta:
        _backup = meta_json.read_text()
        meta_json.unlink()

    try:
        from regime_experiment import run_classification
        df = run_classification(df)
    finally:
        if had_meta:
            meta_json.write_text(_backup)

    results = _evaluate_standalone(df)

    return df, results


# ═══════════════════════════════════════════════════════════════
# DIAGNOSTIC 1: Regime Distribution Analysis
# ═══════════════════════════════════════════════════════════════

def regime_distribution_analysis(df):
    """Count days per regime state across all periods."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 1: Regime Distribution Across Periods")
    print("=" * 70)

    periods = {
        "Train": TRAIN_PERIOD,
        "Val": VAL_PERIOD,
        "OOS1": OOS_PERIOD_1,
        "OOS2": OOS_PERIOD_2,
    }

    report = {}
    clean = df.dropna(subset=["regime_state", "pnl_combined"])

    # Header
    print(f"\n{'State':<20s}", end="")
    for name in periods:
        print(f"  {name:>8s}", end="")
    print(f"  {'Total':>8s}")
    print("-" * 70)

    for state in REGIME_STATES_8:
        row = {"state": state}
        print(f"{state:<20s}", end="")
        for name, period in periods.items():
            mask = _period_mask(clean, period) & (clean["regime_state"] == state)
            count = int(mask.sum())
            row[name] = count
            flag = " *" if count < 5 else ""
            print(f"  {count:>7d}{flag}", end="")
        total = sum(row.get(n, 0) for n in periods)
        row["Total"] = total
        print(f"  {total:>8d}")
        report[state] = row

    # Totals
    print("-" * 70)
    print(f"{'TOTAL':<20s}", end="")
    for name, period in periods.items():
        mask = _period_mask(clean, period) & clean["regime_state"].notna()
        print(f"  {int(mask.sum()):>8d} ", end="")
    print()

    # Flag states with <5 days in any period
    sparse = [(s, r) for s, r in report.items()
              if any(r.get(p, 0) < 5 for p in ["Val", "OOS2"])]
    if sparse:
        print(f"\n  WARNING: States with <5 days in Val or OOS2 (marked with *):")
        for s, r in sparse:
            issues = [f"{p}={r[p]}d" for p in periods if r.get(p, 0) < 5]
            print(f"    {s}: {', '.join(issues)}")

    return report


# ═══════════════════════════════════════════════════════════════
# DIAGNOSTIC 2: Feature Correlation Matrix
# ═══════════════════════════════════════════════════════════════

def feature_correlation_matrix(df):
    """Spearman correlations among all features on training data."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 2: Feature Correlation Matrix (Training Period)")
    print("=" * 70)

    features = [
        "iv_lag", "PK_IV_ratio", "IV_chg_5d", "IV_5d", "PK_5d",
        "IV_10d", "PK_10d", "PK_IV_10d", "IV_chg_1d",
        "IV_percentile_60d", "PK_IV_zscore_30d",
    ]
    # Add computed features if present
    for f in ["PK_IV_smooth3", "pk_iv_pctile", "pk_iv_risk", "RV_today", "VRP_today"]:
        if f in df.columns:
            features.append(f)

    available = [f for f in features if f in df.columns]
    train = df[_period_mask(df, TRAIN_PERIOD)][available].dropna()

    corr = train.corr(method="spearman")

    # Find high correlations
    high_corr_pairs = []
    for i, f1 in enumerate(available):
        for f2 in available[i + 1:]:
            c = abs(corr.loc[f1, f2])
            if c > 0.6:
                high_corr_pairs.append((f1, f2, float(corr.loc[f1, f2])))

    high_corr_pairs.sort(key=lambda x: -abs(x[2]))

    print(f"\n  Features analyzed: {len(available)}")
    print(f"  Training rows: {len(train)}")

    if high_corr_pairs:
        print(f"\n  Highly correlated pairs (|rho| > 0.6):")
        for f1, f2, c in high_corr_pairs:
            flag = "REDUNDANT" if abs(c) > 0.75 else "moderate"
            print(f"    {f1:25s} <-> {f2:25s}  rho={c:+.3f}  [{flag}]")
    else:
        print(f"\n  No highly correlated pairs found.")

    # IC with next-day pnl
    print(f"\n  Univariate IC with next-day pnl_combined:")
    ics = {}
    train_with_pnl = df[_period_mask(df, TRAIN_PERIOD)].dropna(subset=["pnl_combined"])
    for f in available:
        vals = train_with_pnl[[f, "pnl_combined"]].dropna()
        if len(vals) > 20:
            ic = float(spearmanr(vals[f], vals["pnl_combined"]).correlation)
            ics[f] = ic

    for f, ic in sorted(ics.items(), key=lambda x: -abs(x[1])):
        bar = "#" * int(abs(ic) * 50)
        print(f"    {f:25s}  IC={ic:+.4f}  {bar}")

    return {
        "correlation_matrix": corr.to_dict(),
        "high_corr_pairs": [(f1, f2, c) for f1, f2, c in high_corr_pairs],
        "feature_ics": ics,
    }


# ═══════════════════════════════════════════════════════════════
# DIAGNOSTIC 3: Strategy Correlation Matrix
# ═══════════════════════════════════════════════════════════════

def strategy_correlation_matrix(df):
    """Pairwise correlations of pnl_dm/pnl_wc/pnl_orion per regime state."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 3: Strategy Correlation by Regime State")
    print("=" * 70)

    pnl_cols = [c for c in ["pnl_dm", "pnl_wc", "pnl_orion"] if c in df.columns]
    if len(pnl_cols) < 2:
        print("  Insufficient strategy columns for correlation analysis.")
        return {}

    report = {}

    # Overall
    train = df[_period_mask(df, TRAIN_PERIOD)][pnl_cols].dropna()
    overall_corr = train.corr(method="spearman")
    print(f"\n  Overall strategy correlations (training period, {len(train)} days):")
    for i, c1 in enumerate(pnl_cols):
        for c2 in pnl_cols[i + 1:]:
            print(f"    {c1:12s} <-> {c2:12s}  rho={overall_corr.loc[c1, c2]:+.3f}")
    report["overall"] = overall_corr.to_dict()

    # Per regime state
    print(f"\n  Per-state strategy correlations:")
    clean = df.dropna(subset=["regime_state"] + pnl_cols)
    train_clean = clean[_period_mask(clean, TRAIN_PERIOD)]

    for state in REGIME_STATES_8:
        state_data = train_clean[train_clean["regime_state"] == state][pnl_cols]
        if len(state_data) < 10:
            print(f"    {state:20s}  ({len(state_data):3d}d) — too few days")
            continue

        sc = state_data.corr(method="spearman")
        pairs = []
        for i, c1 in enumerate(pnl_cols):
            for c2 in pnl_cols[i + 1:]:
                pairs.append(f"{c1[-2:]}/{c2[-5:]}={sc.loc[c1, c2]:+.2f}")
        print(f"    {state:20s}  ({len(state_data):3d}d)  {', '.join(pairs)}")
        report[state] = sc.to_dict()

    return report


# ═══════════════════════════════════════════════════════════════
# DIAGNOSTIC 4: Rolling Sharpe Analysis
# ═══════════════════════════════════════════════════════════════

def rolling_sharpe_analysis(df):
    """Rolling 60-day Sharpe of pnl_combined."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 4: Rolling 60-Day Sharpe Analysis")
    print("=" * 70)

    clean = df.dropna(subset=["pnl_combined"]).copy()
    pnl = clean.set_index("date")["pnl_combined"]

    window = 60
    rolling_mean = pnl.rolling(window, min_periods=30).mean()
    rolling_std = pnl.rolling(window, min_periods=30).std()
    rolling_sharpe = (rolling_mean * 252 - RISK_FREE_PCT) / (rolling_std * np.sqrt(252))

    # Stats by period
    for name, period in [("Train", TRAIN_PERIOD), ("Val", VAL_PERIOD),
                          ("OOS1", OOS_PERIOD_1), ("OOS2", OOS_PERIOD_2)]:
        start, end = _parse_date(period[0]), _parse_date(period[1])
        mask = (rolling_sharpe.index >= start) & (rolling_sharpe.index <= end)
        period_sharpe = rolling_sharpe[mask].dropna()
        if len(period_sharpe) > 0:
            print(f"\n  {name:6s}: mean={period_sharpe.mean():+.2f}, "
                  f"std={period_sharpe.std():.2f}, "
                  f"min={period_sharpe.min():+.2f}, max={period_sharpe.max():+.2f}, "
                  f"pct_positive={100 * (period_sharpe > 0).mean():.0f}%")
        else:
            print(f"\n  {name:6s}: insufficient data")

    # Check val stability
    val_start = _parse_date(VAL_PERIOD[0])
    val_end = _parse_date(VAL_PERIOD[1])
    val_mask = (rolling_sharpe.index >= val_start) & (rolling_sharpe.index <= val_end)
    val_sharpe = rolling_sharpe[val_mask].dropna()

    if len(val_sharpe) > 0:
        # Split val into halves
        mid = len(val_sharpe) // 2
        first_half = val_sharpe.iloc[:mid].mean()
        second_half = val_sharpe.iloc[mid:].mean()
        print(f"\n  Val period split: first_half={first_half:+.2f}, second_half={second_half:+.2f}")
        if abs(first_half - second_half) > 2.0:
            print(f"  WARNING: Large Sharpe difference between val halves — results may be period-dependent")

    return {
        "val_mean_rolling_sharpe": float(val_sharpe.mean()) if len(val_sharpe) > 0 else None,
    }


# ═══════════════════════════════════════════════════════════════
# DIAGNOSTIC 5: Normalization Sensitivity
# ═══════════════════════════════════════════════════════════════

def normalization_sensitivity(df, results):
    """Sweep normalization constants and recompute composite_score."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 5: Normalization Sensitivity")
    print("=" * 70)

    val_sharpe = results["val_sharpe"]
    safe_sep = results["safe_separation"]
    rank_corr = results["rank_stability"]
    coverage = results["state_coverage"]

    print(f"\n  Baseline metrics: val_sharpe={val_sharpe:.4f}, safe_sep={safe_sep:.2f}, "
          f"rank={rank_corr:.4f}, coverage={coverage:.4f}")

    # Current composite breakdown
    print(f"\n  Current composite breakdown (sharpe_norm=5.0, safe_sep_norm=10.0, sharpe_cap=1.5):")
    sharpe_term = 0.40 * min(val_sharpe / 5.0, 1.5)
    sep_term = 0.25 * min(safe_sep / 10.0, 1.0)
    rank_term = 0.20 * max(rank_corr, 0)
    cov_term = 0.15 * coverage
    print(f"    sharpe:   0.40 * min({val_sharpe:.4f}/5.0, 1.5) = {sharpe_term:.4f}")
    print(f"    safe_sep: 0.25 * min({safe_sep:.2f}/10.0, 1.0) = {sep_term:.4f}")
    print(f"    rank:     0.20 * {rank_corr:.4f}               = {rank_term:.4f}")
    print(f"    coverage: 0.15 * {coverage:.4f}               = {cov_term:.4f}")
    print(f"    TOTAL:    {sharpe_term + sep_term + rank_term + cov_term:.6f}")

    # Sweep sharpe_norm
    print(f"\n  Sharpe norm sensitivity (cap=1.5 vs cap=1.0):")
    print(f"    {'sn':>4s}  {'cap=1.5':>8s}  {'cap=1.0':>8s}  {'saturated?':>10s}")
    report_sn = {}
    for sn in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
        v15 = min(val_sharpe / sn, 1.5)
        v10 = min(val_sharpe / sn, 1.0)
        sat = "YES" if val_sharpe / sn >= 1.0 else "no"
        print(f"    {sn:4.1f}  {v15:8.4f}  {v10:8.4f}  {sat:>10s}")
        report_sn[sn] = {"cap_1.5": v15, "cap_1.0": v10}

    # Sweep safe_sep_norm
    print(f"\n  Safe sep norm sensitivity (cap=1.0):")
    print(f"    {'ssn':>4s}  {'term':>8s}  {'saturated?':>10s}")
    report_ssn = {}
    for ssn in [5.0, 8.0, 10.0, 12.0, 15.0, 20.0]:
        v = min(safe_sep / ssn, 1.0)
        sat = "YES" if safe_sep / ssn >= 1.0 else "no"
        print(f"    {ssn:4.1f}  {v:8.4f}  {sat:>10s}")
        report_ssn[ssn] = {"term": v}

    return {"sharpe_norm_sweep": report_sn, "safe_sep_norm_sweep": report_ssn}


# ═══════════════════════════════════════════════════════════════
# DIAGNOSTIC 6: Rank Stability Deep Dive
# ═══════════════════════════════════════════════════════════════

def rank_stability_deep_dive(df):
    """Investigate the suspicious rank_stability = 1.0."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 6: Rank Stability Deep Dive")
    print("=" * 70)

    clean = df.dropna(subset=["regime_state", "pnl_combined"])
    train = clean[_period_mask(clean, TRAIN_PERIOD)]
    val = clean[_period_mask(clean, VAL_PERIOD)]

    train_means = []
    val_means = []
    print(f"\n  {'State':<20s}  {'Train Mean':>10s}  {'Val Mean':>10s}  {'Train Rank':>10s}  {'Val Rank':>10s}")
    print(f"  {'-' * 65}")

    for s in REGIME_STATES_8:
        t_pnl = train[train["regime_state"] == s]["pnl_combined"]
        v_pnl = val[val["regime_state"] == s]["pnl_combined"]
        t_mean = float(t_pnl.mean()) if len(t_pnl) > 0 else 0
        v_mean = float(v_pnl.mean()) if len(v_pnl) > 0 else 0
        train_means.append(t_mean)
        val_means.append(v_mean)

    # Compute ranks
    train_ranks = pd.Series(train_means).rank().tolist()
    val_ranks = pd.Series(val_means).rank().tolist()

    for i, s in enumerate(REGIME_STATES_8):
        print(f"  {s:<20s}  {train_means[i]:+10.4f}  {val_means[i]:+10.4f}  "
              f"{train_ranks[i]:10.1f}  {val_ranks[i]:10.1f}")

    rho, pval = spearmanr(train_means, val_means)
    print(f"\n  Spearman rho: {rho:.4f}, p-value: {pval:.4f}")

    # Permutation test
    n_perms = 10000
    perm_rhos = []
    rng = np.random.RandomState(42)
    for _ in range(n_perms):
        shuffled = rng.permutation(val_means)
        perm_rho, _ = spearmanr(train_means, shuffled)
        perm_rhos.append(perm_rho)

    perm_rhos = np.array(perm_rhos)
    perm_p = float((perm_rhos >= rho).mean())
    print(f"  Permutation test ({n_perms} shuffles): p={perm_p:.4f}")

    if perm_p > 0.05:
        print(f"  FINDING: rank_stability=1.0 is NOT statistically significant (p={perm_p:.4f})")
        print(f"           With only 8 states, perfect rank correlation can occur by chance ~{perm_p*100:.1f}% of the time")
    else:
        print(f"  FINDING: rank_stability=1.0 IS statistically significant (p={perm_p:.4f})")

    return {
        "rho": rho, "pval": pval, "perm_pval": perm_p,
        "train_means": dict(zip(REGIME_STATES_8, train_means)),
        "val_means": dict(zip(REGIME_STATES_8, val_means)),
    }


# ═══════════════════════════════════════════════════════════════
# DIAGNOSTIC 7: OOS2 Adequacy Check
# ═══════════════════════════════════════════════════════════════

def oos2_adequacy_check(df):
    """Assess whether 34 days of OOS2 provides a reliable signal."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 7: OOS2 Adequacy Check")
    print("=" * 70)

    clean = df.dropna(subset=["pnl_combined"])

    for name, period in [("OOS1", OOS_PERIOD_1), ("OOS2", OOS_PERIOD_2)]:
        subset = clean[_period_mask(clean, period)]
        n = len(subset)
        sharpe = _sharpe(subset["pnl_combined"])

        if n < 5 or sharpe is None:
            print(f"\n  {name}: {n} days — insufficient data")
            continue

        # SE of annualized Sharpe ≈ sqrt((1 + 0.5*S²) / (n-1)) * sqrt(252)
        # Simplified: SE ≈ sqrt(252/n) for Sharpe near 0
        se_simple = np.sqrt(ANNUALIZATION / n)
        se_full = np.sqrt((1 + 0.5 * sharpe ** 2) / max(n - 1, 1)) * np.sqrt(ANNUALIZATION)

        ci_lo = sharpe - 1.96 * se_full
        ci_hi = sharpe + 1.96 * se_full

        significant = abs(sharpe) > 1.96 * se_full

        print(f"\n  {name}: {n} days")
        print(f"    Sharpe:       {sharpe:+.4f}")
        print(f"    SE (simple):  {se_simple:.4f}")
        print(f"    SE (full):    {se_full:.4f}")
        print(f"    95% CI:       [{ci_lo:+.2f}, {ci_hi:+.2f}]")
        print(f"    Significant:  {'YES' if significant else 'NO'}")

        if not significant:
            print(f"    FINDING: {name} Sharpe is NOT distinguishable from zero")

    return {}


# ═══════════════════════════════════════════════════════════════
# DIAGNOSTIC 8: Boundary Sensitivity
# ═══════════════════════════════════════════════════════════════

def boundary_sensitivity(df):
    """Sweep IV boundaries and compute composite_score at each point."""
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 8: IV Boundary Sensitivity")
    print("=" * 70)

    # Import classification functions (no NSQA dependency)
    from regime_experiment import (
        compute_extra_features, compute_thresholds, classify_day,
        apply_strategy_weights, LEVEL_FEATURE, SPLIT_FEATURE_L1,
        SPLIT_FEATURE_L2, SPLIT_FEATURE_L3, L2_DIRECTION_ENABLED,
        L2_DIRECTION_FEATURE, STRATEGY_WEIGHTS, APPLY_STRATEGY_WEIGHTS,
    )

    l1_range = np.arange(7.0, 12.5, 0.5)
    l2_range = np.arange(10.0, 19.0, 1.0)

    print(f"\n  Sweeping IV_L1_UPPER [{l1_range[0]}-{l1_range[-1]}] x IV_L2_UPPER [{l2_range[0]}-{l2_range[-1]}]")
    print(f"  (Current: L1={8.5}, L2={11.0}, composite=0.926)")

    best_score = -999
    best_l1 = None
    best_l2 = None
    grid = {}

    # Pre-compute features once
    df_base = df.copy()
    df_base = compute_extra_features(df_base)

    for l1 in l1_range:
        for l2 in l2_range:
            if l2 <= l1:
                continue

            df_test = df_base.copy()

            # Recompute thresholds with these boundaries
            from datetime import date as dt_date_inner
            train_start = dt_date_inner(2023, 2, 1)
            train_end = dt_date_inner(2025, 6, 30)
            train_data = df_test[(df_test["date"] >= train_start) & (df_test["date"] <= train_end)].copy()
            train_data = train_data.dropna(subset=[LEVEL_FEATURE])

            thresholds = {}
            for lvl_name, mask_fn in [
                ("L1", lambda d, _l1=l1: d[LEVEL_FEATURE] < _l1),
                ("L2", lambda d, _l1=l1, _l2=l2: (d[LEVEL_FEATURE] >= _l1) & (d[LEVEL_FEATURE] < _l2)),
                ("L3", lambda d, _l2=l2: d[LEVEL_FEATURE] >= _l2),
            ]:
                split_feat = {"L1": SPLIT_FEATURE_L1, "L2": SPLIT_FEATURE_L2, "L3": SPLIT_FEATURE_L3}[lvl_name]
                lvl_data = train_data[mask_fn(train_data)]
                if split_feat in lvl_data.columns and len(lvl_data) > 5:
                    thresholds[lvl_name] = float(lvl_data[split_feat].dropna().median())
                else:
                    thresholds[lvl_name] = {"L1": 0.63, "L2": 0.65, "L3": 0.67}[lvl_name]

            df_test["regime_state"] = df_test.apply(
                lambda r: classify_day(r, thresholds, l1, l2), axis=1
            )

            if APPLY_STRATEGY_WEIGHTS:
                df_test = apply_strategy_weights(df_test)

            result = _evaluate_standalone(df_test)
            score = result["composite_score"]
            grid[(l1, l2)] = score

            if score > best_score:
                best_score = score
                best_l1 = l1
                best_l2 = l2

    # Print heatmap
    print(f"\n  Composite Score Grid (L1 rows, L2 columns):")
    print(f"  {'L1\\L2':>6s}", end="")
    for l2 in l2_range:
        print(f"  {l2:5.1f}", end="")
    print()

    for l1 in l1_range:
        print(f"  {l1:6.1f}", end="")
        for l2 in l2_range:
            if l2 <= l1:
                print(f"  {'—':>5s}", end="")
            else:
                score = grid.get((l1, l2), 0)
                marker = " *" if (l1, l2) == (best_l1, best_l2) else "  "
                if score > 0.90:
                    print(f"  {score:5.3f}", end="")
                else:
                    print(f"  {score:5.3f}", end="")
        print()

    print(f"\n  Best: L1={best_l1:.1f}, L2={best_l2:.1f}, composite={best_score:.6f}")
    print(f"  Current: L1=8.5, L2=11.0, composite=0.926")

    return {
        "best_l1": best_l1, "best_l2": best_l2, "best_score": best_score,
        "grid": {f"{l1:.1f}_{l2:.1f}": s for (l1, l2), s in grid.items()},
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def run_all(save=False):
    """Run all diagnostics."""
    print("Loading and classifying data...")
    df, results = _load_classified_data()
    print(f"Loaded {len(df)} rows, date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Baseline composite_score: {results['composite_score']:.6f}")

    report = {}
    report["baseline"] = {k: v for k, v in results.items() if k != "state_metrics"}

    report["regime_distribution"] = regime_distribution_analysis(df)
    report["feature_correlations"] = feature_correlation_matrix(df)
    report["strategy_correlations"] = strategy_correlation_matrix(df)
    report["rolling_sharpe"] = rolling_sharpe_analysis(df)
    report["normalization"] = normalization_sensitivity(df, results)
    report["rank_stability"] = rank_stability_deep_dive(df)
    report["oos2_adequacy"] = oos2_adequacy_check(df)
    report["boundary_sensitivity"] = boundary_sensitivity(df)

    if save:
        out_dir = SCRIPT_DIR / "output"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / "diagnostics_report.json"
        # Convert non-serializable types
        def _clean(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {str(k): _clean(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_clean(i) for i in obj]
            return obj

        with open(out_path, "w") as f:
            json.dump(_clean(report), f, indent=2, default=str)
        print(f"\n\nReport saved to {out_path}")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Framework diagnostics")
    parser.add_argument("--save", action="store_true", help="Save report to JSON")
    parser.add_argument("--only", type=str, help="Run only one diagnostic",
                        choices=["regime", "features", "strategy", "rolling",
                                 "norms", "rank", "oos2", "boundary"])
    args = parser.parse_args()

    if args.only:
        df, results = _load_classified_data()
        funcs = {
            "regime": lambda: regime_distribution_analysis(df),
            "features": lambda: feature_correlation_matrix(df),
            "strategy": lambda: strategy_correlation_matrix(df),
            "rolling": lambda: rolling_sharpe_analysis(df),
            "norms": lambda: normalization_sensitivity(df, results),
            "rank": lambda: rank_stability_deep_dive(df),
            "oos2": lambda: oos2_adequacy_check(df),
            "boundary": lambda: boundary_sensitivity(df),
        }
        funcs[args.only]()
    else:
        run_all(save=args.save)
