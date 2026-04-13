"""
rolling_validation.py — Walk-forward validation for the regime classifier.

Tests whether the classifier consistently adds value across time, or
if val_sharpe is period-specific.

Usage:
    python rolling_validation.py           # Run walk-forward
    python rolling_validation.py --window 120 --step 60
"""

import sys
import numpy as np
import pandas as pd
from datetime import date as dt_date, timedelta
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent  # autoresearch root
CORE_DIR = SCRIPT_DIR / "core"
TOOLS_DIR = Path(__file__).parent
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(CORE_DIR))

from diagnostics import _load_data_standalone, RISK_FREE_PCT, ANNUALIZATION


def _sharpe(pnl):
    pnl = pnl.dropna()
    if len(pnl) < 10:
        return None
    m, s = float(pnl.mean()), float(pnl.std())
    if s == 0:
        return None
    return round((m * ANNUALIZATION - RISK_FREE_PCT) / (s * np.sqrt(ANNUALIZATION)), 4)


def walk_forward_validation(
    val_window_days: int = 120,
    step_days: int = 60,
    train_lookback_days: int = 450,
):
    """Walk-forward validation: slide a window across time.

    For each window position:
    1. Train thresholds on preceding train_lookback_days
    2. Classify the val_window_days
    3. Compute Sharpe of classified portfolio vs equal-weight baseline
    """
    print("Loading data...")
    df = _load_data_standalone("1530")

    # Mock NSQA imports for regime_experiment
    import types
    if "pipeline.nsqa_data" not in sys.modules:
        mock_pipeline = types.ModuleType("pipeline")
        mock_nsqa = types.ModuleType("pipeline.nsqa_data")
        mock_nsqa.fetch_rv_daily = lambda **kw: None
        sys.modules["pipeline"] = mock_pipeline
        sys.modules["pipeline.nsqa_data"] = mock_nsqa
    if "data_management" not in sys.modules:
        sys.modules["data_management"] = types.ModuleType("data_management")
        sys.modules["data_management.market_reader_api"] = types.ModuleType("data_management.market_reader_api")
        sys.modules["data_management.market_reader_api.protos_adapter"] = types.ModuleType("data_management.market_reader_api.protos_adapter")

    for mod in list(sys.modules.keys()):
        if mod in ("regime_experiment", "prepare_rv", "prepare_rv_meta", "meta_config"):
            del sys.modules[mod]

    from regime_experiment import (
        compute_extra_features, classify_day,
        apply_strategy_weights, LEVEL_FEATURE,
        SPLIT_FEATURE_L1, SPLIT_FEATURE_L2, SPLIT_FEATURE_L3,
        APPLY_STRATEGY_WEIGHTS,
    )

    df = compute_extra_features(df)
    clean = df.dropna(subset=["pnl_combined", LEVEL_FEATURE]).copy()
    dates = sorted(clean["date"].unique())

    # Find the range where we have enough history
    min_date_idx = train_lookback_days
    max_date_idx = len(dates) - val_window_days

    if max_date_idx <= min_date_idx:
        print("Insufficient data for walk-forward validation")
        return []

    print(f"\nWalk-forward validation:")
    print(f"  Train lookback: {train_lookback_days} days")
    print(f"  Val window: {val_window_days} days")
    print(f"  Step size: {step_days} days")
    print(f"  Date range: {dates[min_date_idx]} to {dates[-1]}")
    print(f"  Windows: ~{(max_date_idx - min_date_idx) // step_days}")

    results = []

    print(f"\n{'Val Start':>12s}  {'Val End':>12s}  {'Days':>5s}  {'Regime':>8s}  {'Equal':>8s}  {'Delta':>8s}  {'Winner':>8s}")
    print(f"{'-' * 75}")

    for idx in range(min_date_idx, max_date_idx, step_days):
        train_start_idx = max(0, idx - train_lookback_days)
        train_dates = dates[train_start_idx:idx]
        val_dates = dates[idx:idx + val_window_days]

        if len(val_dates) < 20:
            continue

        t_start, t_end = train_dates[0], train_dates[-1]
        v_start, v_end = val_dates[0], val_dates[-1]

        # Train: compute thresholds
        train_data = clean[(clean["date"] >= t_start) & (clean["date"] <= t_end)]
        train_data = train_data.dropna(subset=[LEVEL_FEATURE])

        from regime_experiment import IV_L1_UPPER, IV_L2_UPPER

        thresholds = {}
        for lvl_name, mask_fn in [
            ("L1", lambda d, l1=IV_L1_UPPER: d[LEVEL_FEATURE] < l1),
            ("L2", lambda d, l1=IV_L1_UPPER, l2=IV_L2_UPPER: (d[LEVEL_FEATURE] >= l1) & (d[LEVEL_FEATURE] < l2)),
            ("L3", lambda d, l2=IV_L2_UPPER: d[LEVEL_FEATURE] >= l2),
        ]:
            split_feat = {"L1": SPLIT_FEATURE_L1, "L2": SPLIT_FEATURE_L2, "L3": SPLIT_FEATURE_L3}[lvl_name]
            lvl_data = train_data[mask_fn(train_data)]
            if split_feat in lvl_data.columns and len(lvl_data) > 5:
                thresholds[lvl_name] = float(lvl_data[split_feat].dropna().median())
            else:
                thresholds[lvl_name] = {"L1": 0.63, "L2": 0.65, "L3": 0.67}[lvl_name]

        # Classify val period
        val_data = clean[(clean["date"] >= v_start) & (clean["date"] <= v_end)].copy()
        val_data["regime_state"] = val_data.apply(
            lambda r: classify_day(r, thresholds), axis=1
        )

        if APPLY_STRATEGY_WEIGHTS:
            val_data = apply_strategy_weights(val_data)

        # Regime-weighted Sharpe
        regime_sharpe = _sharpe(val_data["pnl_combined"])

        # Equal-weight baseline (no regime, just average all strategies)
        pnl_cols = [c for c in ["pnl_dm", "pnl_wc", "pnl_orion"] if c in val_data.columns]
        equal_pnl = val_data[pnl_cols].mean(axis=1)
        equal_sharpe = _sharpe(equal_pnl)

        if regime_sharpe is not None and equal_sharpe is not None:
            delta = regime_sharpe - equal_sharpe
            winner = "REGIME" if delta > 0 else "EQUAL" if delta < 0 else "TIE"
        else:
            delta = 0
            winner = "N/A"

        print(f"  {v_start}  {v_end}  {len(val_data):5d}  "
              f"{regime_sharpe or 0:+8.2f}  {equal_sharpe or 0:+8.2f}  "
              f"{delta:+8.2f}  {winner:>8s}")

        results.append({
            "val_start": v_start,
            "val_end": v_end,
            "n_days": len(val_data),
            "regime_sharpe": regime_sharpe,
            "equal_sharpe": equal_sharpe,
            "delta": delta,
            "winner": winner,
        })

    # Summary
    if results:
        regime_wins = sum(1 for r in results if r["winner"] == "REGIME")
        equal_wins = sum(1 for r in results if r["winner"] == "EQUAL")
        avg_delta = np.mean([r["delta"] for r in results if r["delta"] is not None])

        print(f"\n{'='*75}")
        print(f"SUMMARY: {len(results)} windows")
        print(f"  Regime wins: {regime_wins} ({100*regime_wins/len(results):.0f}%)")
        print(f"  Equal wins:  {equal_wins} ({100*equal_wins/len(results):.0f}%)")
        print(f"  Avg delta:   {avg_delta:+.2f}")

        if regime_wins < len(results) * 0.5:
            print(f"\n  WARNING: Regime classifier does NOT consistently beat equal-weight")
            print(f"  The current val_sharpe may be period-specific, not robust")
        else:
            print(f"\n  Regime classifier adds value in {regime_wins}/{len(results)} windows")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Walk-forward validation")
    parser.add_argument("--window", type=int, default=120, help="Validation window (days)")
    parser.add_argument("--step", type=int, default=60, help="Step size (days)")
    parser.add_argument("--lookback", type=int, default=450, help="Training lookback (days)")
    args = parser.parse_args()

    walk_forward_validation(
        val_window_days=args.window,
        step_days=args.step,
        train_lookback_days=args.lookback,
    )
