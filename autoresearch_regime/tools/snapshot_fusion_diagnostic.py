"""
snapshot_fusion_diagnostic.py — Test dual-snapshot classification.

Compares single-snapshot (1530) vs dual-snapshot (1530+0916) classification
to determine if morning data improves regime accuracy on a per-state basis.

Usage:
    python snapshot_fusion_diagnostic.py
"""

import sys
import types
import numpy as np
import pandas as pd
from datetime import date as dt_date
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent  # autoresearch root
CORE_DIR = SCRIPT_DIR / "core"
TOOLS_DIR = Path(__file__).parent
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(CORE_DIR))

from diagnostics import (
    _load_data_standalone, _evaluate_standalone,
    REGIME_STATES_8, TRAIN_PERIOD, VAL_PERIOD,
    _parse_date, _period_mask,
)


def run_snapshot_fusion_diagnostic():
    """Compare single vs dual snapshot classification."""
    print("Loading primary snapshot (1530)...")
    df_primary = _load_data_standalone("1530")

    print("Loading secondary snapshot (0916)...")
    df_secondary = _load_data_standalone("0916")

    # Mock imports for regime_experiment
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

    from regime_experiment import run_classification, run_fusion

    # Classify both snapshots
    print("Classifying primary (1530)...")
    df_p = run_classification(df_primary)

    print("Classifying secondary (0916)...")
    df_s = run_classification(df_secondary)

    # Evaluate primary alone
    results_primary = _evaluate_standalone(df_p)
    print(f"\nPrimary (1530) composite: {results_primary['composite_score']:.6f}")

    # Merge states for comparison
    merged = df_p[["date", "regime_state", "pnl_combined"]].merge(
        df_s[["date", "regime_state"]].rename(columns={"regime_state": "state_0916"}),
        on="date", how="inner"
    )
    merged = merged.dropna(subset=["regime_state", "state_0916"])

    # Agreement analysis
    print(f"\n{'='*70}")
    print(f"SNAPSHOT AGREEMENT ANALYSIS")
    print(f"{'='*70}")

    total = len(merged)
    agree = (merged["regime_state"] == merged["state_0916"]).sum()
    print(f"\n  Overall agreement: {agree}/{total} ({100*agree/total:.1f}%)")

    # Per-state agreement
    print(f"\n  {'Primary State':<20s}  {'Days':>5s}  {'Agree':>5s}  {'Rate':>6s}  {'Most Common 0916':>25s}")
    print(f"  {'-'*70}")

    for state in REGIME_STATES_8:
        mask = merged["regime_state"] == state
        state_rows = merged[mask]
        n = len(state_rows)
        if n == 0:
            continue
        n_agree = (state_rows["state_0916"] == state).sum()
        rate = n_agree / n

        # Most common disagreement
        disagree = state_rows[state_rows["state_0916"] != state]["state_0916"]
        if len(disagree) > 0:
            most_common = disagree.value_counts().index[0]
            mc_count = disagree.value_counts().iloc[0]
            mc_str = f"{most_common} ({mc_count}x)"
        else:
            mc_str = "—"

        print(f"  {state:<20s}  {n:5d}  {n_agree:5d}  {rate:5.0%}  {mc_str:>25s}")

    # Test conservative fusion
    print(f"\n{'='*70}")
    print(f"FUSION METHOD COMPARISON")
    print(f"{'='*70}")

    RISK_ORDER = {s: i for i, s in enumerate([
        "L1 Safe", "L2 Safe", "L2 Caution-A", "L2 Caution-B",
        "L3 Safe", "L3 Exposed", "L2 Risky", "L1 Exposed",
    ])}

    for method in ["majority", "conservative"]:
        df_fused = df_p.copy()

        if method == "majority":
            # If both agree, use it; if disagree, use primary
            fused_states = []
            for _, row in merged.iterrows():
                if row["regime_state"] == row["state_0916"]:
                    fused_states.append(row["regime_state"])
                else:
                    fused_states.append(row["regime_state"])
            # For majority, this is just primary (since we only have 2 snapshots)
            # Real majority would need 3+ snapshots
        elif method == "conservative":
            fused_states = []
            for _, row in merged.iterrows():
                if row["regime_state"] == row["state_0916"]:
                    fused_states.append(row["regime_state"])
                else:
                    r1 = RISK_ORDER.get(row["regime_state"], 99)
                    r2 = RISK_ORDER.get(row["state_0916"], 99)
                    # Pick the MORE cautious (higher risk order = riskier)
                    fused_states.append(row["regime_state"] if r1 >= r2 else row["state_0916"])

        # Apply fused states back to df
        fused_df = merged.copy()
        fused_df["regime_state"] = fused_states

        # Need to restore pnl_combined and all_lose columns for evaluation
        pnl_cols = [c for c in ["pnl_dm", "pnl_wc", "pnl_orion", "pnl_combined",
                                 "all_lose", "all_win"] if c in df_p.columns]
        for col in pnl_cols:
            if col not in fused_df.columns:
                fused_df = fused_df.merge(
                    df_p[["date", col]].drop_duplicates("date"),
                    on="date", how="left"
                )

        results_fused = _evaluate_standalone(fused_df)
        delta = results_fused["composite_score"] - results_primary["composite_score"]
        print(f"\n  {method:15s}: composite={results_fused['composite_score']:.6f} "
              f"(delta={delta:+.4f}) "
              f"sharpe={results_fused['val_sharpe']:.4f} "
              f"safe_sep={results_fused['safe_separation']:.2f}")

    print(f"\n  Primary only:   composite={results_primary['composite_score']:.6f} "
          f"sharpe={results_primary['val_sharpe']:.4f} "
          f"safe_sep={results_primary['safe_separation']:.2f}")


if __name__ == "__main__":
    run_snapshot_fusion_diagnostic()
