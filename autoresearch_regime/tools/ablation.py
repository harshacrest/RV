"""
ablation.py — Systematic ablation study for the regime framework.

Disables one dimension at a time to measure its marginal contribution
to the composite score. Reveals which components are load-bearing.

Usage:
    python ablation.py              # Run all ablations
    python ablation.py --only boundaries
"""

import sys
import copy
import numpy as np
import pandas as pd
from datetime import date as dt_date
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent  # autoresearch root
CORE_DIR = SCRIPT_DIR / "core"
TOOLS_DIR = Path(__file__).parent
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(CORE_DIR))

# Reuse standalone data loading from diagnostics
from diagnostics import _load_data_standalone, _evaluate_standalone, REGIME_STATES_8


def _classify_with_overrides(df, **overrides):
    """Run classification with parameter overrides without modifying regime_experiment.py."""
    import importlib

    _mock_nsqa_imports()
    # Clear module cache
    for mod in list(sys.modules.keys()):
        if mod in ("regime_experiment", "prepare_rv", "prepare_rv_meta", "meta_config"):
            del sys.modules[mod]

    import regime_experiment as re

    # Apply overrides
    originals = {}
    for key, val in overrides.items():
        if hasattr(re, key):
            originals[key] = getattr(re, key)
            setattr(re, key, val)

    try:
        df_copy = df.copy()
        df_copy = re.run_classification(df_copy)
        results = _evaluate_standalone(df_copy)
    finally:
        # Restore originals
        for key, val in originals.items():
            setattr(re, key, val)

    return results


def _mock_nsqa_imports():
    """Mock NSQA modules to avoid import errors."""
    import types
    if "pipeline.nsqa_data" not in sys.modules:
        sys.modules["pipeline"] = types.ModuleType("pipeline")
        mock_nsqa = types.ModuleType("pipeline.nsqa_data")
        mock_nsqa.fetch_rv_daily = lambda **kw: None
        sys.modules["pipeline.nsqa_data"] = mock_nsqa
    if "data_management" not in sys.modules:
        sys.modules["data_management"] = types.ModuleType("data_management")
        sys.modules["data_management.market_reader_api"] = types.ModuleType("data_management.market_reader_api")
        sys.modules["data_management.market_reader_api.protos_adapter"] = types.ModuleType("data_management.market_reader_api.protos_adapter")


def run_ablation_study():
    """Run systematic ablation: disable one dimension at a time."""
    print("Loading data...")
    df = _load_data_standalone("1530")

    print("Running baseline...")
    _mock_nsqa_imports()
    # Clear and import fresh
    for mod in list(sys.modules.keys()):
        if mod in ("regime_experiment", "prepare_rv", "prepare_rv_meta", "meta_config"):
            del sys.modules[mod]
    import regime_experiment as re

    df_base = df.copy()
    df_base = re.run_classification(df_base)
    baseline = _evaluate_standalone(df_base)
    baseline_score = baseline["composite_score"]

    print(f"Baseline composite: {baseline_score:.6f}")
    print(f"  val_sharpe={baseline['val_sharpe']:.4f}, safe_sep={baseline['safe_separation']:.2f}, "
          f"rank={baseline['rank_stability']:.4f}, coverage={baseline['state_coverage']:.4f}")

    ablations = {
        "No strategy weights": {"APPLY_STRATEGY_WEIGHTS": False},
        "No L2 direction": {"L2_DIRECTION_ENABLED": False},
        "Equal boundaries [10,15]": {"IV_L1_UPPER": 10.0, "IV_L2_UPPER": 15.0},
        "Wide boundaries [12,18]": {"IV_L1_UPPER": 12.0, "IV_L2_UPPER": 18.0},
        "Tight boundaries [8,10]": {"IV_L1_UPPER": 8.0, "IV_L2_UPPER": 10.0},
        "PK_IV_ratio everywhere": {
            "SPLIT_FEATURE_L1": "PK_IV_ratio",
            "SPLIT_FEATURE_L2": "PK_IV_ratio",
            "SPLIT_FEATURE_L3": "PK_IV_ratio",
        },
        "Fixed thresholds": {"THRESHOLD_METHOD": "fixed"},
        "No extra features": {},  # Special case below
    }

    print(f"\n{'='*70}")
    print(f"ABLATION STUDY — {len(ablations)} experiments")
    print(f"{'='*70}")
    print(f"\n{'Ablation':<35s}  {'Score':>8s}  {'Delta':>8s}  {'Impact':>8s}")
    print(f"{'-'*70}")

    results = {}

    for name, overrides in ablations.items():
        if name == "No extra features":
            # Special: override compute_extra_features to do nothing
            result = _classify_with_overrides(
                df,
                compute_extra_features=lambda d: d,
            )
        else:
            result = _classify_with_overrides(df, **overrides)

        score = result["composite_score"]
        delta = score - baseline_score
        impact = "CRITICAL" if delta < -0.05 else "moderate" if delta < -0.02 else "minimal"

        print(f"  {name:<35s}  {score:8.4f}  {delta:+8.4f}  {impact:>8s}")
        results[name] = {"score": score, "delta": delta, "impact": impact}

    # Summary
    print(f"\n{'='*70}")
    print(f"LOAD-BEARING COMPONENTS (ablation delta < -0.02):")
    for name, r in sorted(results.items(), key=lambda x: x[1]["delta"]):
        if r["delta"] < -0.02:
            print(f"  {name}: delta={r['delta']:+.4f}")

    print(f"\nNON-ESSENTIAL COMPONENTS (ablation delta >= -0.02):")
    for name, r in sorted(results.items(), key=lambda x: -x[1]["delta"]):
        if r["delta"] >= -0.02:
            print(f"  {name}: delta={r['delta']:+.4f}")

    return results


if __name__ == "__main__":
    run_ablation_study()
