"""
ic_dedup.py — Upgrade 3: IC Deduplication Gate.

Computes pairwise Spearman rank correlation between activated features
across the training period and drops any feature whose max |correlation|
with another activated feature exceeds a threshold (default 0.75).

Runs before the inner loop starts to ensure the agent only sees
orthogonal features.

Usage:
    from ic_dedup import deduplicate_features
    kept, dropped = deduplicate_features(df, ["IV_20d", "PK_20d", "VRP_5d"], threshold=0.75)

    # CLI
    python ic_dedup.py --features "IV_20d,PK_20d,VRP_5d,RV_IV_gap" --threshold 0.75
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))


def compute_correlation_matrix(df: pd.DataFrame, features: list[str],
                                train_start: str = "2023-02-01",
                                train_end: str = "2025-06-30") -> pd.DataFrame:
    """Compute pairwise Spearman rank correlation between features on training data."""
    from datetime import date as dt_date

    start = dt_date(*map(int, train_start.split("-")))
    end = dt_date(*map(int, train_end.split("-")))

    # Filter to training period
    mask = (df["date"] >= start) & (df["date"] <= end)
    train = df.loc[mask, features].dropna()

    if len(train) < 20:
        # Not enough data — return identity (keep everything)
        return pd.DataFrame(np.eye(len(features)), index=features, columns=features)

    # Compute Spearman correlation matrix
    corr_matrix = train.corr(method="spearman")
    return corr_matrix


def deduplicate_features(df: pd.DataFrame, features: list[str],
                          threshold: float = 0.75,
                          train_start: str = "2023-02-01",
                          train_end: str = "2025-06-30") -> tuple[list[str], list[dict]]:
    """Remove redundant features based on pairwise Spearman correlation.

    Returns:
        (kept_features, dropped_info)
        where dropped_info is a list of {"feature": str, "correlated_with": str, "correlation": float}
    """
    if len(features) <= 1:
        return features, []

    # Filter to features that actually exist in the DataFrame
    available = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]

    if len(available) <= 1:
        return available, [{"feature": f, "reason": "not in DataFrame"} for f in missing]

    corr = compute_correlation_matrix(df, available, train_start, train_end)

    kept = []
    dropped = []

    for feat in available:
        # Check if this feature is too correlated with any already-kept feature
        is_redundant = False
        for kept_feat in kept:
            c = abs(corr.loc[feat, kept_feat])
            if c > threshold:
                dropped.append({
                    "feature": feat,
                    "correlated_with": kept_feat,
                    "correlation": round(float(c), 4),
                })
                is_redundant = True
                break

        if not is_redundant:
            kept.append(feat)

    # Add missing features to dropped
    for f in missing:
        dropped.append({"feature": f, "reason": "not in DataFrame"})

    return kept, dropped


def print_correlation_report(df: pd.DataFrame, features: list[str],
                              train_start: str = "2023-02-01",
                              train_end: str = "2025-06-30"):
    """Print a full correlation matrix for the given features."""
    available = [f for f in features if f in df.columns]
    if len(available) < 2:
        print("Need at least 2 available features for correlation report.")
        return

    corr = compute_correlation_matrix(df, available, train_start, train_end)

    print(f"\nSpearman Correlation Matrix ({len(available)} features, training period):\n")

    # Header
    max_name = max(len(f) for f in available)
    header = " " * (max_name + 2) + "  ".join(f"{f[:8]:>8s}" for f in available)
    print(header)

    # Rows
    for f1 in available:
        row = f"{f1:<{max_name}s}  "
        for f2 in available:
            c = corr.loc[f1, f2]
            if f1 == f2:
                row += f"{'---':>8s}  "
            elif abs(c) > 0.75:
                row += f"\033[91m{c:>8.3f}\033[0m  "  # red for high correlation
            elif abs(c) > 0.50:
                row += f"\033[93m{c:>8.3f}\033[0m  "  # yellow for moderate
            else:
                row += f"{c:>8.3f}  "
        print(row)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="IC Deduplication Gate")
    parser.add_argument("--features", type=str, required=True,
                        help="Comma-separated feature names")
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="Correlation threshold for deduplication")
    parser.add_argument("--report", action="store_true",
                        help="Print full correlation matrix")
    args = parser.parse_args()

    features = [f.strip() for f in args.features.split(",")]

    # Load data to compute correlations
    from meta_config import MetaConfig, load_active_config
    cfg = load_active_config()

    # Need to compute features first
    print("Loading data and computing features...")

    # Force all features to be computed
    if cfg is None:
        cfg = MetaConfig(extra_features=features)
        cfg.to_json(SCRIPT_DIR / "meta_config.json")

    # Clear module cache
    for mod in list(sys.modules.keys()):
        if mod in ("prepare_rv", "prepare_rv_meta", "meta_config"):
            del sys.modules[mod]

    from prepare_rv import load_data
    df = load_data("1530")

    # Clean up temp config if we created it
    temp_cfg = SCRIPT_DIR / "meta_config.json"
    if temp_cfg.exists() and cfg.description == "":
        temp_cfg.unlink()

    if args.report:
        print_correlation_report(df, features)
    else:
        kept, dropped = deduplicate_features(df, features, threshold=args.threshold)
        print(f"\nKept ({len(kept)}): {', '.join(kept)}")
        if dropped:
            print(f"Dropped ({len(dropped)}):")
            for d in dropped:
                if "correlated_with" in d:
                    print(f"  {d['feature']} — corr {d['correlation']:.3f} with {d['correlated_with']}")
                else:
                    print(f"  {d['feature']} — {d.get('reason', 'unknown')}")
