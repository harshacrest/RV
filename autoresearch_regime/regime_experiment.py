"""
regime_experiment.py — THE ONLY FILE THE AGENT MODIFIES.

This file defines the regime classification system.
The agent modifies ANYTHING in this file to improve the composite_score
from prepare_rv.evaluate().

Run: python regime_experiment.py
Output: prints results summary to stdout.
"""

import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from prepare_rv import load_data, evaluate, REGIME_STATES_8

# ═══════════════════════════════════════════════════════════════════
# DIMENSION 1: IV LEVEL BOUNDARIES
# ═══════════════════════════════════════════════════════════════════
# Static boundaries — agent can change these values
IV_L1_UPPER = 8.5        # Below this = L1 (low IV)
IV_L2_UPPER = 11       # Below this = L2 (moderate IV), above = L3 (high IV)

# Adaptive boundary parameters
ADAPTIVE_ENABLED = False         # Set True to enable adaptive shifting
ADAPTIVE_LOOKBACK = 45           # Days to look back
ADAPTIVE_HIGH_IV_THRESH = 17     # What counts as "high IV" for the trailing check
ADAPTIVE_TRIGGER_PCT = 0.50      # Fraction of lookback days above threshold to trigger shift
ADAPTIVE_SHIFT_L1 = 17           # Shifted L1 upper when adaptive triggers
ADAPTIVE_SHIFT_L2 = 22           # Shifted L2 upper when adaptive triggers


# ═══════════════════════════════════════════════════════════════════
# DIMENSION 2: FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════
# Which feature to use for level classification
LEVEL_FEATURE = "iv_lag"              # Feature that determines L1/L2/L3

# Which feature to use for Safe/Exposed split within each level
SPLIT_FEATURE_L1 = "PK_IV_ratio"     # L1 splitter
SPLIT_FEATURE_L2 = "PK_IV_ratio"     # L2 primary splitter
SPLIT_FEATURE_L3 = "PK_IV_zscore_30d" # L3 zscore

# L2 secondary signal (IV direction)
L2_DIRECTION_FEATURE = "IV_chg_1d"   # Feature for IV rising/falling at L2
L2_DIRECTION_ENABLED = True           # Whether to use direction at L2

# L1/L3 direction (currently disabled — agent can enable)
L1_DIRECTION_ENABLED = False
L3_DIRECTION_ENABLED = False

# Rolling window for features (agent can create new derived features here)
def compute_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agent can add custom feature computations here.
    Input df already has: iv_lag, PK_IV_ratio, IV_chg_5d, IV_5d, PK_5d,
    IV_10d, PK_10d, PK_IV_10d, IV_chg_1d, IV_percentile_60d, PK_IV_zscore_30d,
    RV_today, VRP_today
    """
    df["PK_IV_smooth3"] = df["PK_IV_ratio"].rolling(3, min_periods=1).mean()
    # IV level percentile blended with PK ratio
    df["pk_iv_pctile"] = df["PK_IV_ratio"] * (1 + 0.3 * (df["IV_percentile_60d"].fillna(50) - 50) / 50)
    # Risk-adjusted PK ratio: penalizes when IV is rising fast
    iv_accel = df["IV_chg_1d"].fillna(0) - df["IV_chg_1d"].shift(1).fillna(0)
    df["pk_iv_risk"] = df["PK_IV_ratio"] - 0.15 * iv_accel.clip(-3, 3)
    return df


# ═══════════════════════════════════════════════════════════════════
# DIMENSION 3: CLASSIFICATION ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════
# Number of states per level (agent can change the classification structure)
# Default: L1=2 states, L2=4 states, L3=2 states = 8 total
# Agent could try: L1=2, L2=2, L3=2 = 6 states
#                  L1=3, L2=4, L3=3 = 10 states
#                  etc.

# Threshold method: "median" (split at median of training data) or "fixed" (use FIXED_THRESHOLD_*)
THRESHOLD_METHOD = "median"

# Fixed thresholds (only used if THRESHOLD_METHOD == "fixed")
FIXED_THRESHOLD_L1 = 0.63
FIXED_THRESHOLD_L2 = 0.65
FIXED_THRESHOLD_L3 = 0.67


# ═══════════════════════════════════════════════════════════════════
# DIMENSION 4: STRATEGY ALLOCATION WEIGHTS
# ═══════════════════════════════════════════════════════════════════
# Weights per state: [dm_weight, wc_weight, orion_weight]
# Default: equal weight [1, 1, 1] everywhere
# Agent can set different weights per state to test portfolio optimization

STRATEGY_WEIGHTS = {
    "L1 Safe":      [0, 1, 0],
    "L1 Exposed":   [0, 1, 0.2],
    "L2 Safe":      [1, 1, 1],
    "L2 Caution-A": [0.2, 0.4, 0.8],
    "L2 Caution-B": [1, 1, 1],
    "L2 Risky":     [1, 1, 1],
    "L3 Safe":      [1, 1, 1],
    "L3 Exposed":   [0.3, 0.3, 1],
}

# Whether to apply strategy weights (False = equal weight as baseline)
APPLY_STRATEGY_WEIGHTS = True


# ═══════════════════════════════════════════════════════════════════
# DIMENSION 5: SNAPSHOT FUSION
# ═══════════════════════════════════════════════════════════════════
# Which snapshot(s) to use
PRIMARY_SNAPSHOT = "1530"
SECONDARY_SNAPSHOT = None  # Set to "0916" to enable dual-snapshot fusion

# Fusion method: "primary_only", "majority", "conservative", "override"
# - primary_only: Use primary snapshot only
# - majority: Use state that both snapshots agree on, else primary
# - conservative: If snapshots disagree, classify as "cautious" variant
# - override: Morning overrides if it's more risky
FUSION_METHOD = "primary_only"


# ═══════════════════════════════════════════════════════════════════
# CLASSIFICATION ENGINE (agent can modify the logic here)
# ═══════════════════════════════════════════════════════════════════

def compute_thresholds(df: pd.DataFrame) -> dict:
    """Compute split thresholds from training data."""
    from datetime import date as dt_date
    train_start = dt_date(2023, 2, 1)
    train_end = dt_date(2025, 6, 30)
    train = df[(df["date"] >= train_start) & (df["date"] <= train_end)].copy()
    train = train.dropna(subset=[LEVEL_FEATURE])

    thresholds = {}

    if THRESHOLD_METHOD == "median":
        for lvl_name, mask_fn in [
            ("L1", lambda d: d[LEVEL_FEATURE] < IV_L1_UPPER),
            ("L2", lambda d: (d[LEVEL_FEATURE] >= IV_L1_UPPER) & (d[LEVEL_FEATURE] < IV_L2_UPPER)),
            ("L3", lambda d: d[LEVEL_FEATURE] >= IV_L2_UPPER),
        ]:
            split_feat = {"L1": SPLIT_FEATURE_L1, "L2": SPLIT_FEATURE_L2, "L3": SPLIT_FEATURE_L3}[lvl_name]
            lvl_data = train[mask_fn(train)]
            if split_feat in lvl_data.columns and len(lvl_data) > 5:
                thresholds[lvl_name] = float(lvl_data[split_feat].dropna().median())
            else:
                thresholds[lvl_name] = {"L1": 0.63, "L2": 0.65, "L3": 0.67}[lvl_name]
    else:
        thresholds = {"L1": FIXED_THRESHOLD_L1, "L2": FIXED_THRESHOLD_L2, "L3": FIXED_THRESHOLD_L3}

    return thresholds


def classify_day(row: pd.Series, thresholds: dict,
                 l1_upper: float = None, l2_upper: float = None) -> str | None:
    """Classify a single day into a regime state."""
    l1_up = l1_upper if l1_upper is not None else IV_L1_UPPER
    l2_up = l2_upper if l2_upper is not None else IV_L2_UPPER

    level_val = row.get(LEVEL_FEATURE)
    if pd.isna(level_val):
        return None

    # Determine level
    if level_val < l1_up:
        level = "L1"
        split_feat = SPLIT_FEATURE_L1
        use_direction = L1_DIRECTION_ENABLED
    elif level_val < l2_up:
        level = "L2"
        split_feat = SPLIT_FEATURE_L2
        use_direction = L2_DIRECTION_ENABLED
    else:
        level = "L3"
        split_feat = SPLIT_FEATURE_L3
        use_direction = L3_DIRECTION_ENABLED

    split_val = row.get(split_feat)
    if pd.isna(split_val):
        return None

    threshold = thresholds.get(level, 0.65)
    low_split = split_val <= threshold

    if level == "L2" and use_direction:
        dir_val = row.get(L2_DIRECTION_FEATURE, 0)
        iv_falling = dir_val <= -1.1 if not pd.isna(dir_val) else True

        if low_split and iv_falling:
            return "L2 Safe"
        elif not low_split and iv_falling:
            return "L2 Caution-A"
        elif low_split and not iv_falling:
            return "L2 Caution-B"
        else:
            return "L2 Risky"

    elif level == "L1":
        if L1_DIRECTION_ENABLED:
            dir_val = row.get(L2_DIRECTION_FEATURE, 0)
            iv_falling = dir_val <= 0 if not pd.isna(dir_val) else True
            # Agent could implement 4-state L1 here
            pass
        return f"L1 Safe" if low_split else f"L1 Exposed"

    elif level == "L3":
        if L3_DIRECTION_ENABLED:
            dir_val = row.get(L2_DIRECTION_FEATURE, 0)
            iv_falling = dir_val <= 0 if not pd.isna(dir_val) else True
            # Agent could implement 4-state L3 here
            pass
        return f"L3 Safe" if low_split else f"L3 Exposed"

    return None


def apply_adaptive_boundaries(df: pd.DataFrame) -> pd.DataFrame:
    """Apply adaptive boundary shifting if enabled."""
    if not ADAPTIVE_ENABLED:
        df["_l1_upper"] = IV_L1_UPPER
        df["_l2_upper"] = IV_L2_UPPER
        return df

    l1_uppers = []
    l2_uppers = []
    for idx in df.index:
        date_val = df.loc[idx, "date"]
        prior = df[df["date"] < date_val].tail(ADAPTIVE_LOOKBACK)
        if len(prior) > 0:
            high_pct = (prior[LEVEL_FEATURE] > ADAPTIVE_HIGH_IV_THRESH).mean()
        else:
            high_pct = 0
        if high_pct > ADAPTIVE_TRIGGER_PCT:
            l1_uppers.append(ADAPTIVE_SHIFT_L1)
            l2_uppers.append(ADAPTIVE_SHIFT_L2)
        else:
            l1_uppers.append(IV_L1_UPPER)
            l2_uppers.append(IV_L2_UPPER)

    df["_l1_upper"] = l1_uppers
    df["_l2_upper"] = l2_uppers
    return df


def apply_strategy_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute pnl_combined using per-state strategy weights."""
    if not APPLY_STRATEGY_WEIGHTS:
        return df

    pnl_cols = [c for c in ["pnl_dm", "pnl_wc", "pnl_orion"] if c in df.columns]
    if not pnl_cols:
        return df

    for idx in df.index:
        state = df.loc[idx, "regime_state"]
        if state in STRATEGY_WEIGHTS:
            weights = STRATEGY_WEIGHTS[state]
            total_w = sum(weights)
            if total_w > 0:
                weighted_pnl = 0
                for col, w in zip(pnl_cols, weights):
                    val = df.loc[idx, col]
                    if not pd.isna(val):
                        weighted_pnl += val * w / total_w
                df.loc[idx, "pnl_combined"] = weighted_pnl
    return df


def run_classification(df: pd.DataFrame) -> pd.DataFrame:
    """Full classification pipeline."""
    # Extra features
    df = compute_extra_features(df)

    # Adaptive boundaries
    df = apply_adaptive_boundaries(df)

    # Compute thresholds from training data
    thresholds = compute_thresholds(df)

    # Classify each day
    if ADAPTIVE_ENABLED:
        df["regime_state"] = df.apply(
            lambda r: classify_day(r, thresholds, r["_l1_upper"], r["_l2_upper"]), axis=1
        )
    else:
        df["regime_state"] = df.apply(
            lambda r: classify_day(r, thresholds), axis=1
        )

    # Apply strategy weights
    df = apply_strategy_weights(df)

    return df


def run_fusion(df_primary: pd.DataFrame, df_secondary: pd.DataFrame) -> pd.DataFrame:
    """Fuse two snapshot classifications if SECONDARY_SNAPSHOT is set."""
    if FUSION_METHOD == "primary_only" or SECONDARY_SNAPSHOT is None:
        return df_primary

    merged = df_primary[["date", "regime_state", "pnl_combined"]].merge(
        df_secondary[["date", "regime_state"]].rename(columns={"regime_state": "regime_secondary"}),
        on="date", how="left"
    )

    if FUSION_METHOD == "majority":
        # If both agree, use it. If they disagree, use primary.
        merged["regime_state"] = merged.apply(
            lambda r: r["regime_state"] if r["regime_state"] == r["regime_secondary"] else r["regime_state"],
            axis=1
        )
    elif FUSION_METHOD == "conservative":
        # If disagree, pick the more cautious state
        RISK_ORDER = {s: i for i, s in enumerate([
            "L1 Safe", "L2 Safe", "L2 Caution-A", "L2 Caution-B",
            "L3 Safe", "L3 Exposed", "L2 Risky", "L1 Exposed",
        ])}
        def pick_cautious(r):
            if r["regime_state"] == r["regime_secondary"]:
                return r["regime_state"]
            r1 = RISK_ORDER.get(r["regime_state"], 99)
            r2 = RISK_ORDER.get(r["regime_secondary"], 99)
            return r["regime_state"] if r1 >= r2 else r["regime_secondary"]
        merged["regime_state"] = merged.apply(pick_cautious, axis=1)

    # Copy back to primary df
    df_primary["regime_state"] = merged["regime_state"].values
    return df_primary


# ═══════════════════════════════════════════════════════════════════
# MAIN — Run experiment and print results
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0 = time.time()

    # Load data
    df = load_data(PRIMARY_SNAPSHOT)

    # Run classification
    df = run_classification(df)

    # Optional: dual-snapshot fusion
    if SECONDARY_SNAPSHOT is not None:
        df2 = load_data(SECONDARY_SNAPSHOT)
        df2 = run_classification(df2)
        df = run_fusion(df, df2)

    # Evaluate
    results = evaluate(df)

    elapsed = time.time() - t0

    # Print results in parseable format
    print("---")
    print(f"composite_score:  {results['composite_score']:.6f}")
    print(f"val_sharpe:       {results['val_sharpe']:.4f}")
    print(f"safe_separation:  {results['safe_separation']:.2f}")
    print(f"rank_stability:   {results['rank_stability']:.4f}")
    print(f"state_coverage:   {results['state_coverage']:.4f}")
    print(f"val_days:         {results['val_days']}")
    print(f"train_days:       {results['train_days']}")
    print(f"n_states_used:    {results['n_states_used']}")
    print(f"elapsed_seconds:  {elapsed:.1f}")

    # Per-state breakdown
    print("\n--- State Breakdown (Validation) ---")
    for s in REGIME_STATES_8:
        m = results["state_metrics"].get(s, {})
        days = m.get("days", 0)
        sh = m.get("sharpe")
        al = m.get("al_pct")
        avg = m.get("port_avg")
        sh_str = f"{sh:.2f}" if sh is not None else "—"
        al_str = f"{al:.1f}%" if al is not None else "—"
        avg_str = f"{avg:+.4f}" if avg is not None else "—"
        print(f"  {s:20s}  {days:3d}d  Sharpe={sh_str:>6s}  AL={al_str:>6s}  Avg={avg_str}")
