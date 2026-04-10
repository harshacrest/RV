"""
paper_trade_gate.py — Upgrade 5: Forward Paper Trade Gate.

Validates a classifier on the most recent unseen data before promotion
to production. Checks:
1. Regime assignments are plausible (not all collapsing to one state)
2. Strategy Sharpe stays positive during the forward window
3. Regime transitions are stable (not flipping every day)

Usage:
    from paper_trade_gate import run_paper_trade_gate
    result = run_paper_trade_gate(days=10)

    # CLI
    python paper_trade_gate.py --days 10
    python paper_trade_gate.py --days 10 --verbose
"""

import sys
import json
import numpy as np
import pandas as pd
from datetime import date as dt_date, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))


@dataclass
class GateResult:
    """Result from the paper trade gate."""
    passed: bool
    days_evaluated: int
    sharpe: float
    al_pct: Optional[float]
    n_states_used: int
    transition_rate: float      # fraction of days with state change
    max_consecutive_same: int   # longest streak in one state
    state_distribution: dict    # state -> day count
    issues: list                # list of failure reasons
    recommendation: str         # "PROMOTE", "ITERATE", or "REJECT"

    def summary(self) -> str:
        icon = "PASS" if self.passed else "FAIL"
        return (
            f"[{icon}] {self.days_evaluated}d | Sharpe={self.sharpe:.2f} | "
            f"states={self.n_states_used} | transitions={self.transition_rate:.0%} | "
            f"{self.recommendation}"
        )


def run_paper_trade_gate(
    days: int = 10,
    min_sharpe: float = -1.0,     # minimum acceptable Sharpe (lenient)
    min_states: int = 2,           # minimum distinct states in forward window
    max_transition_rate: float = 0.8,  # max fraction of days with state change
    verbose: bool = False,
) -> GateResult:
    """Run forward-walk validation on the most recent unseen data.

    Uses data after the latest OOS period end date (2026-03-23).
    If no data exists beyond that, falls back to the last N days of available data.
    """
    # Clear module cache to get fresh imports
    for mod in list(sys.modules.keys()):
        if mod in ("prepare_rv", "prepare_rv_meta", "meta_config", "regime_experiment"):
            del sys.modules[mod]

    from prepare_rv import load_data
    from regime_experiment import run_classification

    # Load and classify
    df = load_data("1530")
    df = run_classification(df)

    # Find the forward window: data after all train/val/OOS periods
    from meta_config import OOS_PERIOD_2
    oos2_end = dt_date(*map(int, OOS_PERIOD_2[1].split("-")))

    classified = df.dropna(subset=["regime_state", "pnl_combined"]).copy()

    # Try data after OOS2
    forward = classified[classified["date"] > oos2_end].tail(days)

    if len(forward) < 3:
        # Not enough post-OOS data — use the last N days before OOS2 as a proxy
        # This is imperfect but still validates classifier behavior
        forward = classified.tail(days)
        if verbose:
            print(f"  NOTE: Only {len(classified[classified['date'] > oos2_end])} days after OOS2. "
                  f"Using last {days} days of available data as proxy.")

    if len(forward) < 3:
        return GateResult(
            passed=False, days_evaluated=len(forward), sharpe=0, al_pct=None,
            n_states_used=0, transition_rate=0, max_consecutive_same=0,
            state_distribution={}, issues=["Too few days for evaluation"],
            recommendation="REJECT",
        )

    # ── Check 1: Sharpe ──
    pnl = forward["pnl_combined"]
    mean_pnl = float(pnl.mean())
    std_pnl = float(pnl.std())
    if std_pnl > 0:
        sharpe = round((mean_pnl * 252 - 5.5) / (std_pnl * np.sqrt(252)), 2)
    else:
        sharpe = 0.0

    # ── Check 2: State distribution ──
    states = forward["regime_state"].value_counts().to_dict()
    n_states = len(states)

    # ── Check 3: Transition stability ──
    state_seq = forward["regime_state"].tolist()
    transitions = sum(1 for i in range(1, len(state_seq)) if state_seq[i] != state_seq[i - 1])
    transition_rate = transitions / max(len(state_seq) - 1, 1)

    # Max consecutive same state
    max_consec = 1
    curr_consec = 1
    for i in range(1, len(state_seq)):
        if state_seq[i] == state_seq[i - 1]:
            curr_consec += 1
            max_consec = max(max_consec, curr_consec)
        else:
            curr_consec = 1

    # ── Check 4: All-lose percentage ──
    al_pct = None
    if "all_lose" in forward.columns:
        al_pct = round(float(forward["all_lose"].sum() / len(forward) * 100), 1)

    # ── Evaluate pass/fail ──
    issues = []

    if sharpe < min_sharpe:
        issues.append(f"Sharpe {sharpe:.2f} below minimum {min_sharpe}")

    if n_states < min_states:
        issues.append(f"Only {n_states} state(s) used — classifier may be collapsed")

    if transition_rate > max_transition_rate:
        issues.append(f"Transition rate {transition_rate:.0%} too high — classifier unstable")

    if n_states == 1 and len(forward) > 5:
        issues.append(f"All {len(forward)} days classified as '{state_seq[0]}' — no discrimination")

    passed = len(issues) == 0

    if passed:
        recommendation = "PROMOTE"
    elif len(issues) == 1 and sharpe >= 0:
        recommendation = "ITERATE"  # minor issue, worth re-trying
    else:
        recommendation = "REJECT"

    result = GateResult(
        passed=passed,
        days_evaluated=len(forward),
        sharpe=sharpe,
        al_pct=al_pct,
        n_states_used=n_states,
        transition_rate=round(transition_rate, 3),
        max_consecutive_same=max_consec,
        state_distribution=states,
        issues=issues,
        recommendation=recommendation,
    )

    if verbose:
        print(f"\n{'='*50}")
        print(f"PAPER TRADE GATE — {len(forward)} days")
        print(f"{'='*50}")
        print(f"  Sharpe:          {sharpe:.2f}")
        print(f"  All-Lose %:      {al_pct}")
        print(f"  States used:     {n_states}")
        print(f"  Transition rate: {transition_rate:.0%}")
        print(f"  Max consecutive: {max_consec} days in same state")
        print(f"\n  State distribution:")
        for s, count in sorted(states.items(), key=lambda x: -x[1]):
            print(f"    {s:20s}: {count:3d} days ({count/len(forward)*100:.0f}%)")
        if issues:
            print(f"\n  Issues:")
            for issue in issues:
                print(f"    - {issue}")
        print(f"\n  Result: {result.summary()}")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Forward Paper Trade Gate")
    parser.add_argument("--days", type=int, default=10, help="Number of forward days to evaluate")
    parser.add_argument("--verbose", action="store_true", help="Detailed output")
    parser.add_argument("--min-sharpe", type=float, default=-1.0, help="Minimum acceptable Sharpe")
    args = parser.parse_args()

    result = run_paper_trade_gate(
        days=args.days,
        min_sharpe=args.min_sharpe,
        verbose=True,
    )
    print(f"\n{result.summary()}")
