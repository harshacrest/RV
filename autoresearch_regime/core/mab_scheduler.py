"""
mab_scheduler.py — Upgrade 2: Multi-Armed Bandit Scheduler.

UCB1-based scheduler for adaptive dimension allocation in the meta-harness.
Each meta-harness dimension (scoring, norms, features, splits) is an arm.
After each iteration, the MAB tracks OOS deltas and allocates the next
iteration to the dimension with highest UCB score.

Usage:
    from mab_scheduler import MABScheduler
    mab = MABScheduler()
    dim = mab.select_dimension()        # "scoring", "norms", "features", "splits"
    mab.record_outcome("scoring", 0.15) # OOS delta from this iteration
    mab.save()

    # CLI
    python mab_scheduler.py --stats
    python mab_scheduler.py --select
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).parent.parent  # autoresearch root
MAB_STATE_FILE = SCRIPT_DIR / "mab_state.json"

# The 5 meta-harness dimensions (Upgrade 4 added "strategy")
DIMENSIONS = ["scoring", "norms", "features", "splits", "strategy"]


@dataclass
class ArmState:
    """State for a single MAB arm (dimension)."""
    name: str
    pulls: int = 0              # number of times this dimension was tried
    total_reward: float = 0.0   # sum of OOS deltas
    rewards: list = field(default_factory=list)  # history of individual rewards
    parameter_history: list = field(default_factory=list)  # [{params: {...}, reward: float}]

    @property
    def mean_reward(self) -> float:
        if self.pulls == 0:
            return 0.0
        return self.total_reward / self.pulls

    def ucb1_score(self, total_pulls: int, c: float = 1.41) -> float:
        """UCB1 score = mean_reward + c * sqrt(ln(total) / pulls).

        Arms with 0 pulls get infinity (must explore first).
        """
        if self.pulls == 0:
            return float("inf")
        exploration = c * math.sqrt(math.log(total_pulls) / self.pulls)
        return self.mean_reward + exploration


class MABScheduler:
    """Multi-Armed Bandit scheduler for meta-harness dimension allocation."""

    def __init__(self, path: Path = MAB_STATE_FILE, exploration_c: float = 1.41):
        self.path = path
        self.exploration_c = exploration_c
        self.arms: dict[str, ArmState] = {}
        self._load()

    def _load(self):
        """Load MAB state from disk."""
        if self.path.exists():
            with open(self.path) as f:
                data = json.load(f)
            for arm_data in data.get("arms", []):
                name = arm_data["name"]
                self.arms[name] = ArmState(
                    name=name,
                    pulls=arm_data.get("pulls", 0),
                    total_reward=arm_data.get("total_reward", 0.0),
                    rewards=arm_data.get("rewards", []),
                    parameter_history=arm_data.get("parameter_history", []),
                )
        # Ensure all dimensions have an arm
        for dim in DIMENSIONS:
            if dim not in self.arms:
                self.arms[dim] = ArmState(name=dim)

    def save(self):
        """Save MAB state to disk."""
        data = {
            "total_pulls": self.total_pulls,
            "exploration_c": self.exploration_c,
            "arms": [
                {
                    "name": arm.name,
                    "pulls": arm.pulls,
                    "total_reward": arm.total_reward,
                    "rewards": arm.rewards[-50:],  # keep last 50
                    "parameter_history": arm.parameter_history[-20:],  # keep last 20
                }
                for arm in self.arms.values()
            ],
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    @property
    def total_pulls(self) -> int:
        return sum(a.pulls for a in self.arms.values())

    def select_dimension(self) -> str:
        """Select the dimension with highest UCB1 score."""
        total = max(self.total_pulls, 1)
        best_dim = None
        best_score = -float("inf")

        for name, arm in self.arms.items():
            score = arm.ucb1_score(total, self.exploration_c)
            if score > best_score:
                best_score = score
                best_dim = name

        return best_dim

    def record_outcome(self, dimension: str, oos_delta: float, params: dict = None):
        """Record the outcome of a meta-iteration.

        oos_delta: change in OOS Sharpe compared to baseline (positive = improvement).
        params: optional dict of specific parameter values tried for this dimension.
        """
        if dimension not in self.arms:
            self.arms[dimension] = ArmState(name=dimension)

        arm = self.arms[dimension]
        arm.pulls += 1
        arm.total_reward += oos_delta
        arm.rewards.append(oos_delta)
        if params is not None:
            arm.parameter_history.append({"params": params, "reward": oos_delta})
        self.save()

    def get_best_params(self, dimension: str) -> dict:
        """Get the parameter values that produced the highest reward for this dimension."""
        if dimension not in self.arms:
            return {}
        arm = self.arms[dimension]
        if not arm.parameter_history:
            return {}
        best = max(arm.parameter_history, key=lambda x: x.get("reward", -999))
        return best.get("params", {})

    def get_stats(self) -> dict:
        """Get stats for all arms."""
        total = max(self.total_pulls, 1)
        stats = {}
        for name, arm in self.arms.items():
            stats[name] = {
                "pulls": arm.pulls,
                "mean_reward": round(arm.mean_reward, 4),
                "ucb1_score": round(arm.ucb1_score(total, self.exploration_c), 4),
                "total_reward": round(arm.total_reward, 4),
                "best_single": round(max(arm.rewards), 4) if arm.rewards else 0,
                "worst_single": round(min(arm.rewards), 4) if arm.rewards else 0,
            }
        return stats

    def get_allocation_summary(self) -> str:
        """Human-readable allocation summary."""
        total = max(self.total_pulls, 1)
        lines = [f"MAB Scheduler — {total} total pulls, c={self.exploration_c}\n"]

        stats = self.get_stats()
        for name in DIMENSIONS:
            s = stats.get(name, {})
            pct = s.get("pulls", 0) / total * 100 if total > 0 else 0
            lines.append(
                f"  {name:12s}: {s.get('pulls', 0):3d} pulls ({pct:5.1f}%) | "
                f"mean={s.get('mean_reward', 0):+.4f} | "
                f"UCB1={s.get('ucb1_score', 0):.4f}"
            )

        next_dim = self.select_dimension()
        lines.append(f"\n  Next dimension: {next_dim}")
        return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MAB Scheduler for meta-harness")
    parser.add_argument("--stats", action="store_true", help="Show MAB statistics")
    parser.add_argument("--select", action="store_true", help="Select next dimension")
    parser.add_argument("--record", nargs=2, metavar=("DIM", "DELTA"),
                        help="Record outcome: dimension and OOS delta")
    args = parser.parse_args()

    mab = MABScheduler()

    if args.record:
        dim, delta = args.record[0], float(args.record[1])
        mab.record_outcome(dim, delta)
        print(f"Recorded: {dim} → {delta:+.4f}")
        print(mab.get_allocation_summary())

    elif args.select:
        dim = mab.select_dimension()
        print(f"Next dimension: {dim}")

    elif args.stats:
        print(mab.get_allocation_summary())

    else:
        print(mab.get_allocation_summary())
