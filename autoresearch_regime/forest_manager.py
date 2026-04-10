"""
forest_manager.py — Upgrade 1: Knowledge Forest.

Persistent experiment memory across all meta-iterations. Every experiment
the inner loop runs gets logged with hypothesis, result, keep/revert status,
and reason. Both the inner loop and meta-harness read this before proposing
new changes, preventing re-discovery of known failures.

Usage:
    # Programmatic
    from forest_manager import ForestManager
    fm = ForestManager()
    fm.add_experiment(...)
    relevant = fm.query("boundary", top_k=10)
    summary = fm.get_summary_for_prompt(max_tokens=2000)

    # CLI
    python forest_manager.py --query "boundary" --top 10
    python forest_manager.py --stats
    python forest_manager.py --summary
"""

import json
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).parent
FOREST_FILE = SCRIPT_DIR / "experiment_forest.json"


@dataclass
class ExperimentEntry:
    """A single experiment record in the knowledge forest."""
    # Identity
    id: str                     # e.g. "meta-003/exp-042"
    meta_iteration: int         # which outer loop iteration
    inner_experiment: int       # which inner loop experiment number
    timestamp: str              # ISO timestamp

    # What was tried
    hypothesis: str             # one-line description of what change was proposed
    dimension: str              # "boundary", "feature", "weight", "architecture", "direction", "meta-scoring", etc.
    parameter_changes: dict     # specific parameters changed, e.g. {"IV_L1_UPPER": 8.5}

    # What happened
    composite_score: float
    val_sharpe: float
    safe_separation: float
    rank_stability: float
    kept: bool                  # was this experiment kept or reverted?

    # Why
    reason: str                 # one-line explanation of why it helped or hurt

    # Meta context
    meta_config_id: str = ""    # which meta config was active
    oos1_sharpe: Optional[float] = None  # only filled after meta-iteration completes
    oos2_sharpe: Optional[float] = None


class ForestManager:
    """Read/write/query the knowledge forest."""

    def __init__(self, path: Path = FOREST_FILE):
        self.path = path
        self._entries: list[dict] = []
        self._load()

    def _load(self):
        """Load forest from disk."""
        if self.path.exists():
            with open(self.path) as f:
                data = json.load(f)
            self._entries = data.get("experiments", [])
        else:
            self._entries = []

    def _save(self):
        """Save forest to disk."""
        data = {
            "version": 1,
            "updated_at": datetime.now().isoformat(),
            "total_experiments": len(self._entries),
            "experiments": self._entries,
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def add_experiment(self, entry: ExperimentEntry):
        """Add a new experiment to the forest."""
        self._entries.append(asdict(entry))
        self._save()

    def add_from_dict(self, d: dict):
        """Add experiment from a raw dict (flexible)."""
        self._entries.append(d)
        self._save()

    def update_oos_for_iteration(self, meta_iteration: int, oos1_sharpe: float, oos2_sharpe: float):
        """Backfill OOS results for all experiments in a meta-iteration."""
        for e in self._entries:
            if e.get("meta_iteration") == meta_iteration:
                e["oos1_sharpe"] = oos1_sharpe
                e["oos2_sharpe"] = oos2_sharpe
        self._save()

    def query(self, keyword: str, top_k: int = 10) -> list[dict]:
        """Find experiments matching a keyword in hypothesis, dimension, or reason."""
        kw = keyword.lower()
        scored = []
        for e in self._entries:
            text = f"{e.get('hypothesis', '')} {e.get('dimension', '')} {e.get('reason', '')}".lower()
            if kw in text:
                scored.append(e)
        # Sort by recency (latest first)
        scored.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return scored[:top_k]

    def get_failures(self, dimension: str = None, top_k: int = 20) -> list[dict]:
        """Get recent failed experiments (reverted), optionally filtered by dimension."""
        failures = [e for e in self._entries if not e.get("kept", True)]
        if dimension:
            failures = [e for e in failures if e.get("dimension", "") == dimension]
        failures.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return failures[:top_k]

    def get_successes(self, top_k: int = 20) -> list[dict]:
        """Get recent successful experiments (kept)."""
        successes = [e for e in self._entries if e.get("kept", False)]
        successes.sort(key=lambda x: x.get("composite_score", 0), reverse=True)
        return successes[:top_k]

    def get_sota(self) -> Optional[dict]:
        """Get the single best experiment by composite score."""
        if not self._entries:
            return None
        kept = [e for e in self._entries if e.get("kept", False)]
        if not kept:
            return max(self._entries, key=lambda x: x.get("composite_score", 0))
        return max(kept, key=lambda x: x.get("composite_score", 0))

    def get_dimension_stats(self) -> dict:
        """Get success rate and avg score by dimension."""
        dims = {}
        for e in self._entries:
            d = e.get("dimension", "unknown")
            if d not in dims:
                dims[d] = {"total": 0, "kept": 0, "scores": []}
            dims[d]["total"] += 1
            if e.get("kept", False):
                dims[d]["kept"] += 1
            dims[d]["scores"].append(e.get("composite_score", 0))

        stats = {}
        for d, info in dims.items():
            avg_score = sum(info["scores"]) / len(info["scores"]) if info["scores"] else 0
            stats[d] = {
                "total": info["total"],
                "kept": info["kept"],
                "success_rate": info["kept"] / info["total"] if info["total"] > 0 else 0,
                "avg_score": round(avg_score, 4),
            }
        return stats

    def get_summary_for_prompt(self, max_entries: int = 30) -> str:
        """Generate a compact summary for injection into the inner loop prompt.

        This is the key function — it gives the agent memory of past experiments
        without overwhelming context.
        """
        lines = ["## Experiment History (Knowledge Forest)\n"]

        # SOTA
        sota = self.get_sota()
        if sota:
            lines.append(f"**Current best**: composite={sota.get('composite_score', 0):.4f}, "
                         f"hypothesis: {sota.get('hypothesis', '?')}")
            if sota.get("oos1_sharpe") is not None:
                lines.append(f"  OOS1 Sharpe: {sota['oos1_sharpe']:.2f}, OOS2 Sharpe: {sota.get('oos2_sharpe', 0):.2f}")
            lines.append("")

        # Dimension stats
        dim_stats = self.get_dimension_stats()
        if dim_stats:
            lines.append("**Success rates by dimension:**")
            for d, s in sorted(dim_stats.items(), key=lambda x: -x[1]["success_rate"]):
                lines.append(f"  {d}: {s['kept']}/{s['total']} kept ({s['success_rate']:.0%}), avg={s['avg_score']:.4f}")
            lines.append("")

        # Recent failures (DON'T retry these)
        failures = self.get_failures(top_k=15)
        if failures:
            lines.append("**Recent failures (DO NOT retry):**")
            for f in failures[:15]:
                lines.append(f"  - {f.get('hypothesis', '?')} → {f.get('reason', '?')}")
            lines.append("")

        # Recent successes (build on these)
        successes = self.get_successes(top_k=10)
        if successes:
            lines.append("**Successful approaches (build on these):**")
            for s in successes[:10]:
                lines.append(f"  - {s.get('hypothesis', '?')} → score={s.get('composite_score', 0):.4f}")
            lines.append("")

        return "\n".join(lines)

    @property
    def total_experiments(self) -> int:
        return len(self._entries)

    def __len__(self) -> int:
        return len(self._entries)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Knowledge Forest manager")
    parser.add_argument("--query", type=str, help="Search experiments by keyword")
    parser.add_argument("--top", type=int, default=10, help="Max results")
    parser.add_argument("--stats", action="store_true", help="Show dimension statistics")
    parser.add_argument("--summary", action="store_true", help="Show prompt summary")
    parser.add_argument("--failures", action="store_true", help="Show recent failures")
    args = parser.parse_args()

    fm = ForestManager()
    print(f"Knowledge Forest: {fm.total_experiments} experiments\n")

    if args.query:
        results = fm.query(args.query, top_k=args.top)
        for r in results:
            kept = "KEPT" if r.get("kept") else "REVERTED"
            print(f"  [{kept}] {r.get('hypothesis', '?')} → score={r.get('composite_score', 0):.4f} ({r.get('reason', '')})")

    elif args.stats:
        stats = fm.get_dimension_stats()
        for d, s in sorted(stats.items(), key=lambda x: -x[1]["total"]):
            print(f"  {d:20s}: {s['kept']}/{s['total']} kept ({s['success_rate']:.0%}), avg={s['avg_score']:.4f}")

    elif args.summary:
        print(fm.get_summary_for_prompt())

    elif args.failures:
        failures = fm.get_failures(top_k=args.top)
        for f in failures:
            print(f"  - {f.get('hypothesis', '?')} → {f.get('reason', '?')}")

    else:
        sota = fm.get_sota()
        if sota:
            print(f"SOTA: {sota.get('composite_score', 0):.4f} — {sota.get('hypothesis', '?')}")
        else:
            print("No experiments yet.")
