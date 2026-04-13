"""
meta_config.py — Meta-level configuration for the outer harness loop.

Controls the scoring formula, normalization constants, train/val split,
feature pipeline, and inner loop parameters. The inner loop (autoresearch)
never sees or modifies this — it only experiences the effects through
prepare_rv.py's parameterized evaluation.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# OOS periods are NEVER modified — these are the ground truth for meta evaluation
OOS_PERIOD_1 = ("2021-01-01", "2023-01-31")  # 515 days (early, high-IV)
OOS_PERIOD_2 = ("2026-02-01", "2026-03-23")  # recent (Feb-Mar 2026)


@dataclass
class MetaConfig:
    """Meta-level configuration that controls the inner loop's evaluation frame."""

    # ── Scoring function weights (must sum to 1.0) ──
    w_sharpe: float = 0.40
    w_safe_sep: float = 0.25
    w_rank_corr: float = 0.25         # increased from 0.20 — rewards generalization
    w_coverage: float = 0.10          # decreased from 0.15 — baseline always hits 1.0

    # ── Scoring normalization constants ──
    sharpe_norm: float = 5.0          # val_sharpe / sharpe_norm
    sharpe_cap: float = 1.0           # lowered from 1.5 — prevents saturation at sharpe=5
    safe_sep_norm: float = 15.0       # raised from 10.0 — baseline 13.82 was saturating at 1.0
    safe_sep_cap: float = 1.0         # max contribution from safe_sep term

    # ── Coverage parameters ──
    min_state_days: int = 5           # days below this penalize coverage
    min_states_used: int = 4          # lowered from 6 — allows fewer-state classifiers

    # ── Reproducibility ──
    random_seed: int = 42             # seed for all random operations

    # ── Train/Validation periods ──
    train_start: str = "2023-02-01"
    train_end: str = "2025-06-30"
    val_start: str = "2025-07-01"
    val_end: str = "2026-01-30"

    # ── Feature pipeline expansion ──
    # Names of additional features to compute beyond the baseline set.
    # Available: IV_20d, PK_20d, IV_momentum_5d, VRP_5d, IV_range_10d, RV_IV_gap
    extra_features: list[str] = field(default_factory=list)

    # ── Strategy co-optimization (Upgrade 4) ──
    # When set, these weights are injected into regime_experiment.py before
    # the inner loop starts, and the inner loop is told NOT to modify them.
    # Format: {"L1 Safe": [dm, wc, orion], ...} or empty dict for no override.
    strategy_weights_override: dict = field(default_factory=dict)
    strategy_lock: bool = False  # If True, inner loop cannot modify strategy weights

    # ── Inner loop parameters ──
    max_inner_experiments: int = 40
    inner_timeout_sec: int = 3600   # 60 minutes max per inner loop
    inner_budget_usd: float = 5.0   # API budget cap for inner loop

    # ── Description (for logging) ──
    description: str = ""

    def validate(self) -> list[str]:
        """Return list of validation errors, empty if valid."""
        errors = []
        weight_sum = self.w_sharpe + self.w_safe_sep + self.w_rank_corr + self.w_coverage
        if abs(weight_sum - 1.0) > 0.001:
            errors.append(f"Scoring weights sum to {weight_sum:.3f}, expected 1.0")
        if self.sharpe_norm <= 0:
            errors.append(f"sharpe_norm must be positive, got {self.sharpe_norm}")
        if self.safe_sep_norm <= 0:
            errors.append(f"safe_sep_norm must be positive, got {self.safe_sep_norm}")
        if self.max_inner_experiments < 5:
            errors.append(f"max_inner_experiments too low: {self.max_inner_experiments}")
        return errors

    def to_json(self, path: str | Path) -> None:
        """Serialize to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "MetaConfig":
        """Deserialize from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def summary(self) -> str:
        """One-line summary for logging."""
        return (
            f"w=[{self.w_sharpe:.2f},{self.w_safe_sep:.2f},{self.w_rank_corr:.2f},{self.w_coverage:.2f}] "
            f"norm=[{self.sharpe_norm:.1f},{self.safe_sep_norm:.1f}] "
            f"train={self.train_start}..{self.train_end} "
            f"val={self.val_start}..{self.val_end} "
            f"features={self.extra_features or 'baseline'}"
        )


# Convenience: load from the autoresearch root (not core/)
META_CONFIG_FILE = Path(__file__).parent.parent / "meta_config.json"


def load_active_config() -> Optional[MetaConfig]:
    """Load meta config if meta_config.json exists, else return None."""
    if META_CONFIG_FILE.exists():
        return MetaConfig.from_json(META_CONFIG_FILE)
    return None
