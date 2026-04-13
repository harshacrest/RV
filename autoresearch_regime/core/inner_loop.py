"""
inner_loop.py — Bounded inner loop runner for meta-harness.

Runs a single inner-loop autoresearch iteration:
1. Writes meta_config.json with the provided config
2. Resets regime_experiment.py to baseline
3. Runs a Claude subprocess for N experiments
4. Parses results and evaluates on OOS

Can be run standalone for testing:
    python inner_loop.py --config meta_configs/iter_001.json
    python inner_loop.py --baseline  # just evaluate current baseline on OOS
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from paths import CORE_DIR, AUTORESEARCH_DIR  # noqa: F401 — sets up sys.path
SCRIPT_DIR = AUTORESEARCH_DIR  # file I/O paths point to autoresearch root

from meta_config import MetaConfig


@dataclass
class InnerLoopResult:
    """Result from a single inner loop run."""
    iteration: int
    best_composite: float
    best_val_sharpe: float
    best_safe_sep: float
    best_rank_stability: float
    best_commit: str
    inner_experiments: int
    oos1_sharpe: float
    oos1_al_pct: Optional[float]
    oos1_days: int
    oos2_sharpe: float
    oos2_al_pct: Optional[float]
    oos2_days: int
    elapsed_sec: float
    status: str  # "ok", "timeout", "error", "no_improvement"

    def summary(self) -> str:
        return (
            f"val={self.best_composite:.4f} "
            f"oos1={self.oos1_sharpe:.2f} oos2={self.oos2_sharpe:.2f} "
            f"exps={self.inner_experiments} "
            f"status={self.status}"
        )


def _parse_results_tsv(results_file: Path) -> list[dict]:
    """Parse results.tsv and return list of experiment dicts."""
    if not results_file.exists():
        return []

    rows = []
    with open(results_file) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                row["composite_score"] = float(row.get("composite_score", 0))
                rows.append(row)
            except (ValueError, KeyError):
                continue
    return rows


def _find_best_result(results_file: Path) -> Optional[dict]:
    """Find the row with highest composite_score in results.tsv."""
    rows = _parse_results_tsv(results_file)
    if not rows:
        return None
    return max(rows, key=lambda r: r["composite_score"])


def _run_oos_evaluation(config_file: Optional[Path] = None) -> dict:
    """Run OOS evaluation on the current regime_experiment.py state.

    Returns dict with oos1 and oos2 results.
    """
    # Ensure meta_config.json is in place if provided
    meta_json = SCRIPT_DIR / "meta_config.json"
    if config_file and config_file.exists():
        shutil.copy(config_file, meta_json)

    # Import fresh (force re-import to pick up meta_config.json)
    import importlib

    # Clear cached modules to force re-read of meta_config.json
    for mod_name in list(sys.modules.keys()):
        if mod_name in ("prepare_rv", "prepare_rv_meta", "meta_config"):
            del sys.modules[mod_name]

    from prepare_rv import load_data, evaluate_oos
    from regime_experiment import run_classification

    df = load_data("1530")
    df = run_classification(df)

    oos1 = evaluate_oos(df, period="oos1")
    oos2 = evaluate_oos(df, period="oos2")

    return {"oos1": oos1, "oos2": oos2}


def _build_inner_prompt(config: MetaConfig) -> str:
    """Build the prompt for the inner loop Claude subprocess.

    Includes knowledge forest summary (Upgrade 1) so the agent
    has memory of prior experiments across meta-iterations.
    """
    # Read the base program spec
    program_file = SCRIPT_DIR / "prompts" / "program_regime.md"
    if program_file.exists():
        program_text = program_file.read_text()
    else:
        program_text = "Optimize regime_experiment.py to maximize composite_score."

    extra_features_note = ""
    if config.extra_features:
        extra_features_note = (
            f"\n\nNOTE: The following additional features are available in the DataFrame "
            f"(computed by prepare_rv.py via meta-config): {', '.join(config.extra_features)}. "
            f"You can use these in SPLIT_FEATURE_*, L2_DIRECTION_FEATURE, or compute_extra_features()."
        )

    # ── Knowledge Forest injection (Upgrade 1) ──
    forest_note = ""
    try:
        from forest_manager import ForestManager
        fm = ForestManager()
        if fm.total_experiments > 0:
            forest_note = (
                "\n\n" + "=" * 60 + "\n"
                "EXPERIMENT HISTORY FROM PRIOR RUNS\n"
                "Read carefully — do NOT retry known failures.\n"
                + "=" * 60 + "\n\n"
                + fm.get_summary_for_prompt(max_entries=30)
            )
    except Exception:
        pass

    # ── Strategy lock note (Upgrade 4: Co-optimization) ──
    strategy_note = ""
    if config.strategy_lock and config.strategy_weights_override:
        strategy_note = (
            "\n\nSTRATEGY WEIGHTS ARE LOCKED BY META-HARNESS.\n"
            "Do NOT modify STRATEGY_WEIGHTS in regime_experiment.py.\n"
            "Focus on optimizing: IV boundaries, feature selection, classification architecture, "
            "direction thresholds, and threshold methods.\n"
            "The outer loop is testing different strategy allocations — your job is to find "
            "the best classifier that works with these fixed weights."
        )

    n = config.max_inner_experiments
    budget_guide = f"""
BUDGET ALLOCATION GUIDANCE ({n} experiments):
- Experiments 1-{max(1, n//8)}: Establish baseline, verify current score
- Experiments {max(2, n//8+1)}-{n//3}: Boundary variations (L1 in [8.0-9.5], L2 in [10.5-12.0])
- Experiments {n//3+1}-{2*n//3}: Feature engineering + architecture (split features, direction)
- Experiments {2*n//3+1}-{5*n//6}: Strategy weight optimization per state
- Experiments {5*n//6+1}-{n}: Fine-tune best configuration from above
DO NOT spend more than 5 experiments on any single dimension without an improvement.
"""

    prompt = f"""You are running a BOUNDED autoresearch session. Follow the program below.

CRITICAL: You must stop after {n} experiments maximum.
After completing your experiments, output "INNER_LOOP_COMPLETE" as the last line.

{program_text}
{budget_guide}
{extra_features_note}
{strategy_note}
{forest_note}

BEGIN. Run the experiment loop. Stop after {n} experiments.
"""
    return prompt


def run_inner_loop(
    config: MetaConfig,
    iteration: int,
    dry_run: bool = False,
) -> InnerLoopResult:
    """Run one complete inner loop iteration.

    1. Write meta_config.json
    2. Reset regime_experiment.py from baseline
    3. Run Claude subprocess
    4. Parse results
    5. Evaluate OOS
    """
    t0 = time.time()
    meta_json = SCRIPT_DIR / "meta_config.json"
    results_file = SCRIPT_DIR / "results.tsv"
    baseline_file = SCRIPT_DIR / "core" / "regime_experiment_baseline.py"
    experiment_file = SCRIPT_DIR / "core" / "regime_experiment.py"

    print(f"\n{'='*60}")
    print(f"INNER LOOP — Iteration {iteration}")
    print(f"Config: {config.summary()}")
    print(f"{'='*60}")

    # 1. Write meta config
    config.to_json(meta_json)
    print(f"  [1/5] Wrote meta_config.json")

    # 2. Reset regime_experiment.py from baseline
    if baseline_file.exists():
        shutil.copy(baseline_file, experiment_file)
        print(f"  [2/5] Reset regime_experiment.py from baseline")
    else:
        print(f"  [2/5] WARNING: No baseline file found, using current regime_experiment.py")

    # 3. Reset results.tsv
    with open(results_file, "w") as f:
        f.write("commit\tcomposite_score\tval_sharpe\tsafe_sep\trank_stability\tstate_coverage\tstatus\tdescription\n")
    print(f"  [3/5] Reset results.tsv")

    if dry_run:
        print(f"  [DRY RUN] Would run Claude for {config.max_inner_experiments} experiments")
        elapsed = time.time() - t0
        return InnerLoopResult(
            iteration=iteration, best_composite=0, best_val_sharpe=0,
            best_safe_sep=0, best_rank_stability=0, best_commit="dry-run",
            inner_experiments=0, oos1_sharpe=0, oos1_al_pct=None, oos1_days=0,
            oos2_sharpe=0, oos2_al_pct=None, oos2_days=0,
            elapsed_sec=elapsed, status="dry_run",
        )

    # 4. Run the inner loop via Claude subprocess
    prompt = _build_inner_prompt(config)
    print(f"  [4/5] Running Claude inner loop (max {config.max_inner_experiments} experiments, timeout {config.inner_timeout_sec}s)...")

    status = "ok"
    try:
        result = subprocess.run(
            [
                "claude",
                "--print",
                "--dangerously-skip-permissions",
                "--max-turns", str(config.max_inner_experiments * 3),  # ~3 turns per experiment
                "-p", prompt,
            ],
            cwd=str(SCRIPT_DIR / "core"),
            capture_output=True,
            text=True,
            timeout=config.inner_timeout_sec,
        )
        print(f"    Claude exited with code {result.returncode}")
        if result.returncode != 0:
            print(f"    stderr: {result.stderr[:500]}")
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT after {config.inner_timeout_sec}s")
        status = "timeout"
    except FileNotFoundError:
        print(f"    ERROR: 'claude' command not found. Install Claude CLI.")
        status = "error"
    except Exception as e:
        print(f"    ERROR: {e}")
        status = "error"

    # 5. Parse results
    best = _find_best_result(results_file)
    if best:
        inner_experiments = len(_parse_results_tsv(results_file))
        best_composite = best["composite_score"]
        best_commit = best.get("commit", "unknown")
        best_val_sharpe = float(best.get("val_sharpe", 0))
        best_safe_sep = float(best.get("safe_sep", 0))
        best_rank_stability = float(best.get("rank_stability", 0))
        print(f"    Best composite: {best_composite:.6f} ({inner_experiments} experiments)")
    else:
        # No results — run current state evaluation
        inner_experiments = 0
        best_composite = 0
        best_commit = "none"
        best_val_sharpe = 0
        best_safe_sep = 0
        best_rank_stability = 0
        if status == "ok":
            status = "no_improvement"
        print(f"    No results found in results.tsv")

    # 6. Evaluate on OOS
    print(f"  [5/5] Running OOS evaluation...")
    try:
        oos = _run_oos_evaluation(meta_json if meta_json.exists() else None)
        oos1 = oos["oos1"]
        oos2 = oos["oos2"]
        print(f"    OOS1: sharpe={oos1.get('sharpe', 0):.2f}, AL={oos1.get('al_pct', '?')}%, days={oos1.get('days', 0)}")
        print(f"    OOS2: sharpe={oos2.get('sharpe', 0):.2f}, AL={oos2.get('al_pct', '?')}%, days={oos2.get('days', 0)}")
    except Exception as e:
        print(f"    OOS evaluation failed: {e}")
        oos1 = {"sharpe": 0, "al_pct": None, "days": 0}
        oos2 = {"sharpe": 0, "al_pct": None, "days": 0}

    # Clean up meta_config.json
    if meta_json.exists():
        meta_json.unlink()

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s")

    return InnerLoopResult(
        iteration=iteration,
        best_composite=best_composite,
        best_val_sharpe=best_val_sharpe,
        best_safe_sep=best_safe_sep,
        best_rank_stability=best_rank_stability,
        best_commit=best_commit,
        inner_experiments=inner_experiments,
        oos1_sharpe=oos1.get("sharpe", 0),
        oos1_al_pct=oos1.get("al_pct"),
        oos1_days=oos1.get("days", 0),
        oos2_sharpe=oos2.get("sharpe", 0),
        oos2_al_pct=oos2.get("al_pct"),
        oos2_days=oos2.get("days", 0),
        elapsed_sec=elapsed,
        status=status,
    )


def evaluate_baseline_oos() -> dict:
    """Quick standalone: evaluate the current baseline on OOS (no inner loop)."""
    print("Evaluating current baseline on OOS periods...")
    meta_json = SCRIPT_DIR / "meta_config.json"

    # Remove meta config to use original scoring
    existed = meta_json.exists()
    if existed:
        backup = meta_json.read_text()
        meta_json.unlink()

    try:
        oos = _run_oos_evaluation()
    finally:
        if existed:
            meta_json.write_text(backup)

    print(f"OOS1: {json.dumps(oos['oos1'], indent=2)}")
    print(f"OOS2: {json.dumps(oos['oos2'], indent=2)}")
    return oos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inner loop runner for meta-harness")
    parser.add_argument("--config", type=str, help="Path to meta config JSON")
    parser.add_argument("--iteration", type=int, default=1, help="Iteration number")
    parser.add_argument("--dry-run", action="store_true", help="Don't run Claude, just setup")
    parser.add_argument("--baseline", action="store_true", help="Just evaluate baseline on OOS")
    args = parser.parse_args()

    if args.baseline:
        evaluate_baseline_oos()
    elif args.config:
        config = MetaConfig.from_json(args.config)
        result = run_inner_loop(config, args.iteration, dry_run=args.dry_run)
        print(f"\nResult: {result.summary()}")
    else:
        # Default: run with default config
        config = MetaConfig(description="default")
        result = run_inner_loop(config, args.iteration, dry_run=args.dry_run)
        print(f"\nResult: {result.summary()}")
