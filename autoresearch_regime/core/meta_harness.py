"""
meta_harness.py — Outer loop orchestrator for meta-harness research.

Upgraded with:
- Knowledge Forest (persistent experiment memory)
- MAB Scheduler (adaptive dimension allocation)
- IC Deduplication Gate (redundant feature filtering)
- Paper Trade Gate (forward-walk validation)

Usage:
    python meta_harness.py --adaptive --budget 20          # MAB-driven adaptive sweep
    python meta_harness.py --sweep scoring --dry-run       # Preview scoring sweep
    python meta_harness.py --sweep scoring                 # Run scoring sweep
    python meta_harness.py --sweep features                # Run feature sweep
    python meta_harness.py --sweep splits                  # Run split sweep
    python meta_harness.py --config meta_configs/custom.json
    python meta_harness.py --baseline                      # Evaluate current baseline on OOS
    python meta_harness.py --paper-trade --days 10         # Run paper trade gate
"""

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

from paths import CORE_DIR, AUTORESEARCH_DIR  # noqa: F401 — sets up sys.path
SCRIPT_DIR = AUTORESEARCH_DIR  # file I/O at autoresearch root

from meta_config import MetaConfig
from inner_loop import run_inner_loop, evaluate_baseline_oos, InnerLoopResult
from forest_manager import ForestManager
from mab_scheduler import MABScheduler, DIMENSIONS
from paper_trade_gate import run_paper_trade_gate
from strategy_coopt import (
    generate_strategy_config, inject_strategy_weights,
    format_weights_summary, STRATEGY_TEMPLATES, REGIME_STATES,
)


META_RESULTS_FILE = SCRIPT_DIR / "meta_results.tsv"
META_CONFIGS_DIR = SCRIPT_DIR / "meta_configs"

# Baseline OOS Sharpe for computing MAB deltas
_BASELINE_OOS1_SHARPE = None  # set on first iteration


class MetaHarness:
    """Outer loop orchestrator with MAB, forest, IC dedup, and paper trade gate."""

    def __init__(self):
        META_CONFIGS_DIR.mkdir(exist_ok=True)
        self._ensure_results_file()
        self.forest = ForestManager()
        self.mab = MABScheduler()

    def _ensure_results_file(self):
        """Create meta_results.tsv with header if it doesn't exist."""
        if not META_RESULTS_FILE.exists():
            with open(META_RESULTS_FILE, "w") as f:
                f.write(
                    "iter\tdescription\tdimension\tbest_val\tval_sharpe\tsafe_sep\trank_stability\t"
                    "oos1_sharpe\toos1_al\toos1_days\t"
                    "oos2_sharpe\toos2_al\toos2_days\t"
                    "inner_exps\telapsed_sec\tstatus\tpaper_trade\tconfig_file\n"
                )

    def next_iteration(self) -> int:
        """Get the next iteration number."""
        if not META_RESULTS_FILE.exists():
            return 1
        with open(META_RESULTS_FILE) as f:
            lines = f.readlines()
        return len(lines)

    def log_result(self, result: InnerLoopResult, config: MetaConfig,
                   config_file: str, dimension: str = "", paper_trade: str = ""):
        """Append result to meta_results.tsv."""
        with open(META_RESULTS_FILE, "a") as f:
            f.write(
                f"{result.iteration}\t"
                f"{config.description}\t"
                f"{dimension}\t"
                f"{result.best_composite:.6f}\t"
                f"{result.best_val_sharpe:.4f}\t"
                f"{result.best_safe_sep:.2f}\t"
                f"{result.best_rank_stability:.4f}\t"
                f"{result.oos1_sharpe:.4f}\t"
                f"{result.oos1_al_pct}\t"
                f"{result.oos1_days}\t"
                f"{result.oos2_sharpe:.4f}\t"
                f"{result.oos2_al_pct}\t"
                f"{result.oos2_days}\t"
                f"{result.inner_experiments}\t"
                f"{result.elapsed_sec:.1f}\t"
                f"{result.status}\t"
                f"{paper_trade}\t"
                f"{config_file}\n"
            )

    def run_iteration(
        self,
        config: MetaConfig,
        dimension: str = "",
        iteration: int = 0,
        dry_run: bool = False,
        run_paper_trade: bool = False,
    ) -> InnerLoopResult:
        """Run one meta-harness iteration with all upgrades."""
        if iteration == 0:
            iteration = self.next_iteration()

        # Seed RNG for reproducibility
        import numpy as np
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)

        # Save config snapshot
        config_file = META_CONFIGS_DIR / f"iter_{iteration:03d}.json"
        config.to_json(config_file)

        # Validate
        errors = config.validate()
        if errors:
            print(f"CONFIG ERRORS: {errors}")
            raise ValueError(f"Invalid config: {errors}")

        # ── IC Deduplication Gate (Upgrade 3) ──
        if config.extra_features and not dry_run:
            from ic_dedup import deduplicate_features
            # Need data to compute correlations — quick load
            for mod in list(sys.modules.keys()):
                if mod in ("prepare_rv", "prepare_rv_meta", "meta_config"):
                    del sys.modules[mod]
            # Write temp config to load features
            config.to_json(SCRIPT_DIR / "meta_config.json")
            from prepare_rv import load_data
            df_temp = load_data("1530")
            kept, dropped = deduplicate_features(
                df_temp, config.extra_features,
                threshold=0.75,
                train_start=config.train_start,
                train_end=config.train_end,
            )
            if dropped:
                print(f"  [IC DEDUP] Dropped {len(dropped)} redundant features: "
                      f"{[d['feature'] for d in dropped]}")
                config.extra_features = kept
                config.to_json(config_file)  # update snapshot
            # Clean up temp
            temp = SCRIPT_DIR / "meta_config.json"
            if temp.exists():
                temp.unlink()
            del df_temp

        # ── Strategy Co-Optimization (Upgrade 4) ──
        if config.strategy_weights_override and not dry_run:
            experiment_file = SCRIPT_DIR / "regime_experiment.py"
            inject_strategy_weights(
                experiment_file,
                config.strategy_weights_override,
                lock_weights=config.strategy_lock,
            )
            print(f"  [STRATEGY] Injected weights: {format_weights_summary(config.strategy_weights_override)}")
            if config.strategy_lock:
                print(f"  [STRATEGY] Weights LOCKED — inner loop will optimize classifier only")

        # ── Run inner loop ──
        result = run_inner_loop(config, iteration, dry_run=dry_run)

        # ── Update knowledge forest (Upgrade 1) ──
        if not dry_run:
            self.forest.update_oos_for_iteration(
                iteration, result.oos1_sharpe, result.oos2_sharpe
            )

        # ── Paper Trade Gate (Upgrade 5) ──
        paper_trade_status = ""
        if run_paper_trade and not dry_run and result.status == "ok":
            print(f"\n  Running paper trade gate...")
            gate = run_paper_trade_gate(days=10, verbose=True)
            paper_trade_status = gate.recommendation

        # ── Record MAB outcome (Upgrade 2) ──
        if dimension and not dry_run:
            global _BASELINE_OOS1_SHARPE
            if _BASELINE_OOS1_SHARPE is None:
                _BASELINE_OOS1_SHARPE = result.oos1_sharpe
                oos_delta = 0.0
            else:
                oos_delta = result.oos1_sharpe - _BASELINE_OOS1_SHARPE
            # Extract key params for within-arm tracking
            dim_params = {}
            if dimension == "scoring":
                dim_params = {"w_sharpe": config.w_sharpe, "w_safe_sep": config.w_safe_sep,
                              "w_rank_corr": config.w_rank_corr, "w_coverage": config.w_coverage}
            elif dimension == "norms":
                dim_params = {"sharpe_norm": config.sharpe_norm, "safe_sep_norm": config.safe_sep_norm}
            elif dimension == "features":
                dim_params = {"extra_features": config.extra_features}
            elif dimension == "splits":
                dim_params = {"train": f"{config.train_start}..{config.train_end}",
                              "val": f"{config.val_start}..{config.val_end}"}
            elif dimension == "strategy":
                dim_params = {"template": config.description}
            self.mab.record_outcome(dimension, oos_delta, params=dim_params)

        # ── Log result ──
        self.log_result(
            result, config,
            str(config_file.relative_to(SCRIPT_DIR)),
            dimension=dimension,
            paper_trade=paper_trade_status,
        )

        return result

    # ════════════════════════════════════════════════════════════
    # Adaptive mode (Upgrade 2: MAB-driven)
    # ════════════════════════════════════════════════════════════

    def run_adaptive(self, budget: int = 20, max_inner: int = 40,
                     dry_run: bool = False, run_paper_trade: bool = False):
        """Run MAB-adaptive meta-harness iterations."""
        print(f"\n{'='*60}")
        print(f"ADAPTIVE META-HARNESS — {budget} iterations, MAB-driven")
        print(f"{'='*60}")
        print(f"Forest: {self.forest.total_experiments} prior experiments")
        print(f"MAB state:\n{self.mab.get_allocation_summary()}\n")

        results = []
        for i in range(1, budget + 1):
            # MAB selects dimension
            dim = self.mab.select_dimension()
            print(f"\n{'─'*50}")
            print(f"Iteration {i}/{budget} — MAB selected: {dim}")
            print(f"{'─'*50}")

            # Generate a config for this dimension
            config = self._generate_config_for_dimension(dim, max_inner)

            result = self.run_iteration(
                config, dimension=dim, dry_run=dry_run,
                run_paper_trade=run_paper_trade,
            )
            results.append((dim, config.description, result))

            if not dry_run:
                print(f"  → {result.summary()}")
                print(f"\nMAB state after iteration {i}:")
                print(self.mab.get_allocation_summary())

        if not dry_run:
            self._print_adaptive_summary(results)
        return results

    def _generate_config_for_dimension(self, dim: str, max_inner: int) -> MetaConfig:
        """Generate a random config variation for the given dimension."""
        config = MetaConfig(max_inner_experiments=max_inner)

        if dim == "scoring":
            # Random weight perturbation
            weights = [0.40, 0.25, 0.20, 0.15]
            perturbation = [random.uniform(-0.10, 0.10) for _ in weights]
            weights = [max(0.05, w + p) for w, p in zip(weights, perturbation)]
            total = sum(weights)
            weights = [w / total for w in weights]  # normalize to 1.0
            config.w_sharpe, config.w_safe_sep, config.w_rank_corr, config.w_coverage = weights
            config.description = (
                f"scoring:w=[{config.w_sharpe:.2f},{config.w_safe_sep:.2f},"
                f"{config.w_rank_corr:.2f},{config.w_coverage:.2f}]"
            )

        elif dim == "norms":
            config.sharpe_norm = random.choice([3.0, 4.0, 5.0, 6.0, 7.0])
            config.safe_sep_norm = random.choice([6.0, 8.0, 10.0, 12.0, 15.0])
            config.description = f"norm:sn={config.sharpe_norm},ssn={config.safe_sep_norm}"

        elif dim == "features":
            all_features = [
                "IV_20d", "PK_20d", "IV_momentum_5d", "VRP_5d",
                "IV_range_10d", "RV_IV_gap", "PK_IV_zscore_60d", "IV_vol_of_vol_20d",
            ]
            k = random.randint(1, 4)
            config.extra_features = random.sample(all_features, k)
            config.description = f"features:{'+'.join(config.extra_features)}"

        elif dim == "splits":
            splits = [
                ("2023-02-01", "2025-06-30", "2025-07-01", "2026-01-30"),
                ("2022-06-01", "2025-03-31", "2025-04-01", "2026-01-30"),
                ("2023-06-01", "2025-09-30", "2025-10-01", "2026-01-30"),
                ("2023-02-01", "2025-03-31", "2025-04-01", "2025-12-31"),
                ("2022-01-01", "2025-06-30", "2025-07-01", "2026-01-30"),
            ]
            ts, te, vs, ve = random.choice(splits)
            config.train_start, config.train_end = ts, te
            config.val_start, config.val_end = vs, ve
            config.description = f"split:train={ts}..{te},val={vs}..{ve}"

        elif dim == "strategy":
            # Upgrade 4: Co-optimization — fix strategy weights, let inner loop optimize classifier
            templates = list(STRATEGY_TEMPLATES.keys())
            tmpl = random.choice(templates)
            weights = generate_strategy_config(template=tmpl, perturbation=0.1)
            config.strategy_weights_override = weights
            config.strategy_lock = True
            config.description = f"strategy:{tmpl}+perturb"

        return config

    def _print_adaptive_summary(self, results: list):
        """Print summary of adaptive run."""
        print(f"\n{'='*80}")
        print(f"ADAPTIVE SWEEP SUMMARY — {len(results)} iterations")
        print(f"{'='*80}")
        print(f"{'Dim':<12s} {'Description':<35s} {'Val':<8s} {'OOS1':<8s} {'OOS2':<8s}")
        print(f"{'-'*80}")
        for dim, desc, result in results:
            print(
                f"{dim:<12s} "
                f"{desc[:34]:<35s} "
                f"{result.best_composite:<8.4f} "
                f"{result.oos1_sharpe:<8.2f} "
                f"{result.oos2_sharpe:<8.2f}"
            )

        print(f"\nFinal MAB state:")
        print(self.mab.get_allocation_summary())

        print(f"\nKnowledge Forest: {self.forest.total_experiments} total experiments")

    # ════════════════════════════════════════════════════════════
    # Pre-built sweep methods (unchanged from original)
    # ════════════════════════════════════════════════════════════

    def run_scoring_sweep(self, dry_run: bool = False, max_inner: int = 40):
        """Dimension 1: Sweep scoring function weights."""
        configs = [
            (0.40, 0.25, 0.20, 0.15, "baseline-weights"),
            (0.50, 0.20, 0.15, 0.15, "sharpe-heavy-50"),
            (0.30, 0.30, 0.25, 0.15, "separation-heavy"),
            (0.35, 0.25, 0.25, 0.15, "rank-heavy"),
            (0.45, 0.15, 0.25, 0.15, "sharpe-45-rank-25"),
            (0.35, 0.30, 0.20, 0.15, "balanced-sep"),
            (0.40, 0.20, 0.30, 0.10, "rank-30"),
            (0.50, 0.25, 0.15, 0.10, "sharpe-50-sep-25"),
        ]

        print(f"\n{'='*60}")
        print(f"SCORING WEIGHT SWEEP — {len(configs)} configurations")
        print(f"{'='*60}\n")

        results = []
        for i, (ws, wss, wr, wc, desc) in enumerate(configs, 1):
            config = MetaConfig(
                w_sharpe=ws, w_safe_sep=wss, w_rank_corr=wr, w_coverage=wc,
                max_inner_experiments=max_inner,
                description=f"scoring:{desc}",
            )
            print(f"\n--- Config {i}/{len(configs)}: {desc} ---")
            result = self.run_iteration(config, dimension="scoring", dry_run=dry_run)
            results.append((desc, result))
            if not dry_run:
                print(f"  Result: {result.summary()}")

        if not dry_run:
            self._print_sweep_summary(results)
        return results

    def run_normalization_sweep(self, dry_run: bool = False, max_inner: int = 40):
        """Dimension 1b: Sweep normalization constants."""
        configs = [
            (5.0, 1.5, 10.0, 1.0, "baseline-norms"),
            (4.0, 1.5, 10.0, 1.0, "easier-sharpe"),
            (6.0, 1.5, 10.0, 1.0, "harder-sharpe"),
            (5.0, 1.5, 8.0, 1.0, "easier-sep"),
            (5.0, 1.5, 15.0, 1.0, "harder-sep"),
            (3.0, 2.0, 8.0, 1.5, "easy-both"),
        ]

        results = []
        for i, (sn, sc, ssn, ssc, desc) in enumerate(configs, 1):
            config = MetaConfig(
                sharpe_norm=sn, sharpe_cap=sc,
                safe_sep_norm=ssn, safe_sep_cap=ssc,
                max_inner_experiments=max_inner,
                description=f"norm:{desc}",
            )
            result = self.run_iteration(config, dimension="norms", dry_run=dry_run)
            results.append((desc, result))

        if not dry_run:
            self._print_sweep_summary(results)
        return results

    def run_feature_sweep(self, dry_run: bool = False, max_inner: int = 40):
        """Dimension 2: Try expanding the feature pipeline."""
        configs = [
            ([], "no-extra-features"),
            (["IV_20d", "PK_20d"], "20d-rolling"),
            (["IV_momentum_5d", "VRP_5d"], "momentum-vrp"),
            (["IV_range_10d", "RV_IV_gap"], "range-gap"),
            (["PK_IV_zscore_60d"], "zscore-60d"),
            (["IV_vol_of_vol_20d"], "vol-of-vol"),
            (["IV_20d", "PK_20d", "IV_momentum_5d", "VRP_5d", "IV_range_10d"], "full-expansion"),
        ]

        results = []
        for i, (features, desc) in enumerate(configs, 1):
            config = MetaConfig(
                extra_features=features,
                max_inner_experiments=max_inner,
                description=f"features:{desc}",
            )
            result = self.run_iteration(config, dimension="features", dry_run=dry_run)
            results.append((desc, result))

        if not dry_run:
            self._print_sweep_summary(results)
        return results

    def run_split_sweep(self, dry_run: bool = False, max_inner: int = 40):
        """Dimension 4: Try different train/val splits."""
        configs = [
            ("2023-02-01", "2025-06-30", "2025-07-01", "2026-01-30", "baseline-split"),
            ("2022-06-01", "2025-03-31", "2025-04-01", "2026-01-30", "earlier-wider-train"),
            ("2023-06-01", "2025-09-30", "2025-10-01", "2026-01-30", "later-split"),
            ("2023-02-01", "2025-03-31", "2025-04-01", "2025-12-31", "shifted-earlier"),
            ("2022-01-01", "2025-06-30", "2025-07-01", "2026-01-30", "longest-train"),
        ]

        results = []
        for i, (ts, te, vs, ve, desc) in enumerate(configs, 1):
            config = MetaConfig(
                train_start=ts, train_end=te, val_start=vs, val_end=ve,
                max_inner_experiments=max_inner,
                description=f"split:{desc}",
            )
            result = self.run_iteration(config, dimension="splits", dry_run=dry_run)
            results.append((desc, result))

        if not dry_run:
            self._print_sweep_summary(results)
        return results

    def run_strategy_sweep(self, dry_run: bool = False, max_inner: int = 40):
        """Upgrade 4: Sweep strategy weight templates with co-optimization."""
        templates = list(STRATEGY_TEMPLATES.keys())

        print(f"\n{'='*60}")
        print(f"STRATEGY CO-OPTIMIZATION SWEEP — {len(templates)} templates")
        print(f"Inner loop will optimize classifier only (weights locked)")
        print(f"{'='*60}\n")

        results = []
        for i, tmpl_name in enumerate(templates, 1):
            weights = generate_strategy_config(template=tmpl_name, perturbation=0)
            config = MetaConfig(
                strategy_weights_override=weights,
                strategy_lock=True,
                max_inner_experiments=max_inner,
                description=f"strategy:{tmpl_name}",
            )
            print(f"\n--- Template {i}/{len(templates)}: {tmpl_name} ---")
            print(f"  {STRATEGY_TEMPLATES[tmpl_name]['description']}")

            result = self.run_iteration(config, dimension="strategy", dry_run=dry_run)
            results.append((tmpl_name, result))
            if not dry_run:
                print(f"  Result: {result.summary()}")

        if not dry_run:
            self._print_sweep_summary(results)
        return results

    def _print_sweep_summary(self, results: list[tuple[str, InnerLoopResult]]):
        """Print a summary table of sweep results."""
        print(f"\n{'='*80}")
        print(f"SWEEP SUMMARY")
        print(f"{'='*80}")
        print(f"{'Config':<30s} {'Val':<8s} {'OOS1':<8s} {'OOS2':<8s} {'Exps':<6s} {'Status'}")
        print(f"{'-'*80}")
        for desc, result in results:
            print(
                f"{desc:<30s} "
                f"{result.best_composite:<8.4f} "
                f"{result.oos1_sharpe:<8.2f} "
                f"{result.oos2_sharpe:<8.2f} "
                f"{result.inner_experiments:<6d} "
                f"{result.status}"
            )

        best_oos1 = max(results, key=lambda x: x[1].oos1_sharpe)
        best_oos2 = max(results, key=lambda x: x[1].oos2_sharpe)
        print(f"\nBest by OOS1 Sharpe: {best_oos1[0]} ({best_oos1[1].oos1_sharpe:.2f})")
        print(f"Best by OOS2 Sharpe: {best_oos2[0]} ({best_oos2[1].oos2_sharpe:.2f})")

        # Statistical significance warning
        import numpy as np
        oos1_days = [r.oos1_days for _, r in results if r.oos1_days > 0]
        oos2_days = [r.oos2_days for _, r in results if r.oos2_days > 0]
        if oos1_days:
            se1 = np.sqrt(252 / np.mean(oos1_days))
            print(f"\n  NOTE: OOS1 Sharpe SE ≈ {se1:.2f} (improvements < {se1:.1f} may be noise)")
        if oos2_days:
            se2 = np.sqrt(252 / np.mean(oos2_days))
            print(f"  NOTE: OOS2 Sharpe SE ≈ {se2:.2f} (improvements < {se2:.1f} may be noise)")


def main():
    parser = argparse.ArgumentParser(description="Meta-harness outer loop orchestrator")
    parser.add_argument("--sweep", choices=["scoring", "norms", "features", "splits", "strategy", "all"],
                        help="Run a pre-built sweep")
    parser.add_argument("--adaptive", action="store_true",
                        help="Run MAB-adaptive iterations (replaces --sweep all)")
    parser.add_argument("--budget", type=int, default=20,
                        help="Number of adaptive iterations")
    parser.add_argument("--config", type=str, help="Path to custom meta config JSON")
    parser.add_argument("--iteration", type=int, default=0, help="Iteration number (0=auto)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without running Claude")
    parser.add_argument("--baseline", action="store_true", help="Evaluate current baseline on OOS")
    parser.add_argument("--max-inner", type=int, default=40, help="Max experiments per inner loop")
    parser.add_argument("--paper-trade", action="store_true",
                        help="Run paper trade gate after each iteration")
    parser.add_argument("--paper-trade-only", action="store_true",
                        help="Just run the paper trade gate on current classifier")
    parser.add_argument("--days", type=int, default=10,
                        help="Days for paper trade gate")
    parser.add_argument("--forest-summary", action="store_true",
                        help="Print knowledge forest summary")
    parser.add_argument("--mab-stats", action="store_true",
                        help="Print MAB scheduler statistics")
    args = parser.parse_args()

    harness = MetaHarness()

    if args.baseline:
        evaluate_baseline_oos()
        return

    if args.paper_trade_only:
        result = run_paper_trade_gate(days=args.days, verbose=True)
        print(f"\n{result.summary()}")
        return

    if args.forest_summary:
        print(harness.forest.get_summary_for_prompt())
        return

    if args.mab_stats:
        print(harness.mab.get_allocation_summary())
        return

    if args.adaptive:
        harness.run_adaptive(
            budget=args.budget, max_inner=args.max_inner,
            dry_run=args.dry_run, run_paper_trade=args.paper_trade,
        )
        return

    if args.config:
        config = MetaConfig.from_json(args.config)
        config.max_inner_experiments = args.max_inner
        result = harness.run_iteration(
            config, iteration=args.iteration, dry_run=args.dry_run,
            run_paper_trade=args.paper_trade,
        )
        print(f"\nFinal: {result.summary()}")
        return

    if args.sweep:
        sweeps = {
            "scoring": harness.run_scoring_sweep,
            "norms": harness.run_normalization_sweep,
            "features": harness.run_feature_sweep,
            "splits": harness.run_split_sweep,
            "strategy": harness.run_strategy_sweep,
        }
        if args.sweep == "all":
            for name, sweep_fn in sweeps.items():
                print(f"\n\n{'#'*60}")
                print(f"# SWEEP: {name.upper()}")
                print(f"{'#'*60}")
                sweep_fn(dry_run=args.dry_run, max_inner=args.max_inner)
        else:
            sweeps[args.sweep](dry_run=args.dry_run, max_inner=args.max_inner)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
