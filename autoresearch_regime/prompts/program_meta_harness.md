# Meta-Harness: Outer Loop for Regime Autoresearch

Optimizes the *frame* that the inner autoresearch loop operates in, evaluated on truly held-out OOS data.

## Architecture

```
META-HARNESS (you — outer loop)
├── Modifies: meta_config.json (scoring weights, norms, features, splits)
├── Evaluates on: OOS holdout (2021-2023 + Feb-Mar 2026)
│
│   INNER LOOP (Claude subprocess — runs autoresearch)
│   ├── Modifies: regime_experiment.py
│   ├── Evaluates on: validation set (controlled by meta_config)
│   └── Runs: N experiments per iteration (default 40)
│
├── After inner loop completes:
│   → Take the best classifier
│   → Evaluate on OOS (never seen by inner loop)
│   → Log to meta_results.tsv
│   → Compare across iterations
└── Propose next meta modification
```

## Setup

1. **Agree on a run tag**: e.g. `meta-apr7`
2. **Read current state**: Check `meta_results.tsv` for prior iterations
3. **Verify baseline**: `python meta_harness.py --baseline` to see OOS performance of current best
4. **Start iterating**

## The Goal

**Maximize OOS performance** — specifically OOS Sharpe ratio across both OOS periods:
- OOS1: Jan 2021 – Jan 2023 (515 days, early high-IV market)
- OOS2: Feb 2026 – Mar 2026 (34 days, recent)

The validation composite score is what the inner loop optimizes. But the meta-harness's job is to find which composite formula, features, and train/val split produces classifiers that **generalize best to unseen data**.

## Statistical Considerations

**OOS2 is unreliable (34 days):**
- SE of annualized Sharpe from 34 days ≈ 2.68
- Any OOS2 Sharpe between -2.7 and +2.7 is statistically indistinguishable from zero
- The current OOS2 Sharpe is -1.37 — well within noise range
- Weight OOS1 (515 days, SE ≈ 0.70) much more heavily in decisions

**OOS1 is also noisy:**
- Even with 515 days, the 95% CI for OOS1 Sharpe spans [-0.1, +8.9]
- Consider an OOS improvement real only if OOS1 delta > 0.5 (roughly 0.7 SE)

**Overfitting signals:**
- If val_composite improves by >0.05 but OOS1 Sharpe drops → overfitting
- Val period is strongly biased: first half Sharpe=2.73, second half=6.25
- Record overfitting patterns in the forest as known failure patterns

**L1 state sparsity:**
- L1 has only 7 training days and 0 OOS1/OOS2 days
- Any L1-specific optimization is fitting to noise
- Consider merging L1 into L2 or widening the L1 boundary

## What You Modify

Only `meta_config.json` parameters, via the `MetaConfig` class. Four dimensions:

### Dimension 1: Scoring Function (HIGHEST PRIORITY)

The composite formula that the inner loop optimizes against:
```python
composite = w_sharpe * min(val_sharpe / sharpe_norm, sharpe_cap)
          + w_safe_sep * min(safe_sep / safe_sep_norm, safe_sep_cap)
          + w_rank_corr * max(rank_corr, 0)
          + w_coverage * coverage
```

**Default**: w=[0.40, 0.25, 0.20, 0.15], sharpe_norm=5.0, safe_sep_norm=10.0

Try:
- Higher Sharpe weight (0.50) — does the inner loop find configs that generalize better?
- Higher rank stability weight (0.30) — does forcing stable rankings prevent overfitting?
- Lower sharpe_norm (3.0) — does easier-to-hit Sharpe target diversify exploration?
- Higher safe_sep_norm (15.0) — does harder separation push the inner loop to better classification?

### Dimension 2: Feature Pipeline

Add features to what the inner loop can use:
```python
extra_features: ["IV_20d", "PK_20d", "IV_momentum_5d", "VRP_5d", ...]
```

Available features in the registry:
- `IV_20d`, `PK_20d` — 20-day rolling averages
- `IV_momentum_5d` — 5-day IV rate of change (%)
- `VRP_5d` — 5-day rolling volatility risk premium
- `IV_range_10d` — 10-day IV high-low range
- `RV_IV_gap` — Realized vol minus implied vol
- `PK_IV_zscore_60d` — 60-day z-scored PK/IV ratio
- `IV_vol_of_vol_20d` — Volatility of IV changes (vol-of-vol)

### Dimension 3: Train/Val Split

Change which data the inner loop trains/validates on:
```python
train_start, train_end, val_start, val_end
```

Try:
- Earlier start (2022-06-01) — more training data
- Later split (val starts Oct 2025) — shorter but more recent validation
- Shifted earlier (val starts Apr 2025) — different market regime in validation

### Dimension 4: Coverage Parameters

```python
min_state_days: int = 5   # threshold for coverage penalty
min_states_used: int = 6  # minimum states for full coverage credit
```

## What You DO NOT Modify

- `regime_experiment.py` — that's the inner loop's domain
- `prepare_rv_meta.py` — frozen parameterized harness
- `inner_loop.py` — frozen subprocess runner
- OOS period definitions — these are the ground truth

## Running an Iteration

### Adaptive mode (RECOMMENDED — uses MAB scheduler):
```bash
python meta_harness.py --adaptive --budget 20 --max-inner 40   # 20 MAB-driven iterations
python meta_harness.py --adaptive --budget 20 --paper-trade    # with forward validation
python meta_harness.py --adaptive --budget 5 --dry-run         # preview
```

### Fixed sweeps (legacy):
```bash
python meta_harness.py --sweep scoring --dry-run     # Preview
python meta_harness.py --sweep scoring --max-inner 40 # Run scoring sweep
python meta_harness.py --sweep features              # Feature sweep
python meta_harness.py --sweep splits                # Split sweep
```

### Manual iteration:
```bash
python meta_harness.py --config meta_configs/custom.json --iteration 1
```

### Diagnostic commands:
```bash
python meta_harness.py --baseline            # Current OOS performance
python meta_harness.py --paper-trade-only    # Paper trade gate on current classifier
python meta_harness.py --forest-summary      # Knowledge forest summary
python meta_harness.py --mab-stats           # MAB scheduler statistics
python forest_manager.py --query "boundary"  # Search experiment history
python ic_dedup.py --features "IV_20d,PK_20d,VRP_5d" --report  # Correlation check
```

## Upgrade Systems

### Knowledge Forest (experiment_forest.json)
All experiments across all meta-iterations are logged persistently. The inner loop
receives a summary of prior experiments in its prompt, preventing re-discovery of
known failures. Check with `python meta_harness.py --forest-summary`.

### MAB Scheduler (mab_state.json)
UCB1 multi-armed bandit adaptively allocates budget across the 4 dimensions
(scoring, norms, features, splits). Dimensions that produce larger OOS deltas
get more pulls. Check with `python meta_harness.py --mab-stats`.

### IC Deduplication Gate (ic_dedup.py)
Before each inner loop, pairwise Spearman correlation is computed on activated
features. Any feature with |corr| > 0.75 to an already-kept feature is dropped.
This prevents the inner loop from seeing redundant signals.

### Paper Trade Gate (paper_trade_gate.py)
After OOS evaluation, optionally run a forward-walk check on the most recent
unseen data. Validates regime plausibility, Sharpe, and transition stability.
Use `--paper-trade` flag or `--paper-trade-only`.

## Output Format

`meta_results.tsv` includes a `dimension` column (which MAB arm was pulled) and
`paper_trade` column (PROMOTE/ITERATE/REJECT).

Config snapshots: `meta_configs/iter_NNN.json`
MAB state: `mab_state.json`
Forest: `experiment_forest.json`

## Strategy Guide

**With adaptive mode**, you don't pick the order — the MAB does. But you should:

1. **Check MAB stats** after each batch — if one dimension is dominating, the system is working
2. **Read the forest summary** periodically — look for patterns in what fails across configs
3. **Watch OOS2 specifically** — that's the most recent data, hardest to generalize to
4. **Run paper trade gate** on promising candidates before promoting to production

**Watch for overfitting**: If a config gives great val_composite but poor OOS, the inner loop overfit. High rank_stability weight in the scoring function may prevent this.

## NEVER STOP

Run as many iterations as possible. Each one takes 20-60 minutes (inner loop + OOS eval). Log everything. Compare OOS performance across configs. The human may be asleep.
