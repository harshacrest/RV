# Regime Classification Autoresearch

Autonomous research loop for optimizing an 8-state regime classification system for options selling strategies (DM, WC, Orion) on Nifty 50.

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr6`). The branch `regime-research/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b regime-research/<tag>` from current state.
3. **Read the in-scope files**:
   - `prepare_rv.py` — Fixed data loading, feature computation, evaluation function. **DO NOT MODIFY.**
   - `regime_experiment.py` — The file you modify. Classification logic, boundaries, features, weights.
4. **Verify data**: Check that `../features/rv_daily.parquet` and `../features/strategy_returns_*.xlsx` exist.
5. **Initialize results.tsv**: Create with header row. Establish baseline with first run.
6. **Confirm and go**.

## The Goal

**Maximize `composite_score`** from `prepare_rv.evaluate()`. This is a weighted combination of:
- **val_sharpe** (30%): Portfolio Sharpe ratio on held-out validation period (Jul 2025 – Jan 2026)
- **monotonicity** (25%): Sharpe must degrade as risk increases within each level. Violations are penalized.
- **safe_separation** (20%): Average AL% gap between Safe and Exposed states (bigger = better discrimination)
- **rank_stability** (15%): Spearman rank correlation of state ordering between train and validation (higher = more robust)
- **state_coverage** (10%): Penalty if any state has <5 days in validation (ensures usable classification)

### CRITICAL: Sharpe Monotonicity Requirement

The classification is **meaningless** if "Risky" or "Exposed" states don't actually perform worse than "Safe" states. The monotonicity component enforces this:

**Expected Sharpe ordering within each level:**
- L1: Safe > Exposed
- L2: Safe > Caution-A > Risky, Safe > Caution-B > Risky
- L3: Safe > Exposed

Every violation (e.g., L2 Caution-B Sharpe > L2 Safe Sharpe) **directly reduces the composite score**. This means:

1. **Naming must match reality** — if a state consistently outperforms, it should be classified as "Safe", not "Caution"
2. **The split logic must create meaningful risk separation** — PK/IV ratio or whatever splitter you use must actually separate good days from bad days
3. **Strategy weights must NOT mask bad classification** — don't fix a broken classifier by overweighting winning strategies in "Risky" states. The classifier itself must produce Sharpe-monotonic states.
4. **If you can't achieve monotonicity, reduce the number of states** — 6 well-separated states beats 8 confused states

A perfect monotonicity score (1.0) means all 7 expected pairs are correctly ordered. Each violation costs ~0.036 from the composite (0.25 × 1/7).

## Baseline Diagnostic Context

**Baseline composite: TBD** (recalculate after monotonicity was added as a 25% weighted component)
Old baseline was 0.891 (val_sharpe=4.12, safe_sep=13.82, rank=0.81, coverage=1.0) — but this did NOT include monotonicity.
The new baseline will be LOWER because multiple monotonicity violations exist (e.g., L2 Caution-B Sharpe > L2 Safe Sharpe in training).

### State Distribution (Training / Validation)
| State | Train | Val | Notes |
|-------|-------|-----|-------|
| L1 Safe | 4 | 13 | VERY sparse in train — thresholds unreliable |
| L1 Exposed | 3 | 6 | VERY sparse — consider merging L1 into L2 |
| L2 Safe | 23 | 9 | |
| L2 Caution-A | 23 | 10 | |
| L2 Caution-B | 42 | 17 | |
| L2 Risky | 42 | 33 | Largest val state |
| L3 Safe | 229 | 25 | Dominates training |
| L3 Exposed | 229 | 30 | Dominates training |

**Key insight**: L3 accounts for 77% of training days. L1 has only 7 training days total — any L1-specific threshold is essentially noise. Consider widening L1 boundary or merging L1 states.

### Feature Correlation Warnings
These pairs are REDUNDANT (|rho| > 0.75) — do NOT use both as split features:
- IV_5d / IV_10d (0.92) — use one or the other
- PK_IV_ratio / PK_IV_smooth3 (0.88) — smooth3 adds no information
- PK_IV_ratio / PK_IV_zscore_30d (0.88) — choose one
- iv_lag / IV_5d (0.82) — effectively the same signal
- PK_5d / PK_10d (0.82) — use one or the other

### Top Predictive Features (IC with next-day pnl)
1. IV_chg_5d (IC=-0.055) — best predictor, negative = falling IV is good
2. VRP_today (IC=-0.050) — vol risk premium signal
3. PK_IV_ratio (IC=-0.048) — current split feature, solid choice
4. PK_IV_10d (IC=-0.047) — alternative to PK_IV_ratio
5. IV_10d (IC=+0.044) — higher IV → better next-day return

All ICs are weak (<0.06) — regime classification adds value through state-conditional strategy allocation, not direct prediction.

### Boundary Sensitivity
- Current: L1=8.5, L2=11.0 → composite=0.891
- Best found: L1=9.0, L2=11.0 → composite=0.907
- Boundaries above L2=14 collapse performance
- Sweet spot: L1 in [8.0-9.5], L2 in [10.5-12.0]

### Walk-Forward Validation Warning
The current regime classifier LOSES to equal-weight in 8/12 rolling windows. It only wins in 2025+ windows. Average Sharpe delta is -0.12 (regime underperforms). This means:
- Strategy weights are overfit to recent conditions
- Improvements must be validated across multiple periods, not just the val set
- Val Sharpe is heavily period-dependent: first half=2.73, second half=6.25

### Ablation Results (what matters most)
1. **Boundaries are CRITICAL** — wrong boundaries destroy performance (delta=-0.75)
2. **Strategy weights matter a lot** — removing them costs -0.12
3. **L2 direction helps moderately** — -0.03 without it
4. **Extra features (PK_IV_smooth3, pk_iv_pctile, pk_iv_risk) add ZERO value** — don't waste experiments on them
5. **PK_IV_ratio everywhere** slightly outperforms using zscore for L3 (+0.006)

### Known Magic Constants (undocumented)
- IV boundaries 8.5/11 — empirically tuned, L1=9.0 may be better
- Direction threshold -1.1 for IV_chg_1d — arbitrary, test alternatives
- PK_IV thresholds computed as training median — no cross-validation

## What You CAN Modify

Only `regime_experiment.py`. Everything is fair game within that file:

### Dimension 1: IV Level Boundaries
- `IV_L1_UPPER`, `IV_L2_UPPER` — the static cutoffs
- Adaptive boundary parameters — enable/tune the trailing-window shift logic
- Try: [10,15], [11,16], [12,18], [13,20], [15,20], etc.
- Try: different lookback windows (30d, 45d, 60d), trigger thresholds (40%, 50%, 60%)

### Dimension 2: Feature Engineering
- `LEVEL_FEATURE` — what determines L1/L2/L3 (default: iv_lag)
- `SPLIT_FEATURE_*` — what splits Safe/Exposed within each level (default: PK_IV_ratio)
- `L2_DIRECTION_FEATURE` — the L2 secondary signal (default: IV_chg_5d)
- `compute_extra_features()` — add ANY derived feature you want
- Available base features: iv_lag, PK_IV_ratio, IV_chg_5d, IV_5d, PK_5d, IV_10d, PK_10d, PK_IV_10d, IV_chg_1d, IV_percentile_60d, PK_IV_zscore_30d, RV_today, VRP_today
- Try: log transforms, z-scores, percentile ranks, interaction terms, different windows

### Dimension 3: Classification Architecture
- Enable/disable direction at L1 or L3 (currently only L2 uses direction)
- Change threshold method from "median" to "fixed"
- Add more states (e.g., 4-state L1 with direction)
- Remove states (e.g., merge L2 Caution-A and L2 Caution-B)
- Try completely different classification trees

### Dimension 4: Strategy Allocation
- Set `APPLY_STRATEGY_WEIGHTS = True`
- Modify `STRATEGY_WEIGHTS` dict to change DM/WC/Orion mix per state
- Try: Orion-heavy in L2 Caution-B, WC-only in L1 Exposed, etc.
- This changes the portfolio PnL the evaluation sees

### Dimension 5: Snapshot Fusion
- Set `SECONDARY_SNAPSHOT = "0916"` to enable morning overlay
- Try different `FUSION_METHOD`: "majority", "conservative", "override"
- Only ~52% of days agree between close and morning — lots of room for alpha

## What You CANNOT Do

- Modify `prepare_rv.py` — it's the ground truth evaluation
- Change the evaluation function or its weights
- Add external dependencies
- Peek at validation/OOS data to set thresholds (thresholds must be computed from training period only)

## Running an Experiment

```bash
cd /Users/harsha/Desktop/Research/RV/autoresearch_regime
python regime_experiment.py > run.log 2>&1
```

Extract key metric:
```bash
grep "^composite_score:" run.log
```

Each run takes ~15-30 seconds (small dataset, CPU only).

## Output Format

The script prints:
```
---
composite_score:  0.654321
val_sharpe:       3.2500
safe_separation:  5.40
rank_stability:   0.5952
state_coverage:   0.8500
val_days:         150
train_days:       600
n_states_used:    8
elapsed_seconds:  12.3

--- State Breakdown (Validation) ---
  L1 Safe               45d  Sharpe= 3.20  AL=  5.2%  Avg=+0.0410
  ...
```

## Logging Results

Log to `results.tsv` (tab-separated):

```
commit	composite_score	val_sharpe	safe_sep	status	description
```

1. git commit hash (7 chars)
2. composite_score (6 decimals)
3. val_sharpe (4 decimals)
4. safe_separation (2 decimals)
5. status: `keep`, `discard`, or `crash`
6. description of what was tried

## The Experiment Loop

LOOP FOREVER:

1. Check git state
2. Edit `regime_experiment.py` with an experimental idea
3. `git commit -am "description"`
4. Run: `python regime_experiment.py > run.log 2>&1`
5. Extract: `grep "^composite_score:\|^val_sharpe:\|^safe_separation:" run.log`
6. If empty output → crash. Read `tail -n 30 run.log`, fix or skip.
7. Log to results.tsv (do NOT commit results.tsv)
8. If composite_score improved → KEEP (advance branch)
9. If equal or worse → DISCARD (`git reset --hard HEAD~1`)

## Strategy Guide

Prioritize experiments roughly in this order:

**Quick wins (try first):**
- Shift boundaries by ±1-2 points (e.g., [11,16], [13,18])
- Try PK_IV_10d instead of PK_IV_ratio as splitter
- Enable L3 direction

**Medium effort:**
- Enable adaptive boundaries with different lookback windows
- Use IV_percentile_60d instead of raw iv_lag for level classification
- Add VRP as a secondary signal at L1

**High effort, high reward:**
- Strategy weight optimization per state
- Dual-snapshot fusion with conservative method
- Custom features in compute_extra_features()
- Completely different classification tree

**Things that probably WON'T work (but try if stuck):**
- Using only 1 feature everywhere (PK_IV_ratio for everything)
- Very small boundaries like [8, 12] (too few L3 days)
- Very large boundaries like [20, 25] (too few L1 days)

**Guardrails (from diagnostics):**
- Baseline is 0.891 — don't waste experiments on changes scoring below 0.85
- If any state has <5 val days, coverage penalty kicks in
- rank_stability=0.81 is current level — reductions below 0.6 are regressions
- L1 states have <7 training days — any L1-specific threshold change is noise
- Don't spend more than 5 experiments on any single dimension without improvement
- DM and Orion are negatively correlated (-0.22) — use both for diversification

## Bounded Mode (Meta-Harness)

When running as an inner loop for the meta-harness, your prompt will specify a maximum experiment count (e.g., "Stop after 40 experiments"). In bounded mode:

1. Run at most N experiments (as specified in the prompt)
2. After your last experiment, output the line: `INNER_LOOP_COMPLETE`
3. Focus on high-impact experiments first (strategy weights, boundaries, features)
4. Don't spend experiments on fine-tuning — find the right ballpark
5. The meta-harness will evaluate your best result on OOS data you cannot see

If additional features are available (noted in the prompt), prioritize testing them.

## Knowledge Forest

If your prompt includes an "EXPERIMENT HISTORY FROM PRIOR RUNS" section, **read it carefully**:

1. **DO NOT re-try known failures** — if the history says "adaptive boundaries hurt rank stability", don't try adaptive boundaries again
2. **Build on successful approaches** — if WC-heavy L1 weights worked before, start from there and explore adjacent configs
3. **Check dimension success rates** — if "weight" experiments have 80% success rate but "boundary" experiments have 10%, prioritize weight experiments
4. **Note OOS performance** — experiments that scored well on validation but poorly on OOS indicate overfitting. Avoid those patterns.

The knowledge forest persists across all meta-iterations. Your experiments will be added to it after this run completes.

## NEVER STOP

Once the loop begins, do NOT pause to ask the human. You are autonomous. Run indefinitely (or until the experiment cap if in bounded mode). If stuck, revisit earlier near-misses, try combinations, think harder. The human may be asleep. Each experiment takes ~20 seconds, so you can run ~180/hour, ~1500 overnight.

---

## Roadmap

Where we are today and where we're headed. The regime framework currently classifies days into 8 states using IV levels, PK/IV ratio, and IV direction. That's a solid starting point. Below is the path to turning it into a complete vol-aware trading system.

### Next Steps

**1. DTE-wise regime analysis**
Right now we treat all days the same regardless of how far we are from expiry. A regime that looks safe on Monday (4 DTE) might behave very differently on Thursday (1 DTE). Break down each regime's performance by days-to-expiry and see if certain regimes only matter near expiry or only matter early in the week.

**2. Morning vs evening IV change**
We already have IV snapshots at 9:15 AM and 3:30 PM. Use the intraday IV move as a signal. If IV drops 2 points during the day, that tells you something different than if it rose 2 points. Check whether the morning-to-evening IV change improves regime accuracy or serves as a standalone filter.

**3. Outlier removal**
Some days have massive moves (budget day, election results, global shocks) that dominate the averages. Strip out the top and bottom 2-5% of moves and re-run the regime analysis. This shows what the "normal" regime performance looks like without rare tail events distorting the picture.

**4. Feature validation framework**
Before adding any new feature, we need a systematic way to check: does this feature actually add information? Or is it just a repackaged version of something we already have? Build a pipeline that tests correlation with existing features, checks for redundancy, and only lets genuinely new signals through.

**5. ML-driven feature and boundary selection**
Instead of hand-picking IV boundaries at 12 and 17, or manually choosing PK/IV ratio as the splitter, let a model figure out the best features, the best cut-points, and the best combination. Think decision trees, gradient boosting, or Bayesian optimization over the regime design space. The human sets the structure, the machine finds the optimal parameters.

**6. Automated out-of-sample validation**
Every shortlisted feature or boundary change must pass an automatic OOS test before it's accepted. No more eyeballing train-period results and hoping they hold. The framework should run the candidate through unseen data, compare against the baseline, and only promote changes that genuinely generalize.

**7. Intraday dynamic vol regimes**
Move beyond daily regimes. Use tick-level or minute-level vol features to classify the market state in real time during the trading day. Methods like Hidden Markov Models (HMM) and transition probability matrices can capture how the market shifts between calm, trending, and volatile states within a single session. This opens the door to intraday position adjustments rather than just daily allocation decisions.

### End Goal

Everything above feeds into three layers of the final system:

**Layer 1 — Vol Score**
A single number (like a credit score but for volatility) that rates how favorable the current environment is for option selling. High score = sell aggressively. Low score = sit on hands or hedge. This combines regime state, DTE, intraday dynamics, and outlier risk into one actionable rating.

**Layer 2 — Regime Classification and Strategy Allocation**
The regime tells you which strategy to run and how much size to give it. DM might be the best in low-IV regimes, Orion might shine when IV is high and falling. The allocation engine maps every regime state to a strategy mix with position sizing rules.

**Layer 3 — Microstructure**
The finest-grained layer, split into two parts:

- **3.1 Strategy execution criteria** — Within a given regime and allocation, what are the exact entry/exit rules? When during the day do you put on the trade? What strike selection? What stop-loss? This is where the intraday vol regime and DTE analysis directly feed in.

- **3.2 Transitions** — How do you handle regime changes mid-position? If you entered a trade in "L2 Safe" and the regime flips to "L2 Risky" the next morning, do you exit, hedge, or hold? Transition rules prevent the system from being caught flat-footed when the market shifts.
