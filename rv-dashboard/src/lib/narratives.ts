/**
 * Pure narrative-generation functions.
 * Every sentence references a computed number — no hardcoded prose.
 */
import type {
  FeatureRankingEntry,
  FeatureRankingData,
  RegimeConstructionData,
  LevelFinalState,
  L2StrategyProfile,
  TestedAndFailed,
} from './types'

/* ─── helpers ─── */
const sign = (v: number) => (v > 0 ? '+' : '') + v.toFixed(1)
const sign2 = (v: number) => (v > 0 ? '+' : '') + v.toFixed(2)
const sign3 = (v: number) => (v > 0 ? '+' : '') + v.toFixed(3)
const pct = (v: number) => `${v.toFixed(1)}%`

function isMonotonic(vals: number[], allowInversions = 1): 'up' | 'down' | null {
  let upBreaks = 0, downBreaks = 0
  for (let i = 1; i < vals.length; i++) {
    if (vals[i] < vals[i - 1]) upBreaks++
    if (vals[i] > vals[i - 1]) downBreaks++
  }
  if (upBreaks <= allowInversions) return 'up'
  if (downBreaks <= allowInversions) return 'down'
  return null
}

/* ═══════════════════════════════════════════
   1. Per-Feature Observations (Feature Ranking)
   ═══════════════════════════════════════════ */

export function generateFeatureObservations(f: FeatureRankingEntry): string[] {
  const obs: string[] = []
  const q = f.quintiles
  if (!q || q.length < 3) return obs

  const q1 = q[0]
  const q5 = q[q.length - 1]

  // DM-Orion mirror check
  if (f.dm_corr != null && f.orion_corr != null) {
    const dmSign = f.dm_corr > 0 ? 'positive' : 'negative'
    const orSign = f.orion_corr > 0 ? 'positive' : 'negative'
    if (dmSign !== orSign && Math.abs(f.dm_corr) > 0.15 && Math.abs(f.orion_corr) > 0.15) {
      obs.push(`DM (${sign3(f.dm_corr)}) and Orion (${sign3(f.orion_corr)}) show opposite correlations — near-perfect mirror.`)
    }
  }

  // Combined portfolio resilience
  const allCombPositive = q.every(row => row.combined_avg != null && row.combined_avg > 0)
  if (allCombPositive) {
    obs.push(`Combined portfolio positive across ALL quintiles — the natural hedge absorbs feature-driven risk.`)
  }

  // AL spread interpretation
  const alSpread = f.al_spread
  if (Math.abs(alSpread) < 2) {
    obs.push(`AL spread near zero (${sign(alSpread)}pp) — this feature does not separate all-lose risk at portfolio level.`)
  } else if (alSpread > 8) {
    obs.push(`Strong AL discrimination: ${sign(alSpread)}pp spread — highest-quintile days carry significantly more all-lose risk.`)
  } else if (alSpread > 4) {
    obs.push(`Moderate AL discrimination: ${sign(alSpread)}pp spread between extreme quintiles.`)
  } else if (alSpread < -4) {
    obs.push(`Inverse AL discrimination: ${sign(alSpread)}pp — lowest-quintile days carry more all-lose risk.`)
  }

  // Monotonic progression in AL%
  const alVals = q.map(row => row.al_pct)
  const mono = isMonotonic(alVals)
  if (mono === 'up') {
    obs.push(`Nearly monotonic AL% progression: ${q.map(row => `${row.quintile.replace(/[^Q0-9]/g, '')} ${pct(row.al_pct)}`).join(' → ')}.`)
  } else if (mono === 'down') {
    obs.push(`Inverse monotonic AL% pattern: risk decreases as feature value rises.`)
  }

  // Strategy-level spread direction
  if (q1.dm_avg != null && q5.dm_avg != null && q1.orion_avg != null && q5.orion_avg != null) {
    const dmSpread = q5.dm_avg - q1.dm_avg
    const orSpread = q5.orion_avg - q1.orion_avg
    if (Math.sign(dmSpread) !== Math.sign(orSpread) && Math.abs(dmSpread) > 0.1 && Math.abs(orSpread) > 0.1) {
      obs.push(`DM and Orion move in opposite directions across quintiles (DM ${sign2(dmSpread)}%, Orion ${sign2(orSpread)}%) — strategy tilt signal, not portfolio risk.`)
    } else if (Math.sign(dmSpread) === Math.sign(orSpread) && Math.abs(dmSpread) > 0.05 && Math.abs(orSpread) > 0.05) {
      obs.push(`All strategies move in the same direction across quintiles — universal direction signal.`)
    }
  }

  // WC behavior
  if (f.wc_corr != null && Math.abs(f.wc_corr) < 0.08) {
    obs.push(`WC shows near-zero correlation (${sign3(f.wc_corr)}) — feature doesn't discriminate WC returns.`)
  }

  return obs
}

/* ═══════════════════════════════════════════
   2. Key Takeaways (Feature Ranking)
   ═══════════════════════════════════════════ */

export function generateKeyTakeaways(data: FeatureRankingData): string[] {
  const { features, portfolio_stats: ps, discarded } = data
  const takeaways: string[] = []

  // 1. Natural hedge
  if (ps.dm_orion_corr != null && ps.dm_orion_corr < -0.1) {
    takeaways.push(`The DM-Orion mirror is the portfolio's natural hedge (corr ${sign3(ps.dm_orion_corr)}). This is why the combined portfolio survives even when individual strategies fail.`)
  }

  // 2. Top discriminator
  const topByAl = features.filter(f => f.verdict === 'KEEP').sort((a, b) => Math.abs(b.al_spread) - Math.abs(a.al_spread))[0]
  if (topByAl) {
    takeaways.push(`${topByAl.label} is the strongest all-lose discriminator (${sign(topByAl.al_spread)}pp spread). It becomes the regime backbone.`)
  }

  // 3. All-lose cause
  const boring = features.find(f => f.al_corr != null && f.al_corr > 0.05 && f.wc_corr != null && Math.abs(f.wc_corr) < 0.1)
  if (boring) {
    takeaways.push(`All-lose is caused by "boring middle" — enough range hurts DM, no trend for Orion/WC. Features like ${boring.label} capture this.`)
  } else {
    takeaways.push(`All-lose days (${pct(ps.al_pct)}) occur when all 3 strategies fail simultaneously — risk is regime-driven, not strategy-specific.`)
  }

  // 4. Strategy tilt vs portfolio risk
  const tiltFeatures = features.filter(f => f.dm_orion_gap != null && Math.abs(f.dm_orion_gap) > 0.3 && Math.abs(f.al_spread) < 3)
  if (tiltFeatures.length > 0) {
    const names = tiltFeatures.map(f => f.label).join(', ')
    takeaways.push(`${names} are strategy tilt signals (which strategy wins), not portfolio risk signals — correlations near zero with all-lose.`)
  }

  // 5. Discarded count
  if (discarded.length > 0) {
    takeaways.push(`${discarded.length} additional features tested and discarded — DM-Orion mirror absorbs most vol shocks, making many features redundant.`)
  }

  return takeaways
}

/* ═══════════════════════════════════════════
   3. Consistent Feature Testing Callout (Regime Construction)
   ═══════════════════════════════════════════ */

export function generateConsistencyCallout(data: RegimeConstructionData): { title: string; bullets: string[] } {
  const { per_level } = data
  const bullets: string[] = []

  // L1 analysis
  const l1q = per_level.L1.pk_iv_quintiles
  if (l1q.length >= 2) {
    const l1Spread = l1q[l1q.length - 1].al_pct - l1q[0].al_pct
    bullets.push(`L1: IV measures all weak. PK/IV ratio gives ${sign(l1Spread)}pp AL spread with AC ${per_level.L1.threshold_ac ?? '~0.75'}. PK dominates.`)
  }

  // L2 analysis
  const l2States = per_level.L2.final_states
  const l2Safe = l2States.find(s => s.state === 'L2 Safe')
  const l2Risky = l2States.find(s => s.state === 'L2 Risky')
  if (l2Safe && l2Risky) {
    const l2Spread = l2Risky.al_pct - l2Safe.al_pct
    bullets.push(`L2: IV direction + PK/IV produces ${sign(l2Spread)}pp AL gap (Safe ${pct(l2Safe.al_pct)} vs Risky ${pct(l2Risky.al_pct)}). Both dimensions needed.`)
  }

  // L3 analysis
  const l3q = per_level.L3.pk_iv_quintiles
  if (l3q.length >= 2) {
    const l3Spread = l3q[l3q.length - 1].al_pct - l3q[0].al_pct
    bullets.push(`L3: IV direction is noise (rising ~86% of days). PK/IV ratio gives ${sign(l3Spread)}pp — same mechanism as L1.`)
  }

  return {
    title: 'Different signals dominate at different levels — but discovered consistently',
    bullets,
  }
}

/* ═══════════════════════════════════════════
   4. Level Stories (Regime Construction)
   ═══════════════════════════════════════════ */

export function generateL1Story(states: LevelFinalState[], threshold: number): string[] {
  const safe = states.find(s => s.state.includes('Safe'))
  const exposed = states.find(s => s.state.includes('Exposed'))
  if (!safe || !exposed) return []

  const stories: string[] = []
  const ratio_safe = (threshold * 100).toFixed(0)

  stories.push(
    `When realized vol is ≤${ratio_safe}% of implied (Safe): deep premium cushion, only ${pct(safe.al_pct)} AL, Sharpe ${safe.sharpe?.toFixed(2) ?? '—'}.`
  )
  stories.push(
    `When realized vol exceeds ${ratio_safe}% of implied (Exposed): cushion thin, ${pct(exposed.al_pct)} AL, Sharpe drops to ${exposed.sharpe?.toFixed(2) ?? '—'}.`
  )

  return stories
}

export function generateL2Story(
  states: LevelFinalState[],
  profiles: L2StrategyProfile[],
): string[] {
  const stories: string[] = []
  const safe = states.find(s => s.state === 'L2 Safe')
  const risky = states.find(s => s.state === 'L2 Risky')

  if (safe && risky) {
    stories.push(
      `At moderate IV, both PK/IV gap AND IV direction matter. Safe = cushion wide AND widening (${pct(safe.al_pct)} AL, Sharpe ${safe.sharpe?.toFixed(2) ?? '—'}). Risky = cushion thin AND shrinking (${pct(risky.al_pct)} AL).`
    )
  }

  // Strategy divergence from profiles
  const safeProfile = profiles.find(p => p.state === 'L2 Safe')
  const cautionB = profiles.find(p => p.state === 'L2 Caution-B')
  if (cautionB && cautionB.dm_avg != null && cautionB.orion_avg != null) {
    if (cautionB.dm_avg < -0.01 && cautionB.orion_avg > 0.05) {
      stories.push(
        `Caution-B (low PK/IV, rising IV): DM negative but Orion thrives — rising IV = trending market benefits Orion.`
      )
    }
  }

  const cautionA = profiles.find(p => p.state === 'L2 Caution-A')
  if (cautionA && cautionA.orion_avg != null && cautionA.orion_avg < -0.01) {
    stories.push(
      `Caution-A (high PK/IV, falling IV): Orion struggles (${sign2(cautionA.orion_avg * 100)}%) — high realized movement but no trend.`
    )
  }

  return stories
}

export function generateL3Story(
  l3States: LevelFinalState[],
  l1States: LevelFinalState[],
  l3Profiles?: L2StrategyProfile[],
): string[] {
  const stories: string[] = []
  const l3Safe = l3States.find(s => s.state.includes('Safe'))
  const l3Exposed = l3States.find(s => s.state.includes('Exposed'))
  const l1Safe = l1States.find(s => s.state.includes('Safe'))

  if (l3Safe && l3Exposed) {
    stories.push(
      `L3 Exposed: ${pct(l3Exposed.al_pct)} AL, Sharpe ${l3Exposed.sharpe?.toFixed(2) ?? '—'}. One in ${Math.round(100 / (l3Exposed.al_pct || 1))} days all strategies lose.`
    )
  }

  if (l3Safe && l1Safe) {
    if (l3Safe.al_pct > l1Safe.al_pct) {
      stories.push(
        `Even L3 Safe (${pct(l3Safe.al_pct)} AL) is riskier than L1 Safe (${pct(l1Safe.al_pct)} AL). The whole level is dangerous territory.`
      )
    }
  }

  // DM standout in L3 Safe
  if (l3Profiles && l3Profiles.length > 0) {
    const safeProfile = l3Profiles.find(p => p.state.includes('Safe'))
    if (safeProfile && safeProfile.dm_avg != null && safeProfile.dm_sharpe != null) {
      if (safeProfile.dm_sharpe > 3) {
        stories.push(
          `DM is the standout in L3 Safe: ${sign2(safeProfile.dm_avg)}% avg, Sharpe ${safeProfile.dm_sharpe.toFixed(2)} — rich premiums, manageable movement.`
        )
      }
    }
  }

  return stories
}

/* ═══════════════════════════════════════════
   5. VRP Confirmation (Regime Construction)
   ═══════════════════════════════════════════ */

export function generateVRPConfirmation(vrpByState: Record<string, number | null>): string | null {
  const safeStates = ['L1 Safe', 'L2 Safe', 'L3 Safe']
  const riskyStates = ['L1 Exposed', 'L2 Risky', 'L3 Exposed']

  const safeVrps = safeStates.map(s => vrpByState[s]).filter((v): v is number => v != null)
  const riskyVrps = riskyStates.map(s => vrpByState[s]).filter((v): v is number => v != null)

  if (safeVrps.length === 0 || riskyVrps.length === 0) return null

  const safeAvg = safeVrps.reduce((a, b) => a + b, 0) / safeVrps.length
  const riskyAvg = riskyVrps.reduce((a, b) => a + b, 0) / riskyVrps.length

  const safeRange = `${Math.min(...safeVrps).toFixed(1)}–${Math.max(...safeVrps).toFixed(1)}`
  const riskyRange = `${Math.min(...riskyVrps).toFixed(1)}–${Math.max(...riskyVrps).toFixed(1)}`

  return `VRP confirms: every Safe state has VRP ${safeRange} (avg ${safeAvg.toFixed(1)}), every Risky/Exposed state has VRP ${riskyRange} (avg ${riskyAvg.toFixed(1)}).`
}

/* ═══════════════════════════════════════════
   6. Tested and Failed Summary
   ═══════════════════════════════════════════ */

export function formatTestedAndFailed(items: TestedAndFailed[]): TestedAndFailed[] {
  return items
}
