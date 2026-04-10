'use client'

import { useState, useEffect, useMemo, Fragment } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ComposedChart, Area, Line, ReferenceLine,
} from 'recharts'
import { fetchAdaptiveOOS } from '@/lib/api'
import type { AdaptiveOOSData, OOSStateMetric, BoundaryTimelineEntry } from '@/lib/types'

interface Props {
  snapshot: string
  dte: number | null
}

const SUB_TABS = [
  { key: '4', label: 'Jan 2026' },
  { key: '2', label: 'Feb 2026' },
  { key: '3', label: 'Mar 2026' },
  { key: '1', label: 'Jan 2021 – Jan 2023' },
] as const

const ALL_STATES = [
  'L1 Safe', 'L1 Exposed',
  'L2 Safe', 'L2 Caution-A', 'L2 Caution-B', 'L2 Risky',
  'L3 Safe', 'L3 Exposed',
]

const STATE_COLORS: Record<string, string> = {
  'L1 Safe': '#00e676', 'L1 Exposed': '#ffab40',
  'L2 Safe': '#448aff', 'L2 Caution-A': '#ffd740', 'L2 Caution-B': '#ff9100', 'L2 Risky': '#ff5252',
  'L3 Safe': '#b388ff', 'L3 Exposed': '#ff8a80',
}

function riskGradient(al: number | null): string {
  if (al == null) return 'var(--text-dim)'
  if (al <= 5) return 'var(--signal-positive)'
  if (al <= 10) return 'var(--signal-warning)'
  if (al <= 15) return '#ff9100'
  return 'var(--signal-negative)'
}

function sharpeColor(sh: number | null): string {
  if (sh == null) return 'var(--text-dim)'
  if (sh >= 4) return 'var(--signal-positive)'
  if (sh >= 2) return '#64ffda'
  if (sh >= 1) return 'var(--signal-warning)'
  return 'var(--signal-negative)'
}

function InsightBox({ children, variant = 'info' }: { children: React.ReactNode; variant?: 'info' | 'warning' | 'positive' | 'success' }) {
  const borderColor = variant === 'warning' ? 'var(--signal-negative)' : variant === 'positive' || variant === 'success' ? 'var(--signal-positive)' : '#bc8cff'
  const bgColor = variant === 'warning' ? 'rgba(248,81,73,0.04)' : variant === 'positive' || variant === 'success' ? 'rgba(63,185,80,0.04)' : 'rgba(188,140,255,0.04)'
  return (
    <div className="rounded-lg px-4 py-3 text-xs font-mono text-[var(--text-secondary)] leading-relaxed" style={{ borderLeft: `3px solid ${borderColor}`, backgroundColor: bgColor }}>
      {children}
    </div>
  )
}

function SectionHeader({ step, title, subtitle }: { step: number; title: string; subtitle?: string }) {
  return (
    <div className="mb-4 flex items-start gap-3">
      <div className="flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold font-mono" style={{ backgroundColor: 'rgba(188,140,255,0.15)', color: '#bc8cff' }}>
        {step}
      </div>
      <div>
        <h3 className="text-xs font-mono uppercase tracking-wider text-[var(--text-primary)] font-semibold">{title}</h3>
        {subtitle && <p className="text-[10px] font-mono text-[var(--text-muted)] mt-0.5">{subtitle}</p>}
      </div>
    </div>
  )
}

function ScoreCard({ label, value, sub, pass: passed }: { label: string; value: string; sub?: string; pass?: boolean | null }) {
  const borderColor = passed === true ? 'var(--signal-positive)' : passed === false ? 'var(--signal-negative)' : 'var(--border)'
  return (
    <div className="rounded-lg border p-3 text-center" style={{ borderColor, backgroundColor: 'var(--bg-elevated)' }}>
      <div className="text-[9px] font-mono text-[var(--text-dim)] uppercase tracking-wider mb-1">{label}</div>
      <div className="text-base font-bold font-mono" style={{ color: passed === true ? 'var(--signal-positive)' : passed === false ? 'var(--signal-negative)' : 'var(--text-primary)' }}>
        {value}
      </div>
      {sub && <div className="text-[9px] text-[var(--text-muted)] font-mono mt-0.5">{sub}</div>}
    </div>
  )
}

function PassFail({ value }: { value: boolean | null }) {
  if (value === null) return <span className="text-[var(--text-dim)]">N/A</span>
  return value
    ? <span className="text-[var(--signal-positive)] font-bold">PASS</span>
    : <span className="text-[var(--signal-negative)] font-bold">FAIL</span>
}

export function AdaptiveOOS({ snapshot, dte }: Props) {
  const [subTab, setSubTab] = useState<string>('4')
  const [data, setData] = useState<AdaptiveOOSData | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    fetchAdaptiveOOS(subTab, snapshot, dte)
      .then(setData)
      .finally(() => setLoading(false))
  }, [subTab, snapshot, dte])

  // Timeline chart data — sample every N points for rendering
  const timelineChart = useMemo(() => {
    if (!data) return []
    const tl = data.boundary_timeline
    const step = Math.max(1, Math.floor(tl.length / 200))
    return tl.filter((_: BoundaryTimelineEntry, i: number) => i % step === 0).map((t: BoundaryTimelineEntry) => ({
      date: t.date.slice(5), // MM-DD
      iv: t.iv_lag,
      highPct: t.trailing_45d_high_pct,
      shifted: t.boundary_shifted ? 1 : 0,
      pnl: t.pnl_combined,
    }))
  }, [data])

  // Safe vs Exposed bar chart data
  const safeVsExposedData = useMemo(() => {
    if (!data) return { fixed: [] as Array<Record<string, string | number | null>>, adaptive: [] as Array<Record<string, string | number | null>>, insample: [] as Array<Record<string, string | number | null>> }
    const mapper = (arr: typeof data.level_comparison_fixed) => arr.map(lc => ({
      level: lc.level,
      'Safe Sharpe': lc.safe_sharpe,
      'Exposed Sharpe': lc.exposed_sharpe,
      'Safe AL%': lc.safe_al,
      'Exposed AL%': lc.exposed_al,
      safe_days: lc.safe_days,
      exposed_days: lc.exposed_days,
    }))
    return {
      fixed: mapper(data.level_comparison_fixed),
      adaptive: mapper(data.level_comparison_adaptive),
      insample: mapper(data.level_comparison_insample),
    }
  }, [data])

  if (loading) return <div className="text-center py-20 text-[var(--text-dim)] text-sm font-mono">Loading adaptive OOS validation...</div>
  if (!data) return <div className="text-center py-20 text-[var(--text-dim)] text-sm font-mono">No data available</div>

  const v = data.validation

  return (
    <div className="space-y-6">
      {/* Sub-tab selector */}
      <div className="flex items-center gap-2">
        {SUB_TABS.map(t => (
          <button
            key={t.key}
            onClick={() => setSubTab(t.key)}
            className="px-4 py-2 rounded-lg text-xs font-mono font-semibold transition-all border"
            style={{
              backgroundColor: subTab === t.key ? 'rgba(188,140,255,0.12)' : 'var(--bg-card)',
              borderColor: subTab === t.key ? '#bc8cff' : 'var(--border)',
              color: subTab === t.key ? '#bc8cff' : 'var(--text-muted)',
            }}
          >
            {t.label}
          </button>
        ))}
        <div className="flex-1" />
        <div className="text-[10px] font-mono text-[var(--text-dim)]">
          Train: <strong className="text-[var(--text-secondary)]">{data.training_period.days}d</strong> ({data.training_period.start.slice(0,7)} → {data.training_period.end.slice(0,7)})
          {' · '}
          Test: <strong className="text-[var(--text-secondary)]">{data.test_period.days}d</strong>
        </div>
      </div>

      {/* ═══ STEP 1: Research — PK/IV Formula ═══ */}
      <div className="rounded-xl border border-[var(--border)] p-5" style={{ backgroundColor: 'var(--bg-card)' }}>
        <SectionHeader step={1} title="PK/IV Ratio — The Core Signal" subtitle="Why Parkinson / IV captures regime risk" />
        <div className="grid grid-cols-3 gap-4">
          <div className="rounded-lg border border-[var(--border)] p-4" style={{ backgroundColor: 'var(--bg-elevated)' }}>
            <div className="text-[10px] font-mono text-[var(--text-dim)] uppercase tracking-wider mb-2">Formula</div>
            <div className="text-sm font-mono text-[var(--text-primary)] mb-2">PK/IV = PK_5d / IV_5d</div>
            <ul className="space-y-1 text-[10px] font-mono text-[var(--text-secondary)]">
              <li><strong>PK_5d</strong> = 5-day avg Parkinson vol (realized intraday range)</li>
              <li><strong>IV_5d</strong> = 5-day avg implied vol (market&apos;s expected vol)</li>
              <li>Ratio &lt; median = &quot;Safe&quot; (IV cushion absorbs realized moves)</li>
              <li>Ratio &gt; median = &quot;Exposed&quot; (realized movement eating into premium)</li>
            </ul>
          </div>
          <div className="rounded-lg border border-[var(--border)] p-4" style={{ backgroundColor: 'var(--bg-elevated)' }}>
            <div className="text-[10px] font-mono text-[var(--text-dim)] uppercase tracking-wider mb-2">Why It Works</div>
            <ul className="space-y-1 text-[10px] font-mono text-[var(--text-secondary)]">
              <li>When IV is high relative to realized movement → sellers have cushion → Safe</li>
              <li>When realized movement catches up to IV → sellers are exposed → Risky</li>
              <li>5-day window smooths daily noise while staying responsive</li>
              <li>The <em>ratio</em> naturally adapts to different IV regimes</li>
            </ul>
          </div>
          <div className="rounded-lg border border-[var(--border)] p-4" style={{ backgroundColor: 'var(--bg-elevated)' }}>
            <div className="text-[10px] font-mono text-[var(--text-dim)] uppercase tracking-wider mb-2">Other Features Tested</div>
            <ul className="space-y-1 text-[10px] font-mono text-[var(--text-secondary)]">
              <li><strong>IV_chg_5d</strong> — IV direction (rising/falling). Adds value at L2 only.</li>
              <li><strong>VRP</strong> (IV − RV) — Confirms PK/IV but redundant as splitter.</li>
              <li><strong>RV_today</strong> — Too noisy alone. PK captures the same info better.</li>
              <li><strong>iv_lag</strong> — Used for level classification, not state splitting.</li>
            </ul>
          </div>
        </div>
      </div>

      {/* ═══ STEP 2: Adaptive IV Boundaries — Timeline ═══ */}
      <div className="rounded-xl border border-[var(--border)] p-5" style={{ backgroundColor: 'var(--bg-card)' }}>
        <SectionHeader step={2} title="Adaptive IV Boundaries" subtitle={`When >50% of trailing 45 days have IV>17, boundaries shift from [12,17] → [17,22]. ${v.shifted_days}/${v.total_test_days} days (${v.shift_pct}%) triggered the shift.`} />

        <div style={{ height: 280 }}>
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={timelineChart}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="date" tick={{ fill: 'var(--text-muted)', fontSize: 9, fontFamily: 'monospace' }} interval={Math.floor(timelineChart.length / 8)} />
              <YAxis yAxisId="left" tick={{ fill: 'var(--text-muted)', fontSize: 9, fontFamily: 'monospace' }} label={{ value: 'IV Lag', angle: -90, position: 'insideLeft', style: { fill: 'var(--text-dim)', fontSize: 9, fontFamily: 'monospace' } }} />
              <YAxis yAxisId="right" orientation="right" domain={[0, 100]} tick={{ fill: 'var(--text-muted)', fontSize: 9, fontFamily: 'monospace' }} label={{ value: 'Trail 45d %', angle: 90, position: 'insideRight', style: { fill: 'var(--text-dim)', fontSize: 9, fontFamily: 'monospace' } }} />
              <Tooltip content={(props) => {
                const { active, payload } = props as unknown as { active?: boolean; payload?: Array<{ payload: Record<string, number | null> }> }
                if (!active || !payload?.[0]) return null
                const d = payload[0].payload
                return (
                  <div className="rounded-lg border border-[var(--border)] px-3 py-2 text-[10px] font-mono" style={{ backgroundColor: 'var(--bg-card)' }}>
                    <div className="text-[var(--text-primary)]">IV: {d.iv?.toFixed(1)} · Trail%: {d.highPct?.toFixed(0)}%</div>
                    <div className="text-[var(--text-muted)]">{d.shifted ? 'Boundaries SHIFTED [17,22]' : 'Fixed [12,17]'}</div>
                  </div>
                )
              }} />
              <Area yAxisId="right" type="stepAfter" dataKey="highPct" fill="rgba(255,171,64,0.1)" stroke="rgba(255,171,64,0.3)" strokeWidth={1} />
              <ReferenceLine yAxisId="right" y={50} stroke="rgba(255,82,82,0.5)" strokeDasharray="5 5" label={{ value: '50% trigger', position: 'right', style: { fill: '#ff5252', fontSize: 9, fontFamily: 'monospace' } }} />
              <ReferenceLine yAxisId="left" y={12} stroke="rgba(0,230,118,0.3)" strokeDasharray="3 3" />
              <ReferenceLine yAxisId="left" y={17} stroke="rgba(255,215,64,0.3)" strokeDasharray="3 3" />
              <Line yAxisId="left" type="monotone" dataKey="iv" stroke="#58a6ff" strokeWidth={1.5} dot={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        <div className="flex items-center justify-center gap-6 mt-2 text-[10px] font-mono">
          <div className="flex items-center gap-1.5"><div className="w-4 h-0.5 rounded" style={{ backgroundColor: '#58a6ff' }} /> IV Lag</div>
          <div className="flex items-center gap-1.5"><div className="w-4 h-3 rounded-sm" style={{ backgroundColor: 'rgba(255,171,64,0.2)' }} /> Trailing 45d IV&gt;17 %</div>
          <div className="flex items-center gap-1.5"><div className="w-4 h-0.5 rounded" style={{ backgroundColor: '#ff5252' }} /> 50% Trigger</div>
          <div className="flex items-center gap-1.5"><div className="w-4 h-0.5 rounded" style={{ backgroundColor: 'rgba(0,230,118,0.3)' }} /> IV=12</div>
          <div className="flex items-center gap-1.5"><div className="w-4 h-0.5 rounded" style={{ backgroundColor: 'rgba(255,215,64,0.3)' }} /> IV=17</div>
        </div>

        {v.shifted_days > 0 && (
          <div className="mt-3 grid grid-cols-2 gap-3">
            <InsightBox>
              <strong className="text-[var(--text-primary)]">Non-shifted periods ({v.fixed_days}d):</strong>{' '}
              Sharpe {data.shift_period_metrics.non_shifted.sharpe?.toFixed(2) ?? '—'} · AL% {data.shift_period_metrics.non_shifted.al_pct ?? '—'}%
            </InsightBox>
            <InsightBox variant="warning">
              <strong className="text-[var(--text-primary)]">Shifted periods ({v.shifted_days}d):</strong>{' '}
              Sharpe {data.shift_period_metrics.shifted.sharpe?.toFixed(2) ?? '—'} · AL% {data.shift_period_metrics.shifted.al_pct ?? '—'}%
            </InsightBox>
          </div>
        )}
      </div>

      {/* ═══ STEP 3: 8-State Comparison — IS vs OOS Adaptive ═══ */}
      <div className="rounded-xl border border-[var(--border)] p-5" style={{ backgroundColor: 'var(--bg-card)' }}>
        <SectionHeader step={3} title="8-State Regime Classification — OOS Validation" subtitle="In-Sample model applied out-of-sample. Do the states hold their character?" />
        <div className="overflow-x-auto">
          <table className="w-full text-[10px] font-mono">
            <thead>
              <tr className="border-b border-[var(--border)]">
                <th className="text-left px-3 py-2 text-[var(--text-dim)] font-normal" rowSpan={2}>State</th>
                <th className="text-center px-2 py-1 text-[#64ffda]" colSpan={4}>In-Sample ({data.training_period.days}d)</th>
                <th className="text-center px-2 py-1 text-[#f0883e]" colSpan={4}>OOS Adaptive</th>
              </tr>
              <tr className="border-b border-[var(--border)]">
                {['Days', 'AL%', 'Sharpe', 'Avg'].map(h => (
                  <th key={`is-${h}`} className="text-right px-2 py-1 text-[var(--text-dim)] font-normal">{h}</th>
                ))}
                {['Days', 'AL%', 'Sharpe', 'Avg'].map(h => (
                  <th key={`ad-${h}`} className="text-right px-2 py-1 text-[var(--text-dim)] font-normal">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {ALL_STATES.map(state => {
                const is_ = data.insample_states.find((s: OOSStateMetric) => s.state === state)
                const ad = data.adaptive_states.find((s: OOSStateMetric) => s.state === state)
                const renderCells = (m: OOSStateMetric | undefined) => (
                  <>
                    <td className="text-right px-2 py-2 text-[var(--text-secondary)]">{m?.days ?? 0}</td>
                    <td className="text-right px-2 py-2 font-semibold" style={{ color: riskGradient(m?.al_pct ?? null) }}>
                      {m?.al_pct != null ? `${m.al_pct}%` : '—'}
                    </td>
                    <td className="text-right px-2 py-2 font-semibold" style={{ color: sharpeColor(m?.sharpe ?? null) }}>
                      {m?.sharpe != null ? m.sharpe.toFixed(2) : '—'}
                    </td>
                    <td className="text-right px-2 py-2 text-[var(--text-secondary)]">
                      {m?.port_avg != null ? `${m.port_avg > 0 ? '+' : ''}${m.port_avg.toFixed(3)}` : '—'}
                    </td>
                  </>
                )
                return (
                  <tr key={state} className="border-b border-[var(--border-subtle)] hover:bg-[var(--bg-hover)]">
                    <td className="px-3 py-2 font-semibold" style={{ color: STATE_COLORS[state] }}>{state}</td>
                    {renderCells(is_)}
                    {renderCells(ad)}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* ═══ STEP 4: PK/IV Ratio Validation — Safe vs Exposed ═══ */}
      <div className="rounded-xl border border-[var(--border)] p-5" style={{ backgroundColor: 'var(--bg-card)' }}>
        <SectionHeader step={4} title="Does PK/IV Ratio Work OOS?" subtitle="Core question: do Safe states (low PK/IV) still beat Exposed states (high PK/IV) out-of-sample?" />

        <div className="grid grid-cols-3 gap-4 mb-4">
          {/* In-Sample */}
          <div>
            <div className="text-[10px] font-mono text-[#64ffda] uppercase tracking-wider mb-2 text-center">In-Sample</div>
            <div style={{ height: 200 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={safeVsExposedData.insample} barCategoryGap="20%">
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="level" tick={{ fill: 'var(--text-muted)', fontSize: 10, fontFamily: 'monospace' }} />
                  <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 9, fontFamily: 'monospace' }} />
                  <Tooltip />
                  <Bar dataKey="Safe Sharpe" fill="#3fb950" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="Exposed Sharpe" fill="#f85149" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          {/* OOS Adaptive */}
          <div>
            <div className="text-[10px] font-mono text-[#f0883e] uppercase tracking-wider mb-2 text-center">OOS — Adaptive</div>
            <div style={{ height: 200 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={safeVsExposedData.adaptive} barCategoryGap="20%">
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="level" tick={{ fill: 'var(--text-muted)', fontSize: 10, fontFamily: 'monospace' }} />
                  <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 9, fontFamily: 'monospace' }} />
                  <Tooltip />
                  <Bar dataKey="Safe Sharpe" fill="#3fb950" radius={[3, 3, 0, 0]} />
                  <Bar dataKey="Exposed Sharpe" fill="#f85149" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* AL% comparison table */}
        <div className="overflow-x-auto">
          <table className="w-full text-[10px] font-mono">
            <thead>
              <tr className="border-b border-[var(--border)]">
                <th className="text-left px-3 py-2 text-[var(--text-dim)] font-normal">Level</th>
                <th className="text-center px-2 py-1 text-[#64ffda]" colSpan={2}>In-Sample AL%</th>
                <th className="text-center px-2 py-1 text-[#f0883e]" colSpan={2}>OOS Adaptive AL%</th>
              </tr>
              <tr className="border-b border-[var(--border)]">
                {[1, 2].map(i => (
                  <Fragment key={i}>
                    <th className="text-right px-2 py-1 text-[var(--text-dim)] font-normal">Safe</th>
                    <th className="text-right px-2 py-1 text-[var(--text-dim)] font-normal">Exposed</th>
                  </Fragment>
                ))}
              </tr>
            </thead>
            <tbody>
              {['L1', 'L2', 'L3'].map(lvl => {
                const isLc = data.level_comparison_insample.find(l => l.level === lvl)
                const adLc = data.level_comparison_adaptive.find(l => l.level === lvl)
                const renderPair = (lc: typeof isLc) => (
                  <>
                    <td className="text-right px-2 py-2 font-semibold" style={{ color: riskGradient(lc?.safe_al ?? null) }}>{lc?.safe_al != null ? `${lc.safe_al}%` : '—'}</td>
                    <td className="text-right px-2 py-2 font-semibold" style={{ color: riskGradient(lc?.exposed_al ?? null) }}>{lc?.exposed_al != null ? `${lc.exposed_al}%` : '—'}</td>
                  </>
                )
                return (
                  <tr key={lvl} className="border-b border-[var(--border-subtle)]">
                    <td className="px-3 py-2 font-semibold text-[var(--text-primary)]">{lvl}</td>
                    {renderPair(isLc)}
                    {renderPair(adLc)}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* ═══ STEP 5: Validation Scorecard ═══ */}
      <div className="rounded-xl border border-[var(--border)] p-5" style={{ backgroundColor: 'var(--bg-card)' }}>
        <SectionHeader step={5} title="Validation Scorecard" subtitle="4 tests the system must pass to be considered robust OOS" />

        <div className="grid grid-cols-3 gap-3 mb-4">
          <ScoreCard
            label="Rank Correlation"
            value={v.rank_corr_adaptive != null ? v.rank_corr_adaptive.toFixed(3) : 'N/A'}
            sub="Spearman rho IS vs OOS"
            pass={v.rank_corr_adaptive != null ? v.rank_corr_adaptive > 0.4 : null}
          />
          <ScoreCard
            label="Boundary Shift"
            value={`${v.shift_pct}%`}
            sub={`${v.shifted_days} / ${v.total_test_days} days`}
          />
          <ScoreCard
            label="Regime Disagreement"
            value={`${v.disagreement_pct}%`}
            sub={`${v.regime_disagreement} days differ`}
          />
        </div>

        {/* Safe vs Exposed checks */}
        <div className="rounded-lg border border-[var(--border)] p-4" style={{ backgroundColor: 'var(--bg-elevated)' }}>
          <div className="text-[10px] font-mono text-[var(--text-dim)] uppercase tracking-wider mb-3">Safe Outperforms Exposed? (Lower AL% = Better)</div>
          <div className="space-y-1">
            {(['L1', 'L2', 'L3'] as const).map(lvl => (
              <div key={lvl} className="flex items-center gap-3 text-xs font-mono">
                <span className="w-6 text-[var(--text-primary)]">{lvl}</span>
                <PassFail value={v.safe_checks_adaptive[lvl] ?? null} />
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ═══ STEP 6: PK/IV Medians Diagnostic ═══ */}
      <div className="rounded-xl border border-[var(--border)] p-5" style={{ backgroundColor: 'var(--bg-card)' }}>
        <SectionHeader step={6} title="PK/IV Threshold Stability" subtitle="Are the median PK/IV ratios stable between in-sample and out-of-sample?" />
        <div className="overflow-x-auto">
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="border-b border-[var(--border)]">
                <th className="text-left px-4 py-2 text-[var(--text-dim)] font-normal">Level</th>
                <th className="text-right px-4 py-2 text-[#64ffda]">IS Median</th>
                <th className="text-right px-4 py-2 text-[#f0883e]">IS Shifted Median</th>
              </tr>
            </thead>
            <tbody>
              {['L1', 'L2', 'L3'].map(lvl => {
                const isMed = data.pkiv_diagnostic.fixed_medians_used[lvl]
                const shiftedMed = data.pkiv_diagnostic.shifted_medians_used[lvl]
                return (
                  <tr key={lvl} className="border-b border-[var(--border-subtle)]">
                    <td className="px-4 py-2.5 font-semibold text-[var(--text-primary)]">{lvl}</td>
                    <td className="text-right px-4 py-2.5 text-[var(--text-primary)]">{isMed?.toFixed(4) ?? '—'}</td>
                    <td className="text-right px-4 py-2.5 text-[var(--text-secondary)]">{shiftedMed?.toFixed(4) ?? '—'}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
        <div className="mt-3">
          <InsightBox>
            <strong className="text-[var(--text-primary)]">Stability check:</strong> If the delta between IS and OOS medians is small (&lt;0.05), the PK/IV ratio is a stable feature —
            it doesn&apos;t just overfit to the training period. Large deltas suggest structural market regime changes that may require adaptive thresholds.
          </InsightBox>
        </div>
      </div>

      {/* ═══ STEP 7: Strategy Deep Dive ═══ */}
      <div className="rounded-xl border border-[var(--border)] p-5" style={{ backgroundColor: 'var(--bg-card)' }}>
        <SectionHeader step={7} title="Strategy Performance by State — OOS" subtitle="Do DM, WC, and Orion maintain their IS character out-of-sample?" />
        <div className="overflow-x-auto">
          <table className="w-full text-[10px] font-mono">
            <thead>
              <tr className="border-b border-[var(--border)]">
                <th className="text-left px-3 py-2 text-[var(--text-dim)] font-normal" rowSpan={2}>State</th>
                <th className="text-center px-1 py-1 text-[#64ffda]" colSpan={3}>In-Sample Sharpe</th>
                <th className="text-center px-1 py-1 text-[#f0883e]" colSpan={3}>OOS Adaptive Sharpe</th>
              </tr>
              <tr className="border-b border-[var(--border)]">
                {['DM', 'WC', 'OR', 'DM', 'WC', 'OR'].map((h, i) => (
                  <th key={i} className="text-center px-2 py-1.5 text-[var(--text-dim)] font-normal">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {ALL_STATES.map(state => {
                const is_ = data.insample_states.find((s: OOSStateMetric) => s.state === state)
                const ad = data.adaptive_states.find((s: OOSStateMetric) => s.state === state)
                const renderStrats = (m: OOSStateMetric | undefined) => (
                  <>
                    <td className="text-center px-2 py-2 font-semibold" style={{ color: sharpeColor(m?.dm_sharpe ?? null) }}>{m?.dm_sharpe?.toFixed(1) ?? '—'}</td>
                    <td className="text-center px-2 py-2 font-semibold" style={{ color: sharpeColor(m?.wc_sharpe ?? null) }}>{m?.wc_sharpe?.toFixed(1) ?? '—'}</td>
                    <td className="text-center px-2 py-2 font-semibold" style={{ color: sharpeColor(m?.orion_sharpe ?? null) }}>{m?.orion_sharpe?.toFixed(1) ?? '—'}</td>
                  </>
                )
                return (
                  <tr key={state} className="border-b border-[var(--border-subtle)] hover:bg-[var(--bg-hover)]">
                    <td className="px-3 py-2 font-semibold" style={{ color: STATE_COLORS[state] }}>{state}</td>
                    {renderStrats(is_)}
                    {renderStrats(ad)}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* ═══ FINAL VERDICT ═══ */}
      <div className="rounded-xl border-2 border-[#bc8cff] overflow-hidden" style={{ background: 'linear-gradient(135deg, rgba(188,140,255,0.05), var(--bg-card))' }}>
        <div className="px-6 py-5">
          <h3 className="text-sm font-mono font-bold text-[#bc8cff] uppercase tracking-wider mb-4">OOS Verdict — {data.test_period.label}</h3>
          <div className="space-y-3">
            <InsightBox variant={
              Object.values(v.safe_checks_adaptive).filter(v => v === true).length >= 2 ? 'success' : 'warning'
            }>
              <strong className="text-[var(--text-primary)]">PK/IV Ratio:</strong>{' '}
              Safe beats Exposed at {Object.values(v.safe_checks_adaptive).filter(v => v === true).length}/3 levels.
              {Object.values(v.safe_checks_adaptive).filter(v => v === true).length >= 2
                ? ' The core signal holds OOS — PK/IV ratio is a genuine regime discriminator.'
                : ' Partial degradation OOS — the signal weakens in some regimes.'}
            </InsightBox>

            {v.rank_corr_adaptive != null && (
              <InsightBox variant={v.rank_corr_adaptive > 0.4 ? 'positive' : 'warning'}>
                <strong className="text-[var(--text-primary)]">State Ordering:</strong>{' '}
                Rank correlation between IS and OOS state performance: <strong>{v.rank_corr_adaptive.toFixed(3)}</strong>.
                {v.rank_corr_adaptive > 0.6 ? ' Strong preservation — state rankings are stable.' :
                  v.rank_corr_adaptive > 0.3 ? ' Moderate preservation — some state re-ordering but overall pattern holds.' :
                    ' Weak preservation — significant state re-ordering OOS.'}
              </InsightBox>
            )}

            <InsightBox variant={v.shift_pct > 30 ? 'warning' : 'info'}>
              <strong className="text-[var(--text-primary)]">Adaptive Boundaries:</strong>{' '}
              Triggered on {v.shift_pct}% of test days ({v.shifted_days}/{v.total_test_days}).
              {v.shift_pct > 40 ? ' Heavy shifting — the adaptive system is actively adjusting boundaries for this period.' :
                v.shift_pct > 10 ? ' Moderate shifting — the market transitions between regimes during this period.' :
                  ' Minimal shifting — base boundaries [12,17] held throughout this period.'}
              {' '}Regime disagreement: {v.disagreement_pct}% of days.
            </InsightBox>

            {data.test_period.days < 50 && (
              <InsightBox variant="warning">
                <strong className="text-[var(--text-primary)]">Sample Size Warning:</strong>{' '}
                Only {data.test_period.days} test days — too few for statistically robust conclusions.
                Most states have &lt;20 observations. Treat these results as directional indicators, not definitive proof.
              </InsightBox>
            )}
          </div>
        </div>
      </div>

      {/* ═══ Day-by-Day Log (collapsible) ═══ */}
      <details className="rounded-xl border border-[var(--border)] overflow-hidden" style={{ backgroundColor: 'var(--bg-card)' }}>
        <summary className="px-5 py-3 cursor-pointer text-xs font-mono text-[var(--text-secondary)] hover:bg-[var(--bg-hover)] transition-colors">
          Day-by-Day Regime Assignment Log ({data.boundary_timeline.length} days) — Click to expand
        </summary>
        <div className="overflow-x-auto max-h-[500px] overflow-y-auto">
          <table className="w-full text-[10px] font-mono">
            <thead className="sticky top-0" style={{ backgroundColor: 'var(--bg-elevated)' }}>
              <tr className="border-b border-[var(--border)]">
                <th className="text-left px-3 py-2 text-[var(--text-dim)] font-normal">Date</th>
                <th className="text-right px-2 py-2 text-[var(--text-dim)] font-normal">IV</th>
                <th className="text-right px-2 py-2 text-[var(--text-dim)] font-normal">PK/IV</th>
                <th className="text-right px-2 py-2 text-[var(--text-dim)] font-normal">Trail%</th>
                <th className="text-center px-2 py-2 text-[var(--text-dim)] font-normal">Shifted?</th>
                <th className="text-left px-2 py-2 text-[var(--text-dim)] font-normal">Regime</th>
                <th className="text-right px-2 py-2 text-[var(--text-dim)] font-normal">PnL</th>
              </tr>
            </thead>
            <tbody>
              {data.boundary_timeline.map((t: BoundaryTimelineEntry) => {
                return (
                  <tr key={t.date} className="border-b border-[var(--border-subtle)]">
                    <td className="px-3 py-1.5 text-[var(--text-secondary)]">{t.date}</td>
                    <td className="text-right px-2 py-1.5 text-[var(--text-primary)]">{t.iv_lag?.toFixed(1) ?? '—'}</td>
                    <td className="text-right px-2 py-1.5 text-[var(--text-primary)]">{t.pk_iv_ratio?.toFixed(3) ?? '—'}</td>
                    <td className="text-right px-2 py-1.5 text-[var(--text-muted)]">{t.trailing_45d_high_pct}%</td>
                    <td className="text-center px-2 py-1.5">
                      {t.boundary_shifted
                        ? <span className="text-[var(--signal-warning)]">YES</span>
                        : <span className="text-[var(--text-dim)]">no</span>}
                    </td>
                    <td className="px-2 py-1.5" style={{ color: STATE_COLORS[t.regime_adaptive ?? ''] ?? 'var(--text-muted)' }}>{t.regime_adaptive ?? '—'}</td>
                    <td className="text-right px-2 py-1.5" style={{ color: t.pnl_combined != null ? (t.pnl_combined > 0 ? 'var(--signal-positive)' : 'var(--signal-negative)') : 'var(--text-dim)' }}>
                      {t.pnl_combined != null ? `${t.pnl_combined > 0 ? '+' : ''}${t.pnl_combined.toFixed(3)}` : '—'}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </details>

    </div>
  )
}
