'use client'

import { useState, useEffect, useMemo } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ScatterChart, Scatter, Cell, Legend,
} from 'recharts'
import { fetchSnapshotComparison } from '@/lib/api'
import type { SnapshotComparisonData, SnapshotResult, SnapshotStateMetric } from '@/lib/types'

interface Props {
  startDate: string
  endDate: string
  dte: number | null
}

const SNAPSHOT_ORDER = ['1530', '1529', '0915', '0916'] as const
const SNAPSHOT_COLORS: Record<string, string> = {
  '1530': '#58a6ff',
  '1529': '#79c0ff',
  '0915': '#f0883e',
  '0916': '#d29922',
}
const SNAPSHOT_LABELS: Record<string, string> = {
  '1530': '3:30 PM',
  '1529': '3:29 PM',
  '0915': '9:15 AM',
  '0916': '9:16 AM',
}

const ALL_STATES = [
  'L1 Safe', 'L1 Exposed',
  'L2 Safe', 'L2 Caution-A', 'L2 Caution-B', 'L2 Risky',
  'L3 Safe', 'L3 Exposed',
]

const STATE_COLORS: Record<string, string> = {
  'L1 Safe': '#3fb950', 'L1 Exposed': '#f85149',
  'L2 Safe': '#58a6ff', 'L2 Caution-A': '#d29922', 'L2 Caution-B': '#f0883e', 'L2 Risky': '#f85149',
  'L3 Safe': '#bc8cff', 'L3 Exposed': '#ff7b72',
}

function riskGradient(alPct: number | null): string {
  if (alPct == null) return 'var(--text-dim)'
  if (alPct <= 5) return 'var(--signal-positive)'
  if (alPct <= 10) return 'var(--signal-warning)'
  if (alPct <= 15) return '#ff9100'
  return 'var(--signal-negative)'
}

function sharpeColor(sh: number | null): string {
  if (sh == null) return 'var(--text-dim)'
  if (sh >= 4) return 'var(--signal-positive)'
  if (sh >= 2) return '#64ffda'
  if (sh >= 1) return 'var(--signal-warning)'
  return 'var(--signal-negative)'
}

function deltaColor(v: number): string {
  if (v > 0.5) return 'var(--signal-positive)'
  if (v < -0.5) return 'var(--signal-negative)'
  return 'var(--text-muted)'
}

function StatCard({ label, value, sub, color }: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <div className="stat-card !p-3 text-center">
      <div className="text-[9px] font-mono text-[var(--text-dim)] uppercase tracking-wider mb-1">{label}</div>
      <div className="text-lg font-bold font-mono" style={{ color: color ?? 'var(--text-primary)' }}>{value}</div>
      {sub && <div className="text-[9px] text-[var(--text-muted)] font-mono mt-0.5">{sub}</div>}
    </div>
  )
}

function SectionHeader({ title, subtitle }: { title: string; subtitle?: string }) {
  return (
    <div className="mb-4">
      <h3 className="text-xs font-mono uppercase tracking-wider text-[var(--text-dim)]">{title}</h3>
      {subtitle && <p className="text-[10px] font-mono text-[var(--text-muted)] mt-1">{subtitle}</p>}
    </div>
  )
}

function InsightBox({ children, variant = 'info' }: { children: React.ReactNode; variant?: 'info' | 'warning' | 'positive' }) {
  const borderColor = variant === 'warning' ? 'var(--signal-negative)' : variant === 'positive' ? 'var(--signal-positive)' : '#bc8cff'
  const bgColor = variant === 'warning' ? 'rgba(248,81,73,0.04)' : variant === 'positive' ? 'rgba(63,185,80,0.04)' : 'rgba(188,140,255,0.04)'
  return (
    <div className="rounded-lg px-4 py-3 text-xs font-mono text-[var(--text-secondary)] leading-relaxed" style={{ borderLeft: `3px solid ${borderColor}`, backgroundColor: bgColor }}>
      {children}
    </div>
  )
}

// Custom tooltip for bar charts
function BarTooltip({ active, payload, label }: { active?: boolean; payload?: Array<{ name: string; value: number; fill: string }>; label?: string }) {
  if (!active || !payload) return null
  return (
    <div className="rounded-lg border border-[var(--border)] px-3 py-2 text-[10px] font-mono" style={{ backgroundColor: 'var(--bg-card)' }}>
      <div className="text-[var(--text-primary)] font-semibold mb-1">{label}</div>
      {payload.map((p, i) => (
        <div key={i} className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full" style={{ backgroundColor: p.fill }} />
          <span className="text-[var(--text-muted)]">{p.name}:</span>
          <span className="text-[var(--text-primary)]">{typeof p.value === 'number' ? p.value.toFixed(1) : p.value}</span>
        </div>
      ))}
    </div>
  )
}

export function SnapshotComparison({ startDate, endDate, dte }: Props) {
  const [data, setData] = useState<SnapshotComparisonData | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    fetchSnapshotComparison(startDate, endDate, dte)
      .then(setData)
      .finally(() => setLoading(false))
  }, [startDate, endDate, dte])

  // Derived data
  const snapshots = useMemo(() => {
    if (!data) return []
    return SNAPSHOT_ORDER.map(s => ({ key: s, ...data.snapshots[s] })).filter(s => !s.error)
  }, [data])

  const levelDistData = useMemo(() => {
    if (!snapshots.length) return []
    return ['L1', 'L2', 'L3'].map(lvl => {
      const row: Record<string, string | number> = { level: lvl }
      snapshots.forEach(s => {
        row[s.key] = s.level_distribution[lvl]?.pct ?? 0
      })
      return row
    })
  }, [snapshots])

  const stateCompareData = useMemo(() => {
    if (!snapshots.length) return []
    return ALL_STATES.map(state => {
      const row: Record<string, string | number | null> = { state }
      snapshots.forEach(s => {
        const m = s.state_metrics.find((sm: SnapshotStateMetric) => sm.state === state)
        row[`${s.key}_al`] = m?.al_pct ?? null
        row[`${s.key}_sharpe`] = m?.sharpe ?? null
        row[`${s.key}_days`] = m?.days ?? 0
        row[`${s.key}_port`] = m?.port_avg ?? null
      })
      return row
    })
  }, [snapshots])

  const vrpCompareData = useMemo(() => {
    if (!snapshots.length) return []
    return ALL_STATES.map(state => {
      const row: Record<string, string | number | null> = { state: state.replace('L1 ', '').replace('L2 ', '').replace('L3 ', '') + ` (${state.split(' ')[0]})` }
      snapshots.forEach(s => {
        row[s.key] = s.vrp_by_state[state] ?? null
      })
      return row
    })
  }, [snapshots])

  // Radar data for overall metrics comparison
  const radarData = useMemo(() => {
    if (!snapshots.length) return []
    // Normalize metrics to 0-100 scale for radar
    const metrics = [
      { metric: 'Sharpe', key: 'sharpe', max: 8 },
      { metric: 'L1 Safe Sharpe', key: 'l1s_sharpe', max: 8 },
      { metric: 'L2 Safe Sharpe', key: 'l2s_sharpe', max: 8 },
      { metric: 'L3 Safe Sharpe', key: 'l3s_sharpe', max: 8 },
      { metric: 'L2 Discrim.', key: 'l2_discrim', max: 15 },
      { metric: 'L3 Volume', key: 'l3_vol', max: 40 },
    ]
    return metrics.map(m => {
      const row: Record<string, string | number> = { metric: m.metric }
      snapshots.forEach(s => {
        let val = 0
        if (m.key === 'sharpe') val = s.overall.sharpe ?? 0
        else if (m.key === 'l1s_sharpe') val = s.state_metrics.find((sm: SnapshotStateMetric) => sm.state === 'L1 Safe')?.sharpe ?? 0
        else if (m.key === 'l2s_sharpe') val = s.state_metrics.find((sm: SnapshotStateMetric) => sm.state === 'L2 Safe')?.sharpe ?? 0
        else if (m.key === 'l3s_sharpe') val = s.state_metrics.find((sm: SnapshotStateMetric) => sm.state === 'L3 Safe')?.sharpe ?? 0
        else if (m.key === 'l2_discrim') {
          const safe = s.state_metrics.find((sm: SnapshotStateMetric) => sm.state === 'L2 Safe')?.al_pct ?? 0
          const risky = s.state_metrics.find((sm: SnapshotStateMetric) => sm.state === 'L2 Risky')?.al_pct ?? 0
          val = Math.abs((risky ?? 0) - (safe ?? 0))
        }
        else if (m.key === 'l3_vol') val = s.level_distribution['L3']?.pct ?? 0
        row[s.key] = Math.min(100, (val / m.max) * 100)
      })
      return row
    })
  }, [snapshots])

  // Scatter data: Sharpe vs AL% for each state across snapshots
  const scatterData = useMemo(() => {
    if (!snapshots.length) return []
    const points: Array<{ x: number; y: number; state: string; snapshot: string; color: string }> = []
    snapshots.forEach(s => {
      s.state_metrics.forEach((m: SnapshotStateMetric) => {
        if (m.sharpe != null && m.al_pct != null) {
          points.push({
            x: m.al_pct,
            y: m.sharpe,
            state: m.state,
            snapshot: SNAPSHOT_LABELS[s.key],
            color: SNAPSHOT_COLORS[s.key],
          })
        }
      })
    })
    return points
  }, [snapshots])

  if (loading) return <div className="text-center py-20 text-[var(--text-dim)] text-sm font-mono">Loading snapshot comparison...</div>
  if (!data || !snapshots.length) return <div className="text-center py-20 text-[var(--text-dim)] text-sm font-mono">No data available</div>

  const agr = data.agreement

  // Find winners for each state
  const getWinner = (state: string, metric: 'sharpe' | 'al_pct'): string => {
    let best = ''
    let bestVal = metric === 'sharpe' ? -Infinity : Infinity
    snapshots.forEach(s => {
      const m = s.state_metrics.find((sm: SnapshotStateMetric) => sm.state === state)
      const v = metric === 'sharpe' ? (m?.sharpe ?? -Infinity) : (m?.al_pct ?? Infinity)
      if (metric === 'sharpe' && v > bestVal) { bestVal = v; best = s.key }
      if (metric === 'al_pct' && v < bestVal) { bestVal = v; best = s.key }
    })
    return best
  }

  return (
    <div className="space-y-6">

      {/* ═══ HERO: Agreement Overview ═══ */}
      <div className="rounded-xl border border-[var(--border)] overflow-hidden" style={{ backgroundColor: 'var(--bg-card)' }}>
        <div className="px-6 py-5">
          <div className="text-[10px] font-mono text-[var(--text-dim)] uppercase tracking-widest mb-3">Snapshot Agreement</div>
          <div className="grid grid-cols-4 gap-4">
            <StatCard label="All 4 Agree" value={`${agr.all_four_agree_pct}%`} sub={`${agr.total_days} overlapping days`} color={agr.all_four_agree_pct > 60 ? 'var(--signal-positive)' : 'var(--signal-warning)'} />
            <StatCard label="Close Pair (3:30/3:29)" value={`${agr.close_pair_agree_pct}%`} color={agr.close_pair_agree_pct > 80 ? 'var(--signal-positive)' : 'var(--signal-warning)'} />
            <StatCard label="Morning Pair (9:15/9:16)" value={`${agr.morning_pair_agree_pct}%`} color={agr.morning_pair_agree_pct > 80 ? 'var(--signal-positive)' : 'var(--signal-warning)'} />
            <StatCard label="Close vs Morning" value={`${agr.close_vs_morning_pct}%`} sub="3:30 PM vs 9:15 AM" color={agr.close_vs_morning_pct > 60 ? 'var(--signal-positive)' : 'var(--signal-warning)'} />
          </div>
        </div>
        <div className="px-6 py-3 border-t border-[var(--border)] bg-[var(--bg-elevated)]">
          <InsightBox variant="info">
            <strong className="text-[var(--text-primary)]">Reading this:</strong> Close pairs (T-1) use yesterday&apos;s close IV separated by 1 minute — high agreement expected.
            Morning pairs (T0) use today&apos;s opening IV. Cross-group disagreement reveals days where overnight/opening moves genuinely change the regime.
            <strong className="text-[var(--text-primary)]"> Days where all 4 agree = highest conviction.</strong>
          </InsightBox>
        </div>
      </div>

      {/* ═══ OVERALL METRICS COMPARISON ═══ */}
      <div className="rounded-xl border border-[var(--border)] p-5" style={{ backgroundColor: 'var(--bg-card)' }}>
        <SectionHeader title="Overall Portfolio Metrics" subtitle="Same strategies, same returns — only the regime classification changes" />
        <div className="overflow-x-auto">
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="border-b border-[var(--border)]">
                <th className="text-left px-4 py-2 text-[var(--text-dim)] font-normal">Metric</th>
                {snapshots.map(s => (
                  <th key={s.key} className="text-right px-4 py-2 font-semibold" style={{ color: SNAPSHOT_COLORS[s.key] }}>
                    {SNAPSHOT_LABELS[s.key]}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-[var(--border-subtle)]">
                <td className="px-4 py-2.5 text-[var(--text-secondary)]">Total Days</td>
                {snapshots.map(s => <td key={s.key} className="text-right px-4 py-2.5 text-[var(--text-primary)]">{s.overall.days}</td>)}
              </tr>
              <tr className="border-b border-[var(--border-subtle)]">
                <td className="px-4 py-2.5 text-[var(--text-secondary)]">Portfolio Sharpe</td>
                {snapshots.map(s => (
                  <td key={s.key} className="text-right px-4 py-2.5 font-semibold" style={{ color: sharpeColor(s.overall.sharpe) }}>
                    {s.overall.sharpe?.toFixed(2) ?? '—'}
                  </td>
                ))}
              </tr>
              <tr className="border-b border-[var(--border-subtle)]">
                <td className="px-4 py-2.5 text-[var(--text-secondary)]">All-Lose %</td>
                {snapshots.map(s => (
                  <td key={s.key} className="text-right px-4 py-2.5 font-semibold" style={{ color: riskGradient(s.overall.al_pct) }}>
                    {s.overall.al_pct != null ? `${s.overall.al_pct}%` : '—'}
                  </td>
                ))}
              </tr>
              <tr className="border-b border-[var(--border-subtle)]">
                <td className="px-4 py-2.5 text-[var(--text-secondary)]">Port Avg Return</td>
                {snapshots.map(s => (
                  <td key={s.key} className="text-right px-4 py-2.5 text-[var(--signal-positive)]">
                    {s.overall.port_avg != null ? `${s.overall.port_avg > 0 ? '+' : ''}${s.overall.port_avg.toFixed(4)}%` : '—'}
                  </td>
                ))}
              </tr>
              <tr className="border-b border-[var(--border-subtle)]">
                <td className="px-4 py-2.5 text-[var(--text-secondary)]">Mean IV (lag)</td>
                {snapshots.map(s => (
                  <td key={s.key} className="text-right px-4 py-2.5 text-[var(--text-primary)]">
                    {s.iv_stats.mean?.toFixed(2) ?? '—'}
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>
        <div className="mt-4">
          <InsightBox variant="positive">
            Portfolio-level metrics are nearly <strong className="text-[var(--text-primary)]">identical</strong> across all 4 snapshots — same returns, same strategies.
            The differences emerge in <em>how days are classified</em>, not in overall performance.
          </InsightBox>
        </div>
      </div>

      {/* ═══ LEVEL DISTRIBUTION BAR CHART ═══ */}
      <div className="rounded-xl border border-[var(--border)] p-5" style={{ backgroundColor: 'var(--bg-card)' }}>
        <SectionHeader title="IV Level Distribution" subtitle="Morning IV inflation pushes days from L1 into L2/L3" />
        <div className="grid grid-cols-2 gap-6">
          <div style={{ height: 280 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={levelDistData} barCategoryGap="25%">
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="level" tick={{ fill: 'var(--text-muted)', fontSize: 11, fontFamily: 'monospace' }} />
                <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 10, fontFamily: 'monospace' }} domain={[0, 50]} tickFormatter={(v: number) => `${v}%`} />
                <Tooltip content={<BarTooltip />} />
                {SNAPSHOT_ORDER.map(s => (
                  <Bar key={s} dataKey={s} name={SNAPSHOT_LABELS[s]} fill={SNAPSHOT_COLORS[s]} radius={[3, 3, 0, 0]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="space-y-3">
            <div className="overflow-x-auto">
              <table className="w-full text-xs font-mono">
                <thead>
                  <tr className="border-b border-[var(--border)]">
                    <th className="text-left px-3 py-2 text-[var(--text-dim)] font-normal">Level</th>
                    {snapshots.map(s => (
                      <th key={s.key} className="text-right px-3 py-2 font-semibold" style={{ color: SNAPSHOT_COLORS[s.key] }}>{SNAPSHOT_LABELS[s.key]}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {['L1', 'L2', 'L3'].map(lvl => (
                    <tr key={lvl} className="border-b border-[var(--border-subtle)]">
                      <td className="px-3 py-2 font-semibold text-[var(--text-primary)]">{lvl}</td>
                      {snapshots.map(s => {
                        const d = s.level_distribution[lvl]
                        return (
                          <td key={s.key} className="text-right px-3 py-2">
                            <span className="text-[var(--text-primary)]">{d?.pct}%</span>
                            <span className="text-[var(--text-dim)] ml-1">({d?.days}d)</span>
                          </td>
                        )
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <InsightBox variant="warning">
              <strong className="text-[var(--text-primary)]">Morning IV inflation:</strong> L1 shrinks from ~31% (close) to ~21% (morning) — 117+ days pushed above IV=12.
              L3 expands from ~27% to ~35%. The same static boundaries [12, 17] capture <em>different populations</em> at different times of day.
            </InsightBox>
          </div>
        </div>
      </div>

      {/* ═══ RADAR CHART: Snapshot Strengths ═══ */}
      <div className="rounded-xl border border-[var(--border)] p-5" style={{ backgroundColor: 'var(--bg-card)' }}>
        <SectionHeader title="Snapshot Strength Profile" subtitle="Normalized metrics — higher = better (except L3 Volume which is just more opportunity)" />
        <div className="grid grid-cols-2 gap-6">
          <div style={{ height: 340 }}>
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="75%">
                <PolarGrid stroke="var(--border)" />
                <PolarAngleAxis dataKey="metric" tick={{ fill: 'var(--text-muted)', fontSize: 10, fontFamily: 'monospace' }} />
                <PolarRadiusAxis tick={false} axisLine={false} />
                {SNAPSHOT_ORDER.map(s => (
                  <Radar key={s} name={SNAPSHOT_LABELS[s]} dataKey={s} stroke={SNAPSHOT_COLORS[s]} fill={SNAPSHOT_COLORS[s]} fillOpacity={0.1} strokeWidth={2} />
                ))}
                <Legend wrapperStyle={{ fontSize: 10, fontFamily: 'monospace' }} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
          <div className="space-y-3 text-xs font-mono text-[var(--text-secondary)]">
            <InsightBox>
              <strong className="text-[var(--text-primary)]">How to read:</strong> Each axis shows a different strength dimension. Close snapshots (blue) tend to dominate
              L2 Safe quality and L2 discrimination. Morning snapshots (orange/yellow) excel at L1 Safe purity and L3 volume.
              The <strong className="text-[var(--text-primary)]">ideal snapshot</strong> would fill the entire radar — none does, which is why a dual-snapshot approach adds value.
            </InsightBox>
            <div className="space-y-2 mt-2">
              <div className="flex items-center gap-2">
                <div className="w-3 h-1 rounded" style={{ backgroundColor: SNAPSHOT_COLORS['1530'] }} />
                <span><strong style={{ color: SNAPSHOT_COLORS['1530'] }}>3:30 PM</strong> — Best L2 Safe (Sharpe), strongest L2 discrimination</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-1 rounded" style={{ backgroundColor: SNAPSHOT_COLORS['0916'] }} />
                <span><strong style={{ color: SNAPSHOT_COLORS['0916'] }}>9:16 AM</strong> — Best L1 Safe purity, more L3 tradeable days</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ═══ 8-STATE COMPARISON TABLE: AL% ═══ */}
      <div className="rounded-xl border border-[var(--border)] p-5" style={{ backgroundColor: 'var(--bg-card)' }}>
        <SectionHeader title="8-State All-Lose % — Cross-Snapshot" subtitle="Lower is better. Highlighted cells show the best snapshot for each state." />
        <div className="overflow-x-auto">
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="border-b border-[var(--border)]">
                <th className="text-left px-4 py-2 text-[var(--text-dim)] font-normal">State</th>
                {snapshots.map(s => (
                  <th key={s.key} className="text-center px-4 py-2" colSpan={1}>
                    <span className="font-semibold" style={{ color: SNAPSHOT_COLORS[s.key] }}>{SNAPSHOT_LABELS[s.key]}</span>
                  </th>
                ))}
                <th className="text-right px-4 py-2 text-[var(--text-dim)] font-normal">Best</th>
              </tr>
            </thead>
            <tbody>
              {ALL_STATES.map(state => {
                const winner = getWinner(state, 'al_pct')
                return (
                  <tr key={state} className="border-b border-[var(--border-subtle)] hover:bg-[var(--bg-hover)] transition-colors">
                    <td className="px-4 py-2.5">
                      <span className="font-semibold" style={{ color: STATE_COLORS[state] }}>{state}</span>
                    </td>
                    {snapshots.map(s => {
                      const m = s.state_metrics.find((sm: SnapshotStateMetric) => sm.state === state)
                      const isWinner = s.key === winner
                      return (
                        <td key={s.key} className="text-center px-4 py-2.5 font-semibold" style={{
                          color: riskGradient(m?.al_pct ?? null),
                          backgroundColor: isWinner ? 'rgba(63,185,80,0.08)' : undefined,
                        }}>
                          {m?.al_pct != null ? `${m.al_pct}%` : '—'}
                        </td>
                      )
                    })}
                    <td className="text-right px-4 py-2.5">
                      <span className="text-[10px] px-2 py-0.5 rounded-full" style={{ backgroundColor: `${SNAPSHOT_COLORS[winner]}20`, color: SNAPSHOT_COLORS[winner] }}>
                        {SNAPSHOT_LABELS[winner]}
                      </span>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* ═══ 8-STATE COMPARISON TABLE: Sharpe ═══ */}
      <div className="rounded-xl border border-[var(--border)] p-5" style={{ backgroundColor: 'var(--bg-card)' }}>
        <SectionHeader title="8-State Sharpe Ratio — Cross-Snapshot" subtitle="Higher is better. Highlighted cells show the best snapshot for each state." />
        <div className="overflow-x-auto">
          <table className="w-full text-xs font-mono">
            <thead>
              <tr className="border-b border-[var(--border)]">
                <th className="text-left px-4 py-2 text-[var(--text-dim)] font-normal">State</th>
                {snapshots.map(s => (
                  <th key={s.key} className="text-center px-4 py-2 font-semibold" style={{ color: SNAPSHOT_COLORS[s.key] }}>{SNAPSHOT_LABELS[s.key]}</th>
                ))}
                <th className="text-right px-4 py-2 text-[var(--text-dim)] font-normal">Best</th>
              </tr>
            </thead>
            <tbody>
              {ALL_STATES.map(state => {
                const winner = getWinner(state, 'sharpe')
                return (
                  <tr key={state} className="border-b border-[var(--border-subtle)] hover:bg-[var(--bg-hover)] transition-colors">
                    <td className="px-4 py-2.5">
                      <span className="font-semibold" style={{ color: STATE_COLORS[state] }}>{state}</span>
                    </td>
                    {snapshots.map(s => {
                      const m = s.state_metrics.find((sm: SnapshotStateMetric) => sm.state === state)
                      const isWinner = s.key === winner
                      return (
                        <td key={s.key} className="text-center px-4 py-2.5 font-semibold" style={{
                          color: sharpeColor(m?.sharpe ?? null),
                          backgroundColor: isWinner ? 'rgba(63,185,80,0.08)' : undefined,
                        }}>
                          {m?.sharpe != null ? m.sharpe.toFixed(2) : '—'}
                        </td>
                      )
                    })}
                    <td className="text-right px-4 py-2.5">
                      <span className="text-[10px] px-2 py-0.5 rounded-full" style={{ backgroundColor: `${SNAPSHOT_COLORS[winner]}20`, color: SNAPSHOT_COLORS[winner] }}>
                        {SNAPSHOT_LABELS[winner]}
                      </span>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* ═══ SCATTER: Sharpe vs AL% ═══ */}
      <div className="rounded-xl border border-[var(--border)] p-5" style={{ backgroundColor: 'var(--bg-card)' }}>
        <SectionHeader title="Risk-Return Landscape — All States × All Snapshots" subtitle="Each dot = one state at one snapshot. Top-left = ideal (high Sharpe, low AL%)" />
        <div style={{ height: 380 }}>
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 10, right: 30, bottom: 30, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis type="number" dataKey="x" name="AL%" tick={{ fill: 'var(--text-muted)', fontSize: 10, fontFamily: 'monospace' }} label={{ value: 'All-Lose %', position: 'bottom', offset: 15, style: { fill: 'var(--text-dim)', fontSize: 10, fontFamily: 'monospace' } }} />
              <YAxis type="number" dataKey="y" name="Sharpe" tick={{ fill: 'var(--text-muted)', fontSize: 10, fontFamily: 'monospace' }} label={{ value: 'Sharpe', angle: -90, position: 'insideLeft', offset: 5, style: { fill: 'var(--text-dim)', fontSize: 10, fontFamily: 'monospace' } }} />
              <Tooltip content={(props) => {
                const { active, payload } = props as unknown as { active?: boolean; payload?: Array<{ payload: { state: string; snapshot: string; x: number; y: number } }> }
                if (!active || !payload?.[0]) return null
                const d = payload[0].payload
                return (
                  <div className="rounded-lg border border-[var(--border)] px-3 py-2 text-[10px] font-mono" style={{ backgroundColor: 'var(--bg-card)' }}>
                    <div className="text-[var(--text-primary)] font-semibold">{d.state}</div>
                    <div className="text-[var(--text-muted)]">{d.snapshot}</div>
                    <div>AL: <span style={{ color: riskGradient(d.x) }}>{d.x.toFixed(1)}%</span> · Sharpe: <span style={{ color: sharpeColor(d.y) }}>{d.y.toFixed(2)}</span></div>
                  </div>
                )
              }} />
              <Scatter data={scatterData}>
                {scatterData.map((d, i) => (
                  <Cell key={i} fill={d.color} fillOpacity={0.8} r={6} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <div className="flex items-center gap-4 mt-3 justify-center">
          {SNAPSHOT_ORDER.map(s => (
            <div key={s} className="flex items-center gap-1.5 text-[10px] font-mono">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: SNAPSHOT_COLORS[s] }} />
              <span style={{ color: SNAPSHOT_COLORS[s] }}>{SNAPSHOT_LABELS[s]}</span>
            </div>
          ))}
        </div>
        <div className="mt-3">
          <InsightBox>
            <strong className="text-[var(--text-primary)]">Cluster interpretation:</strong> States from all snapshots cluster together — L1 Safe always lands top-left (best), L1 Exposed always bottom-right (worst).
            The spread <em>within</em> a state&apos;s cluster shows how much the snapshot choice matters for that particular state.
            Wide spread = snapshot-sensitive state. Tight cluster = robust state.
          </InsightBox>
        </div>
      </div>

      {/* ═══ VRP COMPARISON ═══ */}
      <div className="rounded-xl border border-[var(--border)] p-5" style={{ backgroundColor: 'var(--bg-card)' }}>
        <SectionHeader title="VRP by State — Cross-Snapshot" subtitle="Volatility Risk Premium (IV − RV). Higher VRP = more premium cushion." />
        <div className="grid grid-cols-2 gap-6">
          <div style={{ height: 300 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={vrpCompareData} barCategoryGap="15%" layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis type="number" tick={{ fill: 'var(--text-muted)', fontSize: 10, fontFamily: 'monospace' }} />
                <YAxis dataKey="state" type="category" tick={{ fill: 'var(--text-muted)', fontSize: 9, fontFamily: 'monospace' }} width={100} />
                <Tooltip content={<BarTooltip />} />
                {SNAPSHOT_ORDER.map(s => (
                  <Bar key={s} dataKey={s} name={SNAPSHOT_LABELS[s]} fill={SNAPSHOT_COLORS[s]} radius={[0, 3, 3, 0]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div>
            <InsightBox variant="positive">
              <strong className="text-[var(--text-primary)]">VRP rank ordering is snapshot-invariant:</strong> Across all 4 snapshots, Safe states always have higher VRP than their Risky counterparts.
              L3 Safe consistently has the highest VRP (~5.0) — rich premium is the structural reason this state works.
              Morning VRP is ~0.3 lower on average (inflated IV denominator compresses the gap slightly).
            </InsightBox>
            <div className="mt-3">
              <table className="w-full text-xs font-mono">
                <thead>
                  <tr className="border-b border-[var(--border)]">
                    <th className="text-left px-3 py-1.5 text-[var(--text-dim)] font-normal">State</th>
                    {snapshots.map(s => (
                      <th key={s.key} className="text-right px-3 py-1.5" style={{ color: SNAPSHOT_COLORS[s.key], fontSize: 9 }}>{SNAPSHOT_LABELS[s.key]}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {ALL_STATES.map(state => (
                    <tr key={state} className="border-b border-[var(--border-subtle)]">
                      <td className="px-3 py-1.5" style={{ color: STATE_COLORS[state] }}>{state}</td>
                      {snapshots.map(s => (
                        <td key={s.key} className="text-right px-3 py-1.5 text-[var(--text-primary)]">
                          {s.vrp_by_state[state]?.toFixed(2) ?? '—'}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      {/* ═══ PK/IV THRESHOLDS ═══ */}
      <div className="rounded-xl border border-[var(--border)] p-5" style={{ backgroundColor: 'var(--bg-card)' }}>
        <SectionHeader title="PK/IV Thresholds — The Calibration Shift" subtitle="Morning IV inflates the denominator, compressing all thresholds by ~0.05" />
        <div className="grid grid-cols-2 gap-6">
          <div>
            <table className="w-full text-xs font-mono">
              <thead>
                <tr className="border-b border-[var(--border)]">
                  <th className="text-left px-4 py-2 text-[var(--text-dim)] font-normal">Level</th>
                  {snapshots.map(s => (
                    <th key={s.key} className="text-right px-4 py-2 font-semibold" style={{ color: SNAPSHOT_COLORS[s.key] }}>{SNAPSHOT_LABELS[s.key]}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {['L1', 'L2', 'L3'].map(lvl => (
                  <tr key={lvl} className="border-b border-[var(--border-subtle)]">
                    <td className="px-4 py-2.5 font-semibold text-[var(--text-primary)]">{lvl}</td>
                    {snapshots.map(s => (
                      <td key={s.key} className="text-right px-4 py-2.5 text-[var(--text-primary)]">
                        {s.pk_iv_thresholds[lvl]?.toFixed(4) ?? '—'}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div>
            <InsightBox>
              <strong className="text-[var(--text-primary)]">Why thresholds shift:</strong> PK/IV = Parkinson_Vol / IV.
              Morning IV is 2–5 pts higher (same PK since it&apos;s yesterday&apos;s intraday range), so the ratio compresses mechanically.
              The system adapts by lowering the threshold — this is correct behavior, not a bug.
              Close thresholds cluster around <strong>0.63–0.64</strong>, morning around <strong>0.58–0.59</strong>.
            </InsightBox>
          </div>
        </div>
      </div>

      {/* ═══ STRATEGY PERFORMANCE HEATMAP ═══ */}
      <div className="rounded-xl border border-[var(--border)] p-5" style={{ backgroundColor: 'var(--bg-card)' }}>
        <SectionHeader title="Strategy Sharpe by State × Snapshot" subtitle="Which strategy dominates in which state, and does it change with snapshot timing?" />
        <div className="overflow-x-auto">
          <table className="w-full text-[10px] font-mono">
            <thead>
              <tr className="border-b border-[var(--border)]">
                <th className="text-left px-3 py-2 text-[var(--text-dim)] font-normal" rowSpan={2}>State</th>
                {snapshots.map(s => (
                  <th key={s.key} className="text-center px-1 py-1 font-semibold" style={{ color: SNAPSHOT_COLORS[s.key] }} colSpan={3}>
                    {SNAPSHOT_LABELS[s.key]}
                  </th>
                ))}
              </tr>
              <tr className="border-b border-[var(--border)]">
                {snapshots.map(s => ['DM', 'WC', 'OR'].map(strat => (
                  <th key={`${s.key}-${strat}`} className="text-center px-2 py-1.5 text-[var(--text-dim)] font-normal">{strat}</th>
                )))}
              </tr>
            </thead>
            <tbody>
              {ALL_STATES.map(state => (
                <tr key={state} className="border-b border-[var(--border-subtle)] hover:bg-[var(--bg-hover)]">
                  <td className="px-3 py-2" style={{ color: STATE_COLORS[state] }}>{state}</td>
                  {snapshots.map(s => {
                    const m = s.state_metrics.find((sm: SnapshotStateMetric) => sm.state === state)
                    return ['dm_sharpe', 'wc_sharpe', 'orion_sharpe'].map(k => {
                      const v = m?.[k as keyof SnapshotStateMetric] as number | null
                      return (
                        <td key={`${s.key}-${k}`} className="text-center px-2 py-2 font-semibold" style={{ color: sharpeColor(v) }}>
                          {v != null ? v.toFixed(1) : '—'}
                        </td>
                      )
                    })
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="mt-4 grid grid-cols-2 gap-3">
          <InsightBox>
            <strong className="text-[var(--text-primary)]">L2 Caution-B is Orion&apos;s domain</strong> across all snapshots — Orion Sharpe 3.0–3.7 while DM is 0.2–0.7. Rising IV = trending market = Orion thrives.
          </InsightBox>
          <InsightBox>
            <strong className="text-[var(--text-primary)]">L1 Exposed is universally dead</strong> — DM and Orion near zero across all snapshots. Only WC shows modest life (~1.0–1.2). The premium cushion is too thin.
          </InsightBox>
        </div>
      </div>

      {/* ═══ DEEP INSIGHTS ═══ */}
      <div className="rounded-xl border border-[var(--border)] p-5" style={{ backgroundColor: 'var(--bg-card)' }}>
        <SectionHeader title="Deep Insights" subtitle="What the cross-snapshot analysis reveals about regime stability" />
        <div className="space-y-3">
          <InsightBox variant="positive">
            <strong className="text-[var(--text-primary)]">1. The DM-Orion mirror is structural, not snapshot-dependent.</strong>{' '}
            Correlation stays at −0.22 across all 4 snapshots. This is a property of the strategies themselves (DM sells premium, Orion rides trends) — not of when you measure IV.
          </InsightBox>

          <InsightBox variant="warning">
            <strong className="text-[var(--text-primary)]">2. L2 discrimination degrades at morning snapshots.</strong>{' '}
            The Safe→Risky AL gap shrinks from ~8pp (close) to ~4pp (morning). Morning IV&apos;s inflation pushes borderline L1 days into L2, diluting the signal. This is the strongest argument for using 3:30 PM as canonical.
          </InsightBox>

          <InsightBox>
            <strong className="text-[var(--text-primary)]">3. L1 Safe gets <em>purer</em> at morning.</strong>{' '}
            Fewer days qualify (134 vs 192), but Sharpe jumps from ~3.0 to ~5.6. The borderline days that morning IV pushes out were noise — what remains is genuinely low-vol, high-premium territory.
          </InsightBox>

          <InsightBox>
            <strong className="text-[var(--text-primary)]">4. VRP rank ordering never changes.</strong>{' '}
            L3 Safe &gt; L2 Safe &gt; L2 Caution-A ≈ Caution-B &gt; L1 Safe &gt; L2 Risky &gt; L1 Exposed at every snapshot. VRP is the fundamental confirmation that regime classification captures real premium dynamics.
          </InsightBox>

          <InsightBox variant="positive">
            <strong className="text-[var(--text-primary)]">5. Dual-snapshot alpha opportunity.</strong>{' '}
            Days where close and morning snapshots <em>disagree</em> on the regime are transition days — overnight gaps or morning IV spikes. These are the days where a morning overlay on a close-based system can add genuine value.
            {agr.close_vs_morning_pct < 70 && (
              <span> Only <strong>{agr.close_vs_morning_pct}%</strong> of days agree between 3:30 PM and 9:15 AM — there&apos;s significant room for a confirmation signal.</span>
            )}
          </InsightBox>

          <InsightBox>
            <strong className="text-[var(--text-primary)]">6. Close pairs are near-identical.</strong>{' '}
            3:29 and 3:30 PM agree {agr.close_pair_agree_pct}% of the time. The 1-minute gap is negligible — either close snapshot works.
            Similarly, 9:15 and 9:16 agree {agr.morning_pair_agree_pct}% — the morning pair is also interchangeable.
          </InsightBox>
        </div>
      </div>

      {/* ═══ FINAL VERDICT ═══ */}
      <div className="rounded-xl border-2 border-[#bc8cff] overflow-hidden" style={{ background: 'linear-gradient(135deg, rgba(188,140,255,0.05), var(--bg-card))' }}>
        <div className="px-6 py-5">
          <h3 className="text-sm font-mono font-bold text-[#bc8cff] uppercase tracking-wider mb-4">Final Verdict</h3>
          <div className="grid grid-cols-3 gap-4 text-xs font-mono">
            <div className="rounded-lg border border-[var(--border)] p-4" style={{ backgroundColor: 'var(--bg-elevated)' }}>
              <div className="font-semibold mb-2" style={{ color: SNAPSHOT_COLORS['1530'] }}>Primary: 3:30 PM (T-1)</div>
              <ul className="space-y-1 text-[var(--text-secondary)]">
                <li>Best L2 discrimination (8pp gap)</li>
                <li>Cleanest level distribution</li>
                <li>Highest Safe-state Sharpe (L2, L3)</li>
                <li>More days in L1 for analysis</li>
              </ul>
            </div>
            <div className="rounded-lg border border-[var(--border)] p-4" style={{ backgroundColor: 'var(--bg-elevated)' }}>
              <div className="font-semibold mb-2" style={{ color: SNAPSHOT_COLORS['0916'] }}>Overlay: 9:16 AM (T0)</div>
              <ul className="space-y-1 text-[var(--text-secondary)]">
                <li>Real-time morning signal</li>
                <li>Purest L1 Safe (Sharpe 5.6)</li>
                <li>Catches overnight regime shifts</li>
                <li>More L3 tradeable days</li>
              </ul>
            </div>
            <div className="rounded-lg border border-[var(--border)] p-4" style={{ backgroundColor: 'var(--bg-elevated)' }}>
              <div className="font-semibold mb-2 text-[var(--signal-positive)]">Best of Both</div>
              <ul className="space-y-1 text-[var(--text-secondary)]">
                <li>Use 3:30 PM as canonical classification</li>
                <li>9:16 AM as confirmation/override</li>
                <li>Both agree = highest conviction</li>
                <li>Disagree = caution, regime in flux</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

    </div>
  )
}
