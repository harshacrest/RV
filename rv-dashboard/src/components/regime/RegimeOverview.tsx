'use client'

import { useState, useEffect, useMemo } from 'react'
import { fetchRegimeStates } from '@/lib/api'
import type { RegimeStatesData, RegimeStateMetrics } from '@/lib/types'

interface Props {
  startDate: string
  endDate: string
  snapshot: string
  dte: number | null
}

const IV_LEVEL_GROUPS = [
  { label: 'L1 — Low IV (<12)', states: ['L1 Safe', 'L1 Exposed'] },
  { label: 'L2 — Moderate IV (12–17)', states: ['L2 Safe', 'L2 Caution-A', 'L2 Caution-B', 'L2 Risky'] },
  { label: 'L3 — High IV (>17)', states: ['L3 Safe', 'L3 Exposed'] },
]

const RULE_MAP: Record<string, string> = {
  'L1 Safe': 'PK/IV ≤ 0.63',
  'L1 Exposed': 'PK/IV > 0.63',
  'L2 Safe': 'Lo PK/IV, IV falling',
  'L2 Caution-A': 'Hi PK/IV, IV falling',
  'L2 Caution-B': 'Lo PK/IV, IV rising',
  'L2 Risky': 'Hi PK/IV, IV rising',
  'L3 Safe': 'PK/IV ≤ 0.67',
  'L3 Exposed': 'PK/IV > 0.67',
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

export function RegimeOverview({ startDate, endDate, snapshot, dte }: Props) {
  const [data, setData] = useState<RegimeStatesData | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    fetchRegimeStates(startDate, endDate, snapshot, dte)
      .then(setData)
      .finally(() => setLoading(false))
  }, [startDate, endDate, snapshot, dte])

  const stateMap = useMemo(() => {
    if (!data) return {}
    const map: Record<string, RegimeStateMetrics> = {}
    data.states.forEach(s => { map[s.state] = s })
    return map
  }, [data])

  if (loading) return <div className="text-center py-20 text-[var(--text-dim)] text-sm font-mono">Loading regime data...</div>
  if (!data) return <div className="text-center py-20 text-[var(--text-dim)] text-sm font-mono">No data available</div>

  const { current, overall } = data

  return (
    <div className="space-y-6">
      {/* Current Regime State — Hero Card */}
      <div className="rounded-xl border border-[var(--border)] overflow-hidden" style={{ backgroundColor: 'var(--bg-card)' }}>
        <div className="px-6 py-5 flex items-start gap-6">
          {/* Left: Current State */}
          <div className="flex-1">
            <div className="text-[10px] font-mono text-[var(--text-dim)] uppercase tracking-widest mb-2">Current Regime</div>
            <div className="flex items-center gap-3 mb-2">
              <div className="w-3 h-3 rounded-full animate-pulse" style={{ backgroundColor: current.color ?? '#666' }} />
              <span className="text-2xl font-bold font-mono" style={{ color: current.color ?? 'var(--text-primary)' }}>
                {current.state ?? 'Unknown'}
              </span>
            </div>
            <p className="text-xs text-[var(--text-secondary)] font-mono mb-3">{current.description}</p>
            <div className="text-[10px] font-mono text-[var(--text-dim)]">as of {current.date}</div>
          </div>

          {/* Right: Input Values */}
          <div className="grid grid-cols-5 gap-3">
            {[
              { label: 'IV (lag)', value: current.iv_lag?.toFixed(2) ?? '—', sub: current.iv_lag != null ? (current.iv_lag < 12 ? 'L1' : current.iv_lag < 17 ? 'L2' : 'L3') : '' },
              { label: 'PK/IV Ratio', value: current.pk_iv_ratio?.toFixed(3) ?? '—' },
              { label: 'IV Chg 5d', value: current.iv_chg_5d?.toFixed(3) ?? '—', sub: current.iv_chg_5d != null ? (current.iv_chg_5d <= 0 ? 'Falling' : 'Rising') : '' },
              { label: 'PK 5d', value: current.pk_5d?.toFixed(2) ?? '—' },
              { label: 'IV 5d', value: current.iv_5d?.toFixed(2) ?? '—' },
            ].map(item => (
              <div key={item.label} className="stat-card !p-3">
                <div className="text-[9px] font-mono text-[var(--text-dim)] uppercase tracking-wider mb-1">{item.label}</div>
                <div className="text-sm font-bold text-[var(--text-primary)] font-mono">{item.value}</div>
                {item.sub && <div className="text-[9px] text-[var(--text-muted)] font-mono mt-0.5">{item.sub}</div>}
              </div>
            ))}
          </div>
        </div>

        {/* Overall Summary Bar */}
        <div className="px-6 py-3 border-t border-[var(--border)] flex items-center gap-6 bg-[var(--bg-elevated)]">
          <div className="text-[10px] font-mono text-[var(--text-dim)] uppercase tracking-wider">Overall</div>
          <div className="flex items-center gap-5 text-xs font-mono">
            <span className="text-[var(--text-secondary)]">{overall.days} days</span>
            <span>Port Avg: <strong className="text-[var(--text-primary)]">{overall.port_avg != null ? `${overall.port_avg > 0 ? '+' : ''}${(overall.port_avg).toFixed(4)}%` : '—'}</strong></span>
            <span>Sharpe: <strong style={{ color: sharpeColor(overall.sharpe) }}>{overall.sharpe?.toFixed(2) ?? '—'}</strong></span>
            <span>AL%: <strong style={{ color: riskGradient(overall.al_pct) }}>{overall.al_pct?.toFixed(1) ?? '—'}%</strong></span>
            <span>AW%: <strong className="text-[var(--signal-positive)]">{overall.aw_pct?.toFixed(1) ?? '—'}%</strong></span>
          </div>
        </div>
      </div>

      {/* 8-State Regime Table — Grouped by IV Level */}
      <div className="space-y-4">
        <h3 className="text-xs font-mono uppercase tracking-wider text-[var(--text-dim)]">8-State Regime Classification</h3>

        {IV_LEVEL_GROUPS.map(group => (
          <div key={group.label} className="rounded-xl border border-[var(--border)] overflow-hidden" style={{ backgroundColor: 'var(--bg-card)' }}>
            <div className="px-4 py-2.5 border-b border-[var(--border)] bg-[var(--bg-elevated)]">
              <span className="text-[11px] font-mono font-semibold text-[var(--text-secondary)]">{group.label}</span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-xs font-mono">
                <thead>
                  <tr className="border-b border-[var(--border)]">
                    <th className="text-left px-4 py-2 text-[var(--text-dim)] font-normal">State</th>
                    <th className="text-left px-3 py-2 text-[var(--text-dim)] font-normal">Rule</th>
                    <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Days</th>
                    <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">%</th>
                    <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">AL%</th>
                    <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">AW%</th>
                    <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Port Avg</th>
                    <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Sharpe</th>
                    <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Avg IV</th>
                    <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Avg PK/IV</th>
                  </tr>
                </thead>
                <tbody>
                  {group.states.map(stateName => {
                    const s = stateMap[stateName]
                    if (!s) return null
                    const isCurrent = current.state === stateName
                    return (
                      <tr
                        key={stateName}
                        className="border-b border-[var(--border-subtle)] transition-colors hover:bg-[var(--bg-hover)]"
                        style={isCurrent ? { backgroundColor: `${s.color}08` } : undefined}
                      >
                        <td className="px-4 py-2.5">
                          <div className="flex items-center gap-2">
                            <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: s.color }} />
                            <span className="font-semibold" style={{ color: s.color }}>{s.state}</span>
                            {isCurrent && (
                              <span className="text-[8px] px-1.5 py-0.5 rounded-full bg-white/10 text-white uppercase tracking-wider">now</span>
                            )}
                          </div>
                        </td>
                        <td className="px-3 py-2.5 text-[var(--text-muted)]">{RULE_MAP[stateName] ?? ''}</td>
                        <td className="text-right px-3 py-2.5 text-[var(--text-secondary)]">{s.days}</td>
                        <td className="text-right px-3 py-2.5 text-[var(--text-muted)]">{s.pct_of_total}%</td>
                        <td className="text-right px-3 py-2.5 font-semibold" style={{ color: riskGradient(s.al_pct) }}>
                          {s.al_pct != null ? `${s.al_pct}%` : '—'}
                        </td>
                        <td className="text-right px-3 py-2.5" style={{ color: 'var(--signal-positive)' }}>
                          {s.aw_pct != null ? `${s.aw_pct}%` : '—'}
                        </td>
                        <td className="text-right px-3 py-2.5 font-semibold" style={{ color: s.port_avg != null && s.port_avg > 0 ? 'var(--signal-positive)' : 'var(--signal-negative)' }}>
                          {s.port_avg != null ? `${s.port_avg > 0 ? '+' : ''}${s.port_avg.toFixed(4)}%` : '—'}
                        </td>
                        <td className="text-right px-3 py-2.5 font-semibold" style={{ color: sharpeColor(s.sharpe) }}>
                          {s.sharpe?.toFixed(2) ?? '—'}
                        </td>
                        <td className="text-right px-3 py-2.5 text-[var(--text-secondary)]">{s.iv_lag_mean?.toFixed(1) ?? '—'}</td>
                        <td className="text-right px-3 py-2.5 text-[var(--text-secondary)]">{s.pk_iv_mean?.toFixed(3) ?? '—'}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        ))}
      </div>

      {/* Risk Spectrum Visualization */}
      <div className="rounded-xl border border-[var(--border)] p-5" style={{ backgroundColor: 'var(--bg-card)' }}>
        <h3 className="text-xs font-mono uppercase tracking-wider text-[var(--text-dim)] mb-4">Risk Spectrum — All-Lose %</h3>
        <div className="space-y-2">
          {data.states
            .filter(s => s.days > 0)
            .sort((a, b) => (a.al_pct ?? 0) - (b.al_pct ?? 0))
            .map(s => (
              <div key={s.state} className="flex items-center gap-3">
                <div className="w-28 text-[10px] font-mono font-semibold flex-shrink-0" style={{ color: s.color }}>{s.state}</div>
                <div className="flex-1 h-5 rounded-md overflow-hidden bg-[var(--bg-elevated)] relative">
                  <div
                    className="h-full rounded-md transition-all duration-500"
                    style={{
                      width: `${Math.max((s.al_pct ?? 0) / 25 * 100, 2)}%`,
                      backgroundColor: riskGradient(s.al_pct),
                      opacity: 0.8,
                    }}
                  />
                  <span className="absolute inset-0 flex items-center px-2 text-[10px] font-mono font-semibold text-white">
                    {s.al_pct?.toFixed(1)}%
                  </span>
                </div>
                <div className="w-16 text-right text-[10px] font-mono text-[var(--text-muted)]">Sh {s.sharpe?.toFixed(2) ?? '—'}</div>
              </div>
            ))}
        </div>
      </div>

      {/* The Unified Story */}
      <div className="rounded-xl border border-[var(--border)] p-5" style={{ backgroundColor: 'var(--bg-card)' }}>
        <h3 className="text-xs font-mono uppercase tracking-wider text-[var(--text-dim)] mb-3">The Unified Story</h3>
        <div className="space-y-2 text-xs font-mono text-[var(--text-secondary)]">
          <p><strong className="text-[var(--text-primary)]">Risk at every level = realized movement eating into premium cushion.</strong></p>
          <p>PK/IV ratio is the primary signal at L1 and L3. At L2, IV direction adds because it tells you if the gap is widening or shrinking.</p>
          <p>At L1, IV direction is noise (changes too small). At L3, IV direction is noise (almost always rising).</p>
        </div>
      </div>
    </div>
  )
}
