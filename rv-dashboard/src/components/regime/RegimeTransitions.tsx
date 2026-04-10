'use client'

import { useState, useEffect } from 'react'
import { fetchRegimeTransitions } from '@/lib/api'
import type { RegimeTransitionsData } from '@/lib/types'

interface Props {
  startDate: string
  endDate: string
  snapshot: string
  dte: number | null
}

const STATE_COLORS: Record<string, string> = {
  'L1 Safe': '#00e676',
  'L1 Exposed': '#ffab40',
  'L2 Safe': '#00e676',
  'L2 Caution-A': '#ffab40',
  'L2 Caution-B': '#ff9100',
  'L2 Risky': '#ff5252',
  'L3 Safe': '#64ffda',
  'L3 Exposed': '#ff1744',
}

const SHORT_LABELS: Record<string, string> = {
  'L1 Safe': 'L1S',
  'L1 Exposed': 'L1E',
  'L2 Safe': 'L2S',
  'L2 Caution-A': 'L2A',
  'L2 Caution-B': 'L2B',
  'L2 Risky': 'L2R',
  'L3 Safe': 'L3S',
  'L3 Exposed': 'L3E',
}

type ViewMode = 'probs' | 'counts'

function probColor(p: number | null): string {
  if (p == null || p === 0) return 'transparent'
  if (p >= 0.5) return `rgba(168,85,247,${Math.min(p * 0.8, 0.6)})`
  if (p >= 0.1) return `rgba(100,255,218,${p * 0.5})`
  return `rgba(255,255,255,${p * 0.3})`
}

export function RegimeTransitions({ startDate, endDate, snapshot, dte }: Props) {
  const [data, setData] = useState<RegimeTransitionsData | null>(null)
  const [loading, setLoading] = useState(true)
  const [mode, setMode] = useState<ViewMode>('probs')

  useEffect(() => {
    setLoading(true)
    fetchRegimeTransitions(startDate, endDate, snapshot, dte)
      .then(setData)
      .finally(() => setLoading(false))
  }, [startDate, endDate, snapshot, dte])

  if (loading) return <div className="text-center py-20 text-[var(--text-dim)] text-sm font-mono">Loading transitions...</div>
  if (!data) return null

  const { states, transition_probs, transition_counts, streak_stats, self_transition_rate } = data

  return (
    <div className="space-y-6">
      {/* Mode Toggle */}
      <div className="flex items-center gap-2">
        {([
          { key: 'probs', label: 'Probabilities' },
          { key: 'counts', label: 'Counts' },
        ] as { key: ViewMode; label: string }[]).map(v => (
          <button
            key={v.key}
            onClick={() => setMode(v.key)}
            className="px-3 py-1.5 rounded-lg text-[10px] font-mono font-semibold transition-all"
            style={{
              backgroundColor: mode === v.key ? 'rgba(168,85,247,0.15)' : 'transparent',
              color: mode === v.key ? '#c084fc' : 'var(--text-muted)',
              border: `1px solid ${mode === v.key ? 'rgba(168,85,247,0.3)' : 'var(--border)'}`,
            }}
          >
            {v.label}
          </button>
        ))}
      </div>

      {/* Transition Matrix */}
      <div className="rounded-xl border border-[var(--border)] p-4" style={{ backgroundColor: 'var(--bg-card)' }}>
        <h3 className="text-xs font-mono uppercase tracking-wider text-[var(--text-dim)] mb-4">
          Transition Matrix — {mode === 'probs' ? 'P(Next State | Current State)' : 'Raw Counts'}
        </h3>
        <div className="overflow-x-auto">
          <table className="text-[10px] font-mono">
            <thead>
              <tr>
                <th className="px-2 py-2 text-left text-[var(--text-dim)] font-normal min-w-[70px]">From ↓ / To →</th>
                {states.map(s => (
                  <th key={s} className="px-2 py-2 text-center font-semibold min-w-[48px]" style={{ color: STATE_COLORS[s] }}>
                    {SHORT_LABELS[s]}
                  </th>
                ))}
                <th className="px-2 py-2 text-center text-[var(--text-dim)] font-normal">Self %</th>
              </tr>
            </thead>
            <tbody>
              {states.map(fromState => (
                <tr key={fromState} className="border-t border-[var(--border-subtle)]">
                  <td className="px-2 py-2 font-semibold" style={{ color: STATE_COLORS[fromState] }}>
                    {SHORT_LABELS[fromState]}
                  </td>
                  {states.map(toState => {
                    const prob = transition_probs[fromState]?.[toState]
                    const count = transition_counts[fromState]?.[toState] ?? 0
                    const isDiag = fromState === toState

                    return (
                      <td
                        key={toState}
                        className="px-2 py-2 text-center transition-colors"
                        style={{
                          backgroundColor: probColor(prob),
                          fontWeight: isDiag ? 700 : 400,
                          color: isDiag ? '#ffffff' : (prob != null && prob > 0 ? 'var(--text-primary)' : 'var(--text-dim)'),
                        }}
                      >
                        {mode === 'probs'
                          ? (prob != null && prob > 0 ? `${(prob * 100).toFixed(0)}%` : '·')
                          : (count > 0 ? count : '·')
                        }
                      </td>
                    )
                  })}
                  <td className="px-2 py-2 text-center font-semibold" style={{ color: '#c084fc' }}>
                    {self_transition_rate[fromState] != null ? `${(self_transition_rate[fromState]! * 100).toFixed(0)}%` : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-[9px] font-mono text-[var(--text-dim)] mt-3">Diagonal = self-transition (staying in same state). Higher = more persistent regime.</p>
      </div>

      {/* Streak Analysis */}
      <div className="rounded-xl border border-[var(--border)] p-4" style={{ backgroundColor: 'var(--bg-card)' }}>
        <h3 className="text-xs font-mono uppercase tracking-wider text-[var(--text-dim)] mb-4">Streak Analysis — Consecutive Days in Same Regime</h3>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          {states.map(s => {
            const ss = streak_stats[s]
            if (!ss || ss.count === 0) return null
            return (
              <div key={s} className="stat-card !p-3">
                <div className="flex items-center gap-1.5 mb-2">
                  <div className="w-2 h-2 rounded-full" style={{ backgroundColor: STATE_COLORS[s] }} />
                  <span className="text-[10px] font-mono font-semibold" style={{ color: STATE_COLORS[s] }}>{s}</span>
                </div>
                <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-[10px] font-mono">
                  <div className="text-[var(--text-dim)]">Mean</div>
                  <div className="text-right text-[var(--text-primary)] font-semibold">{ss.mean?.toFixed(1)}d</div>
                  <div className="text-[var(--text-dim)]">Median</div>
                  <div className="text-right text-[var(--text-secondary)]">{ss.median?.toFixed(1)}d</div>
                  <div className="text-[var(--text-dim)]">Min</div>
                  <div className="text-right text-[var(--text-secondary)]">{ss.min}d</div>
                  <div className="text-[var(--text-dim)]">Max</div>
                  <div className="text-right text-[var(--text-secondary)]">{ss.max}d</div>
                  <div className="text-[var(--text-dim)]">Streaks</div>
                  <div className="text-right text-[var(--text-secondary)]">{ss.count}</div>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Persistence Insight */}
      <div className="rounded-xl border border-[var(--border)] p-5" style={{ backgroundColor: 'var(--bg-card)' }}>
        <h3 className="text-xs font-mono uppercase tracking-wider text-[var(--text-dim)] mb-3">Persistence & Predictability</h3>
        <div className="space-y-2 text-xs font-mono text-[var(--text-secondary)]">
          <p>The regime system uses <strong className="text-[var(--text-primary)]">5-day averaging</strong> to transform noisy daily signals into persistent, predictable inputs.</p>
          <p>IV level has the strongest autocorrelation (0.78) — the natural regime backbone. PK_5d AC = 0.90 and VRP_5d AC = 0.83.</p>
          <p>High self-transition rates indicate the regime is stable and actionable — not just noise.</p>
        </div>
      </div>
    </div>
  )
}
