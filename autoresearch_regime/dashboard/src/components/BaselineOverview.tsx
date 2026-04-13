'use client'

import { useState, useEffect } from 'react'
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts'
import { fetchBaseline } from '@/lib/api'
import type { Baseline } from '@/lib/types'

const COLORS = { sharpe: '#c084fc', sep: '#64ffda', rank: '#448aff', coverage: '#ffab40' }

export function BaselineOverview() {
  const [data, setData] = useState<Baseline | null>(null)

  useEffect(() => { fetchBaseline().then(setData) }, [])

  if (!data) return <p style={{ color: 'var(--text-dim)' }}>Loading...</p>

  const breakdown = [
    { name: 'Sharpe', value: 0.40 * Math.min(data.val_sharpe / 5.0, 1.5), color: COLORS.sharpe },
    { name: 'Safe Sep', value: 0.25 * Math.min(data.safe_separation / 10.0, 1.0), color: COLORS.sep },
    { name: 'Rank', value: 0.20 * Math.max(data.rank_stability, 0), color: COLORS.rank },
    { name: 'Coverage', value: 0.15 * data.state_coverage, color: COLORS.coverage },
  ]

  const warnings = [
    data.min_state_days < 5 && `Minimum state has only ${data.min_state_days} val days`,
    'L1 has only 7 training days total — thresholds unreliable',
    'Val Sharpe is period-dependent: first half=2.73, second half=6.25',
    'Regime classifier loses to equal-weight in 8/12 rolling windows',
    'OOS2 (34 days) is statistically meaningless — SE=2.68',
    'Extra features (smooth3, pctile, risk) add zero value',
  ].filter(Boolean)

  return (
    <div className="space-y-6">
      {/* Hero */}
      <div className="flex gap-6 items-start">
        <div className="stat-card flex-1">
          <p className="text-[10px] uppercase tracking-wider mb-2" style={{ color: 'var(--text-dim)' }}>
            Composite Score
          </p>
          <p className="text-4xl font-bold" style={{ color: 'var(--accent-primary)' }}>
            {data.composite_score.toFixed(4)}
          </p>
          <p className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>
            {data.train_days}d train / {data.val_days}d val / {data.n_states_used} states
          </p>
        </div>

        <div className="stat-card" style={{ width: 200, height: 200 }}>
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={breakdown}
                cx="50%" cy="50%"
                innerRadius={50} outerRadius={80}
                dataKey="value"
                stroke="none"
              >
                {breakdown.map((e, i) => <Cell key={i} fill={e.color} />)}
              </Pie>
              <Tooltip
                formatter={(v: number) => v.toFixed(4)}
                contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', fontSize: 11 }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Metric cards */}
      <div className="grid grid-cols-4 gap-4">
        {[
          { label: 'Val Sharpe', value: data.val_sharpe.toFixed(4), color: COLORS.sharpe, desc: '40% weight' },
          { label: 'Safe Separation', value: data.safe_separation.toFixed(2), color: COLORS.sep, desc: '25% weight' },
          { label: 'Rank Stability', value: data.rank_stability.toFixed(4), color: COLORS.rank, desc: '20% weight, p=0.011' },
          { label: 'Coverage', value: data.state_coverage.toFixed(4), color: COLORS.coverage, desc: '15% weight' },
        ].map(m => (
          <div key={m.label} className="stat-card">
            <p className="text-[10px] uppercase tracking-wider" style={{ color: 'var(--text-dim)' }}>{m.label}</p>
            <p className="text-2xl font-bold mt-1" style={{ color: m.color }}>{m.value}</p>
            <p className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>{m.desc}</p>
          </div>
        ))}
      </div>

      {/* Warnings */}
      <div className="stat-card">
        <p className="text-[11px] font-semibold mb-3" style={{ color: 'var(--signal-warning)' }}>
          Key Findings &amp; Warnings
        </p>
        <ul className="space-y-1.5">
          {warnings.map((w, i) => (
            <li key={i} className="text-[11px] flex items-start gap-2">
              <span style={{ color: 'var(--signal-warning)' }}>!</span>
              <span style={{ color: 'var(--text-secondary)' }}>{w}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}
