'use client'

import { useState, useEffect, useMemo } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts'
import { fetchAblation } from '@/lib/api'
import type { AblationData } from '@/lib/types'

const IMPACT_COLORS: Record<string, string> = {
  CRITICAL: 'var(--signal-negative)',
  moderate: 'var(--signal-warning)',
  minimal: 'var(--signal-positive)',
}

export function AblationStudy() {
  const [data, setData] = useState<AblationData | null>(null)

  useEffect(() => { fetchAblation().then(setData) }, [])

  const sorted = useMemo(() => {
    if (!data) return []
    return [...data.ablations].sort((a, b) => a.delta - b.delta)
  }, [data])

  if (!data) return <p style={{ color: 'var(--text-dim)' }}>Loading...</p>

  return (
    <div className="space-y-6">
      {/* Hero */}
      <div className="stat-card">
        <p className="text-[10px] uppercase tracking-wider" style={{ color: 'var(--text-dim)' }}>
          Baseline Composite
        </p>
        <p className="text-3xl font-bold" style={{ color: 'var(--accent-primary)' }}>
          {data.baseline_score.toFixed(4)}
        </p>
        <p className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>
          Each bar shows the score drop when one dimension is disabled
        </p>
      </div>

      {/* Waterfall chart */}
      <div className="stat-card">
        <p className="text-[11px] font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
          Ablation Impact (delta from baseline)
        </p>
        <ResponsiveContainer width="100%" height={350}>
          <BarChart data={sorted} layout="vertical" margin={{ left: 160 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
            <XAxis type="number" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
            <YAxis dataKey="name" type="category" tick={{ fontSize: 10, fill: 'var(--text-secondary)' }} width={150} />
            <Tooltip
              formatter={(v: number) => v.toFixed(4)}
              contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', fontSize: 11 }}
            />
            <ReferenceLine x={0} stroke="var(--text-dim)" strokeDasharray="3 3" />
            <Bar dataKey="delta" radius={[0, 3, 3, 0]}>
              {sorted.map((e, i) => (
                <Cell key={i} fill={IMPACT_COLORS[e.impact] || 'var(--text-dim)'} fillOpacity={0.8} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Impact table */}
      <div className="stat-card overflow-x-auto">
        <table className="w-full text-[11px]">
          <thead>
            <tr style={{ color: 'var(--text-dim)' }}>
              <th className="text-left py-2 px-3">Ablation</th>
              <th className="text-right py-2 px-3">Score</th>
              <th className="text-right py-2 px-3">Delta</th>
              <th className="text-right py-2 px-3">Impact</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map(a => (
              <tr key={a.name} style={{ borderTop: '1px solid var(--border-subtle)' }}>
                <td className="py-2 px-3" style={{ color: 'var(--text-primary)' }}>{a.name}</td>
                <td className="text-right py-2 px-3" style={{ color: 'var(--text-secondary)' }}>{a.score.toFixed(4)}</td>
                <td className="text-right py-2 px-3 font-bold" style={{ color: IMPACT_COLORS[a.impact] }}>
                  {a.delta >= 0 ? '+' : ''}{a.delta.toFixed(4)}
                </td>
                <td className="text-right py-2 px-3">
                  <span className="text-[9px] px-2 py-0.5 rounded" style={{
                    background: a.impact === 'CRITICAL' ? 'rgba(255,82,82,0.15)' : a.impact === 'moderate' ? 'rgba(255,171,64,0.15)' : 'rgba(0,230,118,0.15)',
                    color: IMPACT_COLORS[a.impact],
                  }}>
                    {a.impact.toUpperCase()}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Summary */}
      <div className="grid grid-cols-2 gap-4">
        <div className="stat-card">
          <p className="text-[10px] uppercase tracking-wider mb-2" style={{ color: 'var(--signal-negative)' }}>
            Load-Bearing (keep these)
          </p>
          <ul className="space-y-1 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
            {sorted.filter(a => a.delta < -0.02).map(a => (
              <li key={a.name}>{a.name} ({a.delta.toFixed(3)})</li>
            ))}
          </ul>
        </div>
        <div className="stat-card">
          <p className="text-[10px] uppercase tracking-wider mb-2" style={{ color: 'var(--signal-positive)' }}>
            Non-Essential (can remove)
          </p>
          <ul className="space-y-1 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
            {sorted.filter(a => a.delta >= -0.02).map(a => (
              <li key={a.name}>{a.name} ({a.delta >= 0 ? '+' : ''}{a.delta.toFixed(3)})</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  )
}
