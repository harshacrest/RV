'use client'

import { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { fetchRegimeDistribution } from '@/lib/api'
import type { RegimeDistEntry } from '@/lib/types'

const PERIOD_COLORS: Record<string, string> = {
  Train: '#c084fc', Val: '#64ffda', OOS1: '#448aff', OOS2: '#ffab40',
}

const STATES_ORDER = [
  'L1 Safe', 'L1 Exposed', 'L2 Safe', 'L2 Caution-A',
  'L2 Caution-B', 'L2 Risky', 'L3 Safe', 'L3 Exposed',
]

export function RegimeDistribution() {
  const [data, setData] = useState<Record<string, RegimeDistEntry> | null>(null)

  useEffect(() => { fetchRegimeDistribution().then(setData) }, [])

  if (!data) return <p style={{ color: 'var(--text-dim)' }}>Loading...</p>

  const chartData = STATES_ORDER.map(s => ({
    state: s.replace('L2 Caution-', 'L2C-'),
    Train: data[s]?.Train ?? 0,
    Val: data[s]?.Val ?? 0,
    OOS1: data[s]?.OOS1 ?? 0,
    OOS2: data[s]?.OOS2 ?? 0,
  }))

  return (
    <div className="space-y-6">
      <div className="stat-card">
        <p className="text-[11px] font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
          State Distribution Across Periods
        </p>
        <ResponsiveContainer width="100%" height={350}>
          <BarChart data={chartData} barGap={2}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
            <XAxis dataKey="state" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
            <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
            <Tooltip contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', fontSize: 11 }} />
            <Legend wrapperStyle={{ fontSize: 10 }} />
            {Object.entries(PERIOD_COLORS).map(([key, color]) => (
              <Bar key={key} dataKey={key} fill={color} radius={[2, 2, 0, 0]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Data table */}
      <div className="stat-card overflow-x-auto">
        <table className="w-full text-[11px]">
          <thead>
            <tr style={{ color: 'var(--text-dim)' }}>
              <th className="text-left py-2 px-3">State</th>
              <th className="text-right py-2 px-3">Train</th>
              <th className="text-right py-2 px-3">Val</th>
              <th className="text-right py-2 px-3">OOS1</th>
              <th className="text-right py-2 px-3">OOS2</th>
              <th className="text-right py-2 px-3">Total</th>
            </tr>
          </thead>
          <tbody>
            {STATES_ORDER.map(s => {
              const d = data[s]
              if (!d) return null
              return (
                <tr key={s} style={{ borderTop: '1px solid var(--border-subtle)' }}>
                  <td className="py-2 px-3 font-medium" style={{ color: 'var(--text-primary)' }}>{s}</td>
                  {['Train', 'Val', 'OOS1', 'OOS2'].map(p => {
                    const v = d[p as keyof RegimeDistEntry] as number
                    const warn = v < 5
                    return (
                      <td key={p} className="text-right py-2 px-3" style={{
                        color: warn ? 'var(--signal-negative)' : 'var(--text-secondary)',
                        fontWeight: warn ? 700 : 400,
                      }}>
                        {v}{warn && ' *'}
                      </td>
                    )
                  })}
                  <td className="text-right py-2 px-3" style={{ color: 'var(--text-muted)' }}>{d.Total}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
