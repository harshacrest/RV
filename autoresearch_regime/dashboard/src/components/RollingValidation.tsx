'use client'

import { useState, useEffect } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Cell, ReferenceLine, LineChart, Line, Legend,
} from 'recharts'
import { fetchRollingValidation } from '@/lib/api'
import type { RollingValidation as RollingData } from '@/lib/types'

export function RollingValidation() {
  const [data, setData] = useState<RollingData | null>(null)

  useEffect(() => { fetchRollingValidation().then(setData) }, [])

  if (!data) return <p style={{ color: 'var(--text-dim)' }}>Loading...</p>

  const { windows, summary } = data
  const winRate = summary.total_windows > 0 ? (summary.regime_wins / summary.total_windows * 100) : 0

  const lineData = windows.map(w => ({
    label: w.val_start.slice(2, 7),
    regime: w.regime_sharpe,
    equal: w.equal_sharpe,
  }))

  const deltaData = windows.map(w => ({
    label: w.val_start.slice(2, 7),
    delta: w.delta,
    winner: w.winner,
  }))

  return (
    <div className="space-y-6">
      {/* Summary cards */}
      <div className="grid grid-cols-4 gap-4">
        <div className="stat-card">
          <p className="text-[10px] uppercase tracking-wider" style={{ color: 'var(--text-dim)' }}>Windows</p>
          <p className="text-2xl font-bold mt-1" style={{ color: 'var(--text-primary)' }}>{summary.total_windows}</p>
        </div>
        <div className="stat-card">
          <p className="text-[10px] uppercase tracking-wider" style={{ color: 'var(--text-dim)' }}>Regime Wins</p>
          <p className="text-2xl font-bold mt-1" style={{ color: winRate >= 50 ? 'var(--signal-positive)' : 'var(--signal-negative)' }}>
            {summary.regime_wins} ({winRate.toFixed(0)}%)
          </p>
        </div>
        <div className="stat-card">
          <p className="text-[10px] uppercase tracking-wider" style={{ color: 'var(--text-dim)' }}>Equal Wins</p>
          <p className="text-2xl font-bold mt-1" style={{ color: 'var(--signal-warning)' }}>
            {summary.equal_wins}
          </p>
        </div>
        <div className="stat-card">
          <p className="text-[10px] uppercase tracking-wider" style={{ color: 'var(--text-dim)' }}>Avg Delta</p>
          <p className="text-2xl font-bold mt-1" style={{
            color: summary.avg_delta >= 0 ? 'var(--signal-positive)' : 'var(--signal-negative)'
          }}>
            {summary.avg_delta >= 0 ? '+' : ''}{summary.avg_delta.toFixed(2)}
          </p>
        </div>
      </div>

      {/* Line chart */}
      <div className="stat-card">
        <p className="text-[11px] font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
          Regime vs Equal-Weight Sharpe (120-day windows)
        </p>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={lineData}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
            <XAxis dataKey="label" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
            <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
            <Tooltip contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', fontSize: 11 }} />
            <Legend wrapperStyle={{ fontSize: 10 }} />
            <Line type="monotone" dataKey="regime" stroke="var(--accent-primary)" strokeWidth={2} dot={{ r: 4 }} name="Regime" />
            <Line type="monotone" dataKey="equal" stroke="var(--text-dim)" strokeWidth={2} dot={{ r: 4 }} strokeDasharray="5 5" name="Equal Weight" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Delta bar chart */}
      <div className="stat-card">
        <p className="text-[11px] font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
          Per-Window Delta (Regime - Equal)
        </p>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={deltaData}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
            <XAxis dataKey="label" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
            <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
            <Tooltip
              formatter={(v: number) => v.toFixed(2)}
              contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', fontSize: 11 }}
            />
            <ReferenceLine y={0} stroke="var(--text-dim)" />
            <Bar dataKey="delta" radius={[3, 3, 0, 0]}>
              {deltaData.map((e, i) => (
                <Cell key={i} fill={e.delta >= 0 ? 'var(--signal-positive)' : 'var(--signal-negative)'} fillOpacity={0.7} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Verdict */}
      <div className="stat-card" style={{
        borderColor: winRate >= 50 ? 'rgba(0,230,118,0.3)' : 'rgba(255,82,82,0.3)',
      }}>
        <p className="text-[11px] font-semibold" style={{
          color: winRate >= 50 ? 'var(--signal-positive)' : 'var(--signal-negative)'
        }}>
          {winRate >= 50
            ? `Regime classifier adds value in ${summary.regime_wins}/${summary.total_windows} windows`
            : `WARNING: Regime classifier does NOT consistently beat equal-weight (${summary.regime_wins}/${summary.total_windows} wins)`
          }
        </p>
        <p className="text-[10px] mt-1" style={{ color: 'var(--text-muted)' }}>
          Regime only wins in 2025+ windows. Strategy weights appear overfit to recent conditions.
        </p>
      </div>
    </div>
  )
}
