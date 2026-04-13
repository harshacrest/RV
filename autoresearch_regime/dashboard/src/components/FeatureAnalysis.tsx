'use client'

import { useState, useEffect, useMemo } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { fetchFeatureCorrelations } from '@/lib/api'
import type { FeatureCorrelations } from '@/lib/types'

export function FeatureAnalysis() {
  const [data, setData] = useState<FeatureCorrelations | null>(null)

  useEffect(() => { fetchFeatureCorrelations().then(setData) }, [])

  const icData = useMemo(() => {
    if (!data?.feature_ics) return []
    return Object.entries(data.feature_ics)
      .map(([name, ic]) => ({ name, ic: Number(ic) }))
      .sort((a, b) => Math.abs(b.ic) - Math.abs(a.ic))
  }, [data])

  if (!data) return <p style={{ color: 'var(--text-dim)' }}>Loading...</p>

  return (
    <div className="space-y-6">
      {/* IC Rankings */}
      <div className="stat-card">
        <p className="text-[11px] font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
          Feature Information Coefficients (IC with next-day PnL)
        </p>
        <ResponsiveContainer width="100%" height={Math.max(300, icData.length * 28)}>
          <BarChart data={icData} layout="vertical" margin={{ left: 120 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
            <XAxis type="number" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
            <YAxis dataKey="name" type="category" tick={{ fontSize: 10, fill: 'var(--text-secondary)' }} width={110} />
            <Tooltip
              formatter={(v: number) => v.toFixed(4)}
              contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', fontSize: 11 }}
            />
            <Bar dataKey="ic" radius={[0, 3, 3, 0]}>
              {icData.map((e, i) => (
                <Cell key={i} fill={e.ic > 0 ? 'var(--signal-positive)' : 'var(--signal-negative)'} fillOpacity={0.7} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <p className="text-[10px] mt-2" style={{ color: 'var(--text-dim)' }}>
          All ICs are weak (&lt;0.06). Regime adds value through state-conditional allocation, not direct prediction.
        </p>
      </div>

      {/* High correlation pairs */}
      <div className="stat-card">
        <p className="text-[11px] font-semibold mb-3" style={{ color: 'var(--signal-negative)' }}>
          Redundant Feature Pairs (|rho| &gt; 0.6)
        </p>
        <div className="space-y-1">
          {data.high_corr_pairs.map((pair, i) => {
            const [f1, f2, corr] = pair
            const isRedundant = Math.abs(corr) > 0.75
            return (
              <div key={i} className="flex items-center gap-3 text-[11px] py-1"
                style={{ borderTop: i > 0 ? '1px solid var(--border-subtle)' : undefined }}>
                <span style={{ color: 'var(--text-primary)' }}>{f1}</span>
                <span style={{ color: 'var(--text-dim)' }}>&harr;</span>
                <span style={{ color: 'var(--text-primary)' }}>{f2}</span>
                <span className="ml-auto font-bold" style={{
                  color: isRedundant ? 'var(--signal-negative)' : 'var(--signal-warning)'
                }}>
                  {corr.toFixed(3)}
                </span>
                <span className="text-[9px] px-2 py-0.5 rounded" style={{
                  background: isRedundant ? 'rgba(255,82,82,0.15)' : 'rgba(255,171,64,0.15)',
                  color: isRedundant ? 'var(--signal-negative)' : 'var(--signal-warning)',
                }}>
                  {isRedundant ? 'REDUNDANT' : 'moderate'}
                </span>
              </div>
            )
          })}
        </div>
      </div>

      {/* Recommendations */}
      <div className="stat-card">
        <p className="text-[11px] font-semibold mb-2" style={{ color: 'var(--signal-positive)' }}>
          Feature Recommendations
        </p>
        <ul className="space-y-1 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
          <li>Use: IV_chg_5d (best IC), PK_IV_ratio (solid, already in use), VRP_today</li>
          <li>Avoid: PK_IV_smooth3 (redundant with PK_IV_ratio, rho=0.88)</li>
          <li>Avoid: IV_5d when using iv_lag (rho=0.82)</li>
          <li>Extra features (pk_iv_pctile, pk_iv_risk) add zero value per ablation</li>
        </ul>
      </div>
    </div>
  )
}
