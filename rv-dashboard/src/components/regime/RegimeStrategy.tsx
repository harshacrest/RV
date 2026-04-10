'use client'

import { useState, useEffect } from 'react'
import { fetchRegimeStrategy } from '@/lib/api'
import { STRATEGY_META, STRATEGIES } from '@/lib/config'
import type { RegimeStrategyData, RegimeStrategyState } from '@/lib/types'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'

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

type MetricKey = 'avg_pnl' | 'sharpe' | 'win_rate'

export function RegimeStrategy({ startDate, endDate, snapshot, dte }: Props) {
  const [strategyData, setStrategyData] = useState<Record<string, RegimeStrategyData>>({})
  const [loading, setLoading] = useState(true)
  const [metric, setMetric] = useState<MetricKey>('avg_pnl')

  useEffect(() => {
    setLoading(true)
    Promise.all(
      STRATEGIES.map(s =>
        fetchRegimeStrategy(s, startDate, endDate, snapshot, dte).then(d => [s, d] as const)
      )
    ).then(results => {
      const map: Record<string, RegimeStrategyData> = {}
      results.forEach(([key, data]) => { map[key] = data })
      setStrategyData(map)
      setLoading(false)
    })
  }, [startDate, endDate, snapshot, dte])

  if (loading) return <div className="text-center py-20 text-[var(--text-dim)] text-sm font-mono">Loading strategy breakdown...</div>

  const strategies = STRATEGIES.filter(s => s !== 'dmo') // Only dm, wc, orion per the framework

  // Build comparison chart data
  const chartData = Object.keys(STATE_COLORS).map(state => {
    const row: Record<string, string | number | null> = { state }
    strategies.forEach(s => {
      const stateData = strategyData[s]?.states.find(st => st.state === state)
      if (metric === 'avg_pnl') row[s] = stateData?.avg_pnl ?? null
      else if (metric === 'sharpe') row[s] = stateData?.sharpe ?? null
      else row[s] = stateData?.win_rate != null ? stateData.win_rate * 100 : null
    })
    return row
  })

  return (
    <div className="space-y-6">
      {/* Metric Toggle */}
      <div className="flex items-center gap-2">
        {([
          { key: 'avg_pnl', label: 'Avg Daily %' },
          { key: 'sharpe', label: 'Sharpe' },
          { key: 'win_rate', label: 'Win Rate %' },
        ] as { key: MetricKey; label: string }[]).map(m => (
          <button
            key={m.key}
            onClick={() => setMetric(m.key)}
            className="px-3 py-1.5 rounded-lg text-[10px] font-mono font-semibold transition-all"
            style={{
              backgroundColor: metric === m.key ? 'rgba(168,85,247,0.15)' : 'transparent',
              color: metric === m.key ? '#c084fc' : 'var(--text-muted)',
              border: `1px solid ${metric === m.key ? 'rgba(168,85,247,0.3)' : 'var(--border)'}`,
            }}
          >
            {m.label}
          </button>
        ))}
      </div>

      {/* Grouped Bar Chart — All Strategies by Regime */}
      <div className="rounded-xl border border-[var(--border)] p-4" style={{ backgroundColor: 'var(--bg-card)' }}>
        <h3 className="text-xs font-mono uppercase tracking-wider text-[var(--text-dim)] mb-3">
          Strategy Comparison by Regime — {metric === 'avg_pnl' ? 'Avg Daily %' : metric === 'sharpe' ? 'Sharpe Ratio' : 'Win Rate %'}
        </h3>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={chartData} margin={{ top: 5, right: 10, left: 10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
            <XAxis dataKey="state" tick={{ fontSize: 8, fill: 'var(--text-dim)' }} angle={-20} textAnchor="end" height={50} />
            <YAxis tick={{ fontSize: 9, fill: 'var(--text-dim)' }} tickFormatter={(v: number) => metric === 'win_rate' ? `${v.toFixed(0)}%` : metric === 'avg_pnl' ? `${v.toFixed(3)}%` : v.toFixed(1)} />
            <Tooltip
              contentStyle={{ backgroundColor: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 11, fontFamily: 'monospace' }}
              formatter={(value, name) => {
                const label = STRATEGY_META[String(name)]?.name ?? String(name)
                const v = Number(value)
                if (metric === 'avg_pnl') return [`${v.toFixed(4)}%`, label]
                if (metric === 'win_rate') return [`${v.toFixed(1)}%`, label]
                return [v.toFixed(2), label]
              }}
            />
            {strategies.map(s => (
              <Bar key={s} dataKey={s} name={s} fill={STRATEGY_META[s].accent.startsWith('var') ? (s === 'dm' ? '#ffd740' : s === 'wc' ? '#448aff' : '#64ffda') : STRATEGY_META[s].accent} barSize={18} radius={[2, 2, 0, 0]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
        <div className="flex gap-4 mt-2 justify-center">
          {strategies.map(s => (
            <div key={s} className="flex items-center gap-1.5">
              <div className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: s === 'dm' ? '#ffd740' : s === 'wc' ? '#448aff' : '#64ffda' }} />
              <span className="text-[9px] font-mono text-[var(--text-muted)]">{STRATEGY_META[s].name}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Per-Strategy Detail Tables */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {strategies.map(s => {
          const sData = strategyData[s]
          if (!sData) return null
          const accent = s === 'dm' ? '#ffd740' : s === 'wc' ? '#448aff' : '#64ffda'

          return (
            <div key={s} className="rounded-xl border border-[var(--border)] overflow-hidden" style={{ backgroundColor: 'var(--bg-card)' }}>
              <div className="px-4 py-2.5 border-b border-[var(--border)] bg-[var(--bg-elevated)]">
                <span className="text-[11px] font-mono font-semibold" style={{ color: accent }}>{STRATEGY_META[s].name}</span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-[10px] font-mono">
                  <thead>
                    <tr className="border-b border-[var(--border)]">
                      <th className="text-left px-3 py-1.5 text-[var(--text-dim)] font-normal">State</th>
                      <th className="text-right px-2 py-1.5 text-[var(--text-dim)] font-normal">Days</th>
                      <th className="text-right px-2 py-1.5 text-[var(--text-dim)] font-normal">Avg %</th>
                      <th className="text-right px-2 py-1.5 text-[var(--text-dim)] font-normal">WR</th>
                      <th className="text-right px-2 py-1.5 text-[var(--text-dim)] font-normal">Sh</th>
                      <th className="text-right px-2 py-1.5 text-[var(--text-dim)] font-normal">Total %</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sData.states.map(st => (
                      <tr key={st.state} className="border-b border-[var(--border-subtle)] hover:bg-[var(--bg-hover)]">
                        <td className="px-3 py-1.5">
                          <div className="flex items-center gap-1.5">
                            <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: st.color }} />
                            <span className="text-[var(--text-secondary)]">{st.state.replace('L1 ', '').replace('L2 ', '').replace('L3 ', '')}</span>
                          </div>
                        </td>
                        <td className="text-right px-2 py-1.5 text-[var(--text-muted)]">{st.days}</td>
                        <td className="text-right px-2 py-1.5 font-semibold" style={{ color: st.avg_pnl != null && st.avg_pnl > 0 ? 'var(--signal-positive)' : st.avg_pnl != null && st.avg_pnl < 0 ? 'var(--signal-negative)' : 'var(--text-dim)' }}>
                          {st.avg_pnl != null ? `${st.avg_pnl > 0 ? '+' : ''}${st.avg_pnl.toFixed(3)}` : '—'}
                        </td>
                        <td className="text-right px-2 py-1.5 text-[var(--text-secondary)]">
                          {st.win_rate != null ? `${(st.win_rate * 100).toFixed(0)}%` : '—'}
                        </td>
                        <td className="text-right px-2 py-1.5" style={{ color: st.sharpe != null && st.sharpe >= 2 ? 'var(--signal-positive)' : st.sharpe != null && st.sharpe < 0 ? 'var(--signal-negative)' : 'var(--text-secondary)' }}>
                          {st.sharpe?.toFixed(2) ?? '—'}
                        </td>
                        <td className="text-right px-2 py-1.5 font-semibold" style={{ color: st.total_pct != null && st.total_pct > 0 ? 'var(--signal-positive)' : 'var(--signal-negative)' }}>
                          {st.total_pct != null ? `${st.total_pct > 0 ? '+' : ''}${st.total_pct.toFixed(1)}` : '—'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )
        })}
      </div>

      {/* DM-Orion Mirror Insight */}
      <div className="rounded-xl border border-[var(--border)] p-5" style={{ backgroundColor: 'var(--bg-card)' }}>
        <h3 className="text-xs font-mono uppercase tracking-wider text-[var(--text-dim)] mb-3">Key Insight: DM-Orion Mirror</h3>
        <div className="space-y-2 text-xs font-mono text-[var(--text-secondary)]">
          <p>DM and Orion are near-perfect mirrors (correlation -0.23). This is the portfolio&apos;s natural hedge.</p>
          <p><strong className="text-[#ffd740]">DM</strong> excels in low-vol (L1 Safe) and high-vol with cushion (L3 Safe).</p>
          <p><strong className="text-[#64ffda]">Orion</strong> excels when volatility is trending (L2 Caution-B, L3 Exposed).</p>
          <p>The combined portfolio stays positive in almost all regimes because when one side loses, the other gains.</p>
        </div>
      </div>
    </div>
  )
}
