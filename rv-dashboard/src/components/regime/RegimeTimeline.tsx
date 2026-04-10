'use client'

import { useState, useEffect, useMemo, useCallback } from 'react'
import { fetchRegimeTimeseries } from '@/lib/api'
import type { RegimeTimeseriesPoint } from '@/lib/types'
import {
  ComposedChart, Area, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Legend, ReferenceLine,
} from 'recharts'

interface Props {
  startDate: string
  endDate: string
  snapshot: string
  dte: number | null
}

const STATE_ORDER = [
  'L1 Safe', 'L1 Exposed',
  'L2 Safe', 'L2 Caution-A', 'L2 Caution-B', 'L2 Risky',
  'L3 Safe', 'L3 Exposed',
]

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

const STATE_Y_MAP: Record<string, number> = {
  'L1 Safe': 1,
  'L1 Exposed': 2,
  'L2 Safe': 3,
  'L2 Caution-A': 4,
  'L2 Caution-B': 5,
  'L2 Risky': 6,
  'L3 Safe': 7,
  'L3 Exposed': 8,
}

type ChartView = 'regime_band' | 'iv_level' | 'pk_iv'

export function RegimeTimeline({ startDate, endDate, snapshot, dte }: Props) {
  const [data, setData] = useState<RegimeTimeseriesPoint[]>([])
  const [loading, setLoading] = useState(true)
  const [view, setView] = useState<ChartView>('regime_band')

  useEffect(() => {
    setLoading(true)
    fetchRegimeTimeseries(startDate, endDate, snapshot, dte)
      .then(setData)
      .finally(() => setLoading(false))
  }, [startDate, endDate, snapshot, dte])

  const chartData = useMemo(() => {
    const sampled = data.length > 800 ? data.filter((_, i) => i % Math.ceil(data.length / 800) === 0 || i === data.length - 1) : data
    return sampled.map(d => ({
      ...d,
      regime_y: STATE_Y_MAP[d.regime_state] ?? 0,
      cum_pnl: 0, // will be computed below
    }))
  }, [data])

  // Compute cumulative PnL
  const withCum = useMemo(() => {
    let cum = 0
    return chartData.map(d => {
      cum += d.pnl_combined ?? 0
      return { ...d, cum_pnl: cum }
    })
  }, [chartData])

  const formatDate = useCallback((d: string) => {
    if (!d) return ''
    const parts = d.split('-')
    return `${parts[1]}/${parts[2].substring(0, 2)}`
  }, [])

  if (loading) return <div className="text-center py-20 text-[var(--text-dim)] text-sm font-mono">Loading timeline...</div>

  return (
    <div className="space-y-4">
      {/* View Toggle */}
      <div className="flex items-center gap-2">
        {([
          { key: 'regime_band', label: 'Regime States' },
          { key: 'iv_level', label: 'IV Level & PK/IV' },
          { key: 'pk_iv', label: 'Feature Inputs' },
        ] as { key: ChartView; label: string }[]).map(v => (
          <button
            key={v.key}
            onClick={() => setView(v.key)}
            className="px-3 py-1.5 rounded-lg text-[10px] font-mono font-semibold transition-all"
            style={{
              backgroundColor: view === v.key ? 'rgba(168,85,247,0.15)' : 'transparent',
              color: view === v.key ? '#c084fc' : 'var(--text-muted)',
              border: `1px solid ${view === v.key ? 'rgba(168,85,247,0.3)' : 'var(--border)'}`,
            }}
          >
            {v.label}
          </button>
        ))}
      </div>

      {/* Regime State Band Chart */}
      {view === 'regime_band' && (
        <div className="rounded-xl border border-[var(--border)] p-4" style={{ backgroundColor: 'var(--bg-card)' }}>
          <h3 className="text-xs font-mono uppercase tracking-wider text-[var(--text-dim)] mb-3">Regime State Timeline</h3>
          <ResponsiveContainer width="100%" height={200}>
            <ComposedChart data={withCum} margin={{ top: 5, right: 10, left: 10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
              <XAxis dataKey="date" tickFormatter={formatDate} tick={{ fontSize: 9, fill: 'var(--text-dim)' }} interval="preserveStartEnd" minTickGap={60} />
              <YAxis domain={[0.5, 8.5]} ticks={[1, 2, 3, 4, 5, 6, 7, 8]} tickFormatter={(v: number) => STATE_ORDER[v - 1]?.replace('L1 ', '').replace('L2 ', '').replace('L3 ', '') ?? ''} tick={{ fontSize: 8, fill: 'var(--text-dim)' }} width={65} />
              <Tooltip
                contentStyle={{ backgroundColor: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 11, fontFamily: 'monospace' }}
                formatter={(value, name) => {
                  const v = Number(value)
                  if (name === 'Regime') return [STATE_ORDER[v - 1] ?? '', 'State']
                  return [v, String(name)]
                }}
                labelFormatter={(label) => String(label)}
              />
              <Bar dataKey="regime_y" name="Regime" barSize={3}>
                {withCum.map((entry, i) => (
                  <Cell key={i} fill={entry.color} />
                ))}
              </Bar>
            </ComposedChart>
          </ResponsiveContainer>

          {/* Legend */}
          <div className="flex flex-wrap gap-3 mt-3 px-2">
            {STATE_ORDER.map(s => (
              <div key={s} className="flex items-center gap-1.5">
                <div className="w-2 h-2 rounded-full" style={{ backgroundColor: STATE_COLORS[s] }} />
                <span className="text-[9px] font-mono text-[var(--text-muted)]">{s}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* IV Level + PK/IV Chart */}
      {view === 'iv_level' && (
        <div className="rounded-xl border border-[var(--border)] p-4" style={{ backgroundColor: 'var(--bg-card)' }}>
          <h3 className="text-xs font-mono uppercase tracking-wider text-[var(--text-dim)] mb-3">IV Level & PK/IV Ratio</h3>
          <ResponsiveContainer width="100%" height={300}>
            <ComposedChart data={withCum} margin={{ top: 5, right: 10, left: 10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
              <XAxis dataKey="date" tickFormatter={formatDate} tick={{ fontSize: 9, fill: 'var(--text-dim)' }} interval="preserveStartEnd" minTickGap={60} />
              <YAxis yAxisId="iv" orientation="left" tick={{ fontSize: 9, fill: '#c084fc' }} label={{ value: 'IV', angle: -90, position: 'insideLeft', fill: '#c084fc', fontSize: 10 }} />
              <YAxis yAxisId="pkiv" orientation="right" tick={{ fontSize: 9, fill: '#64ffda' }} label={{ value: 'PK/IV', angle: 90, position: 'insideRight', fill: '#64ffda', fontSize: 10 }} />
              <Tooltip
                contentStyle={{ backgroundColor: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 11, fontFamily: 'monospace' }}
                labelFormatter={(label) => String(label)}
              />
              <ReferenceLine yAxisId="iv" y={12} stroke="rgba(255,255,255,0.2)" strokeDasharray="5 5" label={{ value: 'L1/L2 (12)', fill: 'var(--text-dim)', fontSize: 9, position: 'right' }} />
              <ReferenceLine yAxisId="iv" y={17} stroke="rgba(255,255,255,0.2)" strokeDasharray="5 5" label={{ value: 'L2/L3 (17)', fill: 'var(--text-dim)', fontSize: 9, position: 'right' }} />
              <Area yAxisId="iv" type="monotone" dataKey="iv_lag" name="IV (lag)" stroke="#c084fc" fill="rgba(192,132,252,0.1)" dot={false} strokeWidth={1.5} />
              <Area yAxisId="pkiv" type="monotone" dataKey="pk_iv_ratio" name="PK/IV Ratio" stroke="#64ffda" fill="rgba(100,255,218,0.05)" dot={false} strokeWidth={1.5} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Feature Inputs Chart */}
      {view === 'pk_iv' && (
        <div className="rounded-xl border border-[var(--border)] p-4" style={{ backgroundColor: 'var(--bg-card)' }}>
          <h3 className="text-xs font-mono uppercase tracking-wider text-[var(--text-dim)] mb-3">Feature Inputs (5d Averages)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <ComposedChart data={withCum} margin={{ top: 5, right: 10, left: 10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
              <XAxis dataKey="date" tickFormatter={formatDate} tick={{ fontSize: 9, fill: 'var(--text-dim)' }} interval="preserveStartEnd" minTickGap={60} />
              <YAxis yAxisId="vol" orientation="left" tick={{ fontSize: 9, fill: '#c084fc' }} />
              <YAxis yAxisId="chg" orientation="right" tick={{ fontSize: 9, fill: '#ffab40' }} />
              <Tooltip
                contentStyle={{ backgroundColor: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 11, fontFamily: 'monospace' }}
                labelFormatter={(label) => String(label)}
              />
              <Legend wrapperStyle={{ fontSize: 10 }} />
              <Area yAxisId="vol" type="monotone" dataKey="iv_5d" name="IV 5d" stroke="#c084fc" fill="rgba(192,132,252,0.08)" dot={false} strokeWidth={1.5} />
              <Area yAxisId="vol" type="monotone" dataKey="pk_5d" name="PK 5d" stroke="#64ffda" fill="rgba(100,255,218,0.05)" dot={false} strokeWidth={1.5} />
              <Area yAxisId="chg" type="monotone" dataKey="iv_chg_5d" name="IV Chg 5d" stroke="#ffab40" fill="rgba(255,171,64,0.05)" dot={false} strokeWidth={1} />
              <ReferenceLine yAxisId="chg" y={0} stroke="rgba(255,255,255,0.15)" strokeDasharray="3 3" />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Cumulative PnL with Regime Color Background */}
      <div className="rounded-xl border border-[var(--border)] p-4" style={{ backgroundColor: 'var(--bg-card)' }}>
        <h3 className="text-xs font-mono uppercase tracking-wider text-[var(--text-dim)] mb-3">Combined Portfolio Equity Curve</h3>
        <ResponsiveContainer width="100%" height={250}>
          <ComposedChart data={withCum} margin={{ top: 5, right: 10, left: 10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
            <XAxis dataKey="date" tickFormatter={formatDate} tick={{ fontSize: 9, fill: 'var(--text-dim)' }} interval="preserveStartEnd" minTickGap={60} />
            <YAxis tick={{ fontSize: 9, fill: 'var(--text-dim)' }} tickFormatter={(v: number) => `${v.toFixed(0)}%`} />
            <Tooltip
              contentStyle={{ backgroundColor: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 11, fontFamily: 'monospace' }}
              formatter={(value, name) => {
                const v = Number(value)
                if (name === 'Cumulative %') return [`${v.toFixed(2)}%`, String(name)]
                return [v, String(name)]
              }}
              labelFormatter={(label) => String(label)}
            />
            <Area type="monotone" dataKey="cum_pnl" name="Cumulative %" stroke="rgba(168,85,247,0.8)" fill="rgba(168,85,247,0.15)" dot={false} strokeWidth={1.5} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
