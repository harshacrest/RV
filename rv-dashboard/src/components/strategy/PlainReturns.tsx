'use client'

import { useState, useMemo, useEffect } from 'react'
import {
  AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell, ReferenceLine,
} from 'recharts'
import { fetchPlainReturns } from '@/lib/api'
import type { PlainReturnsData, MonthlyReturn } from '@/lib/types'
import { formatPct, formatVal } from '@/lib/formatters'
import type { DisplayMode } from '@/lib/formatters'
import { STRATEGY_META } from '@/lib/config'
import { ModeToggle } from './ModeToggle'
import { StatCard } from './StatCard'
import { useSortableData, SortableTh } from './SortableHeader'

const MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

function YearlyTable({ yearly }: { yearly: PlainReturnsData['yearly'] }) {
  const { sorted, sort, toggle } = useSortableData(yearly)
  const thR = "text-right py-2 px-3"

  return (
    <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5">
      <h2 className="text-sm font-semibold text-[var(--text-primary)] mb-4 uppercase tracking-wider">Yearly Breakdown</h2>
      <div className="overflow-x-auto">
        <table className="w-full text-[11px] font-mono">
          <thead>
            <tr className="text-[var(--text-dim)] border-b border-[var(--border)]">
              <SortableTh column="year" label="Year" sort={sort} toggle={toggle} className="text-left py-2 pr-4" />
              <SortableTh column="total_pct" label="Total %" sort={sort} toggle={toggle} className={thR} />
              <SortableTh column="mean_daily_pct" label="Avg Daily" sort={sort} toggle={toggle} className={thR} />
              <SortableTh column="std_daily_pct" label="Std" sort={sort} toggle={toggle} className={thR} />
              <SortableTh column="win_rate" label="Win Rate" sort={sort} toggle={toggle} className={thR} />
              <SortableTh column="sharpe_pct" label="Sharpe" sort={sort} toggle={toggle} className={thR} />
              <SortableTh column="max_win_pct" label="Max Win" sort={sort} toggle={toggle} className={thR} />
              <SortableTh column="max_loss_pct" label="Max Loss" sort={sort} toggle={toggle} className={thR} />
              <SortableTh column="trading_days" label="Days" sort={sort} toggle={toggle} className="text-right py-2 pl-3" />
            </tr>
          </thead>
          <tbody>
            {sorted.map(y => (
              <tr key={y.year} className="border-b border-[var(--border-subtle)] hover:bg-[var(--bg-hover)] transition-colors">
                <td className="py-2 pr-4 text-[var(--text-primary)] font-semibold">{y.year}</td>
                <td className="py-2 px-3 text-right" style={{ color: y.total_pct >= 0 ? 'var(--signal-positive)' : 'var(--signal-negative)' }}>{formatPct(y.total_pct)}</td>
                <td className="py-2 px-3 text-right text-[var(--text-secondary)]">{formatPct(y.mean_daily_pct)}</td>
                <td className="py-2 px-3 text-right text-[var(--text-muted)]">{formatPct(y.std_daily_pct)}</td>
                <td className="py-2 px-3 text-right text-[var(--text-secondary)]">{(y.win_rate * 100).toFixed(1)}%</td>
                <td className="py-2 px-3 text-right text-[var(--text-secondary)]">{y.sharpe_pct ?? '—'}</td>
                <td className="py-2 px-3 text-right" style={{ color: 'var(--signal-positive)' }}>{formatPct(y.max_win_pct)}</td>
                <td className="py-2 px-3 text-right" style={{ color: 'var(--signal-negative)' }}>{formatPct(y.max_loss_pct)}</td>
                <td className="py-2 pl-3 text-right text-[var(--text-muted)]">{y.trading_days}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export function PlainReturns({ strategy }: { strategy: string }) {
  const meta = STRATEGY_META[strategy] || STRATEGY_META.dm
  const [data, setData] = useState<PlainReturnsData | null>(null)
  const [mode, setMode] = useState<DisplayMode>('pct')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    fetchPlainReturns(strategy).then(d => { setData(d); setLoading(false) })
  }, [strategy])

  const chartData = useMemo(() => {
    if (!data) return []
    const ts = data.daily_timeseries
    if (ts.length <= 500) return ts
    const sampled = []
    for (let i = 0; i < ts.length; i += 3) sampled.push(ts[i])
    if (sampled[sampled.length - 1] !== ts[ts.length - 1]) sampled.push(ts[ts.length - 1])
    return sampled
  }, [data])

  const heatmapData = useMemo(() => {
    if (!data) return { years: [] as number[], grid: [] as (MonthlyReturn | null)[][] }
    const years = Array.from(new Set(data.monthly.map(m => m.year))).sort()
    const grid = years.map(y => Array.from({ length: 12 }, (_, i) => data.monthly.find(m => m.year === y && m.month === i + 1) || null))
    return { years, grid }
  }, [data])

  const heatmapExtent = useMemo(() => {
    if (!data) return 1
    const vals = data.monthly.map(m => Math.abs(m.total_pct))
    return Math.max(...vals, 1)
  }, [data])

  if (loading || !data) {
    return <div className="flex items-center justify-center h-64"><div className="text-[var(--text-muted)]">Loading {meta.name}...</div></div>
  }

  const s = data.summary

  return (
    <div>
      <div className="mb-6 flex items-start justify-between">
        <div>
          <p className="text-[11px] font-mono text-[var(--text-dim)] mb-1 tracking-wider uppercase">{meta.name} / raw returns</p>
          <h1 className="text-2xl font-bold text-[var(--text-primary)] tracking-tight">Plain Returns</h1>
          <p className="text-sm text-[var(--text-secondary)] mt-1">{data.date_range[0]} to {data.date_range[1]} &middot; {s.total_days.toLocaleString()} trading days</p>
        </div>
        <ModeToggle mode={mode} setMode={setMode} showSharpe={false} />
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3 mb-6">
        <StatCard label="Total Return" value={formatVal(s.total_pct, mode)} />
        <StatCard label="Avg Daily" value={formatPct(s.mean_daily_pct)} sub={`Median: ${formatPct(s.median_daily_pct)}`} />
        <StatCard label="Win Rate" value={`${(s.win_rate * 100).toFixed(1)}%`} sub={`${s.positive_days}W / ${s.negative_days}L`} />
        <StatCard label="Sharpe" value={s.sharpe != null ? s.sharpe.toFixed(2) : '—'} sub="Annualised" />
        <StatCard label="Max Drawdown" value={formatPct(s.max_drawdown_pct)} />
        <StatCard label="Best / Worst" value={formatPct(s.max_win_pct)} sub={`Worst: ${formatPct(s.max_loss_pct)}`} />
      </div>

      {/* Cumulative Returns */}
      <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5 mb-5">
        <h2 className="text-sm font-semibold text-[var(--text-primary)] mb-4 uppercase tracking-wider">Cumulative Returns (%)</h2>
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
            <XAxis dataKey="date" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} tickFormatter={(d: string) => d.slice(0, 7)} interval={Math.floor(chartData.length / 8)} />
            <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }} tickFormatter={(v: number) => `${v.toFixed(1)}%`} />
            <Tooltip contentStyle={{ backgroundColor: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12, color: '#ffffff' }} labelStyle={{ color: '#ffffff' }} itemStyle={{ color: '#ffffff' }} // eslint-disable-next-line @typescript-eslint/no-explicit-any
            formatter={(v: any) => [`${Number(v).toFixed(2)}%`, 'Cumulative %']} />
            <ReferenceLine y={0} stroke="var(--text-dim)" strokeDasharray="3 3" />
            <Area type="monotone" dataKey="cumulative_pct" stroke={meta.accent} fill={`rgba(${meta.accentRgb},0.08)`} strokeWidth={1.5} />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Yearly + Monthly */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 mb-5">
        {/* Yearly Bar */}
        <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5">
          <h2 className="text-sm font-semibold text-[var(--text-primary)] mb-4 uppercase tracking-wider">Yearly Returns</h2>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={data.yearly}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
              <XAxis dataKey="year" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
              <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }} tickFormatter={(v: number) => `${v.toFixed(1)}%`} />
              <Tooltip
                contentStyle={{ backgroundColor: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12, color: '#ffffff' }} labelStyle={{ color: '#ffffff' }} itemStyle={{ color: '#ffffff' }}
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                formatter={(v: any, _: any, entry: any) => {
                  const p = entry?.payload
                  return [`${v.toFixed(2)}% | WR: ${((p?.win_rate as number ?? 0) * 100).toFixed(0)}% | Sharpe: ${p?.sharpe_pct ?? '—'}`, 'Total Return']
                }}
              />
              <ReferenceLine y={0} stroke="var(--text-dim)" strokeDasharray="3 3" />
              <Bar dataKey="total_pct" radius={[4, 4, 0, 0]}>
                {data.yearly.map((e, i) => (
                  <Cell key={i} fill={e.total_pct >= 0 ? 'var(--signal-positive)' : 'var(--signal-negative)'} fillOpacity={0.7} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Monthly Heatmap */}
        <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5">
          <h2 className="text-sm font-semibold text-[var(--text-primary)] mb-4 uppercase tracking-wider">Monthly Returns Heatmap</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-[10px] font-mono">
              <thead>
                <tr>
                  <th className="text-left text-[var(--text-dim)] pb-2 pr-2">Year</th>
                  {MONTH_LABELS.map(m => <th key={m} className="text-center text-[var(--text-dim)] pb-2 px-0.5">{m}</th>)}
                </tr>
              </thead>
              <tbody>
                {heatmapData.years.map((year, yi) => (
                  <tr key={year}>
                    <td className="text-[var(--text-secondary)] pr-2 py-0.5 font-semibold">{year}</td>
                    {heatmapData.grid[yi].map((cell, mi) => {
                      if (!cell) return <td key={mi} className="px-0.5 py-0.5"><div className="w-full h-7 rounded bg-[var(--bg-hover)]" /></td>
                      const intensity = Math.min(Math.abs(cell.total_pct) / heatmapExtent, 1)
                      const bg = cell.total_pct >= 0
                        ? `rgba(0,230,118,${0.05 + intensity * 0.4})`
                        : `rgba(255,82,82,${0.05 + intensity * 0.4})`
                      return (
                        <td key={mi} className="px-0.5 py-0.5">
                          <div className="w-full h-7 rounded flex items-center justify-center text-[9px] font-bold cursor-default transition-all hover:scale-105" style={{ backgroundColor: bg, color: 'var(--text-primary)' }} title={`${MONTH_LABELS[mi]} ${year}: ${cell.total_pct.toFixed(2)}% | WR: ${(cell.win_rate * 100).toFixed(0)}%`}>
                            {formatPct(cell.total_pct)}
                          </div>
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Yearly Table */}
      <YearlyTable yearly={data.yearly} />
    </div>
  )
}
