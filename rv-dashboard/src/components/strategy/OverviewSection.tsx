'use client'

import { useState, useMemo, useEffect } from 'react'
import {
  AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell, ReferenceLine,
} from 'recharts'
import { fetchPlainReturns, fetchRVTimeseries } from '@/lib/api'
import type { PlainReturnsData, MonthlyReturn, RVDataPoint } from '@/lib/types'
import { formatPct } from '@/lib/formatters'
import { STRATEGY_META } from '@/lib/config'
import { StatCard } from './StatCard'
import { RVPlotsSection } from '@/components/RVChart'
import { useSortableData, SortableTh } from './SortableHeader'

const MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

// ── Yearly Table ──
function YearlyTable({ yearly }: { yearly: PlainReturnsData['yearly'] }) {
  const { sorted, sort, toggle } = useSortableData(yearly)
  const thR = "text-right py-2 px-3"

  return (
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
  )
}

// ── Main Overview Section ──
export function OverviewSection({ strategy, startDate, endDate, snapshot }: { strategy: string; startDate: string; endDate: string; snapshot: string }) {
  const meta = STRATEGY_META[strategy] || STRATEGY_META.dm
  const [data, setData] = useState<PlainReturnsData | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    fetchPlainReturns(strategy, startDate, endDate, snapshot).then(d => { setData(d); setLoading(false) })
  }, [strategy, startDate, endDate, snapshot])

  const chartData = useMemo(() => {
    if (!data) return []
    const ts = data.daily_timeseries
    if (ts.length <= 500) return ts
    const sampled = []
    for (let i = 0; i < ts.length; i += 3) sampled.push(ts[i])
    if (sampled[sampled.length - 1] !== ts[ts.length - 1]) sampled.push(ts[ts.length - 1])
    return sampled
  }, [data])

  const ddChartData = useMemo(() => {
    if (!data?.summary?.dd_series) return []
    const dd = data.summary.dd_series
    if (dd.length <= 500) return dd
    const sampled = []
    for (let i = 0; i < dd.length; i += 3) sampled.push(dd[i])
    if (sampled[sampled.length - 1] !== dd[dd.length - 1]) sampled.push(dd[dd.length - 1])
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

  const pnlDistribution = useMemo(() => {
    if (!data) return []
    const pnls = data.daily_timeseries.map(d => d.pnl_pct)
    const min = Math.min(...pnls)
    const max = Math.max(...pnls)
    const binCount = 40
    const binWidth = (max - min) / binCount
    if (binWidth === 0) return []
    const bins: { range: string; center: number; count: number; pct: number }[] = []
    for (let i = 0; i < binCount; i++) {
      const lo = min + i * binWidth
      const hi = lo + binWidth
      const count = pnls.filter(p => i === binCount - 1 ? (p >= lo && p <= hi) : (p >= lo && p < hi)).length
      bins.push({
        range: `${lo.toFixed(2)} – ${hi.toFixed(2)}`,
        center: (lo + hi) / 2,
        count,
        pct: (count / pnls.length) * 100,
      })
    }
    return bins
  }, [data])

  if (loading || !data) {
    return <div className="flex items-center justify-center h-64"><div className="text-[var(--text-muted)]">Loading {meta.name}...</div></div>
  }

  const s = data.summary

  const fv = (v: number | null, suffix = '') => v != null ? `${v.toFixed(2)}${suffix}` : '—'

  return (
    <div>
      <div className="mb-6">
        <p className="text-[11px] font-mono text-[var(--text-dim)] mb-1 tracking-wider uppercase">{meta.name} / overview</p>
        <h1 className="text-2xl font-bold text-[var(--text-primary)] tracking-tight">Strategy Overview</h1>
        <p className="text-sm text-[var(--text-secondary)] mt-1">{data.date_range[0]} to {data.date_range[1]} · {s.total_days.toLocaleString()} trading days</p>
      </div>

      {/* ── Key Performance Metrics ── */}
      <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5 mb-5">
        <h2 className="text-sm font-semibold text-[var(--text-primary)] mb-4 uppercase tracking-wider">Performance</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
          <StatCard label="Total Return" value={`${s.total_pct.toFixed(2)}%`} />
          <StatCard label="CAGR" value={fv(s.cagr_pct, '%')} />
          <StatCard label="Ann. Return" value={`${s.ann_return_pct.toFixed(2)}%`} />
          <StatCard label="Ann. Volatility" value={fv(s.ann_vol_pct, '%')} />
          <StatCard label="Sharpe Ratio" value={fv(s.sharpe)} sub="(Ann. Ret − Rf) / Ann. Vol" />
          <StatCard label="Sortino Ratio" value={fv(s.sortino)} sub="Downside risk adjusted" />
        </div>
      </div>

      {/* ── Risk Metrics ── */}
      <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5 mb-5">
        <h2 className="text-sm font-semibold text-[var(--text-primary)] mb-4 uppercase tracking-wider">Risk</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
          <StatCard label="Max Drawdown" value={formatPct(s.max_drawdown_pct)} />
          <StatCard label="Max DD Duration" value={`${s.max_dd_duration_days}d`} />
          <StatCard label="Calmar Ratio" value={fv(s.calmar)} sub="CAGR / |Max DD|" />
          <StatCard label="Recovery Factor" value={fv(s.recovery_factor)} sub="Total Ret / |Max DD|" />
          <StatCard label="Daily VaR (5%)" value={fv(s.var_95_pct, '%')} sub="Worst 5th percentile" />
          <StatCard label="Tail Ratio" value={fv(s.tail_ratio)} sub="P95 / |P5|" />
        </div>
      </div>

      {/* ── Trade Statistics ── */}
      <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5 mb-5">
        <h2 className="text-sm font-semibold text-[var(--text-primary)] mb-4 uppercase tracking-wider">Trade Statistics</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
          <StatCard label="Win Rate" value={`${(s.win_rate * 100).toFixed(1)}%`} sub={`${s.positive_days}W / ${s.negative_days}L / ${s.flat_days}F`} />
          <StatCard label="Profit Factor" value={fv(s.profit_factor)} sub={`Gross P: ${s.gross_profit_pct.toFixed(1)}% / L: ${s.gross_loss_pct.toFixed(1)}%`} />
          <StatCard label="Expectancy" value={`${s.expectancy_pct.toFixed(4)}%`} sub="Per trade edge" />
          <StatCard label="Payoff Ratio" value={fv(s.payoff_ratio)} sub={`Avg Win: ${formatPct(s.avg_win_pct)} / Loss: ${formatPct(s.avg_loss_pct)}`} />
          <StatCard label="Best / Worst Day" value={formatPct(s.max_win_pct)} sub={`Worst: ${formatPct(s.max_loss_pct)}`} />
          <StatCard label="Avg Daily" value={formatPct(s.mean_daily_pct)} sub={`Median: ${formatPct(s.median_daily_pct)}`} />
        </div>
      </div>

      {/* ── Streaks & Distribution ── */}
      <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5 mb-5">
        <h2 className="text-sm font-semibold text-[var(--text-primary)] mb-4 uppercase tracking-wider">Distribution & Streaks</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
          <StatCard label="Max Consec. Wins" value={`${s.max_consec_wins}`} />
          <StatCard label="Max Consec. Losses" value={`${s.max_consec_losses}`} />
          <StatCard label="Skewness" value={s.skewness != null ? s.skewness.toFixed(3) : '—'} sub={s.skewness != null ? (s.skewness > 0 ? 'Right-tailed' : s.skewness < 0 ? 'Left-tailed' : 'Symmetric') : ''} />
          <StatCard label="Kurtosis" value={s.kurtosis != null ? s.kurtosis.toFixed(3) : '—'} sub={s.kurtosis != null ? (s.kurtosis > 0 ? 'Fat tails' : 'Thin tails') : ''} />
          <StatCard label="P5 / P95" value={`${fv(s.p5_pct, '%')} / ${fv(s.p95_pct, '%')}`} sub="Daily return percentiles" />
          <StatCard label="Daily Std" value={formatPct(s.std_daily_pct)} />
        </div>
      </div>

      {/* ── Cumulative Returns ── */}
      <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5 mb-5">
        <h2 className="text-sm font-semibold text-[var(--text-primary)] mb-4 uppercase tracking-wider">Equity Curve</h2>
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
            <XAxis dataKey="date" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} tickFormatter={(d: string) => d.slice(0, 7)} interval={Math.floor(chartData.length / 8)} />
            <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }} tickFormatter={(v: number) => `${v.toFixed(1)}%`} />
            <Tooltip contentStyle={{ backgroundColor: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12, color: '#ffffff' }} labelStyle={{ color: '#ffffff' }} itemStyle={{ color: '#ffffff' }}
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              formatter={(v: any) => [`${Number(v).toFixed(2)}%`, 'Cumulative %']} />
            <ReferenceLine y={0} stroke="var(--text-dim)" strokeDasharray="3 3" />
            <Area type="monotone" dataKey="cumulative_pct" stroke={meta.accent} fill={`rgba(${meta.accentRgb},0.08)`} strokeWidth={1.5} />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* ── Drawdown Chart ── */}
      <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5 mb-5">
        <h2 className="text-sm font-semibold text-[var(--text-primary)] mb-4 uppercase tracking-wider">Underwater (Drawdown)</h2>
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart data={ddChartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
            <XAxis dataKey="date" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} tickFormatter={(d: string) => d.slice(0, 7)} interval={Math.floor(ddChartData.length / 8)} />
            <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }} tickFormatter={(v: number) => `${v.toFixed(1)}%`} domain={['dataMin', 0]} />
            <Tooltip contentStyle={{ backgroundColor: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12, color: '#ffffff' }} labelStyle={{ color: '#ffffff' }} itemStyle={{ color: '#ffffff' }}
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              formatter={(v: any) => [`${Number(v).toFixed(2)}%`, 'Drawdown']} />
            <ReferenceLine y={0} stroke="var(--text-dim)" strokeDasharray="3 3" />
            <Area type="monotone" dataKey="dd_pct" stroke="var(--signal-negative)" fill="rgba(255,82,82,0.15)" strokeWidth={1.5} />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* ── Daily Returns Distribution ── */}
      <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5 mb-5">
        <h2 className="text-sm font-semibold text-[var(--text-primary)] mb-4 uppercase tracking-wider">Daily Returns Distribution</h2>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={pnlDistribution} barCategoryGap={0}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
            <XAxis dataKey="center" tick={{ fontSize: 9, fill: 'var(--text-muted)' }} tickFormatter={(v: number) => `${v.toFixed(1)}%`} interval={Math.floor(pnlDistribution.length / 8)} />
            <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }} tickFormatter={(v: number) => `${v.toFixed(0)}`} />
            <Tooltip contentStyle={{ backgroundColor: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 11, color: '#ffffff' }} labelStyle={{ color: '#ffffff' }} itemStyle={{ color: '#ffffff' }}
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              formatter={(v: any, _: any, entry: any) => [`${entry.payload.count} days (${Number(v).toFixed(1)}%)`, entry.payload.range]} />
            <Bar dataKey="pct" radius={[2, 2, 0, 0]}>
              {pnlDistribution.map((d, i) => (
                <Cell key={i} fill={d.center >= 0 ? 'var(--signal-positive)' : 'var(--signal-negative)'} fillOpacity={0.6} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* ── Yearly Returns + Monthly Heatmap ── */}
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

      {/* ── Yearly Breakdown Table ── */}
      <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5 mb-5">
        <h2 className="text-sm font-semibold text-[var(--text-primary)] mb-4 uppercase tracking-wider">Yearly Breakdown</h2>
        <YearlyTable yearly={data.yearly} />
      </div>

      {/* ── RV/IV Chart ── */}
      <RVPlotsSection startDate={startDate} endDate={endDate} />
    </div>
  )
}
