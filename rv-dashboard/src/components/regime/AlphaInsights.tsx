'use client'

import { useState, useEffect, useMemo } from 'react'
import { fetchAlphaInsights } from '@/lib/api'
import type {
  AlphaInsightsData, AlphaStateEntry, AlphaFeatureIC,
  AlphaRankDetail, AlphaStrategyCorr, AlphaDistributionWarning,
} from '@/lib/types'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell, ScatterChart, Scatter, ZAxis, CartesianGrid,
  ReferenceLine, Legend, Area, AreaChart,
} from 'recharts'

interface Props {
  snapshot: string
  dte: number | null
}

const REGIME_STATES = [
  'L1 Safe', 'L1 Exposed',
  'L2 Safe', 'L2 Caution-A', 'L2 Caution-B', 'L2 Risky',
  'L3 Safe', 'L3 Exposed',
]

const PERIODS = ['train', 'val', 'oos1', 'oos2'] as const
const PERIOD_LABELS: Record<string, string> = {
  train: 'Train (Feb23–Jun25)',
  val: 'Val (Jul25–Jan26)',
  oos1: 'OOS1 (Jan21–Jan23)',
  oos2: 'OOS2 (Feb–Mar26)',
}

function fmtPct(v: number | null, dp = 1): string {
  if (v == null) return '—'
  return `${v.toFixed(dp)}%`
}
function fmtNum(v: number | null, dp = 2): string {
  if (v == null) return '—'
  return v.toFixed(dp)
}
function fmtSigned(v: number | null, dp = 4): string {
  if (v == null) return '—'
  return v >= 0 ? `+${v.toFixed(dp)}` : v.toFixed(dp)
}

function scoreColor(score: number): string {
  if (score >= 0.9) return '#00e676'
  if (score >= 0.8) return '#64ffda'
  if (score >= 0.6) return '#ffd740'
  return '#ff5252'
}

function sharpeColor(sh: number | null): string {
  if (sh == null) return 'var(--text-dim)'
  if (sh >= 4) return '#00e676'
  if (sh >= 2) return '#64ffda'
  if (sh >= 1) return '#ffd740'
  if (sh >= 0) return '#ff9100'
  return '#ff5252'
}

function icSignalColor(signal: string): string {
  if (signal === 'Strong') return '#00e676'
  if (signal === 'Moderate') return '#ffd740'
  return '#666'
}

// ── Sub Components ──

function CompositeScoreCard({ data }: { data: AlphaInsightsData }) {
  const { baseline } = data
  const bd = baseline.score_breakdown
  const terms = [
    { label: 'Sharpe', value: bd.sharpe_term, max: 0.45, color: '#c084fc' },
    { label: 'Monotonic', value: bd.monotonicity_term ?? 0, max: 0.25, color: baseline.monotonicity >= 0.85 ? '#00e676' : '#ff5252' },
    { label: 'Safe Sep.', value: bd.safe_sep_term, max: 0.20, color: '#00e676' },
    { label: 'Rank Stab.', value: bd.rank_term, max: 0.15, color: '#64b5f6' },
    { label: 'Coverage', value: bd.coverage_term, max: 0.10, color: '#ffd740' },
  ]

  return (
    <div className="rounded-xl border border-[var(--border)] overflow-hidden" style={{ backgroundColor: 'var(--bg-card)' }}>
      <div className="px-6 py-5">
        <div className="text-[10px] font-mono text-[var(--text-dim)] uppercase tracking-widest mb-3">Composite Score — Autoresearch Baseline</div>
        <div className="flex items-end gap-4 mb-4">
          <span className="text-4xl font-black font-mono" style={{ color: scoreColor(baseline.composite_score) }}>
            {baseline.composite_score.toFixed(3)}
          </span>
          <span className="text-xs font-mono text-[var(--text-muted)] pb-1">
            / 1.000 · {baseline.val_days}d val · {baseline.n_states_used} states
          </span>
        </div>

        {/* Score breakdown bars */}
        <div className="space-y-2">
          {terms.map(t => (
            <div key={t.label} className="flex items-center gap-3">
              <span className="text-[10px] font-mono text-[var(--text-muted)] w-16 text-right">{t.label}</span>
              <div className="flex-1 h-4 rounded-full relative overflow-hidden" style={{ backgroundColor: 'rgba(255,255,255,0.04)' }}>
                <div
                  className="h-full rounded-full transition-all"
                  style={{
                    width: `${Math.min(100, (t.value / t.max) * 100)}%`,
                    backgroundColor: t.color,
                    opacity: 0.7,
                  }}
                />
              </div>
              <span className="text-[11px] font-mono font-bold w-12 text-right" style={{ color: t.color }}>
                {t.value.toFixed(3)}
              </span>
            </div>
          ))}
        </div>

        {/* Key metrics row */}
        <div className="mt-4 grid grid-cols-5 gap-3">
          {[
            { label: 'Val Sharpe', value: baseline.val_sharpe.toFixed(2), color: sharpeColor(baseline.val_sharpe) },
            { label: 'Monotonicity', value: `${((baseline.monotonicity ?? 0) * 100).toFixed(0)}%`, color: (baseline.monotonicity ?? 0) >= 0.85 ? '#00e676' : (baseline.monotonicity ?? 0) >= 0.6 ? '#ffd740' : '#ff5252' },
            { label: 'Safe Sep.', value: `${baseline.safe_separation.toFixed(1)}%`, color: baseline.safe_separation > 10 ? '#00e676' : '#ffd740' },
            { label: 'Rank ρ', value: baseline.rank_stability.toFixed(3), color: baseline.rank_stability > 0.7 ? '#64b5f6' : '#ffd740' },
            { label: 'Coverage', value: baseline.state_coverage.toFixed(2), color: baseline.state_coverage >= 1 ? '#ffd740' : '#ff5252' },
          ].map(m => (
            <div key={m.label} className="text-center">
              <div className="text-[10px] font-mono text-[var(--text-dim)] uppercase">{m.label}</div>
              <div className="text-lg font-bold font-mono" style={{ color: m.color }}>{m.value}</div>
            </div>
          ))}
        </div>

        {/* Monotonicity violations detail */}
        {baseline.monotonicity_details && baseline.monotonicity_details.length > 0 && (
          <div className="mt-4 pt-3 border-t border-[var(--border)]">
            <div className="text-[10px] font-mono text-[var(--text-dim)] uppercase tracking-wider mb-2">
              Sharpe Monotonicity Check — {baseline.monotonicity_violations}/{baseline.monotonicity_checked} violations
            </div>
            <div className="grid grid-cols-2 gap-x-4 gap-y-1">
              {baseline.monotonicity_details.map((d, i) => (
                <div key={i} className="flex items-center gap-2 text-[11px] font-mono">
                  <span style={{ color: d.violated ? '#ff5252' : '#00e676' }}>
                    {d.violated ? '✗' : '✓'}
                  </span>
                  <span className="text-[var(--text-muted)]">
                    {d.better} ({d.better_sharpe.toFixed(1)}) {'>'} {d.worse} ({d.worse_sharpe.toFixed(1)})
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function OOSSummaryCards({ data }: { data: AlphaInsightsData }) {
  const { oos_summary } = data
  const periods = [
    { key: 'train', label: 'Train', accent: '#c084fc' },
    { key: 'val', label: 'Validation', accent: '#00e676' },
    { key: 'oos1', label: 'OOS1 (Pre-train)', accent: '#64b5f6' },
    { key: 'oos2', label: 'OOS2 (Recent)', accent: '#ff9100' },
  ]

  return (
    <div className="grid grid-cols-4 gap-3">
      {periods.map(p => {
        const s = oos_summary[p.key]
        if (!s) return null
        return (
          <div key={p.key} className="rounded-lg border border-[var(--border)] p-4" style={{ backgroundColor: 'var(--bg-card)' }}>
            <div className="text-[10px] font-mono uppercase tracking-wider mb-2" style={{ color: p.accent }}>{p.label}</div>
            <div className="text-2xl font-bold font-mono" style={{ color: sharpeColor(s.sharpe) }}>
              {s.sharpe != null ? s.sharpe.toFixed(1) : '—'}
            </div>
            <div className="text-[10px] font-mono text-[var(--text-dim)]">
              Sharpe · {s.days}d · AL {fmtPct(s.al_pct)}
            </div>
          </div>
        )
      })}
    </div>
  )
}

function FeatureImportancePanel({ features }: { features: AlphaFeatureIC[] }) {
  const LABELS: Record<string, string> = {
    iv_lag: 'IV Lag',
    PK_IV_ratio: 'PK/IV Ratio',
    IV_chg_5d: 'IV Δ5d',
    IV_5d: 'IV 5d',
    PK_5d: 'PK 5d',
    IV_10d: 'IV 10d',
    PK_10d: 'PK 10d',
    IV_chg_1d: 'IV Δ1d',
    IV_percentile_60d: 'IV %ile 60d',
    PK_IV_zscore_30d: 'PK/IV Z30d',
    RV_today: 'RV Today',
    VRP_today: 'VRP Today',
  }

  return (
    <div className="rounded-xl border border-[var(--border)] overflow-hidden" style={{ backgroundColor: 'var(--bg-card)' }}>
      <div className="px-6 py-4">
        <div className="text-[10px] font-mono text-[var(--text-dim)] uppercase tracking-widest mb-3">Feature Alpha Ranking (Spearman IC)</div>
        <div className="space-y-1.5">
          {features.map((f, i) => (
            <div key={f.feature} className="flex items-center gap-3">
              <span className="text-[10px] font-mono text-[var(--text-dim)] w-4 text-right">{i + 1}</span>
              <span className="text-[11px] font-mono w-28 truncate" style={{ color: icSignalColor(f.signal) }}>
                {LABELS[f.feature] || f.feature}
              </span>
              <div className="flex-1 h-3 rounded-full relative overflow-hidden" style={{ backgroundColor: 'rgba(255,255,255,0.04)' }}>
                <div
                  className="h-full rounded-full absolute"
                  style={{
                    width: `${Math.min(100, f.abs_ic * 1000)}%`,
                    backgroundColor: f.train_ic < 0 ? '#ff5252' : '#00e676',
                    opacity: 0.6,
                    left: f.train_ic < 0 ? undefined : 0,
                    right: f.train_ic < 0 ? 0 : undefined,
                  }}
                />
              </div>
              <span className="text-[10px] font-mono w-14 text-right" style={{ color: f.train_ic < 0 ? '#ff5252' : '#00e676' }}>
                {fmtSigned(f.train_ic)}
              </span>
              <span className="text-[10px] font-mono w-14 text-right text-[var(--text-dim)]">
                val: {f.val_ic != null ? fmtSigned(f.val_ic) : '—'}
              </span>
              <span
                className="text-[9px] font-mono px-1.5 py-0.5 rounded"
                style={{
                  backgroundColor: f.signal === 'Strong' ? 'rgba(0,230,118,0.12)' : f.signal === 'Moderate' ? 'rgba(255,215,64,0.12)' : 'rgba(255,255,255,0.04)',
                  color: icSignalColor(f.signal),
                }}
              >
                {f.signal}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function StateAlphaTable({ data }: { data: AlphaInsightsData }) {
  const [period, setPeriod] = useState<string>('val')
  const filtered = useMemo(
    () => data.state_alpha.filter(s => s.period === period),
    [data, period]
  )

  return (
    <div className="rounded-xl border border-[var(--border)] overflow-hidden" style={{ backgroundColor: 'var(--bg-card)' }}>
      <div className="px-6 py-4">
        <div className="flex items-center justify-between mb-3">
          <div className="text-[10px] font-mono text-[var(--text-dim)] uppercase tracking-widest">Per-State Alpha Profile</div>
          <div className="flex rounded-lg border border-[var(--border)] overflow-hidden">
            {PERIODS.map(p => (
              <button
                key={p}
                onClick={() => setPeriod(p)}
                className="px-3 py-1 text-[10px] font-mono font-semibold transition-colors"
                style={{
                  backgroundColor: period === p ? 'rgba(192,132,252,0.15)' : 'transparent',
                  color: period === p ? '#c084fc' : 'var(--text-muted)',
                }}
              >
                {p.toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-[11px] font-mono">
            <thead>
              <tr className="border-b border-[var(--border)]">
                <th className="text-left py-2 px-2 text-[var(--text-dim)]">State</th>
                <th className="text-right py-2 px-2 text-[var(--text-dim)]">Days</th>
                <th className="text-right py-2 px-2 text-[var(--text-dim)]">Avg PnL</th>
                <th className="text-right py-2 px-2 text-[var(--text-dim)]">Sharpe</th>
                <th className="text-right py-2 px-2 text-[var(--text-dim)]">Win%</th>
                <th className="text-right py-2 px-2 text-[var(--text-dim)]">AL%</th>
                <th className="text-right py-2 px-2 text-[var(--text-dim)]">DM</th>
                <th className="text-right py-2 px-2 text-[var(--text-dim)]">WC</th>
                <th className="text-right py-2 px-2 text-[var(--text-dim)]">Orion</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map(row => (
                <tr key={row.state} className="border-b border-[var(--border)] border-opacity-30 hover:bg-white/[0.02]">
                  <td className="py-2 px-2">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full" style={{ backgroundColor: row.color }} />
                      <span style={{ color: row.color }}>{row.state}</span>
                    </div>
                  </td>
                  <td className="text-right py-2 px-2 text-[var(--text-muted)]">{row.days}</td>
                  <td className="text-right py-2 px-2" style={{ color: (row.port_avg ?? 0) >= 0 ? '#00e676' : '#ff5252' }}>
                    {fmtSigned(row.port_avg)}
                  </td>
                  <td className="text-right py-2 px-2" style={{ color: sharpeColor(row.sharpe) }}>
                    {fmtNum(row.sharpe)}
                  </td>
                  <td className="text-right py-2 px-2" style={{ color: (row.win_rate ?? 0) >= 55 ? '#00e676' : 'var(--text-muted)' }}>
                    {fmtPct(row.win_rate)}
                  </td>
                  <td className="text-right py-2 px-2" style={{ color: (row.al_pct ?? 0) > 15 ? '#ff5252' : (row.al_pct ?? 0) > 10 ? '#ff9100' : 'var(--text-muted)' }}>
                    {fmtPct(row.al_pct)}
                  </td>
                  <td className="text-right py-2 px-2" style={{ color: sharpeColor(row.strategies?.dm?.sharpe ?? null) }}>
                    {fmtNum(row.strategies?.dm?.sharpe ?? null)}
                  </td>
                  <td className="text-right py-2 px-2" style={{ color: sharpeColor(row.strategies?.wc?.sharpe ?? null) }}>
                    {fmtNum(row.strategies?.wc?.sharpe ?? null)}
                  </td>
                  <td className="text-right py-2 px-2" style={{ color: sharpeColor(row.strategies?.orion?.sharpe ?? null) }}>
                    {fmtNum(row.strategies?.orion?.sharpe ?? null)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

function StrategyWeightsPanel({ data }: { data: AlphaInsightsData }) {
  const weights = data.strategy_weights

  return (
    <div className="rounded-xl border border-[var(--border)] overflow-hidden" style={{ backgroundColor: 'var(--bg-card)' }}>
      <div className="px-6 py-4">
        <div className="text-[10px] font-mono text-[var(--text-dim)] uppercase tracking-widest mb-3">Strategy Allocation Weights (Autoresearch Optimized)</div>
        <div className="space-y-2">
          {REGIME_STATES.map(state => {
            const w = weights[state]
            if (!w) return null
            const total = w.dm + w.wc + w.orion
            return (
              <div key={state} className="flex items-center gap-3">
                <span className="text-[11px] font-mono w-28 truncate text-[var(--text-muted)]">{state}</span>
                <div className="flex-1 flex h-5 rounded overflow-hidden" style={{ backgroundColor: 'rgba(255,255,255,0.04)' }}>
                  {total > 0 && (
                    <>
                      <div className="h-full" style={{ width: `${(w.dm / total) * 100}%`, backgroundColor: 'rgba(255,215,64,0.5)' }} title={`DM: ${w.dm}`} />
                      <div className="h-full" style={{ width: `${(w.wc / total) * 100}%`, backgroundColor: 'rgba(68,138,255,0.5)' }} title={`WC: ${w.wc}`} />
                      <div className="h-full" style={{ width: `${(w.orion / total) * 100}%`, backgroundColor: 'rgba(100,255,218,0.5)' }} title={`Orion: ${w.orion}`} />
                    </>
                  )}
                </div>
                <span className="text-[9px] font-mono w-48 text-[var(--text-dim)] truncate">{w.signal}</span>
              </div>
            )
          })}
        </div>
        <div className="flex gap-4 mt-3">
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-2 rounded-sm" style={{ backgroundColor: 'rgba(255,215,64,0.5)' }} />
            <span className="text-[9px] font-mono text-[var(--text-dim)]">DM</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-2 rounded-sm" style={{ backgroundColor: 'rgba(68,138,255,0.5)' }} />
            <span className="text-[9px] font-mono text-[var(--text-dim)]">WC</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-3 h-2 rounded-sm" style={{ backgroundColor: 'rgba(100,255,218,0.5)' }} />
            <span className="text-[9px] font-mono text-[var(--text-dim)]">Orion</span>
          </div>
        </div>
      </div>
    </div>
  )
}

function RollingSharpeChart({ data }: { data: AlphaInsightsData }) {
  const chartData = useMemo(() => {
    // Downsample if too many points
    const pts = data.rolling_sharpe
    if (pts.length <= 300) return pts
    const step = Math.ceil(pts.length / 300)
    return pts.filter((_, i) => i % step === 0)
  }, [data])

  return (
    <div className="rounded-xl border border-[var(--border)] overflow-hidden" style={{ backgroundColor: 'var(--bg-card)' }}>
      <div className="px-6 py-4">
        <div className="text-[10px] font-mono text-[var(--text-dim)] uppercase tracking-widest mb-3">Rolling 60-Day Sharpe Ratio</div>
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart data={chartData}>
            <defs>
              <linearGradient id="sharpeFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#c084fc" stopOpacity={0.3} />
                <stop offset="100%" stopColor="#c084fc" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis dataKey="date" tick={{ fontSize: 9, fill: '#666' }} interval="preserveStartEnd" />
            <YAxis tick={{ fontSize: 9, fill: '#666' }} domain={['auto', 'auto']} />
            <Tooltip
              contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #333', borderRadius: 8, fontSize: 11, fontFamily: 'monospace' }}
              labelStyle={{ color: '#999' }}
            />
            <ReferenceLine y={0} stroke="#ff5252" strokeDasharray="4 2" strokeOpacity={0.5} />
            <Area type="monotone" dataKey="sharpe" stroke="#c084fc" fill="url(#sharpeFill)" strokeWidth={1.5} dot={false} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function RankStabilityPanel({ data }: { data: AlphaInsightsData }) {
  const chartData = data.rank_detail.map(r => ({
    ...r,
    name: r.state.replace('L1 ', '1').replace('L2 ', '2').replace('L3 ', '3'),
  }))

  return (
    <div className="rounded-xl border border-[var(--border)] overflow-hidden" style={{ backgroundColor: 'var(--bg-card)' }}>
      <div className="px-6 py-4">
        <div className="text-[10px] font-mono text-[var(--text-dim)] uppercase tracking-widest mb-1">
          Rank Stability — Train vs Val Mean PnL (ρ = {data.baseline.rank_stability.toFixed(3)})
        </div>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={chartData} barCategoryGap="15%">
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis dataKey="name" tick={{ fontSize: 9, fill: '#888' }} />
            <YAxis tick={{ fontSize: 9, fill: '#888' }} />
            <Tooltip
              contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #333', borderRadius: 8, fontSize: 11, fontFamily: 'monospace' }}
            />
            <Bar dataKey="train_mean" name="Train" fill="#c084fc" opacity={0.6} radius={[2, 2, 0, 0]} />
            <Bar dataKey="val_mean" name="Val" fill="#64ffda" opacity={0.6} radius={[2, 2, 0, 0]} />
            <Legend wrapperStyle={{ fontSize: 10, fontFamily: 'monospace' }} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function BoundarySensitivityPanel({ data }: { data: AlphaInsightsData }) {
  const { best_boundary, boundary_grid } = data
  if (!boundary_grid.length) return null

  // Build heatmap data
  const l1Values = [...new Set(boundary_grid.map(p => p.l1))].sort((a, b) => a - b)
  const l2Values = [...new Set(boundary_grid.map(p => p.l2))].sort((a, b) => a - b)

  const maxScore = Math.max(...boundary_grid.map(p => p.score))
  const minScore = Math.min(...boundary_grid.map(p => p.score))

  function heatColor(score: number): string {
    const t = (score - minScore) / (maxScore - minScore + 0.001)
    if (t > 0.8) return '#00e676'
    if (t > 0.6) return '#64ffda'
    if (t > 0.4) return '#ffd740'
    if (t > 0.2) return '#ff9100'
    return '#ff5252'
  }

  const gridMap = new Map(boundary_grid.map(p => [`${p.l1}_${p.l2}`, p.score]))

  return (
    <div className="rounded-xl border border-[var(--border)] overflow-hidden" style={{ backgroundColor: 'var(--bg-card)' }}>
      <div className="px-6 py-4">
        <div className="text-[10px] font-mono text-[var(--text-dim)] uppercase tracking-widest mb-1">
          IV Boundary Sensitivity — Composite Score Heatmap
        </div>
        <div className="text-[9px] font-mono text-[var(--text-dim)] mb-3">
          Best: L1={best_boundary.best_l1}, L2={best_boundary.best_l2} → {best_boundary.best_score?.toFixed(3)}
          {' · '}Current: L1={best_boundary.current_l1}, L2={best_boundary.current_l2}
        </div>

        <div className="overflow-x-auto">
          <table className="text-[9px] font-mono">
            <thead>
              <tr>
                <th className="px-1 py-1 text-[var(--text-dim)]">L1\L2</th>
                {l2Values.map(l2 => (
                  <th key={l2} className="px-1 py-1 text-[var(--text-dim)] text-center">{l2}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {l1Values.map(l1 => (
                <tr key={l1}>
                  <td className="px-1 py-1 text-[var(--text-dim)]">{l1}</td>
                  {l2Values.map(l2 => {
                    const key = `${l1}_${l2}`
                    const score = gridMap.get(key)
                    if (score == null || l2 <= l1) {
                      return <td key={l2} className="px-1 py-1 text-center text-[var(--text-dim)]">—</td>
                    }
                    const isBest = l1 === best_boundary.best_l1 && l2 === best_boundary.best_l2
                    const isCurrent = l1 === best_boundary.current_l1 && l2 === best_boundary.current_l2
                    return (
                      <td
                        key={l2}
                        className="px-1 py-1 text-center"
                        style={{
                          color: heatColor(score),
                          fontWeight: isBest || isCurrent ? 700 : 400,
                          textDecoration: isCurrent ? 'underline' : undefined,
                        }}
                        title={`L1=${l1}, L2=${l2}: ${score.toFixed(4)}`}
                      >
                        {score.toFixed(2)}{isBest ? '★' : ''}{isCurrent ? '●' : ''}
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
  )
}

function WarningsPanel({ warnings }: { warnings: AlphaDistributionWarning[] }) {
  if (!warnings.length) return null

  return (
    <div className="rounded-xl border border-[var(--border)] overflow-hidden" style={{ backgroundColor: 'rgba(255,82,82,0.05)', borderColor: 'rgba(255,82,82,0.2)' }}>
      <div className="px-6 py-4">
        <div className="text-[10px] font-mono uppercase tracking-widest mb-2" style={{ color: '#ff5252' }}>
          ⚠ Distribution Warnings — Sparse States
        </div>
        <div className="space-y-1">
          {warnings.map((w, i) => (
            <div key={i} className="flex items-center gap-2">
              <span
                className="text-[9px] font-mono px-1.5 py-0.5 rounded"
                style={{
                  backgroundColor: w.severity === 'critical' ? 'rgba(255,82,82,0.15)' : 'rgba(255,145,0,0.15)',
                  color: w.severity === 'critical' ? '#ff5252' : '#ff9100',
                }}
              >
                {w.severity.toUpperCase()}
              </span>
              <span className="text-[11px] font-mono text-[var(--text-muted)]">{w.message}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function HighCorrPairsPanel({ data }: { data: AlphaInsightsData }) {
  const pairs = data.high_corr_pairs
  if (!pairs.length) return null

  return (
    <div className="rounded-xl border border-[var(--border)] overflow-hidden" style={{ backgroundColor: 'var(--bg-card)' }}>
      <div className="px-6 py-4">
        <div className="text-[10px] font-mono text-[var(--text-dim)] uppercase tracking-widest mb-3">Feature Correlation Pairs (|ρ| &gt; 0.6)</div>
        <div className="space-y-1">
          {pairs.map((p, i) => (
            <div key={i} className="flex items-center gap-2">
              <span className="text-[11px] font-mono text-[var(--text-muted)] w-28 truncate">{p.feature_1}</span>
              <span className="text-[9px] text-[var(--text-dim)]">↔</span>
              <span className="text-[11px] font-mono text-[var(--text-muted)] w-28 truncate">{p.feature_2}</span>
              <span className="text-[10px] font-mono" style={{ color: p.redundant ? '#ff5252' : '#ffd740' }}>
                ρ={p.rho.toFixed(3)}
              </span>
              {p.redundant && (
                <span className="text-[8px] font-mono px-1.5 py-0.5 rounded" style={{ backgroundColor: 'rgba(255,82,82,0.12)', color: '#ff5252' }}>
                  REDUNDANT
                </span>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function StrategyCorrelationPanel({ data }: { data: AlphaInsightsData }) {
  const corrs = data.strategy_correlations
  if (!corrs.length) return null

  return (
    <div className="rounded-xl border border-[var(--border)] overflow-hidden" style={{ backgroundColor: 'var(--bg-card)' }}>
      <div className="px-6 py-4">
        <div className="text-[10px] font-mono text-[var(--text-dim)] uppercase tracking-widest mb-3">Strategy Cross-Correlation by State</div>
        <table className="w-full text-[10px] font-mono">
          <thead>
            <tr className="border-b border-[var(--border)]">
              <th className="text-left py-1.5 px-2 text-[var(--text-dim)]">State</th>
              <th className="text-right py-1.5 px-2 text-[var(--text-dim)]">DM↔WC</th>
              <th className="text-right py-1.5 px-2 text-[var(--text-dim)]">DM↔Orion</th>
              <th className="text-right py-1.5 px-2 text-[var(--text-dim)]">WC↔Orion</th>
            </tr>
          </thead>
          <tbody>
            {corrs.map(c => {
              function corrColor(v: number | null) {
                if (v == null) return 'var(--text-dim)'
                if (v < -0.1) return '#00e676'  // negative = diversified
                if (v < 0.2) return '#ffd740'
                return '#ff5252'  // high positive = concentrated risk
              }
              return (
                <tr key={c.state} className="border-b border-[var(--border)] border-opacity-30">
                  <td className="py-1.5 px-2 text-[var(--text-muted)]">{c.state}</td>
                  <td className="text-right py-1.5 px-2" style={{ color: corrColor(c.dm_wc) }}>{c.dm_wc?.toFixed(2) ?? '—'}</td>
                  <td className="text-right py-1.5 px-2" style={{ color: corrColor(c.dm_orion) }}>{c.dm_orion?.toFixed(2) ?? '—'}</td>
                  <td className="text-right py-1.5 px-2" style={{ color: corrColor(c.wc_orion) }}>{c.wc_orion?.toFixed(2) ?? '—'}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
        <div className="text-[9px] font-mono text-[var(--text-dim)] mt-2">
          <span style={{ color: '#00e676' }}>Green</span> = diversified (negative corr),{' '}
          <span style={{ color: '#ff5252' }}>Red</span> = concentrated risk (high positive corr)
        </div>
      </div>
    </div>
  )
}

// ── Main Component ──

export function AlphaInsights({ snapshot, dte }: Props) {
  const [data, setData] = useState<AlphaInsightsData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    fetchAlphaInsights(snapshot, dte)
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [snapshot, dte])

  if (loading) {
    return (
      <div className="text-center py-20">
        <div className="text-sm font-mono text-[var(--text-dim)]">Loading alpha insights...</div>
        <div className="text-xs font-mono text-[var(--text-dim)] mt-1">Running autoresearch diagnostics</div>
      </div>
    )
  }
  if (error) return <div className="text-center py-20 text-[var(--signal-negative)] text-sm font-mono">Error: {error}</div>
  if (!data) return <div className="text-center py-20 text-[var(--text-dim)] text-sm font-mono">No data</div>

  return (
    <div className="space-y-5">
      {/* Row 1: Composite Score + OOS Summary */}
      <div className="grid grid-cols-[1fr_1fr] gap-5">
        <CompositeScoreCard data={data} />
        <div className="space-y-3">
          <OOSSummaryCards data={data} />
          <WarningsPanel warnings={data.distribution_warnings} />
        </div>
      </div>

      {/* Row 2: State Alpha Table (full width) */}
      <StateAlphaTable data={data} />

      {/* Row 3: Feature IC + Strategy Weights */}
      <div className="grid grid-cols-2 gap-5">
        <FeatureImportancePanel features={data.feature_importance} />
        <StrategyWeightsPanel data={data} />
      </div>

      {/* Row 4: Rolling Sharpe + Rank Stability */}
      <div className="grid grid-cols-2 gap-5">
        <RollingSharpeChart data={data} />
        <RankStabilityPanel data={data} />
      </div>

      {/* Row 5: Boundary Sensitivity */}
      <BoundarySensitivityPanel data={data} />

      {/* Row 6: Correlations */}
      <div className="grid grid-cols-2 gap-5">
        <HighCorrPairsPanel data={data} />
        <StrategyCorrelationPanel data={data} />
      </div>
    </div>
  )
}
