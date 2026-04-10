'use client'

import { useState, useEffect, useMemo } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell, ReferenceLine,
} from 'recharts'
import { fetchFeatureBuckets } from '@/lib/api'
import type { FeatureBucketData, BucketMetrics, DTECross } from '@/lib/types'
import type { DisplayMode } from '@/lib/formatters'
import { formatPct, formatSharpe, annualize } from '@/lib/formatters'
import { STRATEGY_META, FEATURE_LABELS } from '@/lib/config'
import { ModeToggle } from './ModeToggle'
import { useSortableData, SortableTh } from './SortableHeader'

type BucketView = 'raw' | 'percentile'
type DTEView = 'overall' | 'by_dte'

function getCellVal(cell: BucketMetrics, mode: DisplayMode): number {
  if (mode === 'sharpe') return cell.sharpe_pct ?? 0
  if (mode === 'annualized') return annualize(cell.avg_daily_pct)
  return cell.avg_daily_pct
}

function BucketBarChart({ buckets, mode, featureLabel }: { buckets: BucketMetrics[]; mode: DisplayMode; featureLabel: string }) {
  const featureShort = getFeatureShort(featureLabel)
  const data = buckets.map((b, i) => ({
    label: `${getClusterLabel(i, buckets.length, featureShort)}\n${b.label}`,
    value: getCellVal(b, mode),
    win_rate: b.win_rate,
    days: b.trading_days,
    sharpe: b.sharpe_pct,
  }))

  return (
    <ResponsiveContainer width="100%" height={320}>
      <BarChart data={data} barCategoryGap="18%">
        <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
        <XAxis dataKey="label" tick={{ fontSize: 9, fill: '#e0e0e0' }} interval={0} angle={-20} textAnchor="end" height={90} />
        <YAxis tick={{ fontSize: 10, fill: '#e0e0e0' }} tickFormatter={(v: number) => mode === 'sharpe' ? v.toFixed(1) : `${v.toFixed(2)}%`} />
        <Tooltip
          contentStyle={{ backgroundColor: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 11, color: '#ffffff' }}
          labelStyle={{ color: '#ffffff' }}
          itemStyle={{ color: '#ffffff' }}
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          formatter={(v: any, _: any, entry: any) => {
            const p = entry?.payload
            return [`${mode === 'sharpe' ? formatSharpe(v) : formatPct(v)} | WR: ${((p?.win_rate as number ?? 0) * 100).toFixed(0)}% | ${p?.days}d | Sharpe: ${p?.sharpe ?? '—'}`, mode === 'sharpe' ? 'Sharpe' : mode === 'annualized' ? 'Ann. Return' : 'Avg Daily %']
          }}
        />
        <ReferenceLine y={0} stroke="var(--text-dim)" strokeDasharray="3 3" />
        <Bar dataKey="value" radius={[4, 4, 0, 0]}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.value >= 0 ? 'var(--signal-positive)' : 'var(--signal-negative)'} fillOpacity={0.7} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

function fmtStreak(v: number | null): string {
  if (v == null) return '—'
  return Number.isInteger(v) ? String(v) : v.toFixed(1)
}

function getClusterLabel(index: number, total: number, featureShort: string): string {
  if (total <= 1) return featureShort
  if (total === 2) return index === 0 ? `Low ${featureShort}` : `High ${featureShort}`
  if (total === 3) {
    if (index === 0) return `Low ${featureShort}`
    if (index === 1) return `Medium ${featureShort}`
    return `High ${featureShort}`
  }
  // For >3, first = Low, last = High, rest = Medium
  if (index === 0) return `Low ${featureShort}`
  if (index === total - 1) return `High ${featureShort}`
  return `Medium ${featureShort}`
}

function getFeatureShort(featureLabel: string): string {
  // "RV Today (Yang-Zhang)" → "RV", "IV 7d Forward" → "IV", "VRP (IV−RV)" → "VRP"
  return featureLabel.split(/[\s(]/)[0]
}

function BucketTable({ buckets, featureLabel }: { buckets: BucketMetrics[]; featureLabel: string }) {
  const enriched = useMemo(() => buckets.map((b, i) => ({
    ...b,
    _ann: annualize(b.avg_daily_pct),
    _clusterLabel: getClusterLabel(i, buckets.length, getFeatureShort(featureLabel)),
    _origIdx: i,
  })), [buckets, featureLabel])

  const { sorted, sort, toggle } = useSortableData(enriched)

  const thR = "text-right py-2 px-2"
  const thL = "text-left py-2 pr-3"

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-[11px] font-mono">
        <thead>
          <tr className="text-[var(--text-dim)] border-b border-[var(--border)]">
            <SortableTh column="label" label="Bucket" sort={sort} toggle={toggle} className={thL} />
            <SortableTh column="trading_days" label="Days" sort={sort} toggle={toggle} className={thR} />
            <SortableTh column="streak_mean" label="Mean Days" sort={sort} toggle={toggle} className={thR} />
            <SortableTh column="streak_median" label="Med Days" sort={sort} toggle={toggle} className={thR} />
            <SortableTh column="streak_min" label="Min Days" sort={sort} toggle={toggle} className={thR} />
            <SortableTh column="streak_max" label="Max Days" sort={sort} toggle={toggle} className={thR} />
            <SortableTh column="_ann" label="Ann. Return" sort={sort} toggle={toggle} className={thR} />
            <SortableTh column="ann_vol_pct" label="Ann. Vol" sort={sort} toggle={toggle} className={thR} />
            <SortableTh column="win_rate" label="Win %" sort={sort} toggle={toggle} className={thR} />
            <SortableTh column="loss_rate" label="Loss %" sort={sort} toggle={toggle} className={thR} />
            <SortableTh column="sharpe_pct" label="Sharpe" sort={sort} toggle={toggle} className={thR} />
            <SortableTh column="total_pct" label="Total %" sort={sort} toggle={toggle} className={thR} />
            <SortableTh column="avg_daily_pct" label="Avg Daily %" sort={sort} toggle={toggle} className={thR} />
            <SortableTh column="max_win_pct" label="Max Win" sort={sort} toggle={toggle} className={thR} />
            <SortableTh column="max_loss_pct" label="Max Loss" sort={sort} toggle={toggle} className="text-right py-2 pl-2" />
          </tr>
        </thead>
        <tbody>
          {sorted.map((b) => (
            <tr key={b._origIdx} className="border-b border-[var(--border-subtle)] hover:bg-[var(--bg-hover)] transition-colors" style={{ opacity: b.trading_days < 5 ? 0.4 : 1 }}>
              <td className="py-1.5 pr-3 text-[var(--text-primary)] font-semibold whitespace-nowrap">
                <div className="flex flex-col">
                  <span className="text-[10px] text-[var(--text-primary)] opacity-70">{b._clusterLabel}</span>
                  <span>{b.label}</span>
                </div>
              </td>
              <td className="py-1.5 px-2 text-right text-[var(--text-muted)]">{b.trading_days}</td>
              <td className="py-1.5 px-2 text-right text-[var(--text-secondary)]">{fmtStreak(b.streak_mean)}</td>
              <td className="py-1.5 px-2 text-right text-[var(--text-secondary)]">{fmtStreak(b.streak_median)}</td>
              <td className="py-1.5 px-2 text-right text-[var(--text-secondary)]">{fmtStreak(b.streak_min)}</td>
              <td className="py-1.5 px-2 text-right text-[var(--text-secondary)]">{fmtStreak(b.streak_max)}</td>
              <td className="py-1.5 px-2 text-right" style={{ color: b._ann >= 0 ? 'var(--signal-positive)' : 'var(--signal-negative)' }}>{formatPct(b._ann)}</td>
              <td className="py-1.5 px-2 text-right text-[var(--text-secondary)]">{b.ann_vol_pct != null ? formatPct(b.ann_vol_pct) : '—'}</td>
              <td className="py-1.5 px-2 text-right text-[var(--text-secondary)]">{(b.win_rate * 100).toFixed(1)}%</td>
              <td className="py-1.5 px-2 text-right text-[var(--text-secondary)]">{(b.loss_rate * 100).toFixed(1)}%</td>
              <td className="py-1.5 px-2 text-right text-[var(--text-secondary)]">{formatSharpe(b.sharpe_pct)}</td>
              <td className="py-1.5 px-2 text-right" style={{ color: b.total_pct >= 0 ? 'var(--signal-positive)' : 'var(--signal-negative)' }}>{formatPct(b.total_pct)}</td>
              <td className="py-1.5 px-2 text-right text-[var(--text-secondary)]">{formatPct(b.avg_daily_pct)}</td>
              <td className="py-1.5 px-2 text-right" style={{ color: 'var(--signal-positive)' }}>{formatPct(b.max_win_pct)}</td>
              <td className="py-1.5 pl-2 text-right" style={{ color: 'var(--signal-negative)' }}>{formatPct(b.max_loss_pct)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ── DTE Cross-tab Heatmap ──

function DTEHeatmap({ cross, mode, featureLabel }: { cross: DTECross; mode: DisplayMode; featureLabel: string }) {
  const extent = useMemo(() => {
    let maxAbs = 0.01
    for (const row of cross.grid) {
      for (const cell of row) {
        if (cell && cell.trading_days > 0) maxAbs = Math.max(maxAbs, Math.abs(getCellVal(cell, mode)))
      }
    }
    return maxAbs
  }, [cross.grid, mode])

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-[11px] font-mono">
        <thead>
          <tr>
            <th className="text-left text-[var(--text-dim)] pb-3 pr-3 text-[10px]">{featureLabel} ↓ / DTE →</th>
            {cross.dte_labels.map(d => (
              <th key={d} className="text-center text-[var(--text-dim)] pb-3 px-1 min-w-[110px] text-[10px]">DTE {d}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {cross.feature_labels.map((fLabel, fi) => {
            const clusterTag = getClusterLabel(fi, cross.feature_labels.length, getFeatureShort(featureLabel))
            return (
            <tr key={fLabel}>
              <td className="text-[var(--text-secondary)] pr-3 py-1 font-semibold whitespace-nowrap">
                <div className="flex flex-col">
                  <span className="text-[9px] text-[var(--text-primary)] opacity-70">{clusterTag}</span>
                  <span>{fLabel}</span>
                </div>
              </td>
              {cross.grid[fi]?.map((cell, di) => {
                if (!cell || cell.trading_days === 0) {
                  return <td key={di} className="px-1 py-1"><div className="w-full h-16 rounded-lg bg-[var(--bg-hover)] flex items-center justify-center text-[var(--text-dim)] text-[10px]">—</div></td>
                }
                const val = getCellVal(cell, mode)
                const intensity = Math.min(Math.abs(val) / extent, 1)
                const bg = val >= 0
                  ? `rgba(0,230,118,${0.06 + intensity * 0.45})`
                  : `rgba(255,82,82,${0.06 + intensity * 0.45})`
                const dimmed = cell.trading_days < 5
                return (
                  <td key={di} className="px-1 py-1">
                    <div
                      className="w-full h-16 rounded-lg flex flex-col items-center justify-center cursor-default transition-all hover:scale-[1.03] hover:shadow-lg"
                      style={{ backgroundColor: bg, color: 'var(--text-primary)', opacity: dimmed ? 0.35 : 1 }}
                      title={`${clusterTag} (${fLabel}) × DTE ${cross.dte_labels[di]}\nWR: ${(cell.win_rate * 100).toFixed(0)}% | Vol: ${cell.ann_vol_pct != null ? cell.ann_vol_pct.toFixed(1) + '%' : '—'} | Days: ${cell.trading_days}`}
                    >
                      <span className="text-[12px] font-bold leading-tight">
                        {mode === 'sharpe' ? formatSharpe(val) : mode === 'annualized' ? `${val.toFixed(1)}%` : formatPct(val)}
                      </span>
                      <span className="text-[9px] text-white/80 mt-0.5">WR {(cell.win_rate * 100).toFixed(0)}%</span>
                      <span className="text-[8px] text-white/60">{cell.trading_days}d</span>
                    </div>
                  </td>
                )
              })}
            </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function DTEDetailTable({ cross, mode }: { cross: DTECross; mode: DisplayMode }) {
  const rows = useMemo(() => {
    const result: { bucket: string; dte: string; trading_days: number; _ann: number; ann_vol_pct: number | null; win_rate: number; loss_rate: number; sharpe_pct: number | null; avg_daily_pct: number; _origIdx: number; _isFirstDte: boolean }[] = []
    let idx = 0
    cross.feature_labels.forEach((fLabel, fi) => {
      cross.dte_labels.forEach((dLabel, di) => {
        const cell = cross.grid[fi]?.[di]
        if (cell && cell.trading_days > 0) {
          result.push({
            bucket: fLabel, dte: dLabel, trading_days: cell.trading_days,
            _ann: annualize(cell.avg_daily_pct), ann_vol_pct: cell.ann_vol_pct,
            win_rate: cell.win_rate, loss_rate: cell.loss_rate,
            sharpe_pct: cell.sharpe_pct, avg_daily_pct: cell.avg_daily_pct,
            _origIdx: idx++, _isFirstDte: di === 0,
          })
        }
      })
    })
    return result
  }, [cross])

  const { sorted, sort, toggle } = useSortableData(rows)
  const thR = "text-right py-2 px-2"

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-[10px] font-mono">
        <thead>
          <tr className="text-[var(--text-dim)] border-b border-[var(--border)]">
            <SortableTh column="bucket" label="Bucket" sort={sort} toggle={toggle} className="text-left py-2 pr-2" />
            <SortableTh column="dte" label="DTE" sort={sort} toggle={toggle} className="text-left py-2 px-2" />
            <SortableTh column="trading_days" label="Days" sort={sort} toggle={toggle} className={thR} />
            <SortableTh column="_ann" label="Ann. Return" sort={sort} toggle={toggle} className={thR} />
            <SortableTh column="ann_vol_pct" label="Ann. Vol" sort={sort} toggle={toggle} className={thR} />
            <SortableTh column="win_rate" label="Win %" sort={sort} toggle={toggle} className={thR} />
            <SortableTh column="loss_rate" label="Loss %" sort={sort} toggle={toggle} className={thR} />
            <SortableTh column="sharpe_pct" label="Sharpe" sort={sort} toggle={toggle} className={thR} />
            <SortableTh column="avg_daily_pct" label="Avg Daily %" sort={sort} toggle={toggle} className={thR} />
          </tr>
        </thead>
        <tbody>
          {sorted.map((r) => (
            <tr key={r._origIdx} className="border-b border-[var(--border-subtle)] hover:bg-[var(--bg-hover)] transition-colors" style={{ opacity: r.trading_days < 5 ? 0.4 : 1 }}>
              <td className="py-1.5 pr-2 text-[var(--text-primary)] font-semibold">{sort.column ? r.bucket : (r._isFirstDte ? r.bucket : '')}</td>
              <td className="py-1.5 px-2 text-[var(--text-secondary)]">DTE {r.dte}</td>
              <td className="py-1.5 px-2 text-right text-[var(--text-muted)]">{r.trading_days}</td>
              <td className="py-1.5 px-2 text-right" style={{ color: r._ann >= 0 ? 'var(--signal-positive)' : 'var(--signal-negative)' }}>{formatPct(r._ann)}</td>
              <td className="py-1.5 px-2 text-right text-[var(--text-secondary)]">{r.ann_vol_pct != null ? formatPct(r.ann_vol_pct) : '—'}</td>
              <td className="py-1.5 px-2 text-right text-[var(--text-secondary)]">{(r.win_rate * 100).toFixed(1)}%</td>
              <td className="py-1.5 px-2 text-right text-[var(--text-secondary)]">{(r.loss_rate * 100).toFixed(1)}%</td>
              <td className="py-1.5 px-2 text-right text-[var(--text-secondary)]">{formatSharpe(r.sharpe_pct)}</td>
              <td className="py-1.5 px-2 text-right text-[var(--text-secondary)]">{formatPct(r.avg_daily_pct)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ── Main Component ──

export function FeatureAnalysis({ strategy, startDate, endDate, snapshot }: { strategy: string; startDate: string; endDate: string; snapshot: string }) {
  const meta = STRATEGY_META[strategy] || STRATEGY_META.dm
  const features = Object.keys(FEATURE_LABELS)
  const [feature, setFeature] = useState(features[0])
  const [bucketView, setBucketView] = useState<BucketView>('raw')
  const [dteView, setDteView] = useState<DTEView>('overall')
  const [mode, setMode] = useState<DisplayMode>('sharpe')
  const [data, setData] = useState<FeatureBucketData | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    fetchFeatureBuckets(strategy, feature, startDate, endDate, snapshot).then(d => { setData(d); setLoading(false) })
  }, [strategy, feature, startDate, endDate, snapshot])

  const buckets = useMemo(() => {
    if (!data) return []
    return bucketView === 'raw' ? data.raw_buckets : data.percentile_buckets
  }, [data, bucketView])

  const featureLabel = FEATURE_LABELS[feature] || feature

  return (
    <div>
      <div className="mb-2">
        <p className="text-[11px] font-mono text-[var(--text-dim)] mb-1 tracking-wider uppercase">{meta.name} / feature analysis</p>
        <h1 className="text-2xl font-bold text-[var(--text-primary)] tracking-tight">Individual Feature Buckets</h1>
        <p className="text-sm text-[var(--text-secondary)] mt-1">Strategy performance bucketed by RV feature values</p>
      </div>

      {/* Controls */}
      <div className="sticky top-0 z-30 py-3 mb-5 backdrop-blur-md" style={{ backgroundColor: 'rgba(10, 10, 16, 0.92)' }}>
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-mono text-[var(--text-dim)] uppercase tracking-wider">Feature</span>
            <select
              value={feature}
              onChange={e => setFeature(e.target.value)}
              className="text-[11px] font-mono px-3 py-1.5 rounded-lg border border-[var(--border)] bg-[var(--bg-card)] text-[var(--text-primary)] outline-none focus:border-[var(--text-muted)]"
            >
              {features.map(f => <option key={f} value={f}>{FEATURE_LABELS[f]}</option>)}
            </select>
          </div>

          <div className="h-5 w-px bg-[var(--border)]" />

          {/* Bucket view toggle */}
          <div className="inline-flex rounded-lg border border-[var(--border)] overflow-hidden text-[11px] font-mono">
            {(['raw', 'percentile'] as BucketView[]).map((v, i) => (
              <button key={v} onClick={() => setBucketView(v)} className="px-3 py-1.5 transition-colors capitalize"
                style={{ backgroundColor: bucketView === v ? 'var(--bg-elevated)' : 'transparent', color: bucketView === v ? 'var(--text-primary)' : 'var(--text-muted)', borderLeft: i > 0 ? '1px solid var(--border)' : 'none' }}>
                {v}
              </button>
            ))}
          </div>

          <div className="h-5 w-px bg-[var(--border)]" />

          {/* DTE view toggle */}
          <div className="inline-flex rounded-lg border border-[var(--border)] overflow-hidden text-[11px] font-mono">
            {([['overall', 'Overall'], ['by_dte', 'By DTE']] as [DTEView, string][]).map(([v, label], i) => (
              <button key={v} onClick={() => setDteView(v)} className="px-3 py-1.5 transition-colors"
                style={{ backgroundColor: dteView === v ? 'var(--bg-elevated)' : 'transparent', color: dteView === v ? 'var(--text-primary)' : 'var(--text-muted)', borderLeft: i > 0 ? '1px solid var(--border)' : 'none' }}>
                {label}
              </button>
            ))}
          </div>

          <ModeToggle mode={mode} setMode={setMode} showSharpe />
        </div>
      </div>

      {loading && <div className="flex items-center justify-center h-64"><div className="text-[var(--text-muted)]">Loading...</div></div>}

      {/* Overall view */}
      {!loading && dteView === 'overall' && buckets.length > 0 && (
        <>
          <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5 mb-5">
            <h2 className="text-sm font-semibold text-[var(--text-primary)] mb-1 uppercase tracking-wider">
              {featureLabel} — {mode === 'sharpe' ? 'Sharpe' : mode === 'annualized' ? 'Ann. Return' : 'Avg Daily %'}
            </h2>
            <p className="text-[11px] text-[var(--text-muted)] mb-4">
              {bucketView === 'raw' ? 'Raw value' : 'Percentile'} buckets &middot; {buckets.reduce((a, b) => a + b.trading_days, 0)} total days
            </p>
            <BucketBarChart buckets={buckets} mode={mode} featureLabel={featureLabel} />
          </div>

          <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5">
            <h2 className="text-sm font-semibold text-[var(--text-primary)] mb-4 uppercase tracking-wider">Bucket Breakdown</h2>
            <BucketTable buckets={buckets} featureLabel={featureLabel} />
          </div>
        </>
      )}

      {/* By DTE view */}
      {!loading && dteView === 'by_dte' && data?.dte_cross && (
        <>
          <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5 mb-5">
            <h2 className="text-sm font-semibold text-[var(--text-primary)] mb-1 uppercase tracking-wider">
              {featureLabel} x DTE — {mode === 'sharpe' ? 'Sharpe' : mode === 'annualized' ? 'Ann. Return' : 'Avg Daily %'}
            </h2>
            <p className="text-[11px] text-[var(--text-muted)] mb-4">
              Feature buckets vs Days to Expiry. Faded cells &lt;5 trading days.
            </p>
            <DTEHeatmap cross={data.dte_cross} mode={mode} featureLabel={featureLabel} />
          </div>

          <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5">
            <h2 className="text-sm font-semibold text-[var(--text-primary)] mb-4 uppercase tracking-wider">DTE Breakdown</h2>
            <DTEDetailTable cross={data.dte_cross} mode={mode} />
          </div>
        </>
      )}
    </div>
  )
}
