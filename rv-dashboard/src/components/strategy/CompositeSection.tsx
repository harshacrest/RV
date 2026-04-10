'use client'

import { useState, useEffect, useMemo } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts'
import { fetchComposite } from '@/lib/api'
import type { CompositeData, BucketMetrics, CompositeDTECross } from '@/lib/types'
import type { DisplayMode } from '@/lib/formatters'
import { formatPct, formatSharpe, annualize } from '@/lib/formatters'
import { STRATEGY_META, FEATURE_LABELS, FEATURES } from '@/lib/config'
import { ModeToggle } from './ModeToggle'
import { useSortableData, SortableTh } from './SortableHeader'

type BucketView = 'raw' | 'percentile'
type DTEView = 'overall' | 'by_dte'

function getCellVal(cell: BucketMetrics, mode: DisplayMode): number {
  if (mode === 'sharpe') return cell.sharpe_pct ?? 0
  if (mode === 'annualized') return annualize(cell.avg_daily_pct)
  return cell.avg_daily_pct
}

function formatCellValue(v: number | null, mode: DisplayMode): string {
  if (v == null) return '—'
  if (mode === 'sharpe') return formatSharpe(v)
  return formatPct(v)
}

function getFeatureShort(featureLabel: string | undefined): string {
  if (!featureLabel) return ''
  return featureLabel.split(/[\s(]/)[0]
}

function getClusterLabel(index: number, total: number, featureShort: string): string {
  if (total <= 1) return featureShort
  if (total === 2) return index === 0 ? `Low ${featureShort}` : `High ${featureShort}`
  if (total === 3) {
    if (index === 0) return `Low ${featureShort}`
    if (index === 1) return `Medium ${featureShort}`
    return `High ${featureShort}`
  }
  if (index === 0) return `Low ${featureShort}`
  if (index === total - 1) return `High ${featureShort}`
  return `Medium ${featureShort}`
}

// ── Heatmap ──
function CrossHeatmap({ data, mode, bucketView, rowLabel, colLabel }: {
  data: CompositeData; mode: DisplayMode; bucketView: BucketView; rowLabel: string; colLabel: string
}) {
  const useP = bucketView === 'percentile'
  const fLabels = useP ? data.pct_feature_labels : data.feature_labels
  const grid = useP ? data.pct_grid : data.grid
  const sLabels = data.static_labels
  const rowShort = getFeatureShort(rowLabel)
  const colShort = getFeatureShort(colLabel)

  const extent = useMemo(() => {
    let maxAbs = 0.01
    for (const row of grid) {
      for (const cell of row) {
        if (cell && cell.trading_days > 0) maxAbs = Math.max(maxAbs, Math.abs(getCellVal(cell, mode)))
      }
    }
    return maxAbs
  }, [grid, mode])

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-[11px] font-mono">
        <thead>
          <tr>
            <th className="text-left text-[var(--text-dim)] pb-3 pr-3 text-[10px]">{useP ? 'Percentile' : rowLabel} ↓ / {colLabel} →</th>
            {sLabels.map((v, i) => (
              <th key={v} className="text-center text-[var(--text-dim)] pb-3 px-1 min-w-[120px] text-[10px]">
                <div className="flex flex-col items-center">
                  <span className="text-[9px] text-[var(--text-primary)] opacity-70">{getClusterLabel(i, sLabels.length, colShort)}</span>
                  <span>{v}</span>
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {fLabels.map((fLabel, fi) => {
            const rowTag = getClusterLabel(fi, fLabels.length, rowShort)
            return (
              <tr key={fLabel}>
                <td className="text-[var(--text-secondary)] pr-3 py-1 font-semibold whitespace-nowrap">
                  <div className="flex flex-col">
                    <span className="text-[9px] text-[var(--text-primary)] opacity-70">{rowTag}</span>
                    <span>{fLabel}</span>
                  </div>
                </td>
                {grid[fi]?.map((cell, vi) => {
                  if (!cell || cell.trading_days === 0) {
                    return <td key={vi} className="px-1 py-1"><div className="w-full h-16 rounded-lg bg-[var(--bg-hover)] flex items-center justify-center text-[var(--text-dim)] text-[10px]">—</div></td>
                  }
                  const val = getCellVal(cell, mode)
                  const intensity = Math.min(Math.abs(val) / extent, 1)
                  const bg = val >= 0 ? `rgba(0,230,118,${0.06 + intensity * 0.45})` : `rgba(255,82,82,${0.06 + intensity * 0.45})`
                  const dimmed = cell.trading_days < 5
                  const colTag = getClusterLabel(vi, sLabels.length, colShort)
                  return (
                    <td key={vi} className="px-1 py-1">
                      <div className="w-full h-16 rounded-lg flex flex-col items-center justify-center cursor-default transition-all hover:scale-[1.03] hover:shadow-lg" style={{ backgroundColor: bg, color: 'var(--text-primary)', opacity: dimmed ? 0.35 : 1 }} title={`${rowTag} (${fLabel}) × ${colTag} (${sLabels[vi]})\nWR: ${(cell.win_rate * 100).toFixed(0)}% | Days: ${cell.trading_days}`}>
                        <span className="text-[12px] font-bold leading-tight">{mode === 'sharpe' ? formatSharpe(val) : mode === 'annualized' ? `${val.toFixed(1)}%` : formatPct(val)}</span>
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

// ── Grouped Bar Chart ──
function CrossBarChart({ data, mode, bucketView, rowLabel, colLabel }: {
  data: CompositeData; mode: DisplayMode; bucketView: BucketView; rowLabel: string; colLabel: string
}) {
  const useP = bucketView === 'percentile'
  const fLabels = useP ? data.pct_feature_labels : data.feature_labels
  const grid = useP ? data.pct_grid : data.grid
  const sLabels = data.static_labels
  const rowShort = getFeatureShort(rowLabel)
  const colShort = getFeatureShort(colLabel)

  const chartData = fLabels.map((fLabel, fi) => {
    const rowTag = getClusterLabel(fi, fLabels.length, rowShort)
    const row: Record<string, unknown> = { label: `${rowTag}\n${fLabel}` }
    sLabels.forEach((sLabel, si) => {
      const cell = grid[fi]?.[si]
      row[sLabel] = cell && cell.trading_days > 0 ? getCellVal(cell, mode) : 0
      row[`${sLabel}_days`] = cell?.trading_days ?? 0
      row[`${sLabel}_wr`] = cell ? (cell.win_rate * 100).toFixed(0) : '0'
    })
    return row
  })

  const colors = ['#42a5f5', '#ffc107', '#ef5350', '#ab47bc', '#66bb6a']

  return (
    <ResponsiveContainer width="100%" height={320}>
      <BarChart data={chartData} barCategoryGap="18%">
        <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
        <XAxis dataKey="label" tick={{ fontSize: 9, fill: '#e0e0e0' }} interval={0} angle={-20} textAnchor="end" height={80} />
        <YAxis tick={{ fontSize: 10, fill: '#e0e0e0' }} tickFormatter={(v: number) => mode === 'sharpe' ? v.toFixed(1) : `${v.toFixed(2)}%`} />
        <Tooltip
          contentStyle={{ backgroundColor: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 11, color: '#ffffff' }}
          labelStyle={{ color: '#ffffff' }}
          itemStyle={{ color: '#ffffff' }}
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          formatter={(v: any, name: any, entry: any) => {
            const p = entry?.payload
            return [`${formatCellValue(v, mode)} | WR: ${p?.[`${name}_wr`]}% | ${p?.[`${name}_days`]}d`, name]
          }}
        />
        <ReferenceLine y={0} stroke="var(--text-dim)" strokeDasharray="3 3" />
        {sLabels.map((sLabel, i) => (
          <Bar key={sLabel} dataKey={sLabel} fill={colors[i % colors.length]} fillOpacity={0.75} radius={[3, 3, 0, 0]} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  )
}

// ── Detail Table ──
function CrossDetailTable({ data, mode, bucketView, colLabel, rowLabel }: {
  data: CompositeData; mode: DisplayMode; bucketView: BucketView; colLabel: string; rowLabel: string
}) {
  const useP = bucketView === 'percentile'
  const fLabels = useP ? data.pct_feature_labels : data.feature_labels
  const grid = useP ? data.pct_grid : data.grid
  const sLabels = data.static_labels
  const rowShort = getFeatureShort(rowLabel)
  const colShort = getFeatureShort(colLabel)

  const rows = useMemo(() => {
    const result: { bucket: string; _rowTag: string; colBucket: string; _colTag: string; _primary: number; avg_daily_pct: number; win_rate: number; sharpe_pct: number | null; trading_days: number; _origIdx: number; _isFirstCol: boolean }[] = []
    let idx = 0
    fLabels.forEach((fLabel, fi) => {
      sLabels.forEach((sLabel, si) => {
        const cell = grid[fi]?.[si]
        if (cell && cell.trading_days > 0) {
          const primary = mode === 'sharpe' ? (cell.sharpe_pct ?? 0) : mode === 'annualized' ? annualize(cell.avg_daily_pct) : cell.total_pct
          result.push({
            bucket: fLabel, _rowTag: getClusterLabel(fi, fLabels.length, rowShort),
            colBucket: sLabel, _colTag: getClusterLabel(si, sLabels.length, colShort),
            _primary: primary,
            avg_daily_pct: cell.avg_daily_pct, win_rate: cell.win_rate,
            sharpe_pct: cell.sharpe_pct, trading_days: cell.trading_days,
            _origIdx: idx++, _isFirstCol: si === 0,
          })
        }
      })
    })
    return result
  }, [fLabels, grid, sLabels, mode, rowShort, colShort])

  const { sorted, sort, toggle } = useSortableData(rows)
  const thR = "text-right py-2 px-2"

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-[10px] font-mono">
        <thead>
          <tr className="text-[var(--text-dim)] border-b border-[var(--border)]">
            <SortableTh column="bucket" label="Bucket" sort={sort} toggle={toggle} className="text-left py-2 pr-2" />
            <SortableTh column="colBucket" label={colLabel} sort={sort} toggle={toggle} className="text-left py-2 px-2" />
            <SortableTh column="_primary" label={mode === 'sharpe' ? 'Sharpe' : mode === 'annualized' ? 'Ann. Return' : 'Total %'} sort={sort} toggle={toggle} className={thR} />
            <SortableTh column="avg_daily_pct" label="Avg Daily %" sort={sort} toggle={toggle} className={thR} />
            <SortableTh column="win_rate" label="Win Rate" sort={sort} toggle={toggle} className={thR} />
            {mode !== 'sharpe' && <SortableTh column="sharpe_pct" label="Sharpe" sort={sort} toggle={toggle} className={thR} />}
            <SortableTh column="trading_days" label="Days" sort={sort} toggle={toggle} className="text-right py-2 pl-2" />
          </tr>
        </thead>
        <tbody>
          {sorted.map((r) => (
            <tr key={r._origIdx} className="border-b border-[var(--border-subtle)] hover:bg-[var(--bg-hover)] transition-colors" style={{ opacity: r.trading_days < 5 ? 0.4 : 1 }}>
              <td className="py-1.5 pr-2 text-[var(--text-primary)] font-semibold">
                {(sort.column ? true : r._isFirstCol) && (
                  <div className="flex flex-col">
                    <span className="text-[9px] text-[var(--text-primary)] opacity-70">{r._rowTag}</span>
                    <span>{r.bucket}</span>
                  </div>
                )}
              </td>
              <td className="py-1.5 px-2 text-[var(--text-secondary)]">
                <div className="flex flex-col">
                  <span className="text-[9px] text-[var(--text-primary)] opacity-70">{r._colTag}</span>
                  <span>{r.colBucket}</span>
                </div>
              </td>
              <td className="py-1.5 px-2 text-right" style={{ color: r._primary >= 0 ? 'var(--signal-positive)' : 'var(--signal-negative)' }}>{formatCellValue(r._primary, mode)}</td>
              <td className="py-1.5 px-2 text-right text-[var(--text-secondary)]">{formatPct(r.avg_daily_pct)}</td>
              <td className="py-1.5 px-2 text-right text-[var(--text-secondary)]">{(r.win_rate * 100).toFixed(1)}%</td>
              {mode !== 'sharpe' && <td className="py-1.5 px-2 text-right text-[var(--text-secondary)]">{formatSharpe(r.sharpe_pct)}</td>}
              <td className="py-1.5 pl-2 text-right text-[var(--text-muted)]">{r.trading_days}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ── DTE Heatmap for Composite (9 combos × DTE) ──

function CompositeDTEHeatmap({ cross, mode, rowLabel, colLabel }: {
  cross: CompositeDTECross; mode: DisplayMode; rowLabel: string; colLabel: string
}) {
  const rowShort = getFeatureShort(rowLabel)
  const colShort = getFeatureShort(colLabel)

  // Count unique row/col labels for cluster tagging
  const uniqueRowLabels = useMemo(() => {
    const seen = new Set<string>()
    cross.combo_labels.forEach(c => seen.add(c.row_label))
    return [...seen]
  }, [cross.combo_labels])
  const uniqueColLabels = useMemo(() => {
    const seen = new Set<string>()
    cross.combo_labels.forEach(c => seen.add(c.col_label))
    return [...seen]
  }, [cross.combo_labels])

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
            <th className="text-left text-[var(--text-muted)] pb-3 pr-3 text-[10px]">{rowLabel} × {colLabel} ↓ / DTE →</th>
            {cross.dte_labels.map(d => (
              <th key={d} className="text-center text-[var(--text-muted)] pb-3 px-1 min-w-[100px] text-[10px]">DTE {d}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {cross.combo_labels.map((combo, ci) => {
            const rowIdx = uniqueRowLabels.indexOf(combo.row_label)
            const colIdx = uniqueColLabels.indexOf(combo.col_label)
            const rowTag = getClusterLabel(rowIdx, uniqueRowLabels.length, rowShort)
            const colTag = getClusterLabel(colIdx, uniqueColLabels.length, colShort)
            const isNewRowGroup = ci === 0 || cross.combo_labels[ci - 1].row_label !== combo.row_label

            return (
              <tr key={ci} className={isNewRowGroup && ci > 0 ? 'border-t-2 border-[var(--border)]' : ''}>
                <td className="text-[var(--text-secondary)] pr-3 py-1 font-semibold whitespace-nowrap">
                  <div className="flex flex-col">
                    <span className="text-[9px] text-[var(--text-primary)] opacity-70">{rowTag} × {colTag}</span>
                    <span className="text-[10px]">{combo.row_label} × {combo.col_label}</span>
                  </div>
                </td>
                {cross.grid[ci]?.map((cell, di) => {
                  if (!cell || cell.trading_days === 0) {
                    return <td key={di} className="px-1 py-1"><div className="w-full h-16 rounded-lg bg-[var(--bg-hover)] flex items-center justify-center text-[var(--text-dim)] text-[10px]">—</div></td>
                  }
                  const val = getCellVal(cell, mode)
                  const intensity = Math.min(Math.abs(val) / extent, 1)
                  const bg = val >= 0
                    ? `rgba(0,230,118,${0.06 + intensity * 0.45})`
                    : `rgba(255,82,82,${0.06 + intensity * 0.45})`
                  return (
                    <td key={di} className="px-1 py-1">
                      <div
                        className="w-full h-16 rounded-lg flex flex-col items-center justify-center cursor-default transition-all hover:scale-[1.03] hover:shadow-lg"
                        style={{ backgroundColor: bg, color: 'var(--text-primary)', opacity: cell.trading_days < 5 ? 0.35 : 1 }}
                        title={`${rowTag} × ${colTag} (${combo.combo_label}) | DTE ${cross.dte_labels[di]}\nWR: ${(cell.win_rate * 100).toFixed(0)}% | Days: ${cell.trading_days}`}
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

// ── Main Component ──
export function CompositeSection({ strategy, startDate, endDate, snapshot }: { strategy: string; startDate: string; endDate: string; snapshot: string }) {
  const meta = STRATEGY_META[strategy] || STRATEGY_META.dm
  const [rowFeature, setRowFeature] = useState<string>(FEATURES[0])
  const [colFeature, setColFeature] = useState<string>(FEATURES[1])
  const [bucketView, setBucketView] = useState<BucketView>('raw')
  const [dteView, setDteView] = useState<DTEView>('overall')
  const [mode, setMode] = useState<DisplayMode>('sharpe')
  const [data, setData] = useState<CompositeData | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    fetchComposite(strategy, rowFeature, colFeature, startDate, endDate, snapshot).then(d => { setData(d); setLoading(false) })
  }, [strategy, rowFeature, colFeature, startDate, endDate, snapshot])

  const rowLabel = FEATURE_LABELS[rowFeature] || rowFeature
  const colLabel = FEATURE_LABELS[colFeature] || colFeature

  return (
    <div>
      <div className="mb-2">
        <p className="text-[11px] font-mono text-[var(--text-dim)] mb-1 tracking-wider uppercase">{meta.name} / composite</p>
        <h1 className="text-2xl font-bold text-[var(--text-primary)] tracking-tight">Feature x Feature Composite</h1>
        <p className="text-sm text-[var(--text-secondary)] mt-1">Cross-tabulate RV feature buckets</p>
      </div>

      {/* Controls */}
      <div className="sticky top-0 z-30 py-3 mb-5 backdrop-blur-md" style={{ backgroundColor: 'rgba(10, 10, 16, 0.92)' }}>
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-mono text-[var(--text-dim)] uppercase tracking-wider">Row</span>
            <select value={rowFeature} onChange={e => setRowFeature(e.target.value)} className="text-[11px] font-mono px-3 py-1.5 rounded-lg border border-[var(--border)] bg-[var(--bg-card)] text-[var(--text-primary)] outline-none">
              {FEATURES.map(f => <option key={f} value={f}>{FEATURE_LABELS[f]}</option>)}
            </select>
          </div>

          <span className="text-[10px] font-mono text-[var(--text-dim)]">x</span>

          <div className="flex items-center gap-2">
            <span className="text-[10px] font-mono text-[var(--text-dim)] uppercase tracking-wider">Col</span>
            <select value={colFeature} onChange={e => setColFeature(e.target.value)} className="text-[11px] font-mono px-3 py-1.5 rounded-lg border border-[var(--border)] bg-[var(--bg-card)] text-[var(--text-primary)] outline-none">
              {FEATURES.filter(f => f !== rowFeature).map(f => <option key={f} value={f}>{FEATURE_LABELS[f]}</option>)}
            </select>
          </div>

          <div className="h-5 w-px bg-[var(--border)]" />

          <div className="inline-flex rounded-lg border border-[var(--border)] overflow-hidden text-[11px] font-mono">
            {(['raw', 'percentile'] as BucketView[]).map((v, i) => (
              <button key={v} onClick={() => setBucketView(v)} className="px-3 py-1.5 transition-colors capitalize" style={{ backgroundColor: bucketView === v ? 'var(--bg-elevated)' : 'transparent', color: bucketView === v ? 'var(--text-primary)' : 'var(--text-muted)', borderLeft: i > 0 ? '1px solid var(--border)' : 'none' }}>
                {v}
              </button>
            ))}
          </div>

          <div className="h-5 w-px bg-[var(--border)]" />

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

      {loading && <div className="flex items-center justify-center h-64"><div className="text-[var(--text-muted)]">Loading composite...</div></div>}

      {/* Overall view */}
      {!loading && data && dteView === 'overall' && (
        <>
          <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5 mb-5">
            <h2 className="text-sm font-semibold text-[var(--text-primary)] mb-1 uppercase tracking-wider">{rowLabel} x {colLabel} — {mode === 'sharpe' ? 'Sharpe' : mode === 'annualized' ? 'Ann. Return' : 'Avg Daily %'}</h2>
            <div className="flex items-center gap-4 text-[11px] text-[var(--text-secondary)] mb-4">
              <span>{bucketView === 'raw' ? 'Raw buckets' : 'Percentile buckets'}</span>
              {data.static_labels.map((label, i) => {
                const colShort = getFeatureShort(colLabel)
                const tag = getClusterLabel(i, data.static_labels.length, colShort)
                return (
                  <span key={label} className="flex items-center gap-1">
                    <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ backgroundColor: ['#42a5f5', '#ffc107', '#ef5350', '#ab47bc', '#66bb6a'][i % 5] }} />
                    <span className="text-[9px] text-[var(--text-primary)] opacity-70">{tag}</span> {label}
                  </span>
                )
              })}
            </div>
            <CrossBarChart data={data} mode={mode} bucketView={bucketView} rowLabel={rowLabel} colLabel={colLabel} />
          </div>

          <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5 mb-5">
            <h2 className="text-sm font-semibold text-[var(--text-primary)] mb-1 uppercase tracking-wider">{rowLabel} x {colLabel} Heatmap</h2>
            <p className="text-[11px] text-[var(--text-muted)] mb-4">
              {mode === 'sharpe' ? 'Sharpe ratio' : mode === 'annualized' ? 'Annualised return' : 'Avg daily return'} by {bucketView} buckets. Faded cells &lt;5 days.
            </p>
            <CrossHeatmap data={data} mode={mode} bucketView={bucketView} rowLabel={rowLabel} colLabel={colLabel} />
          </div>

          <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5">
            <h2 className="text-sm font-semibold text-[var(--text-primary)] mb-4 uppercase tracking-wider">Full Breakdown</h2>
            <CrossDetailTable data={data} mode={mode} bucketView={bucketView} colLabel={colLabel} rowLabel={rowLabel} />
          </div>
        </>
      )}

      {/* By DTE view */}
      {!loading && data?.composite_dte_cross && dteView === 'by_dte' && (
        <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5">
          <h2 className="text-sm font-semibold text-[var(--text-primary)] mb-1 uppercase tracking-wider">
            {rowLabel} x {colLabel} x DTE — {mode === 'sharpe' ? 'Sharpe' : mode === 'annualized' ? 'Ann. Return' : 'Avg Daily %'}
          </h2>
          <p className="text-[11px] text-[var(--text-muted)] mb-4">
            All 9 feature combinations vs Days to Expiry. Faded cells &lt;5 trading days.
          </p>
          <CompositeDTEHeatmap cross={data.composite_dte_cross} mode={mode} rowLabel={rowLabel} colLabel={colLabel} />
        </div>
      )}
    </div>
  )
}
