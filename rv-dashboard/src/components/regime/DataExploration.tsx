'use client'

import { useState, useEffect, useMemo, useRef, useCallback } from 'react'
import { fetchExplorationFeatures, fetchDataExploration } from '@/lib/api'
import { STRATEGY_META } from '@/lib/config'
import type { ExplorationFeatureMeta, DataExplorationResult, EDAQuintileBucket } from '@/lib/types'
import {
  BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ComposedChart, Area, Line, ReferenceLine, ScatterChart, Scatter,
} from 'recharts'
import { createChart, LineSeries, HistogramSeries, type IChartApi, type ISeriesApi, ColorType } from 'lightweight-charts'

interface Props {
  startDate: string
  endDate: string
  snapshot: string
  dte: number | null
}

const STRATEGY_COLORS: Record<string, string> = {
  dm: '#ffd740', wc: '#448aff', orion: '#64ffda', dmo: '#ff80ab', combined: '#c084fc',
}
const STRATEGY_LABELS: Record<string, string> = {
  dm: 'DM', wc: 'WC', orion: 'Orion', dmo: 'DMO', combined: 'Combined',
}

const FEATURE_FORMULAS: Record<string, { formula: string; terms: string[] }> = {
  RV_today: {
    formula: 'sqrt(ro² + RS) × sqrt(252) × 100',
    terms: [
      'ro = log(Open / previous day Close) — overnight return',
      'rc = log(Close / Open) — intraday close-to-open return',
      'rh = log(High / Open) — high deviation from open',
      'rl = log(Low / Open) — low deviation from open',
      'RS = rh×(rh − rc) + rl×(rl − rc) — Rogers-Satchell intraday variance (clamped ≥ 0)',
      '252 — annualisation factor (trading days per year)',
    ],
  },
  IV_7d: {
    formula: 'IV_7d = IV_near × (T_far − 7)/(T_far − T_near) + IV_far × (7 − T_near)/(T_far − T_near)',
    terms: [
      'IV_near — ATM implied volatility of the nearest weekly expiry (DTE < 7)',
      'IV_far — ATM implied volatility of the next weekly expiry (DTE > 7)',
      'T_near — days to expiry of the near contract',
      'T_far — days to expiry of the far contract',
      '7 — target constant maturity (7 calendar days)',
      'Linear interpolation between two bracketing expiries to synthesise a fixed 7-day IV',
      'Sampled at the selected snapshot time (e.g. 09:15 or 15:30)',
    ],
  },
  IV_change_1d: {
    formula: 'IV_7d[t] − IV_7d[t−1]',
    terms: [
      'IV_7d[t] — today\'s 7-day forward implied volatility',
      'IV_7d[t−1] — previous trading day\'s 7-day forward implied volatility',
    ],
  },
  VRP_today: {
    formula: 'IV_7d − RV_today',
    terms: [
      'IV_7d — 7-day forward implied volatility (market\'s expectation of future vol)',
      'RV_today — Yang-Zhang realised volatility (actual movement observed today)',
      'Positive VRP — market is pricing in more vol than realised (normal state)',
    ],
  },
  IV_intraday_change: {
    formula: 'IV_7d[09:15] − IV_7d[15:30]',
    terms: [
      'IV_7d[09:15] — 7-day forward IV at market open (9:15 AM)',
      'IV_7d[15:30] — 7-day forward IV at market close (3:30 PM)',
      'Positive — IV fell during the day; Negative — IV rose during the day',
    ],
  },
  PK_today: {
    formula: 'sqrt(log(High/Low)² / (4·ln2)) × sqrt(252) × 100',
    terms: [
      'High — intraday high price of Nifty',
      'Low — intraday low price of Nifty',
      'log(High/Low) — log range, captures intraday price spread',
      '4·ln2 — Parkinson normalisation constant (≈ 2.773)',
      '252 — annualisation factor (trading days per year)',
    ],
  },
  iv_lag: {
    formula: 'IV_7d[t−1]',
    terms: [
      'IV_7d[t−1] — previous trading day\'s 7-day forward implied volatility',
      'Lagged by 1 day so it is known before today\'s trading begins',
    ],
  },
  IV_5d: {
    formula: 'mean(iv_lag[t−1], iv_lag[t−2], … iv_lag[t−5])',
    terms: [
      'iv_lag — previous day\'s 7-day forward IV (already lagged once)',
      'Rolling 5-day average of lagged IV values (minimum 3 observations required)',
      'Smooths out day-to-day noise in IV level',
    ],
  },
  PK_5d: {
    formula: 'mean(PK_today[t−1], PK_today[t−2], … PK_today[t−5])',
    terms: [
      'PK_today — Parkinson volatility computed from each day\'s High/Low',
      'Rolling 5-day average of lagged Parkinson vol (minimum 3 observations required)',
      'Smooths out day-to-day noise in realised movement',
    ],
  },
  IV_chg_5d: {
    formula: 'mean(IV_change_1d[t−1], IV_change_1d[t−2], … IV_change_1d[t−5])',
    terms: [
      'IV_change_1d — daily change in 7-day forward IV',
      'Rolling 5-day average of lagged IV daily changes (minimum 3 observations required)',
      'Captures the recent direction of IV — rising or falling trend',
    ],
  },
  PK_IV_ratio: {
    formula: 'PK_5d / IV_5d',
    terms: [
      'PK_5d — 5-day average Parkinson realised vol (how much the market actually moved)',
      'IV_5d — 5-day average implied vol (how much the market expected to move)',
      'Ratio > 1 — realised vol exceeds implied (market moved more than priced)',
      'Ratio < 1 — implied vol exceeds realised (premium cushion exists)',
    ],
  },
}

function valColor(v: number | null): string {
  if (v == null) return 'var(--text-dim)'
  if (v > 0) return 'var(--signal-positive)'
  if (v < 0) return 'var(--signal-negative)'
  return 'var(--text-secondary)'
}

function wrColor(wr: number): string {
  if (wr >= 0.7) return 'var(--signal-positive)'
  if (wr >= 0.5) return 'var(--text-secondary)'
  if (wr >= 0.3) return 'var(--signal-warning)'
  return 'var(--signal-negative)'
}

/* ═══ TradingView Lightweight Chart ═══ */
function TVChart({ data, featureName }: { data: { date: string; value: number }[]; featureName: string }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)

  useEffect(() => {
    if (!containerRef.current || !data?.length) return

    // Clean up previous chart
    if (chartRef.current) {
      chartRef.current.remove()
      chartRef.current = null
    }

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: 'rgba(255,255,255,0.5)',
        fontFamily: 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace',
        fontSize: 10,
      },
      grid: {
        vertLines: { color: 'rgba(255,255,255,0.04)' },
        horzLines: { color: 'rgba(255,255,255,0.04)' },
      },
      crosshair: {
        vertLine: { color: 'rgba(168,85,247,0.4)', width: 1, style: 2, labelBackgroundColor: '#7c3aed' },
        horzLine: { color: 'rgba(168,85,247,0.4)', width: 1, style: 2, labelBackgroundColor: '#7c3aed' },
      },
      rightPriceScale: {
        borderColor: 'rgba(255,255,255,0.08)',
      },
      timeScale: {
        borderColor: 'rgba(255,255,255,0.08)',
        timeVisible: false,
      },
      handleScroll: { vertTouchDrag: false },
    })

    chartRef.current = chart

    const lineSeries = chart.addSeries(LineSeries, {
      color: '#a855f7',
      lineWidth: 2,
      crosshairMarkerRadius: 4,
      crosshairMarkerBackgroundColor: '#a855f7',
      priceLineVisible: false,
      lastValueVisible: true,
    })

    const chartData = data
      .filter(d => d.date && d.value != null && !isNaN(d.value))
      .map(d => ({
        time: d.date as string,
        value: d.value,
      }))

    lineSeries.setData(chartData)
    chart.timeScale().fitContent()

    // Resize observer
    const ro = new ResizeObserver(entries => {
      for (const entry of entries) {
        const { width } = entry.contentRect
        if (width > 0) chart.applyOptions({ width })
      }
    })
    ro.observe(containerRef.current)

    return () => {
      ro.disconnect()
      chart.remove()
      chartRef.current = null
    }
  }, [data, featureName])

  return (
    <div className="rounded-xl border border-[var(--border)] p-4" style={{ backgroundColor: 'var(--bg-card)' }}>
      <div ref={containerRef} style={{ width: '100%', height: 350 }} />
    </div>
  )
}

type ChartMetric = 'returns' | 'win_rate' | 'sharpe' | 'al_spread'

const METRIC_OPTIONS: { key: ChartMetric; label: string }[] = [
  { key: 'returns', label: 'Returns' },
  { key: 'win_rate', label: 'Win Rate' },
  { key: 'sharpe', label: 'Sharpe' },
  { key: 'al_spread', label: 'All Lose %' },
]

export function DataExploration({ startDate, endDate, snapshot, dte }: Props) {
  const [features, setFeatures] = useState<ExplorationFeatureMeta[]>([])
  const [selectedFeature, setSelectedFeature] = useState<string>('')
  const [data, setData] = useState<DataExplorationResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [chartMetric, setChartMetric] = useState<ChartMetric>('returns')
  const [dteFilter, setDteFilter] = useState<string>('all')  // 'all' or '0','1','2','3','4','5'

  // Load feature list
  useEffect(() => {
    fetchExplorationFeatures().then(feats => {
      setFeatures(feats)
      if (feats.length > 0 && !selectedFeature) setSelectedFeature(feats[0].key)
    })
  }, [])

  // Load data when feature changes
  useEffect(() => {
    if (!selectedFeature) return
    setLoading(true)
    fetchDataExploration(selectedFeature, startDate, endDate, snapshot, dte)
      .then(setData)
      .finally(() => setLoading(false))
  }, [selectedFeature, startDate, endDate, snapshot, dte])

  const formatDate = (d: string) => {
    if (!d) return ''
    const parts = d.split('-')
    return `${parts[1]}/${parts[2]?.substring(0, 2)}`
  }

  // Available DTE values from the data
  const availableDteValues = useMemo(() => {
    if (!data?.dte_quintile_analysis) return []
    return Object.keys(data.dte_quintile_analysis).sort((a, b) => {
      const aNum = Number(a), bNum = Number(b)
      const aIsNum = !isNaN(aNum), bIsNum = !isNaN(bNum)
      if (aIsNum && bIsNum) return aNum - bNum
      if (aIsNum) return -1  // individual DTEs first
      if (bIsNum) return 1
      return a.localeCompare(b)  // grouped buckets after
    })
  }, [data])

  // Quintile cross-strategy table data (respects DTE filter)
  const quintileCross = useMemo(() => {
    const qa = dteFilter !== 'all' && data?.dte_quintile_analysis?.[dteFilter]
      ? data.dte_quintile_analysis[dteFilter]
      : data?.quintile_analysis
    if (!qa) return null
    const labels = ['Q1 (Low)', 'Q2', 'Q3', 'Q4', 'Q5 (High)']
    const strats = ['dm', 'wc', 'orion', 'combined']

    return labels.map(q => {
      const row: Record<string, unknown> = { quintile: q }
      let n = 0
      let alPct = 0
      strats.forEach(s => {
        const entry = qa[s]
        if (!entry) return
        const buckets = Array.isArray(entry) ? entry : entry.buckets
        const bucket = buckets?.find((b: EDAQuintileBucket) => b.quintile === q)
        if (bucket) {
          row[`${s}_avg`] = bucket.pnl_mean
          row[`${s}_wr`] = bucket.win_rate
          row[`${s}_sharpe`] = bucket.sharpe
          row[`${s}_range`] = bucket.feature_range
          if (n === 0) {
            n = bucket.count
            alPct = bucket.al_pct ?? 0
          }
        }
      })
      row['n'] = n
      row['al_pct'] = alPct
      return row
    })
  }, [data, dteFilter])

  // Extract quintile metadata (n_used vs n_total) for display
  const quintileMeta = useMemo(() => {
    const qa = dteFilter !== 'all' && data?.dte_quintile_analysis?.[dteFilter]
      ? data.dte_quintile_analysis[dteFilter]
      : data?.quintile_analysis
    if (!qa) return null
    // Use first available strategy's metadata
    for (const s of ['dm', 'wc', 'orion', 'combined']) {
      const entry = qa[s]
      if (entry && !Array.isArray(entry) && entry.n_total != null) {
        return { n_total: entry.n_total, n_used: entry.n_used, n_dropped: entry.n_dropped_missing_pnl }
      }
    }
    return null
  }, [data, dteFilter])

  if (!features.length) return <div className="text-center py-20 text-[var(--text-dim)] text-sm font-mono">Loading features...</div>

  return (
    <div className="space-y-6">
      {/* Feature Selector */}
      <div className="flex items-center gap-4">
        <label className="text-[10px] font-mono text-[var(--text-dim)] uppercase tracking-wider">Feature</label>
        <select
          value={selectedFeature}
          onChange={e => setSelectedFeature(e.target.value)}
          className="text-[11px] font-mono px-3 py-1.5 rounded-lg border border-[var(--border)] bg-[var(--bg-card)] text-[var(--text-primary)] outline-none min-w-[220px]"
        >
          {features.map(f => (
            <option key={f.key} value={f.key}>{f.label}</option>
          ))}
        </select>
      </div>

      {loading && <div className="text-center py-20 text-[var(--text-dim)] text-sm font-mono">Loading exploration data...</div>}

      {data && !loading && (
        <>
          {/* ═══ TradingView Interactive Time Series (full width, right below dropdown) ═══ */}
          <TVChart data={data.timeseries} featureName={selectedFeature} />

          {/* ═══ Row 1: Descriptive Stats + Correlations ═══ */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Descriptive Stats */}
            <div className="rounded-xl border border-[var(--border)] p-4" style={{ backgroundColor: 'var(--bg-card)' }}>
              <h3 className="text-xs font-mono uppercase tracking-wider text-[var(--text-dim)] mb-3">Descriptive Statistics</h3>
              <div className="text-[10px] font-mono flex justify-between">
                <span className="text-[var(--text-dim)]">No. of Days</span>
                <span className="text-[var(--text-primary)] font-semibold">{data.descriptive_stats.count}</span>
              </div>
              {FEATURE_FORMULAS[selectedFeature] && (
                <div className="mt-3 pt-3 border-t border-[var(--border-subtle)]">
                  <div className="text-[9px] font-mono text-[var(--text-dim)] uppercase tracking-wider mb-1.5">Formula</div>
                  <div className="text-[10px] font-mono text-[var(--text-primary)] font-semibold mb-2">
                    {FEATURE_FORMULAS[selectedFeature].formula}
                  </div>
                  <div className="space-y-1">
                    {FEATURE_FORMULAS[selectedFeature].terms.map((term, i) => (
                      <div key={i} className="text-[9px] font-mono text-[var(--text-muted)] leading-relaxed pl-2 border-l border-[var(--border-subtle)]">
                        {term}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Strategy Correlations */}
            <div className="rounded-xl border border-[var(--border)] p-4" style={{ backgroundColor: 'var(--bg-card)' }}>
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xs font-mono uppercase tracking-wider text-[var(--text-dim)]">Strategy Correlations</h3>
                <div className="flex items-center gap-1">
                  <span className="text-[9px] font-mono text-[var(--text-dim)] mr-1">DTE:</span>
                  {['all', ...availableDteValues].map(d => (
                    <button
                      key={`corr_${d}`}
                      onClick={() => setDteFilter(d)}
                      className="px-2 py-0.5 text-[9px] font-mono rounded transition-colors"
                      style={{
                        backgroundColor: dteFilter === d ? 'rgba(168,85,247,0.2)' : 'transparent',
                        color: dteFilter === d ? '#c084fc' : 'var(--text-dim)',
                        border: `1px solid ${dteFilter === d ? '#c084fc' : 'var(--border-subtle)'}`,
                      }}
                    >
                      {d === 'all' ? 'All' : d}
                    </button>
                  ))}
                </div>
              </div>
              <div className="space-y-2">
                {['dm', 'wc', 'orion', 'combined'].map(skey => {
                  const corrSource = dteFilter !== 'all' && data.dte_feature_vs_pnl?.[dteFilter]
                    ? data.dte_feature_vs_pnl[dteFilter]
                    : data.feature_vs_pnl
                  const corr = corrSource?.[skey]
                  if (!corr) return null
                  const r = corr.pearson_r
                  const barWidth = Math.abs(r) * 100
                  return (
                    <div key={skey} className="flex items-center gap-3">
                      <div className="w-20 text-[10px] font-mono font-semibold" style={{ color: STRATEGY_COLORS[skey] }}>
                        {STRATEGY_LABELS[skey]}
                      </div>
                      <div className="flex-1 h-5 relative">
                        {/* Center line */}
                        <div className="absolute left-1/2 top-0 bottom-0 w-px bg-[var(--border)]" />
                        <div className="absolute inset-0 flex items-center">
                          <div
                            className="h-4 rounded-sm transition-all duration-300"
                            style={{
                              width: `${barWidth / 2}%`,
                              backgroundColor: STRATEGY_COLORS[skey],
                              opacity: 0.6,
                              marginLeft: r >= 0 ? '50%' : `${50 - barWidth / 2}%`,
                            }}
                          />
                        </div>
                      </div>
                      <div className="w-16 text-right text-[10px] font-mono font-semibold" style={{ color: STRATEGY_COLORS[skey] }}>
                        {r > 0 ? '+' : ''}{r.toFixed(3)}
                      </div>
                      <div className="w-12 text-right text-[9px] font-mono text-[var(--text-dim)]">
                        p={corr.pearson_p < 0.001 ? '<.001' : corr.pearson_p.toFixed(3)}
                      </div>
                    </div>
                  )
                })}
              </div>
              {/* Spearman rank correlations */}
              <div className="mt-3 pt-3 border-t border-[var(--border-subtle)]">
                <div className="text-[9px] font-mono text-[var(--text-dim)] mb-1">Spearman (rank)</div>
                <div className="flex gap-3 text-[9px] font-mono">
                  {['dm', 'wc', 'orion'].map(skey => {
                    const corrSrc = dteFilter !== 'all' && data.dte_feature_vs_pnl?.[dteFilter]
                      ? data.dte_feature_vs_pnl[dteFilter]
                      : data.feature_vs_pnl
                    const corr = corrSrc?.[skey]
                    if (!corr) return null
                    return (
                      <span key={skey}>
                        <span style={{ color: STRATEGY_COLORS[skey] }}>{STRATEGY_LABELS[skey]}</span>
                        : {corr.spearman_r > 0 ? '+' : ''}{corr.spearman_r.toFixed(3)}
                      </span>
                    )
                  })}
                </div>
              </div>
            </div>
          </div>

          {/* ═══ Distribution ═══ */}
          <div className="rounded-xl border border-[var(--border)] p-4" style={{ backgroundColor: 'var(--bg-card)' }}>
            <h3 className="text-xs font-mono uppercase tracking-wider text-[var(--text-dim)] mb-3">Distribution</h3>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={data.histogram} margin={{ top: 5, right: 10, left: 10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
                <XAxis dataKey="bin_label" tick={{ fontSize: 8, fill: 'var(--text-dim)' }} interval={Math.max(0, Math.floor(data.histogram.length / 8))} />
                <YAxis tick={{ fontSize: 9, fill: 'var(--text-dim)' }} />
                <Tooltip contentStyle={{ backgroundColor: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 11, fontFamily: 'monospace' }} formatter={(v) => [v, 'Count']} />
                <Bar dataKey="count" fill="rgba(168,85,247,0.6)" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* ═══ Row 2: Quintile Analysis Cross-Strategy Table (THE KEY TABLE from PDF Step 1) ═══ */}
          <div className="rounded-xl border border-[var(--border)] overflow-hidden" style={{ backgroundColor: 'var(--bg-card)' }}>
            <div className="px-4 py-2.5 border-b border-[var(--border)] bg-[var(--bg-elevated)] flex items-center justify-between flex-wrap gap-2">
              <span className="text-[11px] font-mono font-semibold text-[var(--text-secondary)]">
                Quintile Analysis — Equal-Count (5 bins)
                {quintileMeta && quintileMeta.n_dropped > 0 && (
                  <span className="ml-2 text-[9px] font-normal text-[var(--signal-warning)]">
                    N={quintileMeta.n_used}/{quintileMeta.n_total} ({quintileMeta.n_dropped} rows missing PnL)
                  </span>
                )}
              </span>
              <div className="flex items-center gap-1">
                <span className="text-[9px] font-mono text-[var(--text-dim)] mr-1">DTE:</span>
                {['all', ...availableDteValues].map(d => (
                  <button
                    key={d}
                    onClick={() => setDteFilter(d)}
                    className="px-2 py-0.5 text-[9px] font-mono rounded transition-colors"
                    style={{
                      backgroundColor: dteFilter === d ? 'rgba(168,85,247,0.2)' : 'transparent',
                      color: dteFilter === d ? '#c084fc' : 'var(--text-dim)',
                      border: `1px solid ${dteFilter === d ? '#c084fc' : 'var(--border-subtle)'}`,
                    }}
                  >
                    {d === 'all' ? 'All' : d}
                  </button>
                ))}
              </div>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-[10px] font-mono">
                <thead>
                  <tr className="border-b border-[var(--border)]">
                    <th className="text-left px-3 py-2 text-[var(--text-dim)] font-normal">Q</th>
                    <th className="text-left px-2 py-2 text-[var(--text-dim)] font-normal">Range</th>
                    <th className="text-right px-2 py-2 text-[var(--text-dim)] font-normal">N</th>
                    <th className="text-right px-2 py-2 font-normal" style={{ color: '#ffd740' }}>DM</th>
                    <th className="text-right px-2 py-2 font-normal" style={{ color: '#ffd740' }}>DM WR</th>
                    <th className="text-right px-2 py-2 font-normal" style={{ color: '#ffd740' }}>DM Sh</th>
                    <th className="text-right px-2 py-2 font-normal" style={{ color: '#448aff' }}>WC</th>
                    <th className="text-right px-2 py-2 font-normal" style={{ color: '#448aff' }}>WC WR</th>
                    <th className="text-right px-2 py-2 font-normal" style={{ color: '#448aff' }}>WC Sh</th>
                    <th className="text-right px-2 py-2 font-normal" style={{ color: '#64ffda' }}>Orion</th>
                    <th className="text-right px-2 py-2 font-normal" style={{ color: '#64ffda' }}>Or WR</th>
                    <th className="text-right px-2 py-2 font-normal" style={{ color: '#64ffda' }}>Or Sh</th>
                    <th className="text-right px-2 py-2 font-normal" style={{ color: '#c084fc' }}>Comb</th>
                    <th className="text-right px-2 py-2 font-normal" style={{ color: '#c084fc' }}>Comb Sh</th>
                    <th className="text-right px-2 py-2 text-[var(--text-dim)] font-normal">AL%</th>
                  </tr>
                </thead>
                <tbody>
                  {quintileCross?.map((row, i) => (
                    <tr key={i} className="border-b border-[var(--border-subtle)] hover:bg-[var(--bg-hover)]">
                      <td className="px-3 py-2 text-[var(--text-primary)] font-semibold">{String(row.quintile)}</td>
                      <td className="px-2 py-2 text-[var(--text-muted)] text-[9px]">{String(row.dm_range ?? '')}</td>
                      <td className="text-right px-2 py-2 text-[var(--text-secondary)]">{String(row.n)}</td>
                      {['dm', 'wc', 'orion', 'combined'].flatMap(s => {
                        const cells = [
                          <td key={`${i}_${s}_avg`} className="text-right px-2 py-2 font-semibold" style={{ color: valColor(row[`${s}_avg`] as number | null) }}>
                            {row[`${s}_avg`] != null ? `${(row[`${s}_avg`] as number) > 0 ? '+' : ''}${(row[`${s}_avg`] as number).toFixed(2)}%` : '—'}
                          </td>
                        ]
                        if (s !== 'combined') {
                          cells.push(
                            <td key={`${i}_${s}_wr`} className="text-right px-2 py-2" style={{ color: wrColor(row[`${s}_wr`] as number ?? 0.5) }}>
                              {row[`${s}_wr`] != null ? `${((row[`${s}_wr`] as number) * 100).toFixed(0)}%` : '—'}
                            </td>
                          )
                        }
                        cells.push(
                          <td key={`${i}_${s}_sh`} className="text-right px-2 py-2 text-[var(--text-secondary)]">
                            {row[`${s}_sharpe`] != null ? (row[`${s}_sharpe`] as number).toFixed(2) : '—'}
                          </td>
                        )
                        return cells
                      })}
                      <td className="text-right px-2 py-2 font-semibold" style={{ color: (row.al_pct as number) > 15 ? 'var(--signal-negative)' : (row.al_pct as number) > 8 ? 'var(--signal-warning)' : 'var(--text-secondary)' }}>
                        {(row.al_pct as number).toFixed(1)}%
                      </td>
                    </tr>
                  ))}
                  {/* Spread Row */}
                  {quintileCross && quintileCross.length >= 5 && (
                    <tr className="border-t-2 border-[var(--border-active)] bg-[var(--bg-elevated)]">
                      <td className="px-3 py-2 text-[var(--text-dim)] font-semibold">Spread</td>
                      <td className="px-2 py-2 text-[var(--text-dim)] text-[9px]">Q5−Q1</td>
                      <td className="text-right px-2 py-2" />
                      {['dm', 'wc', 'orion', 'combined'].flatMap(s => {
                        const q1 = quintileCross[0]?.[`${s}_avg`] as number | undefined
                        const q5 = quintileCross[4]?.[`${s}_avg`] as number | undefined
                        const spread = q1 != null && q5 != null ? q5 - q1 : null
                        const q1wr = quintileCross[0]?.[`${s}_wr`] as number | undefined
                        const q5wr = quintileCross[4]?.[`${s}_wr`] as number | undefined
                        const wrSpread = q1wr != null && q5wr != null ? (q5wr - q1wr) * 100 : null
                        const cells = [
                          <td key={`sp_${s}_avg`} className="text-right px-2 py-2 font-bold" style={{ color: valColor(spread) }}>
                            {spread != null ? `${spread > 0 ? '+' : ''}${spread.toFixed(2)}%` : '—'}
                          </td>
                        ]
                        if (s !== 'combined') {
                          cells.push(
                            <td key={`sp_${s}_wr`} className="text-right px-2 py-2 font-semibold" style={{ color: valColor(wrSpread) }}>
                              {wrSpread != null ? `${wrSpread > 0 ? '+' : ''}${wrSpread.toFixed(0)}pp` : '—'}
                            </td>
                          )
                        }
                        cells.push(<td key={`sp_${s}_sh`} className="text-right px-2 py-2" />)
                        return cells
                      })}
                      {/* AL% spread */}
                      {(() => {
                        const q1al = quintileCross[0]?.al_pct as number | undefined
                        const q5al = quintileCross[4]?.al_pct as number | undefined
                        const alSpread = q1al != null && q5al != null ? q5al - q1al : null
                        return (
                          <td className="text-right px-2 py-2 font-bold" style={{ color: valColor(alSpread) }}>
                            {alSpread != null ? `${alSpread > 0 ? '+' : ''}${alSpread.toFixed(1)}%` : '—'}
                          </td>
                        )
                      })()}
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* ═══ Row 3: Metric-Selectable Quintile Bar Chart ═══ */}
          <div className="rounded-xl border border-[var(--border)] p-4" style={{ backgroundColor: 'var(--bg-card)' }}>
            <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
              <h3 className="text-xs font-mono uppercase tracking-wider text-[var(--text-dim)]">Quintile Breakdown by Strategy</h3>
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-1">
                  <span className="text-[9px] font-mono text-[var(--text-dim)] mr-1">DTE:</span>
                  {['all', ...availableDteValues].map(d => (
                    <button
                      key={`chart_dte_${d}`}
                      onClick={() => setDteFilter(d)}
                      className="px-2 py-0.5 text-[9px] font-mono rounded transition-colors"
                      style={{
                        backgroundColor: dteFilter === d ? 'rgba(168,85,247,0.2)' : 'transparent',
                        color: dteFilter === d ? '#c084fc' : 'var(--text-dim)',
                        border: `1px solid ${dteFilter === d ? '#c084fc' : 'var(--border-subtle)'}`,
                      }}
                    >
                      {d === 'all' ? 'All' : d}
                    </button>
                  ))}
                </div>
                <div className="flex gap-1">
                  {METRIC_OPTIONS.map(opt => (
                    <button
                      key={opt.key}
                      onClick={() => setChartMetric(opt.key)}
                      className="px-2.5 py-1 text-[9px] font-mono rounded-md transition-colors"
                      style={{
                        backgroundColor: chartMetric === opt.key ? 'rgba(168,85,247,0.2)' : 'transparent',
                        color: chartMetric === opt.key ? '#c084fc' : 'var(--text-dim)',
                        border: `1px solid ${chartMetric === opt.key ? '#c084fc' : 'var(--border-subtle)'}`,
                      }}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
              </div>
            </div>
            {(() => {
              const strats = chartMetric === 'al_spread' ? ['combined'] : ['dm', 'wc', 'orion', 'combined']
              const suffix = chartMetric === 'returns' ? '_avg' : chartMetric === 'win_rate' ? '_wr' : chartMetric === 'sharpe' ? '_sharpe' : ''
              const chartData = (quintileCross ?? []).map(row => {
                if (chartMetric === 'al_spread') {
                  const qa = data?.quintile_analysis
                  const buckets = qa?.['dm']
                  const arr = Array.isArray(buckets) ? buckets : buckets?.buckets
                  const bucket = arr?.find((b: EDAQuintileBucket) => b.quintile === row.quintile)
                  return { ...row, al_pct: bucket?.al_pct ?? 0 }
                }
                return row
              })
              const fmtY = (v: number) => {
                if (chartMetric === 'returns') return `${v.toFixed(2)}%`
                if (chartMetric === 'win_rate') return `${(v * 100).toFixed(0)}%`
                if (chartMetric === 'sharpe') return v.toFixed(1)
                if (chartMetric === 'al_spread') return `${v.toFixed(0)}%`
                return String(v)
              }
              const fmtTooltip = (value: number, name: string) => {
                const sKey = name.replace(suffix, '').replace('al_pct', 'AL%')
                const label = STRATEGY_LABELS[sKey] ?? sKey
                if (chartMetric === 'returns') return [`${Number(value).toFixed(4)}%`, label]
                if (chartMetric === 'win_rate') return [`${(Number(value) * 100).toFixed(1)}%`, label]
                if (chartMetric === 'sharpe') return [Number(value).toFixed(2), label]
                if (chartMetric === 'al_spread') return [`${Number(value).toFixed(1)}%`, 'All Lose %']
                return [String(value), label]
              }
              return (
                <>
                  <ResponsiveContainer width="100%" height={280}>
                    <BarChart data={chartData} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
                      <XAxis dataKey="quintile" tick={{ fontSize: 9, fill: 'var(--text-dim)' }} />
                      <YAxis tick={{ fontSize: 9, fill: 'var(--text-dim)' }} tickFormatter={fmtY} />
                      <Tooltip
                        contentStyle={{ backgroundColor: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 11, fontFamily: 'monospace' }}
                        formatter={fmtTooltip as never}
                      />
                      <ReferenceLine y={0} stroke="rgba(255,255,255,0.15)" />
                      {chartMetric === 'al_spread' ? (
                        <Bar dataKey="al_pct" name="al_pct" fill="#ef5350" barSize={28} radius={[2, 2, 0, 0]} opacity={0.85} />
                      ) : (
                        strats.map(s => (
                          <Bar key={s} dataKey={`${s}${suffix}`} name={`${s}${suffix}`} fill={STRATEGY_COLORS[s]} barSize={14} radius={[2, 2, 0, 0]} opacity={0.85} />
                        ))
                      )}
                    </BarChart>
                  </ResponsiveContainer>
                  <div className="flex gap-4 mt-2 justify-center">
                    {chartMetric === 'al_spread' ? (
                      <div className="flex items-center gap-1.5">
                        <div className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: '#ef5350' }} />
                        <span className="text-[9px] font-mono text-[var(--text-muted)]">All Lose %</span>
                      </div>
                    ) : (
                      strats.map(s => (
                        <div key={s} className="flex items-center gap-1.5">
                          <div className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: STRATEGY_COLORS[s] }} />
                          <span className="text-[9px] font-mono text-[var(--text-muted)]">{STRATEGY_LABELS[s]}</span>
                        </div>
                      ))
                    )}
                  </div>
                </>
              )
            })()}
          </div>

        </>
      )}
    </div>
  )
}
