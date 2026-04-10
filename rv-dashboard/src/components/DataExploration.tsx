'use client'

import { useState, useEffect, useMemo } from 'react'
import { fetchExplorationFeatures, fetchDataExploration } from '@/lib/api'
import type {
  ExplorationFeatureMeta, DataExplorationResult,
  EDAHistogramBin, EDAAutocorrelation, EDAScatterPoint,
  EDAQuintileBucket, EDARegimeDistribution, EDAOutlierDay,
} from '@/lib/types'

interface Props {
  startDate: string
  endDate: string
  snapshot: string
}

/* ── Helpers ── */
function fmtNum(v: number | null | undefined, dp: number = 4): string {
  if (v == null) return '—'
  return v.toFixed(dp)
}
function signColor(v: number | null | undefined): string {
  if (v == null) return 'var(--text-dim)'
  return v >= 0 ? 'var(--signal-positive)' : 'var(--signal-negative)'
}
function corrColor(r: number): string {
  const abs = Math.abs(r)
  if (abs >= 0.3) return r > 0 ? '#00e676' : '#ff5252'
  if (abs >= 0.15) return r > 0 ? '#69f0ae' : '#ff8a80'
  return 'var(--text-muted)'
}
function sigBadge(p: number): string {
  if (p < 0.001) return '***'
  if (p < 0.01) return '**'
  if (p < 0.05) return '*'
  return ''
}

type EDASection = 'stats' | 'timeseries' | 'distribution' | 'acf' | 'correlation' | 'quintiles' | 'regime' | 'outliers'

const SECTIONS: { key: EDASection; label: string }[] = [
  { key: 'stats', label: 'Descriptive Stats' },
  { key: 'timeseries', label: 'Time Series' },
  { key: 'distribution', label: 'Distribution' },
  { key: 'acf', label: 'Autocorrelation' },
  { key: 'correlation', label: 'Feature vs PnL' },
  { key: 'quintiles', label: 'Quintile Buckets' },
  { key: 'regime', label: 'Regime Conditional' },
  { key: 'outliers', label: 'Outliers' },
]

export function DataExploration({ startDate, endDate, snapshot }: Props) {
  const [features, setFeatures] = useState<ExplorationFeatureMeta[]>([])
  const [selectedFeature, setSelectedFeature] = useState<string>('')
  const [data, setData] = useState<DataExplorationResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [activeSection, setActiveSection] = useState<EDASection>('stats')

  // Load available features
  useEffect(() => {
    fetchExplorationFeatures().then(f => {
      setFeatures(f)
      if (f.length > 0 && !selectedFeature) setSelectedFeature(f[0].key)
    })
  }, [])

  // Fetch EDA when feature or params change
  useEffect(() => {
    if (!selectedFeature) return
    setLoading(true)
    fetchDataExploration(selectedFeature, startDate, endDate, snapshot)
      .then(setData)
      .finally(() => setLoading(false))
  }, [selectedFeature, startDate, endDate, snapshot])

  return (
    <div className="space-y-5">
      {/* Feature Selector + Section Nav */}
      <div className="rounded-xl border border-[var(--border)] overflow-hidden" style={{ backgroundColor: 'var(--bg-card)' }}>
        <div className="px-5 py-4 flex items-center gap-4">
          <div className="text-[10px] font-mono text-[var(--text-dim)] uppercase tracking-widest flex-shrink-0">Feature</div>
          <select
            value={selectedFeature}
            onChange={e => setSelectedFeature(e.target.value)}
            className="px-3 py-1.5 rounded-lg text-[11px] font-mono font-semibold border border-[var(--border)] bg-[var(--bg-elevated)] text-[var(--text-primary)] outline-none cursor-pointer min-w-[200px]"
          >
            {features.map(f => (
              <option key={f.key} value={f.key}>{f.label}</option>
            ))}
          </select>

          {data && (
            <div className="ml-auto flex items-center gap-3 text-[10px] font-mono text-[var(--text-muted)]">
              <span>{data.descriptive_stats.count} obs</span>
              <span className="text-[var(--border)]">|</span>
              <span>μ = {fmtNum(data.descriptive_stats.mean)}</span>
              <span className="text-[var(--border)]">|</span>
              <span>σ = {fmtNum(data.descriptive_stats.std)}</span>
              <span className="text-[var(--border)]">|</span>
              <span style={{ color: data.stationarity?.is_stationary_5pct ? '#00e676' : '#ff5252' }}>
                {data.stationarity?.is_stationary_5pct ? 'Stationary' : 'Non-Stationary'}
              </span>
            </div>
          )}
        </div>

        {/* Section tabs */}
        <div className="px-5 pb-2 flex gap-1 overflow-x-auto">
          {SECTIONS.map(s => (
            <button
              key={s.key}
              onClick={() => setActiveSection(s.key)}
              className="px-3 py-1.5 rounded-lg text-[10px] font-mono font-semibold transition-all flex-shrink-0"
              style={{
                backgroundColor: activeSection === s.key ? 'rgba(168,85,247,0.15)' : 'transparent',
                color: activeSection === s.key ? '#c084fc' : 'var(--text-muted)',
                border: `1px solid ${activeSection === s.key ? 'rgba(168,85,247,0.3)' : 'transparent'}`,
              }}
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>

      {loading && <div className="text-center py-16 text-[var(--text-dim)] text-sm font-mono">Computing EDA for {selectedFeature}...</div>}

      {!loading && data && (
        <>
          {activeSection === 'stats' && <StatsPanel data={data} />}
          {activeSection === 'timeseries' && <TimeSeriesPanel data={data} />}
          {activeSection === 'distribution' && <DistributionPanel data={data} />}
          {activeSection === 'acf' && <ACFPanel data={data} />}
          {activeSection === 'correlation' && <CorrelationPanel data={data} />}
          {activeSection === 'quintiles' && <QuintilePanel data={data} />}
          {activeSection === 'regime' && <RegimePanel data={data} />}
          {activeSection === 'outliers' && <OutlierPanel data={data} />}
        </>
      )}
    </div>
  )
}


/* ═══════════════════════════════════════════
   1. DESCRIPTIVE STATS PANEL
   ═══════════════════════════════════════════ */
function StatsPanel({ data }: { data: DataExplorationResult }) {
  const s = data.descriptive_stats
  const rs = data.rolling_stats
  const st = data.stationarity

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
      {/* Central Tendency */}
      <div className="stat-card !p-4">
        <h4 className="text-[10px] font-mono text-[var(--text-dim)] uppercase tracking-wider mb-3">Central Tendency</h4>
        <div className="space-y-2 text-[11px] font-mono">
          <StatRow label="Mean" value={fmtNum(s.mean, 6)} />
          <StatRow label="Median" value={fmtNum(s.median, 6)} />
          <StatRow label="Std Dev" value={fmtNum(s.std, 6)} />
          <StatRow label="Min" value={fmtNum(s.min, 6)} />
          <StatRow label="Max" value={fmtNum(s.max, 6)} />
          <StatRow label="IQR" value={fmtNum(s.iqr, 6)} />
          <StatRow label="Count" value={String(s.count)} />
        </div>
      </div>

      {/* Shape */}
      <div className="stat-card !p-4">
        <h4 className="text-[10px] font-mono text-[var(--text-dim)] uppercase tracking-wider mb-3">Distribution Shape</h4>
        <div className="space-y-2 text-[11px] font-mono">
          <StatRow label="Skewness" value={fmtNum(s.skew)} highlight={Math.abs(s.skew) > 1} />
          <StatRow label="Kurtosis" value={fmtNum(s.kurtosis)} highlight={Math.abs(s.kurtosis) > 3} />
          <StatRow label="P5" value={fmtNum(s.p5, 6)} />
          <StatRow label="P10" value={fmtNum(s.p10, 6)} />
          <StatRow label="P25" value={fmtNum(s.p25, 6)} />
          <StatRow label="P75" value={fmtNum(s.p75, 6)} />
          <StatRow label="P90" value={fmtNum(s.p90, 6)} />
          <StatRow label="P95" value={fmtNum(s.p95, 6)} />
        </div>
        {s.jb_pvalue != null && (
          <div className="mt-3 pt-2 border-t border-[var(--border-subtle)] text-[10px] font-mono">
            <span className="text-[var(--text-dim)]">Jarque-Bera: </span>
            <span className="text-[var(--text-secondary)]">{fmtNum(s.jb_stat)}</span>
            <span className="text-[var(--text-dim)]"> (p=</span>
            <span style={{ color: s.is_normal ? '#00e676' : '#ff5252' }}>{s.jb_pvalue?.toFixed(4)}</span>
            <span className="text-[var(--text-dim)]">) → </span>
            <span style={{ color: s.is_normal ? '#00e676' : '#ff5252' }}>{s.is_normal ? 'Normal' : 'Non-Normal'}</span>
          </div>
        )}
      </div>

      {/* Current State + Stationarity */}
      <div className="stat-card !p-4">
        <h4 className="text-[10px] font-mono text-[var(--text-dim)] uppercase tracking-wider mb-3">Current State & Stationarity</h4>
        <div className="space-y-2 text-[11px] font-mono">
          <StatRow label="20d Mean" value={fmtNum(rs.current_20d_mean, 6)} />
          <StatRow label="20d Std" value={fmtNum(rs.current_20d_std, 6)} />
          <StatRow label="50d Mean" value={fmtNum(rs.current_50d_mean, 6)} />
          <StatRow label="Z-Score (20d)" value={fmtNum(rs.current_z_20d, 2)} highlight={rs.current_z_20d != null && Math.abs(rs.current_z_20d) > 2} />
          <StatRow label="Percentile Rank" value={`${rs.percentile_rank}%`} />
        </div>
        {st && (
          <div className="mt-3 pt-2 border-t border-[var(--border-subtle)]">
            <div className="text-[10px] font-mono mb-2">
              <span className="text-[var(--text-dim)]">ADF Test: </span>
              <span style={{ color: st.is_stationary_5pct ? '#00e676' : '#ff5252' }}>
                {st.is_stationary_5pct ? 'STATIONARY' : 'NON-STATIONARY'}
              </span>
            </div>
            <div className="space-y-1 text-[10px] font-mono">
              <StatRow label="ADF Stat" value={fmtNum(st.adf_statistic)} />
              <StatRow label="p-value" value={st.p_value.toFixed(4)} highlight={st.p_value < 0.05} />
              <StatRow label="Crit 1%" value={fmtNum(st.critical_1pct)} />
              <StatRow label="Crit 5%" value={fmtNum(st.critical_5pct)} />
              <StatRow label="Lags" value={String(st.lags_used)} />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function StatRow({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className="flex justify-between items-center">
      <span className="text-[var(--text-dim)]">{label}</span>
      <span className={highlight ? 'text-[#ffab40] font-semibold' : 'text-[var(--text-secondary)]'}>{value}</span>
    </div>
  )
}


/* ═══════════════════════════════════════════
   2. TIME SERIES PANEL
   ═══════════════════════════════════════════ */
function TimeSeriesPanel({ data }: { data: DataExplorationResult }) {
  const ts = data.timeseries
  if (ts.length === 0) return <Empty msg="No timeseries data" />

  const values = ts.map(t => t.value)
  const minV = Math.min(...values)
  const maxV = Math.max(...values)
  const range = maxV - minV || 1

  const W = 1200
  const H = 280
  const pad = { t: 10, b: 30, l: 60, r: 10 }
  const plotW = W - pad.l - pad.r
  const plotH = H - pad.t - pad.b

  const toX = (i: number) => pad.l + (i / (ts.length - 1)) * plotW
  const toY = (v: number) => pad.t + (1 - (v - minV) / range) * plotH

  const valuePath = ts.map((t, i) => `${i === 0 ? 'M' : 'L'}${toX(i).toFixed(1)},${toY(t.value).toFixed(1)}`).join(' ')
  const ma20Path = ts.filter(t => t.roll_20_mean != null)
    .map((t, i, arr) => `${i === 0 ? 'M' : 'L'}${toX(ts.indexOf(t)).toFixed(1)},${toY(t.roll_20_mean!).toFixed(1)}`)
    .join(' ')
  const ma50Path = ts.filter(t => t.roll_50_mean != null)
    .map((t, i) => `${i === 0 ? 'M' : 'L'}${toX(ts.indexOf(t)).toFixed(1)},${toY(t.roll_50_mean!).toFixed(1)}`)
    .join(' ')

  // Bollinger bands (20d mean ± 2*20d std)
  const bbUpper = ts.filter(t => t.roll_20_mean != null && t.roll_20_std != null)
    .map(t => ({ idx: ts.indexOf(t), y: t.roll_20_mean! + 2 * t.roll_20_std! }))
  const bbLower = ts.filter(t => t.roll_20_mean != null && t.roll_20_std != null)
    .map(t => ({ idx: ts.indexOf(t), y: t.roll_20_mean! - 2 * t.roll_20_std! }))

  const bbPath = bbUpper.length > 2 ? [
    ...bbUpper.map((p, i) => `${i === 0 ? 'M' : 'L'}${toX(p.idx).toFixed(1)},${toY(p.y).toFixed(1)}`),
    ...bbLower.reverse().map((p, i) => `${i === 0 ? 'L' : 'L'}${toX(p.idx).toFixed(1)},${toY(p.y).toFixed(1)}`),
    'Z'
  ].join(' ') : ''

  // Y-axis ticks
  const yTicks = 5
  const yStep = range / yTicks

  // X-axis date labels
  const dateStep = Math.max(1, Math.floor(ts.length / 6))

  return (
    <div className="rounded-xl border border-[var(--border)] p-5" style={{ backgroundColor: 'var(--bg-card)' }}>
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs font-mono uppercase tracking-wider text-[var(--text-dim)]">
          {data.label} — Time Series with Rolling Stats
        </h3>
        <div className="flex items-center gap-4 text-[9px] font-mono">
          <span className="flex items-center gap-1"><span className="w-3 h-px bg-[#c084fc] inline-block" /> Value</span>
          <span className="flex items-center gap-1"><span className="w-3 h-px bg-[#ffab40] inline-block" /> MA20</span>
          <span className="flex items-center gap-1"><span className="w-3 h-px bg-[#64b5f6] inline-block" /> MA50</span>
          <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-[rgba(168,85,247,0.2)] inline-block" /> 2σ Band</span>
        </div>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: 320 }}>
        {/* Grid lines */}
        {Array.from({ length: yTicks + 1 }, (_, i) => {
          const val = minV + i * yStep
          const y = toY(val)
          return (
            <g key={i}>
              <line x1={pad.l} x2={W - pad.r} y1={y} y2={y} stroke="var(--border)" strokeWidth={0.5} strokeDasharray="4,4" />
              <text x={pad.l - 6} y={y + 3} textAnchor="end" fill="var(--text-dim)" fontSize={8} fontFamily="monospace">{val.toFixed(2)}</text>
            </g>
          )
        })}

        {/* Bollinger band */}
        {bbPath && <path d={bbPath} fill="rgba(168,85,247,0.08)" stroke="none" />}

        {/* Lines */}
        <path d={valuePath} fill="none" stroke="#c084fc" strokeWidth={1.2} opacity={0.9} />
        {ma20Path && <path d={ma20Path} fill="none" stroke="#ffab40" strokeWidth={1} opacity={0.7} />}
        {ma50Path && <path d={ma50Path} fill="none" stroke="#64b5f6" strokeWidth={1} opacity={0.7} />}

        {/* X-axis dates */}
        {ts.filter((_, i) => i % dateStep === 0).map((t, i) => (
          <text key={i} x={toX(ts.indexOf(t))} y={H - 5} textAnchor="middle" fill="var(--text-dim)" fontSize={7} fontFamily="monospace">
            {t.date.slice(0, 7)}
          </text>
        ))}
      </svg>
    </div>
  )
}


/* ═══════════════════════════════════════════
   3. DISTRIBUTION PANEL
   ═══════════════════════════════════════════ */
function DistributionPanel({ data }: { data: DataExplorationResult }) {
  const hist = data.histogram
  const s = data.descriptive_stats
  const maxCount = Math.max(...hist.map(h => h.count), 1)

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {/* Histogram */}
      <div className="rounded-xl border border-[var(--border)] p-5" style={{ backgroundColor: 'var(--bg-card)' }}>
        <h3 className="text-xs font-mono uppercase tracking-wider text-[var(--text-dim)] mb-4">Histogram</h3>
        <div className="space-y-0.5">
          {hist.map((b, i) => (
            <div key={i} className="flex items-center gap-2">
              <div className="w-16 text-right text-[8px] font-mono text-[var(--text-muted)] flex-shrink-0">{b.bin_label}</div>
              <div className="flex-1 h-4 rounded overflow-hidden bg-[var(--bg-elevated)] relative">
                <div
                  className="h-full rounded transition-all duration-300"
                  style={{
                    width: `${(b.count / maxCount) * 100}%`,
                    backgroundColor: `rgba(168,85,247,${0.3 + (b.count / maxCount) * 0.5})`,
                  }}
                />
                {b.count > 0 && (
                  <span className="absolute inset-0 flex items-center px-1.5 text-[8px] font-mono font-semibold text-white">{b.count}</span>
                )}
              </div>
            </div>
          ))}
        </div>
        {/* Mean/Median markers */}
        <div className="mt-3 flex items-center gap-4 text-[9px] font-mono text-[var(--text-muted)]">
          <span>μ = {s.mean.toFixed(4)}</span>
          <span>med = {s.median.toFixed(4)}</span>
          <span>skew = {s.skew.toFixed(2)}</span>
          <span>kurt = {s.kurtosis.toFixed(2)}</span>
        </div>
      </div>

      {/* Box-plot style summary */}
      <div className="rounded-xl border border-[var(--border)] p-5" style={{ backgroundColor: 'var(--bg-card)' }}>
        <h3 className="text-xs font-mono uppercase tracking-wider text-[var(--text-dim)] mb-4">Percentile Map</h3>
        <PercentileBar stats={s} />

        {/* Normality */}
        <div className="mt-6">
          <h4 className="text-[10px] font-mono text-[var(--text-dim)] uppercase tracking-wider mb-3">Normality Assessment</h4>
          <div className="space-y-2 text-[11px] font-mono">
            <div className="flex justify-between">
              <span className="text-[var(--text-dim)]">Jarque-Bera</span>
              <span className="text-[var(--text-secondary)]">{fmtNum(s.jb_stat)} {s.jb_pvalue != null && `(p=${s.jb_pvalue.toFixed(4)})`}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[var(--text-dim)]">Conclusion</span>
              <span style={{ color: s.is_normal ? '#00e676' : '#ff5252' }}>
                {s.is_normal ? 'Cannot reject normality' : 'Reject normality (non-Gaussian)'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-[var(--text-dim)]">Skewness</span>
              <span className={Math.abs(s.skew) > 1 ? 'text-[#ffab40]' : 'text-[var(--text-secondary)]'}>
                {s.skew > 0.5 ? 'Right-skewed' : s.skew < -0.5 ? 'Left-skewed' : 'Roughly symmetric'} ({s.skew.toFixed(3)})
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-[var(--text-dim)]">Kurtosis</span>
              <span className={Math.abs(s.kurtosis) > 3 ? 'text-[#ffab40]' : 'text-[var(--text-secondary)]'}>
                {s.kurtosis > 1 ? 'Leptokurtic (fat tails)' : s.kurtosis < -1 ? 'Platykurtic (thin tails)' : 'Mesokurtic'} ({s.kurtosis.toFixed(3)})
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function PercentileBar({ stats: s }: { stats: DataExplorationResult['descriptive_stats'] }) {
  const range = s.max - s.min || 1
  const pct = (v: number) => ((v - s.min) / range) * 100

  const markers = [
    { label: 'P5', value: s.p5, color: '#ff5252' },
    { label: 'P25', value: s.p25, color: '#ffab40' },
    { label: 'Med', value: s.median, color: '#c084fc' },
    { label: 'μ', value: s.mean, color: '#64b5f6' },
    { label: 'P75', value: s.p75, color: '#ffab40' },
    { label: 'P95', value: s.p95, color: '#ff5252' },
  ]

  return (
    <div className="relative h-16 mt-4">
      {/* Track */}
      <div className="absolute left-0 right-0 top-6 h-2 rounded-full bg-[var(--bg-elevated)]" />
      {/* IQR box */}
      <div
        className="absolute top-4 h-6 rounded border border-[rgba(168,85,247,0.4)]"
        style={{
          left: `${pct(s.p25)}%`,
          width: `${pct(s.p75) - pct(s.p25)}%`,
          backgroundColor: 'rgba(168,85,247,0.1)',
        }}
      />
      {/* Markers */}
      {markers.map(m => (
        <div key={m.label} className="absolute" style={{ left: `${pct(m.value)}%`, transform: 'translateX(-50%)' }}>
          <div className="w-px h-4 mx-auto" style={{ backgroundColor: m.color }} />
          <div className="text-[8px] font-mono text-center mt-0.5" style={{ color: m.color }}>{m.label}</div>
          <div className="text-[7px] font-mono text-center text-[var(--text-dim)]">{m.value.toFixed(2)}</div>
        </div>
      ))}
    </div>
  )
}


/* ═══════════════════════════════════════════
   4. AUTOCORRELATION PANEL
   ═══════════════════════════════════════════ */
function ACFPanel({ data }: { data: DataExplorationResult }) {
  const acf = data.autocorrelation
  const conf = data.acf_confidence_bound ?? 0
  const st = data.stationarity

  if (acf.length === 0) return <Empty msg="Not enough data for ACF" />

  const maxAcf = Math.max(...acf.map(a => Math.abs(a.acf)), conf, 0.3)
  const barW = Math.min(30, 600 / acf.length)

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
      {/* ACF Bar Chart */}
      <div className="lg:col-span-2 rounded-xl border border-[var(--border)] p-5" style={{ backgroundColor: 'var(--bg-card)' }}>
        <h3 className="text-xs font-mono uppercase tracking-wider text-[var(--text-dim)] mb-4">
          Autocorrelation Function (ACF)
        </h3>
        <div className="relative" style={{ height: 200 }}>
          {/* Zero line */}
          <div className="absolute left-0 right-0 top-1/2 h-px bg-[var(--border)]" />
          {/* Confidence bounds */}
          <div
            className="absolute left-0 right-0 border-t border-b border-dashed border-[rgba(255,171,64,0.3)]"
            style={{
              top: `${50 - (conf / maxAcf) * 50}%`,
              bottom: `${50 - (conf / maxAcf) * 50}%`,
            }}
          />

          {/* Bars */}
          <div className="absolute inset-0 flex items-center justify-center gap-px">
            {acf.map(a => {
              const height = Math.abs(a.acf) / maxAcf * 90
              const isUp = a.acf >= 0
              return (
                <div key={a.lag} className="flex flex-col items-center" style={{ width: barW }}>
                  <div className="relative" style={{ height: 180 }}>
                    <div
                      className="absolute rounded-sm transition-all"
                      style={{
                        width: Math.max(barW - 4, 3),
                        height: `${height}%`,
                        backgroundColor: a.significant ? (a.acf > 0 ? '#00e676' : '#ff5252') : 'rgba(168,85,247,0.3)',
                        [isUp ? 'bottom' : 'top']: '50%',
                        left: '50%',
                        transform: 'translateX(-50%)',
                      }}
                    />
                  </div>
                  <span className="text-[7px] font-mono text-[var(--text-dim)] mt-0.5">{a.lag}</span>
                </div>
              )
            })}
          </div>
        </div>
        <div className="mt-2 text-[9px] font-mono text-[var(--text-muted)] text-center">
          95% confidence bound: ±{conf.toFixed(4)} · Colored bars = significant
        </div>
      </div>

      {/* Stationarity Summary */}
      <div className="stat-card !p-4">
        <h4 className="text-[10px] font-mono text-[var(--text-dim)] uppercase tracking-wider mb-3">Stationarity Test (ADF)</h4>
        {st ? (
          <div className="space-y-2 text-[11px] font-mono">
            <div className="text-center py-2 rounded-lg mb-3" style={{
              backgroundColor: st.is_stationary_5pct ? 'rgba(0,230,118,0.1)' : 'rgba(255,82,82,0.1)',
              color: st.is_stationary_5pct ? '#00e676' : '#ff5252',
            }}>
              <div className="text-sm font-bold">{st.is_stationary_5pct ? 'STATIONARY' : 'NON-STATIONARY'}</div>
              <div className="text-[9px] mt-0.5">p = {st.p_value.toFixed(4)}</div>
            </div>
            <StatRow label="ADF Stat" value={fmtNum(st.adf_statistic)} />
            <StatRow label="Critical 1%" value={fmtNum(st.critical_1pct)} />
            <StatRow label="Critical 5%" value={fmtNum(st.critical_5pct)} />
            <StatRow label="Critical 10%" value={fmtNum(st.critical_10pct)} />
            <StatRow label="Lags Used" value={String(st.lags_used)} />
            <StatRow label="Observations" value={String(st.n_obs)} />
          </div>
        ) : (
          <div className="text-[var(--text-dim)] text-xs font-mono">ADF test unavailable</div>
        )}

        {/* ACF summary */}
        <div className="mt-4 pt-3 border-t border-[var(--border-subtle)]">
          <h4 className="text-[10px] font-mono text-[var(--text-dim)] uppercase tracking-wider mb-2">ACF Highlights</h4>
          <div className="space-y-1 text-[10px] font-mono">
            <div className="flex justify-between">
              <span className="text-[var(--text-dim)]">Lag-1</span>
              <span style={{ color: acf[0]?.significant ? '#ffab40' : 'var(--text-secondary)' }}>
                {acf[0]?.acf.toFixed(3) ?? '—'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-[var(--text-dim)]">Significant lags</span>
              <span className="text-[var(--text-secondary)]">{acf.filter(a => a.significant).length} / {acf.length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[var(--text-dim)]">Max |ACF|</span>
              <span className="text-[var(--text-secondary)]">
                {Math.max(...acf.map(a => Math.abs(a.acf))).toFixed(3)} (lag {acf.reduce((m, a) => Math.abs(a.acf) > Math.abs(m.acf) ? a : m, acf[0]).lag})
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}


/* ═══════════════════════════════════════════
   5. CORRELATION PANEL (Feature vs PnL)
   ═══════════════════════════════════════════ */
function CorrelationPanel({ data }: { data: DataExplorationResult }) {
  const pnlMap = data.feature_vs_pnl
  const strats = Object.keys(pnlMap)
  const [activePnl, setActivePnl] = useState(strats[0] ?? 'combined')

  const active = pnlMap[activePnl]

  const STRAT_LABELS: Record<string, string> = { dm: 'DM', wc: 'WC', orion: 'Orion', dmo: 'DMO', combined: 'Combined' }

  return (
    <div className="space-y-4">
      {/* Correlation Summary Table */}
      <div className="rounded-xl border border-[var(--border)] p-5" style={{ backgroundColor: 'var(--bg-card)' }}>
        <h3 className="text-xs font-mono uppercase tracking-wider text-[var(--text-dim)] mb-4">
          {data.label} vs Strategy PnL — Correlation Matrix
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-[11px] font-mono">
            <thead>
              <tr className="border-b border-[var(--border)]">
                <th className="text-left px-3 py-2 text-[var(--text-dim)] font-normal">Strategy</th>
                <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Pearson r</th>
                <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">p-value</th>
                <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Spearman ρ</th>
                <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">p-value</th>
                <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Sig</th>
              </tr>
            </thead>
            <tbody>
              {strats.map(s => {
                const d = pnlMap[s]
                return (
                  <tr key={s} className="border-b border-[var(--border-subtle)] hover:bg-[var(--bg-hover)] transition-colors cursor-pointer" onClick={() => setActivePnl(s)}>
                    <td className="px-3 py-2 text-[var(--text-secondary)] font-semibold">{STRAT_LABELS[s] ?? s}</td>
                    <td className="text-right px-3 py-2 font-semibold" style={{ color: corrColor(d.pearson_r) }}>{d.pearson_r.toFixed(3)}</td>
                    <td className="text-right px-3 py-2 text-[var(--text-muted)]">{d.pearson_p.toFixed(4)}</td>
                    <td className="text-right px-3 py-2 font-semibold" style={{ color: corrColor(d.spearman_r) }}>{d.spearman_r.toFixed(3)}</td>
                    <td className="text-right px-3 py-2 text-[var(--text-muted)]">{d.spearman_p.toFixed(4)}</td>
                    <td className="text-right px-3 py-2 text-[#ffab40]">{sigBadge(d.pearson_p)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Scatter Plot for Selected Strategy */}
      {active && (
        <div className="rounded-xl border border-[var(--border)] p-5" style={{ backgroundColor: 'var(--bg-card)' }}>
          <h3 className="text-xs font-mono uppercase tracking-wider text-[var(--text-dim)] mb-4">
            Scatter: {data.label} vs {STRAT_LABELS[activePnl] ?? activePnl} PnL
            <span className="ml-2 text-[var(--text-muted)]">(r={active.pearson_r.toFixed(3)}, ρ={active.spearman_r.toFixed(3)})</span>
          </h3>
          <ScatterChart scatter={active.scatter} xLabel={data.label} yLabel="PnL %" />
        </div>
      )}
    </div>
  )
}


/* ═══════════════════════════════════════════
   6. QUINTILE PANEL
   ═══════════════════════════════════════════ */
function QuintilePanel({ data }: { data: DataExplorationResult }) {
  const qa = data.quintile_analysis
  const strats = Object.keys(qa)
  const [activeStrat, setActiveStrat] = useState(strats.includes('combined') ? 'combined' : strats[0] ?? '')

  const STRAT_LABELS: Record<string, string> = { dm: 'DM', wc: 'WC', orion: 'Orion', dmo: 'DMO', combined: 'Combined' }

  const activeEntry = qa[activeStrat]
  const buckets = activeEntry ? (Array.isArray(activeEntry) ? activeEntry : activeEntry.buckets) : []
  const maxPnlAbs = Math.max(...buckets.map(b => Math.abs(b.pnl_mean ?? 0)), 0.01)

  return (
    <div className="space-y-4">
      {/* Strategy selector */}
      <div className="flex items-center gap-2">
        {strats.map(s => (
          <button
            key={s}
            onClick={() => setActiveStrat(s)}
            className="px-3 py-1.5 rounded-lg text-[10px] font-mono font-semibold transition-all"
            style={{
              backgroundColor: activeStrat === s ? 'rgba(168,85,247,0.15)' : 'transparent',
              color: activeStrat === s ? '#c084fc' : 'var(--text-muted)',
              border: `1px solid ${activeStrat === s ? 'rgba(168,85,247,0.3)' : 'var(--border)'}`,
            }}
          >
            {STRAT_LABELS[s] ?? s}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Quintile Bar Chart */}
        <div className="rounded-xl border border-[var(--border)] p-5" style={{ backgroundColor: 'var(--bg-card)' }}>
          <h3 className="text-xs font-mono uppercase tracking-wider text-[var(--text-dim)] mb-4">
            Mean PnL by {data.label} Quintile
          </h3>
          <div className="space-y-2">
            {buckets.map(b => {
              const pm = b.pnl_mean ?? 0
              return (
                <div key={b.quintile} className="flex items-center gap-3">
                  <div className="w-20 text-[10px] font-mono text-[var(--text-muted)] flex-shrink-0">{b.quintile}</div>
                  <div className="flex-1 h-6 relative">
                    <div className="absolute inset-0 flex items-center">
                      <div className="absolute left-1/2 w-px h-full bg-[var(--border)]" />
                      <div
                        className="absolute h-5 rounded transition-all"
                        style={{
                          left: pm < 0 ? `${50 - (Math.abs(pm) / maxPnlAbs) * 50}%` : '50%',
                          width: `${(Math.abs(pm) / maxPnlAbs) * 50}%`,
                          backgroundColor: pm >= 0 ? 'rgba(0,230,118,0.5)' : 'rgba(255,82,82,0.5)',
                        }}
                      />
                    </div>
                    <div className="absolute inset-0 flex items-center justify-center text-[9px] font-mono font-bold" style={{ color: signColor(b.pnl_mean) }}>
                      {b.pnl_mean != null ? `${pm > 0 ? '+' : ''}${pm.toFixed(3)}%` : '—'}
                    </div>
                  </div>
                  <div className="w-16 text-right text-[9px] font-mono text-[var(--text-muted)]">{b.count}d</div>
                </div>
              )
            })}
          </div>
        </div>

        {/* Quintile Detail Table */}
        <div className="rounded-xl border border-[var(--border)] overflow-hidden" style={{ backgroundColor: 'var(--bg-card)' }}>
          <div className="overflow-x-auto">
            <table className="w-full text-[10px] font-mono">
              <thead>
                <tr className="border-b border-[var(--border)] bg-[var(--bg-elevated)]">
                  <th className="text-left px-3 py-2 text-[var(--text-dim)] font-normal">Quintile</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Feat μ</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">PnL μ</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">PnL med</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Win%</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Sharpe</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Total%</th>
                </tr>
              </thead>
              <tbody>
                {buckets.map(b => (
                  <tr key={b.quintile} className="border-b border-[var(--border-subtle)]">
                    <td className="px-3 py-2 text-[var(--text-secondary)] font-semibold">{b.quintile}</td>
                    <td className="text-right px-3 py-2 text-[var(--text-secondary)]">{b.feature_mean.toFixed(3)}</td>
                    <td className="text-right px-3 py-2 font-semibold" style={{ color: signColor(b.pnl_mean) }}>{b.pnl_mean != null ? b.pnl_mean.toFixed(3) : '—'}</td>
                    <td className="text-right px-3 py-2" style={{ color: signColor(b.pnl_median) }}>{b.pnl_median != null ? b.pnl_median.toFixed(3) : '—'}</td>
                    <td className="text-right px-3 py-2" style={{ color: (b.win_rate ?? 0) >= 0.5 ? '#00e676' : '#ff5252' }}>
                      {b.win_rate != null ? `${(b.win_rate * 100).toFixed(1)}%` : '—'}
                    </td>
                    <td className="text-right px-3 py-2" style={{ color: signColor(b.sharpe) }}>{fmtNum(b.sharpe, 2)}</td>
                    <td className="text-right px-3 py-2" style={{ color: signColor(b.pnl_sum) }}>{b.pnl_sum != null ? b.pnl_sum.toFixed(2) : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}


/* ═══════════════════════════════════════════
   7. REGIME CONDITIONAL PANEL
   ═══════════════════════════════════════════ */
function RegimePanel({ data }: { data: DataExplorationResult }) {
  const rd = data.regime_distributions
  const active = rd.filter(r => r.count >= 3 && r.mean != null)

  if (active.length === 0) return <Empty msg="Not enough data for regime-conditional analysis" />

  const maxMean = Math.max(...active.map(r => Math.abs(r.mean!)), 0.01)

  return (
    <div className="space-y-4">
      {/* Diverging bar chart of means */}
      <div className="rounded-xl border border-[var(--border)] p-5" style={{ backgroundColor: 'var(--bg-card)' }}>
        <h3 className="text-xs font-mono uppercase tracking-wider text-[var(--text-dim)] mb-4">
          {data.label} Mean by Regime State
        </h3>
        <div className="space-y-2">
          {active.sort((a, b) => (a.mean ?? 0) - (b.mean ?? 0)).map(r => (
            <div key={r.state} className="flex items-center gap-3 group">
              <div className="w-28 flex items-center gap-1.5 flex-shrink-0">
                <div className="w-2 h-2 rounded-full" style={{ backgroundColor: r.color }} />
                <span className="text-[10px] font-mono font-semibold" style={{ color: r.color }}>{r.state}</span>
              </div>
              <div className="flex-1 h-6 relative">
                <div className="absolute left-1/2 w-px h-full bg-[var(--border)]" />
                <div
                  className="absolute h-4 rounded top-1 transition-all"
                  style={{
                    left: r.mean! < 0 ? `${50 - (Math.abs(r.mean!) / maxMean) * 50}%` : '50%',
                    width: `${(Math.abs(r.mean!) / maxMean) * 50}%`,
                    backgroundColor: `${r.color}66`,
                  }}
                />
                <div className="absolute inset-0 flex items-center justify-center text-[9px] font-mono font-bold" style={{ color: r.color }}>
                  {r.mean!.toFixed(4)}
                </div>
              </div>
              <div className="w-12 text-right text-[9px] font-mono text-[var(--text-muted)]">{r.count}d</div>
            </div>
          ))}
        </div>
      </div>

      {/* Regime detail table */}
      <div className="rounded-xl border border-[var(--border)] overflow-hidden" style={{ backgroundColor: 'var(--bg-card)' }}>
        <table className="w-full text-[10px] font-mono">
          <thead>
            <tr className="border-b border-[var(--border)] bg-[var(--bg-elevated)]">
              <th className="text-left px-3 py-2 text-[var(--text-dim)] font-normal">State</th>
              <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Count</th>
              <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Mean</th>
              <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Median</th>
              <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Std</th>
              <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Min</th>
              <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Max</th>
              <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">P25</th>
              <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">P75</th>
            </tr>
          </thead>
          <tbody>
            {active.sort((a, b) => a.state.localeCompare(b.state)).map(r => (
              <tr key={r.state} className="border-b border-[var(--border-subtle)] hover:bg-[var(--bg-hover)]">
                <td className="px-3 py-2">
                  <div className="flex items-center gap-1">
                    <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: r.color }} />
                    <span style={{ color: r.color }}>{r.state}</span>
                  </div>
                </td>
                <td className="text-right px-3 py-2 text-[var(--text-secondary)]">{r.count}</td>
                <td className="text-right px-3 py-2 font-semibold" style={{ color: r.color }}>{fmtNum(r.mean, 4)}</td>
                <td className="text-right px-3 py-2 text-[var(--text-secondary)]">{fmtNum(r.median, 4)}</td>
                <td className="text-right px-3 py-2 text-[var(--text-secondary)]">{fmtNum(r.std, 4)}</td>
                <td className="text-right px-3 py-2 text-[var(--text-muted)]">{fmtNum(r.min, 4)}</td>
                <td className="text-right px-3 py-2 text-[var(--text-muted)]">{fmtNum(r.max, 4)}</td>
                <td className="text-right px-3 py-2 text-[var(--text-muted)]">{fmtNum(r.p25, 4)}</td>
                <td className="text-right px-3 py-2 text-[var(--text-muted)]">{fmtNum(r.p75, 4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}


/* ═══════════════════════════════════════════
   8. OUTLIER PANEL
   ═══════════════════════════════════════════ */
function OutlierPanel({ data }: { data: DataExplorationResult }) {
  const o = data.outliers

  return (
    <div className="space-y-4">
      {/* Summary cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <div className="stat-card !p-3">
          <div className="text-[9px] font-mono text-[var(--text-dim)] uppercase tracking-wider mb-1">2.5σ Outliers</div>
          <div className="text-lg font-bold text-[#ffab40] font-mono">{o.sigma_count}</div>
          <div className="text-[9px] text-[var(--text-muted)] font-mono mt-0.5">
            bounds: [{o.sigma_bounds.lower.toFixed(3)}, {o.sigma_bounds.upper.toFixed(3)}]
          </div>
        </div>
        <div className="stat-card !p-3">
          <div className="text-[9px] font-mono text-[var(--text-dim)] uppercase tracking-wider mb-1">IQR Outliers</div>
          <div className="text-lg font-bold text-[#ffab40] font-mono">{o.iqr_count}</div>
          <div className="text-[9px] text-[var(--text-muted)] font-mono mt-0.5">
            bounds: [{o.iqr_bounds.lower.toFixed(3)}, {o.iqr_bounds.upper.toFixed(3)}]
          </div>
        </div>
        <div className="stat-card !p-3">
          <div className="text-[9px] font-mono text-[var(--text-dim)] uppercase tracking-wider mb-1">σ Threshold</div>
          <div className="text-lg font-bold text-[var(--text-secondary)] font-mono">{o.sigma_threshold}</div>
        </div>
        <div className="stat-card !p-3">
          <div className="text-[9px] font-mono text-[var(--text-dim)] uppercase tracking-wider mb-1">Union Count</div>
          <div className="text-lg font-bold text-[#ff5252] font-mono">{o.days.length}</div>
          <div className="text-[9px] text-[var(--text-muted)] font-mono mt-0.5">σ ∪ IQR (shown below)</div>
        </div>
      </div>

      {/* Outlier day table */}
      {o.days.length > 0 && (
        <div className="rounded-xl border border-[var(--border)] overflow-hidden" style={{ backgroundColor: 'var(--bg-card)' }}>
          <div className="px-4 py-3 border-b border-[var(--border)] bg-[var(--bg-elevated)]">
            <span className="text-[11px] font-mono font-semibold text-[var(--text-secondary)]">
              Outlier Days — {o.days.length} records (capped at 100)
            </span>
          </div>
          <div className="overflow-x-auto max-h-[400px] overflow-y-auto">
            <table className="w-full text-[10px] font-mono">
              <thead className="sticky top-0 bg-[var(--bg-card)]">
                <tr className="border-b border-[var(--border)]">
                  <th className="text-left px-3 py-2 text-[var(--text-dim)] font-normal">Date</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Value</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Z-Score</th>
                  <th className="text-left px-3 py-2 text-[var(--text-dim)] font-normal">Regime</th>
                </tr>
              </thead>
              <tbody>
                {o.days.map(d => (
                  <tr key={d.date} className="border-b border-[var(--border-subtle)] hover:bg-[var(--bg-hover)]">
                    <td className="px-3 py-1.5 text-[var(--text-secondary)]">{d.date}</td>
                    <td className="text-right px-3 py-1.5 font-semibold text-[#ffab40]">{d.value.toFixed(4)}</td>
                    <td className="text-right px-3 py-1.5" style={{ color: Math.abs(d.z_score ?? 0) > 3 ? '#ff1744' : '#ff5252' }}>
                      {d.z_score?.toFixed(2) ?? '—'}
                    </td>
                    <td className="px-3 py-1.5 text-[var(--text-muted)]">{d.regime_state ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}


/* ═══════════════════════════════════════════
   Shared components
   ═══════════════════════════════════════════ */
function Empty({ msg }: { msg: string }) {
  return <div className="text-center py-16 text-[var(--text-dim)] text-xs font-mono">{msg}</div>
}

function ScatterChart({ scatter, xLabel, yLabel }: { scatter: EDAScatterPoint[]; xLabel: string; yLabel: string }) {
  if (scatter.length === 0) return <Empty msg="No scatter data" />

  const xs = scatter.map(p => p.x)
  const ys = scatter.map(p => p.y)
  const xMin = Math.min(...xs), xMax = Math.max(...xs)
  const yMin = Math.min(...ys), yMax = Math.max(...ys)
  const xRange = xMax - xMin || 1
  const yRange = yMax - yMin || 1

  const H = 260

  return (
    <div className="relative" style={{ height: H }}>
      <div className="absolute left-0 top-0 bottom-0 w-12 flex flex-col justify-between text-[8px] font-mono text-[var(--text-dim)]">
        <span>{yMax.toFixed(2)}</span>
        <span>{((yMax + yMin) / 2).toFixed(2)}</span>
        <span>{yMin.toFixed(2)}</span>
      </div>
      <div className="absolute bottom-0 left-12 right-0 flex justify-between text-[8px] font-mono text-[var(--text-dim)]" style={{ transform: 'translateY(14px)' }}>
        <span>{xMin.toFixed(2)}</span>
        <span>{xLabel}</span>
        <span>{xMax.toFixed(2)}</span>
      </div>
      <div className="absolute left-12 right-0 top-0 rounded-lg border border-[var(--border-subtle)]" style={{ height: H, backgroundColor: 'var(--bg-elevated)' }}>
        {/* Zero lines */}
        {xMin < 0 && xMax > 0 && (
          <div className="absolute top-0 bottom-0 w-px bg-[var(--border)]" style={{ left: `${((0 - xMin) / xRange) * 100}%` }} />
        )}
        {yMin < 0 && yMax > 0 && (
          <div className="absolute left-0 right-0 h-px bg-[var(--border)]" style={{ top: `${((yMax - 0) / yRange) * 100}%` }} />
        )}
        {scatter.map((p, i) => {
          const x = ((p.x - xMin) / xRange) * 100
          const y = ((yMax - p.y) / yRange) * 100
          return (
            <div
              key={i}
              className="absolute w-1.5 h-1.5 rounded-full bg-[#c084fc] opacity-50 hover:opacity-100 hover:w-2.5 hover:h-2.5 transition-all hover:z-10"
              style={{ left: `calc(${x}% - 3px)`, top: `calc(${y}% - 3px)` }}
              title={`${xLabel}: ${p.x.toFixed(4)}\n${yLabel}: ${p.y.toFixed(4)}`}
            />
          )
        })}
      </div>
    </div>
  )
}
