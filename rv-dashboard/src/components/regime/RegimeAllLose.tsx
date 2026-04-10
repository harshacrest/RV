'use client'

import { useState, useEffect, useMemo } from 'react'
import { fetchRegimeAllLose } from '@/lib/api'
import type { AllLoseData, AllLoseDayRecord } from '@/lib/types'

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

type ViewMode = 'summary' | 'scatter' | 'table'

function spotColor(v: number | null): string {
  if (v == null) return 'var(--text-dim)'
  if (v <= -1.5) return '#ff1744'
  if (v <= -0.5) return '#ff5252'
  if (v < 0) return '#ff8a80'
  if (v === 0) return 'var(--text-dim)'
  if (v < 0.5) return '#b9f6ca'
  if (v < 1.5) return '#69f0ae'
  return '#00e676'
}

function pnlColor(v: number | null): string {
  if (v == null) return 'var(--text-dim)'
  return v >= 0 ? 'var(--signal-positive)' : 'var(--signal-negative)'
}

function barWidthPct(value: number, maxAbsValue: number): number {
  if (maxAbsValue === 0) return 0
  return Math.min(Math.abs(value) / maxAbsValue * 100, 100)
}

export function RegimeAllLose({ startDate, endDate, snapshot, dte }: Props) {
  const [data, setData] = useState<AllLoseData | null>(null)
  const [loading, setLoading] = useState(true)
  const [mode, setMode] = useState<ViewMode>('summary')
  const [selectedState, setSelectedState] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    fetchRegimeAllLose(startDate, endDate, snapshot, dte)
      .then(setData)
      .finally(() => setLoading(false))
  }, [startDate, endDate, snapshot, dte])

  const filteredDays = useMemo(() => {
    if (!data) return []
    if (!selectedState) return data.days
    return data.days.filter(d => d.regime_state === selectedState)
  }, [data, selectedState])

  const maxSpotAbs = useMemo(() => {
    if (!data) return 3
    const vals = data.states
      .filter(s => s.al_days > 0)
      .flatMap(s => [Math.abs(s.spot_chg_mean ?? 0), Math.abs(s.spot_chg_min ?? 0), Math.abs(s.spot_chg_max ?? 0)])
    return Math.max(...vals, 1)
  }, [data])

  if (loading) return <div className="text-center py-20 text-[var(--text-dim)] text-sm font-mono">Loading all-lose analysis...</div>
  if (!data) return null

  const { states, distribution, overall } = data
  const activeStates = states.filter(s => s.al_days > 0)

  return (
    <div className="space-y-6">
      {/* Overall Hero Card */}
      <div className="rounded-xl border border-[var(--border)] overflow-hidden" style={{ backgroundColor: 'var(--bg-card)' }}>
        <div className="px-6 py-5">
          <div className="text-[10px] font-mono text-[var(--text-dim)] uppercase tracking-widest mb-3">All-Lose Day Spot Analysis</div>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="stat-card !p-3">
              <div className="text-[9px] font-mono text-[var(--text-dim)] uppercase tracking-wider mb-1">All-Lose Days</div>
              <div className="text-lg font-bold text-[var(--signal-negative)] font-mono">{overall.total_al_days}</div>
              <div className="text-[9px] text-[var(--text-muted)] font-mono mt-0.5">
                {overall.al_pct}% of {overall.total_trading_days} trading days
              </div>
            </div>
            <div className="stat-card !p-3">
              <div className="text-[9px] font-mono text-[var(--text-dim)] uppercase tracking-wider mb-1">Mean Spot Move</div>
              <div className="text-lg font-bold font-mono" style={{ color: spotColor(overall.spot_chg_mean) }}>
                {overall.spot_chg_mean != null ? `${overall.spot_chg_mean > 0 ? '+' : ''}${overall.spot_chg_mean.toFixed(2)}%` : '—'}
              </div>
              <div className="text-[9px] text-[var(--text-muted)] font-mono mt-0.5">
                σ = {overall.spot_chg_std?.toFixed(2) ?? '—'}%
              </div>
            </div>
            <div className="stat-card !p-3">
              <div className="text-[9px] font-mono text-[var(--text-dim)] uppercase tracking-wider mb-1">Median Spot Move</div>
              <div className="text-lg font-bold font-mono" style={{ color: spotColor(overall.spot_chg_median) }}>
                {overall.spot_chg_median != null ? `${overall.spot_chg_median > 0 ? '+' : ''}${overall.spot_chg_median.toFixed(2)}%` : '—'}
              </div>
            </div>
            <div className="stat-card !p-3">
              <div className="text-[9px] font-mono text-[var(--text-dim)] uppercase tracking-wider mb-1">Spot Direction</div>
              <div className="text-sm font-bold font-mono">
                <span style={{ color: 'var(--signal-negative)' }}>↓ {overall.spot_down_pct ?? 0}%</span>
                <span className="text-[var(--text-dim)] mx-1">/</span>
                <span style={{ color: 'var(--signal-positive)' }}>↑ {overall.spot_up_pct ?? 0}%</span>
              </div>
              <div className="text-[9px] text-[var(--text-muted)] font-mono mt-0.5">down / up split</div>
            </div>
          </div>
        </div>
      </div>

      {/* Mode Toggle + State Filter */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {([
            { key: 'summary', label: 'By Regime' },
            { key: 'scatter', label: 'Distribution' },
            { key: 'table', label: 'Day Log' },
          ] as { key: ViewMode; label: string }[]).map(v => (
            <button
              key={v.key}
              onClick={() => setMode(v.key)}
              className="px-3 py-1.5 rounded-lg text-[10px] font-mono font-semibold transition-all"
              style={{
                backgroundColor: mode === v.key ? 'rgba(168,85,247,0.15)' : 'transparent',
                color: mode === v.key ? '#c084fc' : 'var(--text-muted)',
                border: `1px solid ${mode === v.key ? 'rgba(168,85,247,0.3)' : 'var(--border)'}`,
              }}
            >
              {v.label}
            </button>
          ))}
        </div>

        {/* State filter chips */}
        {(mode === 'scatter' || mode === 'table') && (
          <div className="flex items-center gap-1">
            <button
              onClick={() => setSelectedState(null)}
              className="px-2 py-1 rounded text-[9px] font-mono font-semibold transition-all"
              style={{
                backgroundColor: selectedState === null ? 'rgba(168,85,247,0.15)' : 'transparent',
                color: selectedState === null ? '#c084fc' : 'var(--text-muted)',
                border: `1px solid ${selectedState === null ? 'rgba(168,85,247,0.3)' : 'var(--border)'}`,
              }}
            >
              ALL
            </button>
            {activeStates.map(s => (
              <button
                key={s.state}
                onClick={() => setSelectedState(selectedState === s.state ? null : s.state)}
                className="px-2 py-1 rounded text-[9px] font-mono font-semibold transition-all"
                style={{
                  backgroundColor: selectedState === s.state ? `${s.color}22` : 'transparent',
                  color: selectedState === s.state ? s.color : 'var(--text-muted)',
                  border: `1px solid ${selectedState === s.state ? `${s.color}44` : 'var(--border)'}`,
                }}
              >
                {s.state.replace('L1 ', 'L1').replace('L2 ', 'L2').replace('L3 ', 'L3').replace('Caution-', 'C')}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* === SUMMARY VIEW === */}
      {mode === 'summary' && (
        <>
          {/* Spot Movement by Regime — Horizontal Bars */}
          <div className="rounded-xl border border-[var(--border)] p-5" style={{ backgroundColor: 'var(--bg-card)' }}>
            <h3 className="text-xs font-mono uppercase tracking-wider text-[var(--text-dim)] mb-4">
              Spot Movement on All-Lose Days — By Regime State
            </h3>
            <div className="space-y-3">
              {activeStates
                .sort((a, b) => (a.spot_chg_mean ?? 0) - (b.spot_chg_mean ?? 0))
                .map(s => {
                  const mean = s.spot_chg_mean ?? 0
                  const isNeg = mean < 0
                  return (
                    <div key={s.state} className="group">
                      <div className="flex items-center gap-3">
                        <div className="w-28 flex-shrink-0 flex items-center gap-1.5">
                          <div className="w-2 h-2 rounded-full" style={{ backgroundColor: s.color }} />
                          <span className="text-[10px] font-mono font-semibold" style={{ color: s.color }}>{s.state}</span>
                        </div>

                        {/* Diverging bar — centered at 0 */}
                        <div className="flex-1 h-7 relative">
                          <div className="absolute inset-0 flex items-center">
                            {/* Center line */}
                            <div className="absolute left-1/2 w-px h-full bg-[var(--border)]" />

                            {/* Bar */}
                            <div
                              className="absolute h-5 rounded transition-all duration-500"
                              style={{
                                left: isNeg ? `${50 - barWidthPct(mean, maxSpotAbs) / 2}%` : '50%',
                                width: `${barWidthPct(mean, maxSpotAbs) / 2}%`,
                                backgroundColor: isNeg ? 'rgba(255,82,82,0.5)' : 'rgba(0,230,118,0.5)',
                              }}
                            />
                          </div>

                          {/* Value label */}
                          <div className="absolute inset-0 flex items-center justify-center">
                            <span className="text-[10px] font-mono font-bold" style={{ color: spotColor(mean) }}>
                              {mean > 0 ? '+' : ''}{mean.toFixed(3)}%
                            </span>
                          </div>
                        </div>

                        <div className="w-24 text-right flex-shrink-0">
                          <span className="text-[9px] font-mono text-[var(--text-muted)]">
                            {s.al_days}d ({s.al_pct}%)
                          </span>
                        </div>
                      </div>

                      {/* Expanded detail row on hover */}
                      <div className="h-0 group-hover:h-8 overflow-hidden transition-all duration-200">
                        <div className="flex items-center gap-4 mt-1 pl-[124px] text-[9px] font-mono text-[var(--text-muted)]">
                          <span>range: [{s.spot_chg_min?.toFixed(2)}, {s.spot_chg_max?.toFixed(2)}]%</span>
                          <span>p25: {s.spot_chg_p25?.toFixed(2)}%</span>
                          <span>p75: {s.spot_chg_p75?.toFixed(2)}%</span>
                          <span>σ: {s.spot_chg_std?.toFixed(2)}%</span>
                          <span>intraday: {s.intraday_range_mean?.toFixed(2)}%</span>
                        </div>
                      </div>
                    </div>
                  )
                })}
            </div>
          </div>

          {/* Regime Detail Cards */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            {activeStates.map(s => (
              <div key={s.state} className="stat-card !p-3">
                <div className="flex items-center gap-1.5 mb-2">
                  <div className="w-2 h-2 rounded-full" style={{ backgroundColor: s.color }} />
                  <span className="text-[10px] font-mono font-semibold" style={{ color: s.color }}>{s.state}</span>
                </div>
                <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-[10px] font-mono">
                  <div className="text-[var(--text-dim)]">AL Days</div>
                  <div className="text-right text-[var(--signal-negative)] font-semibold">{s.al_days} ({s.al_pct}%)</div>
                  <div className="text-[var(--text-dim)]">Mean Spot</div>
                  <div className="text-right font-semibold" style={{ color: spotColor(s.spot_chg_mean) }}>
                    {s.spot_chg_mean != null ? `${s.spot_chg_mean > 0 ? '+' : ''}${s.spot_chg_mean.toFixed(3)}%` : '—'}
                  </div>
                  <div className="text-[var(--text-dim)]">Median</div>
                  <div className="text-right" style={{ color: spotColor(s.spot_chg_median) }}>
                    {s.spot_chg_median != null ? `${s.spot_chg_median > 0 ? '+' : ''}${s.spot_chg_median.toFixed(3)}%` : '—'}
                  </div>
                  <div className="text-[var(--text-dim)]">Range</div>
                  <div className="text-right text-[var(--text-secondary)]">
                    {s.intraday_range_mean?.toFixed(2) ?? '—'}%
                  </div>
                  <div className="text-[var(--text-dim)]">Gap</div>
                  <div className="text-right" style={{ color: spotColor(s.gap_mean) }}>
                    {s.gap_mean != null ? `${s.gap_mean > 0 ? '+' : ''}${s.gap_mean.toFixed(3)}%` : '—'}
                  </div>
                  <div className="text-[var(--text-dim)]">Port PnL</div>
                  <div className="text-right font-semibold" style={{ color: pnlColor(s.pnl_combined_mean) }}>
                    {s.pnl_combined_mean != null ? `${s.pnl_combined_mean.toFixed(3)}%` : '—'}
                  </div>
                </div>

                {/* Per-strategy PnL mini bar */}
                <div className="mt-2 pt-2 border-t border-[var(--border-subtle)]">
                  <div className="flex items-center gap-2 text-[9px] font-mono">
                    <span className="text-[var(--text-dim)]">DM</span>
                    <span style={{ color: pnlColor(s.pnl_dm_mean) }}>{s.pnl_dm_mean?.toFixed(3) ?? '—'}</span>
                    <span className="text-[var(--text-dim)] ml-1">WC</span>
                    <span style={{ color: pnlColor(s.pnl_wc_mean) }}>{s.pnl_wc_mean?.toFixed(3) ?? '—'}</span>
                    <span className="text-[var(--text-dim)] ml-1">OR</span>
                    <span style={{ color: pnlColor(s.pnl_orion_mean) }}>{s.pnl_orion_mean?.toFixed(3) ?? '—'}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      {/* === DISTRIBUTION VIEW === */}
      {mode === 'scatter' && (
        <>
          {/* Histogram */}
          <div className="rounded-xl border border-[var(--border)] p-5" style={{ backgroundColor: 'var(--bg-card)' }}>
            <h3 className="text-xs font-mono uppercase tracking-wider text-[var(--text-dim)] mb-4">
              Spot Change Distribution — All-Lose Days
              {selectedState && <span style={{ color: STATE_COLORS[selectedState] }}> ({selectedState})</span>}
            </h3>
            <HistogramChart
              days={filteredDays}
              distribution={selectedState ? undefined : distribution}
            />
          </div>

          {/* Spot vs PnL scatter */}
          <div className="rounded-xl border border-[var(--border)] p-5" style={{ backgroundColor: 'var(--bg-card)' }}>
            <h3 className="text-xs font-mono uppercase tracking-wider text-[var(--text-dim)] mb-4">
              Spot Move vs Portfolio Loss — Each Dot = 1 All-Lose Day
            </h3>
            <ScatterPlot days={filteredDays} />
          </div>
        </>
      )}

      {/* === TABLE VIEW === */}
      {mode === 'table' && (
        <div className="rounded-xl border border-[var(--border)] overflow-hidden" style={{ backgroundColor: 'var(--bg-card)' }}>
          <div className="px-4 py-3 border-b border-[var(--border)] bg-[var(--bg-elevated)]">
            <span className="text-[11px] font-mono font-semibold text-[var(--text-secondary)]">
              All-Lose Day Log — {filteredDays.length} days
              {selectedState && <span style={{ color: STATE_COLORS[selectedState] }}> ({selectedState})</span>}
            </span>
          </div>
          <div className="overflow-x-auto max-h-[600px] overflow-y-auto">
            <table className="w-full text-[10px] font-mono">
              <thead className="sticky top-0 bg-[var(--bg-card)]">
                <tr className="border-b border-[var(--border)]">
                  <th className="text-left px-3 py-2 text-[var(--text-dim)] font-normal">Date</th>
                  <th className="text-left px-3 py-2 text-[var(--text-dim)] font-normal">Regime</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Close</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Spot Chg%</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Gap%</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Range%</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">IV</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">PK/IV</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">DM</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">WC</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Orion</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Port</th>
                </tr>
              </thead>
              <tbody>
                {filteredDays.map(d => (
                  <tr key={d.date} className="border-b border-[var(--border-subtle)] hover:bg-[var(--bg-hover)] transition-colors">
                    <td className="px-3 py-2 text-[var(--text-secondary)]">{d.date}</td>
                    <td className="px-3 py-2">
                      <div className="flex items-center gap-1">
                        <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: d.color }} />
                        <span style={{ color: d.color }}>{d.regime_state}</span>
                      </div>
                    </td>
                    <td className="text-right px-3 py-2 text-[var(--text-secondary)]">{d.close.toLocaleString()}</td>
                    <td className="text-right px-3 py-2 font-semibold" style={{ color: spotColor(d.spot_chg_pct) }}>
                      {d.spot_chg_pct != null ? `${d.spot_chg_pct > 0 ? '+' : ''}${d.spot_chg_pct.toFixed(2)}%` : '—'}
                    </td>
                    <td className="text-right px-3 py-2" style={{ color: spotColor(d.gap_pct) }}>
                      {d.gap_pct != null ? `${d.gap_pct > 0 ? '+' : ''}${d.gap_pct.toFixed(2)}%` : '—'}
                    </td>
                    <td className="text-right px-3 py-2 text-[var(--text-secondary)]">
                      {d.intraday_range_pct?.toFixed(2) ?? '—'}%
                    </td>
                    <td className="text-right px-3 py-2 text-[var(--text-secondary)]">{d.iv_lag?.toFixed(1) ?? '—'}</td>
                    <td className="text-right px-3 py-2 text-[var(--text-secondary)]">{d.pk_iv_ratio?.toFixed(3) ?? '—'}</td>
                    <td className="text-right px-3 py-2" style={{ color: pnlColor(d.pnl_dm) }}>
                      {d.pnl_dm?.toFixed(3) ?? '—'}
                    </td>
                    <td className="text-right px-3 py-2" style={{ color: pnlColor(d.pnl_wc) }}>
                      {d.pnl_wc?.toFixed(3) ?? '—'}
                    </td>
                    <td className="text-right px-3 py-2" style={{ color: pnlColor(d.pnl_orion) }}>
                      {d.pnl_orion?.toFixed(3) ?? '—'}
                    </td>
                    <td className="text-right px-3 py-2 font-semibold" style={{ color: pnlColor(d.pnl_combined) }}>
                      {d.pnl_combined?.toFixed(3) ?? '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Insight Card */}
      <div className="rounded-xl border border-[var(--border)] p-5" style={{ backgroundColor: 'var(--bg-card)' }}>
        <h3 className="text-xs font-mono uppercase tracking-wider text-[var(--text-dim)] mb-3">Key Insight</h3>
        <div className="space-y-2 text-xs font-mono text-[var(--text-secondary)]">
          <p><strong className="text-[var(--text-primary)]">All-lose days are when every strategy bleeds simultaneously</strong> — understanding the spot dynamics on these days reveals the market conditions that overwhelm the entire portfolio.</p>
          <p>Large spot moves (either direction) with high intraday range indicate gamma-eating environments. Days with small spot change but wide range suggest whipsaw — the worst for short-vol portfolios.</p>
        </div>
      </div>
    </div>
  )
}


/* ── Inline Histogram ── */
function HistogramChart({ days, distribution }: {
  days: AllLoseDayRecord[]
  distribution?: { bucket: string; count: number }[]
}) {
  const buckets = useMemo(() => {
    if (distribution) return distribution

    // Compute from filtered days
    const bins = [-999, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 999]
    const labels = ['<-3%', '-3 to -2%', '-2 to -1.5%', '-1.5 to -1%', '-1 to -0.5%',
      '-0.5 to 0%', '0 to 0.5%', '0.5 to 1%', '1 to 1.5%', '1.5 to 2%', '2 to 3%', '>3%']
    const counts = new Array(labels.length).fill(0)
    days.forEach(d => {
      const v = d.spot_chg_pct
      if (v == null) return
      for (let i = 0; i < bins.length - 1; i++) {
        if (v >= bins[i] && v < bins[i + 1]) { counts[i]++; break }
      }
    })
    return labels.map((label, i) => ({ bucket: label, count: counts[i] }))
  }, [days, distribution])

  const maxCount = Math.max(...buckets.map(b => b.count), 1)

  return (
    <div className="space-y-1">
      {buckets.map((b, i) => {
        const isNeg = i < 6
        return (
          <div key={b.bucket} className="flex items-center gap-2">
            <div className="w-24 text-right text-[9px] font-mono text-[var(--text-muted)] flex-shrink-0">{b.bucket}</div>
            <div className="flex-1 h-5 rounded overflow-hidden bg-[var(--bg-elevated)] relative">
              <div
                className="h-full rounded transition-all duration-300"
                style={{
                  width: `${(b.count / maxCount) * 100}%`,
                  backgroundColor: isNeg
                    ? `rgba(255,82,82,${0.3 + (b.count / maxCount) * 0.4})`
                    : `rgba(0,230,118,${0.3 + (b.count / maxCount) * 0.4})`,
                }}
              />
              {b.count > 0 && (
                <span className="absolute inset-0 flex items-center px-2 text-[9px] font-mono font-semibold text-white">
                  {b.count}
                </span>
              )}
            </div>
          </div>
        )
      })}
    </div>
  )
}


/* ── Inline CSS Scatter Plot ── */
function ScatterPlot({ days }: { days: AllLoseDayRecord[] }) {
  const filtered = days.filter(d => d.spot_chg_pct != null && d.pnl_combined != null)

  if (filtered.length === 0) {
    return <div className="text-center py-10 text-[var(--text-dim)] text-xs font-mono">No data to display</div>
  }

  const spotVals = filtered.map(d => d.spot_chg_pct!)
  const pnlVals = filtered.map(d => d.pnl_combined!)

  const spotMin = Math.min(...spotVals)
  const spotMax = Math.max(...spotVals)
  const pnlMin = Math.min(...pnlVals)
  const pnlMax = Math.max(...pnlVals)

  const spotRange = spotMax - spotMin || 1
  const pnlRange = pnlMax - pnlMin || 1

  const W = 100 // percentage
  const H = 240 // px

  return (
    <div className="relative" style={{ height: H }}>
      {/* Y-axis labels */}
      <div className="absolute left-0 top-0 bottom-0 w-14 flex flex-col justify-between text-[8px] font-mono text-[var(--text-dim)]">
        <span>{pnlMax.toFixed(2)}%</span>
        <span>{((pnlMax + pnlMin) / 2).toFixed(2)}%</span>
        <span>{pnlMin.toFixed(2)}%</span>
      </div>
      {/* X-axis labels */}
      <div className="absolute bottom-0 left-14 right-0 flex justify-between text-[8px] font-mono text-[var(--text-dim)]" style={{ transform: 'translateY(14px)' }}>
        <span>{spotMin.toFixed(1)}%</span>
        <span>Spot Δ</span>
        <span>{spotMax.toFixed(1)}%</span>
      </div>
      {/* Plot area */}
      <div className="absolute left-14 right-0 top-0 rounded-lg border border-[var(--border-subtle)]" style={{ height: H, backgroundColor: 'var(--bg-elevated)' }}>
        {/* Zero lines */}
        {spotMin < 0 && spotMax > 0 && (
          <div
            className="absolute top-0 bottom-0 w-px bg-[var(--border)]"
            style={{ left: `${((0 - spotMin) / spotRange) * 100}%` }}
          />
        )}
        {pnlMin < 0 && pnlMax > 0 && (
          <div
            className="absolute left-0 right-0 h-px bg-[var(--border)]"
            style={{ top: `${((pnlMax - 0) / pnlRange) * 100}%` }}
          />
        )}

        {/* Dots */}
        {filtered.map((d, i) => {
          const x = ((d.spot_chg_pct! - spotMin) / spotRange) * 100
          const y = ((pnlMax - d.pnl_combined!) / pnlRange) * 100
          return (
            <div
              key={i}
              className="absolute w-2 h-2 rounded-full transition-all duration-150 hover:w-3 hover:h-3 hover:z-10"
              style={{
                left: `calc(${x}% - 4px)`,
                top: `calc(${y}% - 4px)`,
                backgroundColor: d.color,
                opacity: 0.7,
              }}
              title={`${d.date}\n${d.regime_state}\nSpot: ${d.spot_chg_pct?.toFixed(2)}%\nPnL: ${d.pnl_combined?.toFixed(3)}%`}
            />
          )
        })}
      </div>
    </div>
  )
}
