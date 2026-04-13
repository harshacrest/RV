'use client'

import { useState, useEffect, useMemo } from 'react'
import { fetchRegimeConstruction } from '@/lib/api'
import type { RegimeConstructionData, CompleteTableRow, LevelFinalState, DteBreakdownRow, TestedAndFailed, L2StrategyProfile } from '@/lib/types'
import { generateConsistencyCallout, generateL1Story, generateL2Story, generateL3Story, generateVRPConfirmation } from '@/lib/narratives'

interface Props {
  startDate: string
  endDate: string
  snapshot: string
  dte: number | null
}

/* ── State color map ── */
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

const IV_LEVEL_GROUPS: { key: string; label: string; states: string[] }[] = [
  { key: 'L1', label: 'L1 -- Low IV (<12)', states: ['L1 Safe', 'L1 Exposed'] },
  { key: 'L2', label: 'L2 -- Moderate IV (12-17)', states: ['L2 Safe', 'L2 Caution-A', 'L2 Caution-B', 'L2 Risky'] },
  { key: 'L3', label: 'L3 -- High IV (>17)', states: ['L3 Safe', 'L3 Exposed'] },
]

/* ── Color helpers ── */
function alColor(alPct: number | null): string {
  if (alPct == null) return 'var(--text-dim)'
  if (alPct <= 5) return 'var(--signal-positive)'
  if (alPct <= 10) return 'var(--signal-warning)'
  if (alPct <= 15) return '#ff9100'
  return 'var(--signal-negative)'
}

function sharpeColor(sh: number | null): string {
  if (sh == null) return 'var(--text-dim)'
  if (sh >= 4) return 'var(--signal-positive)'
  if (sh >= 2) return '#64ffda'
  if (sh >= 1) return 'var(--signal-warning)'
  return 'var(--signal-negative)'
}

function pnlColor(v: number | null): string {
  if (v == null) return 'var(--text-dim)'
  return v >= 0 ? 'var(--signal-positive)' : 'var(--signal-negative)'
}

function fmtPct(v: number | null, decimals = 1): string {
  if (v == null) return '\u2014'
  return `${v.toFixed(decimals)}%`
}

function fmtNum(v: number | null, decimals = 2): string {
  if (v == null) return '\u2014'
  return v.toFixed(decimals)
}

function fmtPnl(v: number | null, decimals = 4): string {
  if (v == null) return '\u2014'
  return `${v > 0 ? '+' : ''}${v.toFixed(decimals)}%`
}

/* ── Strategy accent colors for L2 profile headers ── */
const STRAT_ACCENT: Record<string, string> = {
  dm: '#ffd740',
  wc: '#448aff',
  orion: '#64ffda',
}

function StratMetricToggle({ value, onChange }: { value: 'avg' | 'sharpe'; onChange: (v: 'avg' | 'sharpe') => void }) {
  return (
    <div className="inline-flex items-center rounded-md border border-[var(--border)] overflow-hidden text-[9px] font-mono">
      <button
        onClick={() => onChange('avg')}
        className="px-2.5 py-1 transition-colors"
        style={{
          backgroundColor: value === 'avg' ? 'rgba(192, 132, 252, 0.15)' : 'transparent',
          color: value === 'avg' ? '#c084fc' : 'var(--text-dim)',
        }}
      >
        Avg Return
      </button>
      <button
        onClick={() => onChange('sharpe')}
        className="px-2.5 py-1 transition-colors border-l border-[var(--border)]"
        style={{
          backgroundColor: value === 'sharpe' ? 'rgba(192, 132, 252, 0.15)' : 'transparent',
          color: value === 'sharpe' ? '#c084fc' : 'var(--text-dim)',
        }}
      >
        Sharpe
      </button>
    </div>
  )
}

/* ── Strategy table helpers ── */
function stratHeader(strat: string, metric: 'avg' | 'sharpe'): string {
  const names: Record<string, string> = { dm: 'DM', wc: 'WC', orion: 'Orion' }
  return `${names[strat] ?? strat} ${metric === 'avg' ? 'avg' : 'Sh'}`
}

function stratVal(avg: number | null, sh: number | null, metric: 'avg' | 'sharpe'): { text: string; color: string } {
  if (metric === 'avg') return { text: fmtPnl(avg), color: pnlColor(avg) }
  return { text: fmtNum(sh), color: sharpeColor(sh) }
}

type StratMetric = 'avg' | 'sharpe'

export function RegimeConstruction({ startDate, endDate, snapshot, dte }: Props) {
  const [data, setData] = useState<RegimeConstructionData | null>(null)
  const [loading, setLoading] = useState(true)
  const [stratMetric, setStratMetric] = useState<StratMetric>('avg')
  const [dteView, setDteView] = useState<string | null>(null) // null = all DTEs

  useEffect(() => {
    setLoading(true)
    fetchRegimeConstruction(startDate, endDate, snapshot, dte)
      .then(setData)
      .finally(() => setLoading(false))
  }, [startDate, endDate, snapshot, dte])

  const selectedConfig = useMemo(() => {
    if (!data) return null
    return data.boundary_configs.find(c => c.is_selected) ?? null
  }, [data])

  if (loading) return <div className="text-center py-20 text-[var(--text-dim)] text-sm font-mono">Loading regime construction...</div>
  if (!data) return <div className="text-center py-20 text-[var(--text-dim)] text-sm font-mono">No data available</div>

  const { boundary_configs, per_level, complete_table, overall, dte_breakdown } = data

  return (
    <div className="space-y-6">

      {/* ── 1. Intro ── */}
      <div className="rounded-xl border border-[var(--border)] p-5" style={{ backgroundColor: 'var(--bg-card)' }}>
        <h3 className="text-xs font-mono uppercase tracking-wider text-[var(--text-dim)] mb-3">Step 4: Regime Construction</h3>
        <div className="space-y-2 text-xs font-mono text-[var(--text-secondary)]">
          <p>Build the classification. Step 1: IV level boundaries. Step 2: find best combined score within each level.</p>
          <p><strong className="text-[var(--text-primary)]">IV has the strongest autocorrelation (0.78)</strong> &mdash; it becomes the backbone for level boundaries.</p>
        </div>
      </div>

      {/* ── 2. IV Level Boundaries: 12 and 17 ── */}
      <div className="rounded-xl border border-[var(--border)] overflow-hidden" style={{ backgroundColor: 'var(--bg-card)' }}>
        <div className="px-5 py-4 border-b border-[var(--border)]">
          <h3 className="text-sm font-mono font-semibold text-[var(--text-primary)] mb-1">IV Level Boundaries: 12 and 17</h3>
          <p className="text-[10px] font-mono text-[var(--text-dim)]">{boundary_configs.length} configs tested on all-lose spread and persistence (avg streak length).</p>
        </div>

        <div className="overflow-x-auto">
          {(() => {
            const maxLevels = Math.max(...boundary_configs.map(c => c.levels.length))
            return (
              <table className="w-full text-[10px] font-mono">
                <thead>
                  <tr className="border-b border-[var(--border)]">
                    <th className="text-left px-4 py-2 text-[var(--text-dim)] font-normal">Config</th>
                    {Array.from({ length: maxLevels }, (_, i) => (
                      <th key={`lh-${i}`} className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">L{i + 1}%</th>
                    ))}
                    <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Spread</th>
                    <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Avg Streak</th>
                    <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Self-trans</th>
                  </tr>
                </thead>
                <tbody>
                  {boundary_configs.map(cfg => (
                    <tr
                      key={cfg.label}
                      className="border-b border-[var(--border-subtle)] transition-colors hover:bg-[var(--bg-hover)]"
                      style={cfg.is_selected
                        ? { borderLeft: '3px solid #00e676', backgroundColor: 'rgba(0, 230, 118, 0.06)' }
                        : undefined
                      }
                    >
                      <td className="px-4 py-2 text-[var(--text-secondary)]">
                        {cfg.label}
                        {cfg.is_selected && <span className="ml-2 text-[8px] px-1.5 py-0.5 rounded-full bg-[#00e676]/15 text-[#00e676] uppercase tracking-wider">selected</span>}
                      </td>
                      {Array.from({ length: maxLevels }, (_, i) => (
                        <td key={`pct-${i}`} className="text-right px-3 py-2 text-[var(--text-secondary)]">
                          {i < cfg.levels.length ? fmtPct(cfg.levels[i].pct) : '—'}
                        </td>
                      ))}
                      <td className="text-right px-3 py-2 font-semibold" style={{ color: alColor(cfg.spread) }}>{fmtPct(cfg.spread)}</td>
                      <td className="text-right px-3 py-2 text-[var(--text-primary)]">{fmtNum(cfg.avg_streak, 1)}d</td>
                      <td className="text-right px-3 py-2 text-[var(--text-muted)]">{fmtPct(cfg.self_trans_pct)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )
          })()}
        </div>

        {selectedConfig && (
          <div className="px-5 py-3 border-t border-[var(--border)] bg-[var(--bg-elevated)]">
            <p className="text-[10px] font-mono text-[var(--text-secondary)]">
              <strong className="text-[var(--text-primary)]">12/17 chosen:</strong>{' '}
              {fmtPct(selectedConfig.spread)} spread with {fmtNum(selectedConfig.avg_streak, 1)}d streaks. Balanced distribution.
            </p>
          </div>
        )}
      </div>

      {/* ── 2b. Consistent Feature Testing Callout ── */}
      {(() => {
        const callout = generateConsistencyCallout(data)
        return (
          <div className="rounded-xl border border-[var(--border)] p-4" style={{ backgroundColor: 'var(--bg-card)', borderLeft: '3px solid #00e676' }}>
            <p className="text-[11px] font-mono font-bold text-[#00e676] mb-2">
              {callout.title}
            </p>
            <p className="text-[10px] font-mono text-[var(--text-muted)] mb-3">
              36+ single features tested at ALL levels. The SAME features were evaluated at L1, L2, L3 — no level-specific picking.
            </p>
            <div className="space-y-1.5">
              {callout.bullets.map((b, i) => (
                <p key={i} className="text-[10px] font-mono text-[var(--text-secondary)] leading-relaxed pl-3 border-l-2 border-[var(--border-subtle)]">
                  {b}
                </p>
              ))}
            </div>
          </div>
        )
      })()}

      {/* ── 3A. L1 (IV < 12): PK/IV Ratio is the Signal ── */}
      <div className="rounded-xl border border-[var(--border)] overflow-hidden" style={{ backgroundColor: 'var(--bg-card)' }}>
        <div className="px-5 py-4 border-b border-[var(--border)]">
          <h3 className="text-sm font-mono font-semibold text-[var(--text-primary)]">L1 (IV &lt; 12): PK/IV Ratio is the Signal</h3>
          <p className="text-[10px] font-mono text-[var(--text-dim)] mt-1">{per_level.L1.description}</p>
        </div>

        {/* PK/IV Quintile Table */}
        <div className="px-5 py-3">
          <div className="text-[10px] font-mono uppercase tracking-wider text-[var(--text-dim)] mb-2">PK/IV Quintile Breakdown</div>
          <div className="overflow-x-auto">
            <table className="w-full text-[10px] font-mono">
              <thead>
                <tr className="border-b border-[var(--border)]">
                  <th className="text-left px-3 py-2 text-[var(--text-dim)] font-normal">Q</th>
                  <th className="text-left px-3 py-2 text-[var(--text-dim)] font-normal">PK/IV range</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Days</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">AL%</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Sharpe</th>
                </tr>
              </thead>
              <tbody>
                {per_level.L1.pk_iv_quintiles.map(q => (
                  <tr key={q.quintile} className="border-b border-[var(--border-subtle)] hover:bg-[var(--bg-hover)]">
                    <td className="px-3 py-2 text-[var(--text-primary)] font-semibold">{q.quintile}</td>
                    <td className="px-3 py-2 text-[var(--text-secondary)]">{q.pk_iv_range}</td>
                    <td className="text-right px-3 py-2 text-[var(--text-secondary)]">{q.days}</td>
                    <td className="text-right px-3 py-2 font-semibold" style={{ color: alColor(q.al_pct) }}>{fmtPct(q.al_pct)}</td>
                    <td className="text-right px-3 py-2 font-semibold" style={{ color: sharpeColor(q.sharpe) }}>{fmtNum(q.sharpe)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-[10px] font-mono text-[var(--text-muted)] mt-2">
            Nearly monotonic: {per_level.L1.pk_iv_quintiles.map(q => `${q.quintile} ${fmtPct(q.al_pct)}`).join(' \u2192 ')}
          </p>
        </div>

        {/* L1 Final: 2 states */}
        <div className="px-5 py-3 border-t border-[var(--border)]">
          <div className="text-[10px] font-mono uppercase tracking-wider text-[var(--text-dim)] mb-2">
            L1 Final: 2 states (split at median {fmtNum(per_level.L1.threshold, 2)})
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-[10px] font-mono">
              <thead>
                <tr className="border-b border-[var(--border)]">
                  <th className="text-left px-3 py-2 text-[var(--text-dim)] font-normal">State</th>
                  <th className="text-left px-3 py-2 text-[var(--text-dim)] font-normal">Rule</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Days</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">%</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">AL%</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">AW%</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Port avg</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Port Sh</th>
                </tr>
              </thead>
              <tbody>
                {per_level.L1.final_states.map(st => {
                  const color = STATE_COLORS[st.state] ?? 'var(--text-secondary)'
                  return (
                    <tr
                      key={st.state}
                      className="border-b border-[var(--border-subtle)] hover:bg-[var(--bg-hover)]"
                      style={{ borderLeft: `3px solid ${color}` }}
                    >
                      <td className="px-3 py-2 font-semibold" style={{ color }}>{st.state}</td>
                      <td className="px-3 py-2 text-[var(--text-muted)]">{st.rule ?? '\u2014'}</td>
                      <td className="text-right px-3 py-2 text-[var(--text-secondary)]">{st.days}</td>
                      <td className="text-right px-3 py-2 text-[var(--text-muted)]">{fmtPct(st.pct)}</td>
                      <td className="text-right px-3 py-2 font-semibold" style={{ color: alColor(st.al_pct) }}>{fmtPct(st.al_pct)}</td>
                      <td className="text-right px-3 py-2" style={{ color: 'var(--signal-positive)' }}>{fmtPct(st.aw_pct)}</td>
                      <td className="text-right px-3 py-2 font-semibold" style={{ color: pnlColor(st.port_avg) }}>{fmtPnl(st.port_avg)}</td>
                      <td className="text-right px-3 py-2 font-semibold" style={{ color: sharpeColor(st.sharpe) }}>{fmtNum(st.sharpe)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          {/* L1 Strategy Profiles */}
          {per_level.L1.strategy_profiles && per_level.L1.strategy_profiles.length > 0 && (
            <div className="mt-3">
              <div className="flex items-center justify-between mb-2">
                <div className="text-[10px] font-mono uppercase tracking-wider text-[var(--text-dim)]">Strategy Breakdown</div>
                <StratMetricToggle value={stratMetric} onChange={setStratMetric} />
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-[10px] font-mono">
                  <thead>
                    <tr className="border-b border-[var(--border)]">
                      <th className="text-left px-3 py-2 text-[var(--text-dim)] font-normal">State</th>
                      <th className="text-right px-3 py-2 font-normal" style={{ color: STRAT_ACCENT.dm }}>{stratHeader('dm', stratMetric)}</th>
                      <th className="text-right px-3 py-2 font-normal" style={{ color: STRAT_ACCENT.wc }}>{stratHeader('wc', stratMetric)}</th>
                      <th className="text-right px-3 py-2 font-normal" style={{ color: STRAT_ACCENT.orion }}>{stratHeader('orion', stratMetric)}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {per_level.L1.strategy_profiles.map((sp: L2StrategyProfile) => {
                      const color = STATE_COLORS[sp.state] ?? 'var(--text-secondary)'
                      const dm = stratVal(sp.dm_avg, sp.dm_sharpe, stratMetric)
                      const wc = stratVal(sp.wc_avg, sp.wc_sharpe, stratMetric)
                      const or = stratVal(sp.orion_avg, sp.orion_sharpe, stratMetric)
                      return (
                        <tr key={sp.state} className="border-b border-[var(--border-subtle)] hover:bg-[var(--bg-hover)]">
                          <td className="px-3 py-2 font-semibold" style={{ color }}>{sp.state}</td>
                          <td className="text-right px-3 py-2 font-semibold" style={{ color: dm.color }}>{dm.text}</td>
                          <td className="text-right px-3 py-2 font-semibold" style={{ color: wc.color }}>{wc.text}</td>
                          <td className="text-right px-3 py-2 font-semibold" style={{ color: or.color }}>{or.text}</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
          {/* L1 Story Narrative */}
          {(() => {
            const stories = generateL1Story(per_level.L1.final_states, per_level.L1.threshold)
            const tested = per_level.L1.tested_and_failed
            if (stories.length === 0 && (!tested || tested.length === 0)) return null
            return (
              <div className="px-5 py-3 border-t border-[var(--border-subtle)]">
                {stories.map((s, i) => (
                  <p key={i} className="text-[10px] font-mono text-[var(--text-muted)] leading-relaxed mb-1">{s}</p>
                ))}
                {tested && tested.length > 0 && (
                  <div className="mt-2 pt-2 border-t border-[var(--border-subtle)]">
                    <div className="text-[9px] font-mono text-[var(--text-dim)] uppercase tracking-wider mb-1">Also tested at L1:</div>
                    {tested.map((t, i) => (
                      <p key={i} className="text-[9px] font-mono text-[var(--text-dim)] leading-relaxed">
                        <span className="text-[var(--signal-negative)]">&#x2717;</span> {t.approach}: {t.result}
                      </p>
                    ))}
                  </div>
                )}
              </div>
            )
          })()}
        </div>
      </div>

      {/* ── 3B. L2 (IV 12-17): IV Direction + PK/IV Ratio ── */}
      <div className="rounded-xl border border-[var(--border)] overflow-hidden" style={{ backgroundColor: 'var(--bg-card)' }}>
        <div className="px-5 py-4 border-b border-[var(--border)]">
          <h3 className="text-sm font-mono font-semibold text-[var(--text-primary)]">L2 (IV 12-17): IV Direction + PK/IV Ratio</h3>
          <p className="text-[10px] font-mono text-[var(--text-dim)] mt-1">{per_level.L2.description}</p>
        </div>

        {/* L2 Final: 4 states */}
        <div className="px-5 py-3">
          <div className="text-[10px] font-mono uppercase tracking-wider text-[var(--text-dim)] mb-2">
            L2 Final: 4 states (PK/IV median x IV_chg_5d direction)
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-[10px] font-mono">
              <thead>
                <tr className="border-b border-[var(--border)]">
                  <th className="text-left px-3 py-2 text-[var(--text-dim)] font-normal">State</th>
                  <th className="text-left px-3 py-2 text-[var(--text-dim)] font-normal">PK/IV</th>
                  <th className="text-left px-3 py-2 text-[var(--text-dim)] font-normal">IV dir</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Days</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">%</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">AL%</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">AW%</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Port avg</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Sharpe</th>
                </tr>
              </thead>
              <tbody>
                {per_level.L2.final_states.map(st => {
                  const color = STATE_COLORS[st.state] ?? 'var(--text-secondary)'
                  return (
                    <tr
                      key={st.state}
                      className="border-b border-[var(--border-subtle)] hover:bg-[var(--bg-hover)]"
                      style={{ borderLeft: `3px solid ${color}` }}
                    >
                      <td className="px-3 py-2 font-semibold" style={{ color }}>{st.state}</td>
                      <td className="px-3 py-2 text-[var(--text-muted)]">{st.pk_iv ?? '\u2014'}</td>
                      <td className="px-3 py-2 text-[var(--text-muted)]">{st.iv_dir ?? '\u2014'}</td>
                      <td className="text-right px-3 py-2 text-[var(--text-secondary)]">{st.days}</td>
                      <td className="text-right px-3 py-2 text-[var(--text-muted)]">{fmtPct(st.pct)}</td>
                      <td className="text-right px-3 py-2 font-semibold" style={{ color: alColor(st.al_pct) }}>{fmtPct(st.al_pct)}</td>
                      <td className="text-right px-3 py-2" style={{ color: 'var(--signal-positive)' }}>{fmtPct(st.aw_pct)}</td>
                      <td className="text-right px-3 py-2 font-semibold" style={{ color: pnlColor(st.port_avg) }}>{fmtPnl(st.port_avg)}</td>
                      <td className="text-right px-3 py-2 font-semibold" style={{ color: sharpeColor(st.sharpe) }}>{fmtNum(st.sharpe)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Strategy profiles by state */}
        <div className="px-5 py-3 border-t border-[var(--border)]">
          <div className="flex items-center justify-between mb-2">
            <div className="text-[10px] font-mono uppercase tracking-wider text-[var(--text-dim)]">Strategy Profiles by State</div>
            <StratMetricToggle value={stratMetric} onChange={setStratMetric} />
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-[10px] font-mono">
              <thead>
                <tr className="border-b border-[var(--border)]">
                  <th className="text-left px-3 py-2 text-[var(--text-dim)] font-normal">State</th>
                  <th className="text-right px-3 py-2 font-normal" style={{ color: STRAT_ACCENT.dm }}>{stratHeader('dm', stratMetric)}</th>
                  <th className="text-right px-3 py-2 font-normal" style={{ color: STRAT_ACCENT.wc }}>{stratHeader('wc', stratMetric)}</th>
                  <th className="text-right px-3 py-2 font-normal" style={{ color: STRAT_ACCENT.orion }}>{stratHeader('orion', stratMetric)}</th>
                </tr>
              </thead>
              <tbody>
                {per_level.L2.strategy_profiles.map(sp => {
                  const color = STATE_COLORS[sp.state] ?? 'var(--text-secondary)'
                  const dm = stratVal(sp.dm_avg, sp.dm_sharpe, stratMetric)
                  const wc = stratVal(sp.wc_avg, sp.wc_sharpe, stratMetric)
                  const or = stratVal(sp.orion_avg, sp.orion_sharpe, stratMetric)
                  return (
                    <tr key={sp.state} className="border-b border-[var(--border-subtle)] hover:bg-[var(--bg-hover)]">
                      <td className="px-3 py-2 font-semibold" style={{ color }}>{sp.state}</td>
                      <td className="text-right px-3 py-2 font-semibold" style={{ color: dm.color }}>{dm.text}</td>
                      <td className="text-right px-3 py-2 font-semibold" style={{ color: wc.color }}>{wc.text}</td>
                      <td className="text-right px-3 py-2 font-semibold" style={{ color: or.color }}>{or.text}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
        {/* L2 Story Narrative */}
        {(() => {
          const stories = generateL2Story(per_level.L2.final_states, per_level.L2.strategy_profiles)
          const tested = per_level.L2.tested_and_failed
          if (stories.length === 0 && (!tested || tested.length === 0)) return null
          return (
            <div className="px-5 py-3 border-t border-[var(--border-subtle)]">
              {stories.map((s, i) => (
                <p key={i} className="text-[10px] font-mono text-[var(--text-muted)] leading-relaxed mb-1">{s}</p>
              ))}
              {tested && tested.length > 0 && (
                <div className="mt-2 pt-2 border-t border-[var(--border-subtle)]">
                  <div className="text-[9px] font-mono text-[var(--text-dim)] uppercase tracking-wider mb-1">Also tested at L2:</div>
                  {tested.map((t, i) => (
                    <p key={i} className="text-[9px] font-mono text-[var(--text-dim)] leading-relaxed">
                      <span className="text-[var(--signal-negative)]">&#x2717;</span> {t.approach}: {t.result}
                    </p>
                  ))}
                </div>
              )}
            </div>
          )
        })()}
      </div>

      {/* ── 3C. L3 (IV > 17): PK/IV Ratio -- Higher Stakes ── */}
      <div className="rounded-xl border border-[var(--border)] overflow-hidden" style={{ backgroundColor: 'var(--bg-card)' }}>
        <div className="px-5 py-4 border-b border-[var(--border)]">
          <h3 className="text-sm font-mono font-semibold text-[var(--text-primary)]">L3 (IV &gt; 17): PK/IV Ratio &mdash; Higher Stakes</h3>
          <p className="text-[10px] font-mono text-[var(--text-dim)] mt-1">{per_level.L3.description}</p>
        </div>

        {/* PK/IV Quintile Table */}
        <div className="px-5 py-3">
          <div className="text-[10px] font-mono uppercase tracking-wider text-[var(--text-dim)] mb-2">PK/IV Quintile Breakdown</div>
          <div className="overflow-x-auto">
            <table className="w-full text-[10px] font-mono">
              <thead>
                <tr className="border-b border-[var(--border)]">
                  <th className="text-left px-3 py-2 text-[var(--text-dim)] font-normal">Q</th>
                  <th className="text-left px-3 py-2 text-[var(--text-dim)] font-normal">PK/IV range</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Days</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">AL%</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Sharpe</th>
                </tr>
              </thead>
              <tbody>
                {per_level.L3.pk_iv_quintiles.map(q => (
                  <tr key={q.quintile} className="border-b border-[var(--border-subtle)] hover:bg-[var(--bg-hover)]">
                    <td className="px-3 py-2 text-[var(--text-primary)] font-semibold">{q.quintile}</td>
                    <td className="px-3 py-2 text-[var(--text-secondary)]">{q.pk_iv_range}</td>
                    <td className="text-right px-3 py-2 text-[var(--text-secondary)]">{q.days}</td>
                    <td className="text-right px-3 py-2 font-semibold" style={{ color: alColor(q.al_pct) }}>{fmtPct(q.al_pct)}</td>
                    <td className="text-right px-3 py-2 font-semibold" style={{ color: sharpeColor(q.sharpe) }}>{fmtNum(q.sharpe)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-[10px] font-mono text-[var(--text-muted)] mt-2">
            Nearly monotonic: {per_level.L3.pk_iv_quintiles.map(q => `${q.quintile} ${fmtPct(q.al_pct)}`).join(' \u2192 ')}
          </p>
        </div>

        {/* L3 Final: 2 states */}
        <div className="px-5 py-3 border-t border-[var(--border)]">
          <div className="text-[10px] font-mono uppercase tracking-wider text-[var(--text-dim)] mb-2">
            L3 Final: 2 states (split at median {fmtNum(per_level.L3.threshold, 2)})
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-[10px] font-mono">
              <thead>
                <tr className="border-b border-[var(--border)]">
                  <th className="text-left px-3 py-2 text-[var(--text-dim)] font-normal">State</th>
                  <th className="text-left px-3 py-2 text-[var(--text-dim)] font-normal">Rule</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Days</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">%</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">AL%</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">AW%</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Port avg</th>
                  <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Port Sh</th>
                </tr>
              </thead>
              <tbody>
                {per_level.L3.final_states.map(st => {
                  const color = STATE_COLORS[st.state] ?? 'var(--text-secondary)'
                  return (
                    <tr
                      key={st.state}
                      className="border-b border-[var(--border-subtle)] hover:bg-[var(--bg-hover)]"
                      style={{ borderLeft: `3px solid ${color}` }}
                    >
                      <td className="px-3 py-2 font-semibold" style={{ color }}>{st.state}</td>
                      <td className="px-3 py-2 text-[var(--text-muted)]">{st.rule ?? '\u2014'}</td>
                      <td className="text-right px-3 py-2 text-[var(--text-secondary)]">{st.days}</td>
                      <td className="text-right px-3 py-2 text-[var(--text-muted)]">{fmtPct(st.pct)}</td>
                      <td className="text-right px-3 py-2 font-semibold" style={{ color: alColor(st.al_pct) }}>{fmtPct(st.al_pct)}</td>
                      <td className="text-right px-3 py-2" style={{ color: 'var(--signal-positive)' }}>{fmtPct(st.aw_pct)}</td>
                      <td className="text-right px-3 py-2 font-semibold" style={{ color: pnlColor(st.port_avg) }}>{fmtPnl(st.port_avg)}</td>
                      <td className="text-right px-3 py-2 font-semibold" style={{ color: sharpeColor(st.sharpe) }}>{fmtNum(st.sharpe)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          {/* L3 Strategy Profiles */}
          {per_level.L3.strategy_profiles && per_level.L3.strategy_profiles.length > 0 && (
            <div className="mt-3">
              <div className="flex items-center justify-between mb-2">
                <div className="text-[10px] font-mono uppercase tracking-wider text-[var(--text-dim)]">Strategy Breakdown</div>
                <StratMetricToggle value={stratMetric} onChange={setStratMetric} />
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-[10px] font-mono">
                  <thead>
                    <tr className="border-b border-[var(--border)]">
                      <th className="text-left px-3 py-2 text-[var(--text-dim)] font-normal">State</th>
                      <th className="text-right px-3 py-2 font-normal" style={{ color: STRAT_ACCENT.dm }}>{stratHeader('dm', stratMetric)}</th>
                      <th className="text-right px-3 py-2 font-normal" style={{ color: STRAT_ACCENT.wc }}>{stratHeader('wc', stratMetric)}</th>
                      <th className="text-right px-3 py-2 font-normal" style={{ color: STRAT_ACCENT.orion }}>{stratHeader('orion', stratMetric)}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {per_level.L3.strategy_profiles.map((sp: L2StrategyProfile) => {
                      const color = STATE_COLORS[sp.state] ?? 'var(--text-secondary)'
                      const dm = stratVal(sp.dm_avg, sp.dm_sharpe, stratMetric)
                      const wc = stratVal(sp.wc_avg, sp.wc_sharpe, stratMetric)
                      const or = stratVal(sp.orion_avg, sp.orion_sharpe, stratMetric)
                      return (
                        <tr key={sp.state} className="border-b border-[var(--border-subtle)] hover:bg-[var(--bg-hover)]">
                          <td className="px-3 py-2 font-semibold" style={{ color }}>{sp.state}</td>
                          <td className="text-right px-3 py-2 font-semibold" style={{ color: dm.color }}>{dm.text}</td>
                          <td className="text-right px-3 py-2 font-semibold" style={{ color: wc.color }}>{wc.text}</td>
                          <td className="text-right px-3 py-2 font-semibold" style={{ color: or.color }}>{or.text}</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
          {/* L3 Story Narrative */}
          {(() => {
            const stories = generateL3Story(per_level.L3.final_states, per_level.L1.final_states, per_level.L3.strategy_profiles)
            const tested = per_level.L3.tested_and_failed
            return (
              <div className="mt-3">
                {stories.map((s, i) => (
                  <p key={i} className="text-[10px] font-mono text-[var(--text-muted)] leading-relaxed mb-1">{s}</p>
                ))}
                {tested && tested.length > 0 && (
                  <div className="mt-2 pt-2 border-t border-[var(--border-subtle)]">
                    <div className="text-[9px] font-mono text-[var(--text-dim)] uppercase tracking-wider mb-1">Also tested at L3:</div>
                    {tested.map((t, i) => (
                      <p key={i} className="text-[9px] font-mono text-[var(--text-dim)] leading-relaxed">
                        <span className="text-[var(--signal-negative)]">&#x2717;</span> {t.approach}: {t.result}
                      </p>
                    ))}
                  </div>
                )}
              </div>
            )
          })()}
        </div>
      </div>

      {/* ── 4. Complete Regime Table ── */}
      <div className="rounded-xl border border-[var(--border)] overflow-hidden" style={{ backgroundColor: 'var(--bg-card)' }}>
        <div className="px-5 py-4 border-b border-[var(--border)]">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-mono font-semibold text-[var(--text-primary)]">Complete Regime Table</h3>
            <StratMetricToggle value={stratMetric} onChange={setStratMetric} />
          </div>
          <p className="text-[10px] font-mono text-[var(--text-dim)] mt-1">8 states across 3 IV levels. 3 inputs: IV_5d, PK_5d, IV_chg_5d.</p>
        </div>

        {IV_LEVEL_GROUPS.map(group => {
          const groupRows = complete_table.filter(r => group.states.includes(r.state))
          if (groupRows.length === 0) return null
          return (
            <div key={group.key}>
              <div className="px-4 py-2 bg-[var(--bg-elevated)] border-b border-[var(--border)]">
                <span className="text-[10px] font-mono font-semibold text-[var(--text-secondary)]">{group.label}</span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-[10px] font-mono">
                  <thead>
                    <tr className="border-b border-[var(--border)]">
                      <th className="text-left px-4 py-2 text-[var(--text-dim)] font-normal">State</th>
                      <th className="text-left px-3 py-2 text-[var(--text-dim)] font-normal">Rule</th>
                      <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Days</th>
                      <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">%</th>
                      <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">AL%</th>
                      <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">AW%</th>
                      <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Port avg</th>
                      <th className="text-right px-3 py-2 text-[var(--text-dim)] font-normal">Sharpe</th>
                      <th className="text-right px-2 py-2 font-normal" style={{ color: STRAT_ACCENT.dm, borderBottom: `1px solid ${STRAT_ACCENT.dm}` }}>{stratHeader('dm', stratMetric)}</th>
                      <th className="text-right px-2 py-2 font-normal" style={{ color: STRAT_ACCENT.wc, borderBottom: `1px solid ${STRAT_ACCENT.wc}` }}>{stratHeader('wc', stratMetric)}</th>
                      <th className="text-right px-2 py-2 font-normal" style={{ color: STRAT_ACCENT.orion, borderBottom: `1px solid ${STRAT_ACCENT.orion}` }}>{stratHeader('orion', stratMetric)}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {groupRows.map(row => (
                      <tr key={row.state} className="border-b border-[var(--border-subtle)] hover:bg-[var(--bg-hover)]">
                        <td className="px-4 py-2.5">
                          <div className="flex items-center gap-2">
                            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: row.color }} />
                            <span className="font-semibold" style={{ color: row.color }}>{row.state}</span>
                          </div>
                        </td>
                        <td className="px-3 py-2.5 text-[var(--text-muted)]">{row.rule}</td>
                        <td className="text-right px-3 py-2.5 text-[var(--text-secondary)]">{row.days}</td>
                        <td className="text-right px-3 py-2.5 text-[var(--text-muted)]">{fmtPct(row.pct)}</td>
                        <td className="text-right px-3 py-2.5 font-semibold" style={{ color: alColor(row.al_pct) }}>{fmtPct(row.al_pct)}</td>
                        <td className="text-right px-3 py-2.5" style={{ color: 'var(--signal-positive)' }}>{fmtPct(row.aw_pct)}</td>
                        <td className="text-right px-3 py-2.5 font-semibold" style={{ color: pnlColor(row.port_avg) }}>{fmtPnl(row.port_avg)}</td>
                        <td className="text-right px-3 py-2.5 font-semibold" style={{ color: sharpeColor(row.sharpe) }}>{fmtNum(row.sharpe)}</td>
                        {(() => {
                          const dm = stratVal(row.dm_avg, row.dm_sharpe, stratMetric)
                          const wc = stratVal(row.wc_avg, row.wc_sharpe, stratMetric)
                          const or = stratVal(row.orion_avg, row.orion_sharpe, stratMetric)
                          return (<>
                            <td className="text-right px-2 py-2.5 font-semibold" style={{ color: dm.color }}>{dm.text}</td>
                            <td className="text-right px-2 py-2.5 font-semibold" style={{ color: wc.color }}>{wc.text}</td>
                            <td className="text-right px-2 py-2.5 font-semibold" style={{ color: or.color }}>{or.text}</td>
                          </>)
                        })()}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )
        })}

        {/* Overall row */}
        <div className="px-4 py-3 border-t border-[var(--border)] bg-[var(--bg-elevated)] flex items-center gap-6">
          <div className="text-[10px] font-mono text-[var(--text-dim)] uppercase tracking-wider">Overall</div>
          <div className="flex items-center gap-5 text-[10px] font-mono">
            <span className="text-[var(--text-secondary)]">{overall.days} days</span>
            <span>AL%: <strong style={{ color: alColor(overall.al_pct) }}>{fmtPct(overall.al_pct)}</strong></span>
            <span>AW%: <strong className="text-[var(--signal-positive)]">{fmtPct(overall.aw_pct)}</strong></span>
            <span>Port Avg: <strong style={{ color: pnlColor(overall.port_avg) }}>{fmtPnl(overall.port_avg)}</strong></span>
            <span>Sharpe: <strong style={{ color: sharpeColor(overall.sharpe) }}>{fmtNum(overall.sharpe)}</strong></span>
          </div>
        </div>
      </div>

      {/* ── 4b. DTE Breakdown per Regime ── */}
      {dte_breakdown && Object.keys(dte_breakdown).length > 0 && (
        <div className="rounded-xl border border-[var(--border)] overflow-hidden" style={{ backgroundColor: 'var(--bg-card)' }}>
          <div className="px-5 py-4 border-b border-[var(--border)]">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-mono font-semibold text-[var(--text-primary)]">DTE Breakdown by Regime</h3>
              <div className="flex items-center gap-3">
                {/* DTE filter */}
                <div className="flex items-center gap-1.5">
                  <span className="text-[9px] font-mono uppercase tracking-wider text-[var(--text-dim)]">DTE</span>
                  <div className="flex items-center rounded-md border border-[var(--border)] overflow-hidden">
                    {([null, '4', '3', '2', '1', '0', '0,2', '1,3,4'] as const).map(val => (
                      <button
                        key={val ?? 'all'}
                        onClick={() => setDteView(val)}
                        className="px-2 py-0.5 text-[9px] font-mono font-semibold transition-all"
                        style={{
                          backgroundColor: dteView === val ? 'rgba(251,191,36,0.15)' : 'transparent',
                          color: dteView === val ? '#fbbf24' : 'var(--text-muted)',
                          borderRight: '1px solid var(--border)',
                        }}
                      >
                        {val === null ? 'All' : `${val}d`}
                      </button>
                    ))}
                  </div>
                </div>
                <StratMetricToggle value={stratMetric} onChange={setStratMetric} />
              </div>
            </div>
            <p className="text-[10px] font-mono text-[var(--text-dim)] mt-1">Does regime behavior change near expiry? Each DTE value analysed separately.</p>
          </div>
          <div className="overflow-x-auto overflow-y-auto" style={{ maxHeight: '600px' }}>
            <table className="w-full text-[10px] font-mono">
              <thead className="sticky top-0 z-10" style={{ backgroundColor: 'var(--bg-card)' }}>
                <tr className="border-b border-[var(--border)]">
                  <th className="text-left px-4 py-2 text-[var(--text-dim)] font-normal" style={{ backgroundColor: 'var(--bg-card)' }}>State</th>
                  <th className="text-right px-2 py-2 text-[var(--text-dim)] font-normal" style={{ backgroundColor: 'var(--bg-card)' }}>DTE</th>
                  <th className="text-right px-2 py-2 text-[var(--text-dim)] font-normal" style={{ backgroundColor: 'var(--bg-card)' }}>Days</th>
                  <th className="text-right px-2 py-2 text-[var(--text-dim)] font-normal" style={{ backgroundColor: 'var(--bg-card)' }}>AL%</th>
                  <th className="text-right px-2 py-2 text-[var(--text-dim)] font-normal" style={{ backgroundColor: 'var(--bg-card)' }}>AW%</th>
                  <th className="text-right px-2 py-2 text-[var(--text-dim)] font-normal" style={{ backgroundColor: 'var(--bg-card)' }}>Port avg</th>
                  <th className="text-right px-2 py-2 text-[var(--text-dim)] font-normal" style={{ backgroundColor: 'var(--bg-card)' }}>Sharpe</th>
                  <th className="text-right px-2 py-2 font-normal" style={{ color: STRAT_ACCENT.dm, borderBottom: `1px solid ${STRAT_ACCENT.dm}`, backgroundColor: 'var(--bg-card)' }}>{stratHeader('dm', stratMetric)}</th>
                  <th className="text-right px-2 py-2 font-normal" style={{ color: STRAT_ACCENT.wc, borderBottom: `1px solid ${STRAT_ACCENT.wc}`, backgroundColor: 'var(--bg-card)' }}>{stratHeader('wc', stratMetric)}</th>
                  <th className="text-right px-2 py-2 font-normal" style={{ color: STRAT_ACCENT.orion, borderBottom: `1px solid ${STRAT_ACCENT.orion}`, backgroundColor: 'var(--bg-card)' }}>{stratHeader('orion', stratMetric)}</th>
                </tr>
              </thead>
              <tbody>
                {complete_table.map(regime => {
                  const allRows = dte_breakdown[regime.state] as DteBreakdownRow[] | undefined
                  if (!allRows || allRows.length === 0) return null
                  const rows = dteView !== null ? allRows.filter(r => String(r.dte) === dteView) : allRows
                  if (rows.length === 0) return null
                  return rows.map((dr, idx) => {
                    const dm = stratVal(dr.dm_avg, dr.dm_sharpe, stratMetric)
                    const wc = stratVal(dr.wc_avg, dr.wc_sharpe, stratMetric)
                    const or = stratVal(dr.orion_avg, dr.orion_sharpe, stratMetric)
                    return (
                      <tr key={`${regime.state}-${dr.dte}`} className="border-b border-[var(--border-subtle)] hover:bg-[var(--bg-hover)]">
                        {idx === 0 ? (
                          <td className="px-4 py-2" rowSpan={rows.length}>
                            <div className="flex items-center gap-2">
                              <div className="w-2 h-2 rounded-full" style={{ backgroundColor: regime.color }} />
                              <span className="font-semibold text-[10px]" style={{ color: regime.color }}>{regime.state}</span>
                            </div>
                          </td>
                        ) : null}
                        <td className="text-right px-2 py-2 text-[var(--text-primary)] font-semibold">{dr.dte}</td>
                        <td className="text-right px-2 py-2 text-[var(--text-secondary)]">{dr.days}</td>
                        <td className="text-right px-2 py-2 font-semibold" style={{ color: alColor(dr.al_pct) }}>{fmtPct(dr.al_pct)}</td>
                        <td className="text-right px-2 py-2" style={{ color: 'var(--signal-positive)' }}>{fmtPct(dr.aw_pct)}</td>
                        <td className="text-right px-2 py-2 font-semibold" style={{ color: pnlColor(dr.port_avg) }}>{fmtPnl(dr.port_avg)}</td>
                        <td className="text-right px-2 py-2 font-semibold" style={{ color: sharpeColor(dr.sharpe) }}>{fmtNum(dr.sharpe)}</td>
                        <td className="text-right px-2 py-2 font-semibold" style={{ color: dm.color }}>{dm.text}</td>
                        <td className="text-right px-2 py-2 font-semibold" style={{ color: wc.color }}>{wc.text}</td>
                        <td className="text-right px-2 py-2 font-semibold" style={{ color: or.color }}>{or.text}</td>
                      </tr>
                    )
                  })
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── 5. The Unified Story ── */}
      <div className="rounded-xl border border-[var(--border)] p-5" style={{ backgroundColor: 'var(--bg-card)' }}>
        <h3 className="text-xs font-mono uppercase tracking-wider text-[var(--text-dim)] mb-3">The Unified Story</h3>
        <div className="space-y-2 text-xs font-mono text-[var(--text-secondary)]">
          <p><strong className="text-[var(--text-primary)]">Risk at every level = realized movement eating into premium cushion.</strong></p>
          <p>PK/IV ratio is the primary signal at L1 and L3.</p>
          <p>At L2, IV direction adds because it tells you if the gap is widening or shrinking.</p>
          <p className="text-[var(--text-dim)]">At L1 and L3, IV direction is noise — at L1 because changes are too small, at L3 because IV is almost always rising.</p>
          {/* VRP Confirmation */}
          {data.vrp_by_state && (() => {
            const vrpLine = generateVRPConfirmation(data.vrp_by_state)
            if (!vrpLine) return null
            return <p className="text-[var(--signal-positive)]">{vrpLine}</p>
          })()}
        </div>
      </div>

      {/* ── 6. Inputs Reference ── */}
      <div className="rounded-xl border border-[var(--border)] p-5" style={{ backgroundColor: 'var(--bg-card)' }}>
        <h3 className="text-xs font-mono uppercase tracking-wider text-[var(--text-dim)] mb-3">Regime Inputs</h3>
        <p className="text-[10px] font-mono text-[var(--text-muted)] mb-3">3 inputs, all 5-day averages, all lagged (known before 9:15 AM). PK/IV = PK_5d / IV_5d. iv_lag = IV_7d[t-1] for level classification.</p>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
          {[
            { name: 'iv_lag', formula: 'IV_7d[t-1]', role: 'Level classification (L1/L2/L3)' },
            { name: 'IV_5d', formula: 'mean(IV_7d, t-5 to t-1)', role: 'PK/IV denominator' },
            { name: 'PK_5d', formula: 'mean(Parkinson vol, t-5 to t-1)', role: 'PK/IV numerator' },
            { name: 'IV_chg_5d', formula: 'mean(daily IV change, t-5 to t-1)', role: 'L2 direction only' },
          ].map(inp => (
            <div key={inp.name} className="rounded-lg border border-[var(--border-subtle)] p-3" style={{ backgroundColor: 'var(--bg-elevated)' }}>
              <div className="text-[11px] font-mono font-bold text-[#c084fc] mb-1">{inp.name}</div>
              <div className="text-[9px] font-mono text-[var(--text-muted)] mb-1.5 bg-[var(--bg-card)] rounded px-2 py-1">{inp.formula}</div>
              <div className="text-[9px] font-mono text-[var(--text-dim)]">{inp.role}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
