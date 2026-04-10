'use client'

import { useState, useEffect, useMemo } from 'react'
import { fetchFeatureRanking } from '@/lib/api'
import type { FeatureRankingData, FeatureRankingEntry } from '@/lib/types'
import { generateFeatureObservations, generateKeyTakeaways } from '@/lib/narratives'

interface Props {
  startDate: string
  endDate: string
  snapshot: string
  dte: number | null
}

function valColor(v: number | null): string {
  if (v == null) return 'var(--text-dim)'
  if (v > 0) return 'var(--signal-positive)'
  if (v < 0) return 'var(--signal-negative)'
  return 'var(--text-secondary)'
}

function wrColor(wr: number | null): string {
  if (wr == null) return 'var(--text-dim)'
  if (wr >= 0.7) return 'var(--signal-positive)'
  if (wr >= 0.5) return 'var(--text-secondary)'
  if (wr >= 0.3) return 'var(--signal-warning)'
  return 'var(--signal-negative)'
}

function fmt(v: number | null, decimals = 2): string {
  if (v == null) return '\u2014'
  return v.toFixed(decimals)
}

function fmtPct(v: number | null, decimals = 1): string {
  if (v == null) return '\u2014'
  return `${v.toFixed(decimals)}%`
}

function fmtSign(v: number | null, decimals = 2): string {
  if (v == null) return '\u2014'
  return `${v > 0 ? '+' : ''}${v.toFixed(decimals)}`
}

export function FeatureRanking({ startDate, endDate, snapshot, dte }: Props) {
  const [data, setData] = useState<FeatureRankingData | null>(null)
  const [loading, setLoading] = useState(true)
  const [expandedFeatures, setExpandedFeatures] = useState<Set<string>>(new Set())

  useEffect(() => {
    setLoading(true)
    fetchFeatureRanking(startDate, endDate, snapshot, dte)
      .then(setData)
      .finally(() => setLoading(false))
  }, [startDate, endDate, snapshot, dte])

  const toggleFeature = (key: string) => {
    setExpandedFeatures(prev => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

  if (loading) {
    return <div className="text-center py-20 text-[var(--text-dim)] text-sm font-mono">Loading feature ranking...</div>
  }

  if (!data) {
    return <div className="text-center py-20 text-[var(--signal-negative)] text-sm font-mono">Failed to load feature ranking data.</div>
  }

  const { features, portfolio_stats: ps, two_jobs, discarded } = data

  return (
    <div className="space-y-6">

      {/* ═══ 1. Header Stats Bar ═══ */}
      <div className="rounded-xl border border-[var(--border)] p-4" style={{ backgroundColor: 'var(--bg-card)' }}>
        <div className="flex flex-wrap gap-x-6 gap-y-2 text-[11px] font-mono">
          <span className="text-[var(--text-secondary)]">
            Dataset: <span className="text-[var(--text-primary)] font-semibold">{ps.days}</span> days
          </span>
          <span className="text-[var(--text-secondary)]">
            Combined Sharpe: <span className="text-[var(--text-primary)] font-semibold">{fmt(ps.combined_sharpe)}</span>
          </span>
          <span className="text-[var(--text-secondary)]">
            AL: <span style={{ color: 'var(--signal-negative)' }} className="font-semibold">{fmtPct(ps.al_pct)}</span>
          </span>
          <span className="text-[var(--text-secondary)]">
            AW: <span style={{ color: 'var(--signal-positive)' }} className="font-semibold">{fmtPct(ps.aw_pct)}</span>
          </span>
        </div>
        <div className="mt-2 flex flex-wrap gap-x-6 gap-y-1 text-[10px] font-mono text-[var(--text-muted)]">
          <span>
            Strategy correlations:
          </span>
          <span>
            DM vs Orion = <span className="text-[var(--text-primary)] font-semibold">{fmtSign(ps.dm_orion_corr, 3)}</span>
            <span className="text-[var(--text-dim)] ml-1">(natural hedge)</span>
          </span>
          <span>
            DM vs WC = <span className="text-[var(--text-primary)] font-semibold">{fmtSign(ps.dm_wc_corr, 3)}</span>
          </span>
          <span>
            WC vs Orion = <span className="text-[var(--text-primary)] font-semibold">{fmtSign(ps.wc_orion_corr, 3)}</span>
          </span>
        </div>
      </div>

      {/* ═══ 2. Main Ranking Table (Section 2.1) ═══ */}
      <div className="rounded-xl border border-[var(--border)] overflow-hidden" style={{ backgroundColor: 'var(--bg-card)' }}>
        <div className="px-4 py-2.5 border-b border-[var(--border)] bg-[var(--bg-elevated)]">
          <span className="text-xs font-mono uppercase tracking-wider text-[var(--text-dim)]">Ranked by All-Lose Spread</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-[10px] font-mono">
            <thead>
              <tr className="border-b border-[var(--border)]">
                <th className="text-left px-3 py-2 text-[var(--text-dim)] font-normal w-8">#</th>
                <th className="text-left px-2 py-2 text-[var(--text-dim)] font-normal">Feature</th>
                <th className="text-right px-2 py-2 text-[var(--text-dim)] font-normal">AL Spread</th>
                <th className="text-right px-2 py-2 text-[var(--text-dim)] font-normal">Best Q</th>
                <th className="text-right px-2 py-2 text-[var(--text-dim)] font-normal">Worst Q</th>
                <th className="text-right px-2 py-2 text-[var(--text-dim)] font-normal">Corr Gap</th>
                <th className="text-left px-2 py-2 text-[var(--text-dim)] font-normal">Verdict</th>
              </tr>
            </thead>
            <tbody>
              {features.map((f: FeatureRankingEntry) => (
                <tr key={f.feature_key} className="border-b border-[var(--border-subtle)] hover:bg-[var(--bg-hover)]">
                  <td className="px-3 py-2 text-[var(--text-dim)]">{f.rank}</td>
                  <td className="px-2 py-2 text-[var(--text-primary)] font-semibold">{f.label}</td>
                  <td className="text-right px-2 py-2 font-semibold" style={{ color: f.al_spread > 0 ? 'var(--signal-warning)' : 'var(--signal-positive)' }}>
                    {fmtSign(f.al_spread)}%
                  </td>
                  <td className="text-right px-2 py-2 text-[var(--text-secondary)]">{f.best_q}</td>
                  <td className="text-right px-2 py-2 text-[var(--text-secondary)]">{f.worst_q}</td>
                  <td className="text-right px-2 py-2 font-semibold" style={{ color: valColor(f.dm_orion_gap) }}>
                    {fmtSign(f.dm_orion_gap)}
                  </td>
                  <td className="px-2 py-2">
                    <span className="font-semibold" style={{ color: f.verdict === 'KEEP' ? 'var(--signal-positive)' : 'var(--signal-negative)' }}>
                      {f.verdict}
                    </span>
                    {f.verdict_reason && (
                      <span className="ml-1.5 text-[9px] text-[var(--text-dim)]">{f.verdict_reason}</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ═══ 2b. Key Takeaways ═══ */}
      {(() => {
        const takeaways = generateKeyTakeaways(data)
        if (takeaways.length === 0) return null
        return (
          <div className="rounded-xl border border-[var(--border)] p-4" style={{ backgroundColor: 'var(--bg-card)', borderLeft: '3px solid #c084fc' }}>
            <h3 className="text-[11px] font-mono font-bold text-[#c084fc] mb-3">Key Takeaways</h3>
            <div className="space-y-2">
              {takeaways.map((t, i) => (
                <div key={i} className="flex items-start gap-2.5">
                  <span className="text-[10px] font-mono font-bold text-[#c084fc] mt-0.5 shrink-0">{i + 1}.</span>
                  <p className="text-[10px] font-mono text-[var(--text-secondary)] leading-relaxed">{t}</p>
                </div>
              ))}
            </div>
          </div>
        )
      })()}

      {/* ═══ 3. Two-Job Framework (Section 2.2) ═══ */}
      <div>
        <h3 className="text-xs font-mono uppercase tracking-wider text-[var(--text-dim)] mb-3">Two-Job Framework</h3>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {[two_jobs.job1, two_jobs.job2].map((job, idx) => (
            <div
              key={idx}
              className="rounded-xl border border-[var(--border)] p-4"
              style={{ backgroundColor: 'var(--bg-card)', borderLeftWidth: 3, borderLeftColor: '#a78bfa' }}
            >
              <div className="text-[11px] font-mono font-semibold text-[var(--text-primary)] mb-0.5">
                Job {idx + 1}: &ldquo;{job.title}&rdquo;
              </div>
              <div className="text-[10px] font-mono text-[var(--text-secondary)] mb-2">{job.subtitle}</div>
              <p className="text-[10px] font-mono text-[var(--text-muted)] mb-3 leading-relaxed">{job.description}</p>
              <div className="flex flex-wrap gap-1.5">
                {job.features.map(feat => (
                  <span
                    key={feat}
                    className="px-2 py-0.5 rounded text-[9px] font-mono font-semibold text-[var(--text-primary)] border border-[var(--border-subtle)]"
                    style={{ backgroundColor: 'var(--bg-elevated)' }}
                  >
                    {feat}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
        <p className="mt-3 text-[10px] font-mono text-[var(--text-dim)] italic">
          Observation: Job 1 and Job 2 use almost completely different feature sets. A regime system needs both layers.
        </p>
      </div>

      {/* ═══ 4. Per-Feature Expandable Quintile Tables ═══ */}
      <div>
        <h3 className="text-xs font-mono uppercase tracking-wider text-[var(--text-dim)] mb-3">Per-Feature Quintile Breakdown</h3>
        <div className="space-y-2">
          {features.map((f: FeatureRankingEntry) => {
            const isExpanded = expandedFeatures.has(f.feature_key)
            const quintiles = f.quintiles ?? []
            // Compute spread row (last - first)
            const q1 = quintiles.length > 0 ? quintiles[0] : null
            const q5 = quintiles.length > 0 ? quintiles[quintiles.length - 1] : null

            return (
              <div key={f.feature_key} className="rounded-xl border border-[var(--border)] overflow-hidden" style={{ backgroundColor: 'var(--bg-card)' }}>
                {/* Clickable header */}
                <button
                  className="w-full px-4 py-2.5 flex items-center justify-between text-left hover:bg-[var(--bg-hover)] transition-colors"
                  onClick={() => toggleFeature(f.feature_key)}
                >
                  <div className="flex items-center gap-3">
                    <span className="text-[10px] font-mono text-[var(--text-dim)] w-4">{isExpanded ? '\u25BC' : '\u25B6'}</span>
                    <span className="text-[11px] font-mono font-semibold text-[var(--text-primary)]">{f.label}</span>
                    <span className="text-[9px] font-mono text-[var(--text-muted)]">
                      <span style={{ color: '#ffd740' }}>DM corr: {fmtSign(f.dm_corr, 3)}</span>
                      <span className="mx-1.5">|</span>
                      <span style={{ color: '#448aff' }}>WC corr: {fmtSign(f.wc_corr, 3)}</span>
                      <span className="mx-1.5">|</span>
                      <span style={{ color: '#64ffda' }}>Orion corr: {fmtSign(f.orion_corr, 3)}</span>
                      <span className="mx-1.5">|</span>
                      <span className="text-[var(--text-muted)]">AL corr: {fmtSign(f.al_corr, 3)}</span>
                    </span>
                  </div>
                  <span
                    className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded"
                    style={{
                      color: f.verdict === 'KEEP' ? 'var(--signal-positive)' : 'var(--signal-negative)',
                      backgroundColor: f.verdict === 'KEEP' ? 'rgba(76,175,80,0.1)' : 'rgba(244,67,54,0.1)',
                    }}
                  >
                    {f.verdict}
                  </span>
                </button>

                {/* Expanded quintile table */}
                {isExpanded && quintiles.length > 0 && (
                  <div className="border-t border-[var(--border)] overflow-x-auto">
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
                          <th className="text-right px-2 py-2 text-[var(--text-dim)] font-normal">Comb</th>
                          <th className="text-right px-2 py-2 text-[var(--text-dim)] font-normal">Comb Sh</th>
                          <th className="text-right px-2 py-2 text-[var(--text-dim)] font-normal">AL%</th>
                          <th className="text-right px-2 py-2 text-[var(--text-dim)] font-normal">AW%</th>
                        </tr>
                      </thead>
                      <tbody>
                        {quintiles.map(q => (
                          <tr key={q.quintile} className="border-b border-[var(--border-subtle)] hover:bg-[var(--bg-hover)]">
                            <td className="px-3 py-2 text-[var(--text-primary)] font-semibold">{q.quintile}</td>
                            <td className="px-2 py-2 text-[var(--text-muted)] text-[9px]">{q.range}</td>
                            <td className="text-right px-2 py-2 text-[var(--text-secondary)]">{q.n}</td>
                            <td className="text-right px-2 py-2 font-semibold" style={{ color: valColor(q.dm_avg) }}>
                              {q.dm_avg != null ? `${fmtSign(q.dm_avg)}%` : '\u2014'}
                            </td>
                            <td className="text-right px-2 py-2" style={{ color: wrColor(q.dm_wr) }}>
                              {q.dm_wr != null ? `${(q.dm_wr * 100).toFixed(0)}%` : '\u2014'}
                            </td>
                            <td className="text-right px-2 py-2 text-[var(--text-secondary)]">
                              {q.dm_sharpe != null ? fmt(q.dm_sharpe) : '\u2014'}
                            </td>
                            <td className="text-right px-2 py-2 font-semibold" style={{ color: valColor(q.wc_avg) }}>
                              {q.wc_avg != null ? `${fmtSign(q.wc_avg)}%` : '\u2014'}
                            </td>
                            <td className="text-right px-2 py-2" style={{ color: wrColor(q.wc_wr) }}>
                              {q.wc_wr != null ? `${(q.wc_wr * 100).toFixed(0)}%` : '\u2014'}
                            </td>
                            <td className="text-right px-2 py-2 text-[var(--text-secondary)]">
                              {q.wc_sharpe != null ? fmt(q.wc_sharpe) : '\u2014'}
                            </td>
                            <td className="text-right px-2 py-2 font-semibold" style={{ color: valColor(q.orion_avg) }}>
                              {q.orion_avg != null ? `${fmtSign(q.orion_avg)}%` : '\u2014'}
                            </td>
                            <td className="text-right px-2 py-2" style={{ color: wrColor(q.orion_wr) }}>
                              {q.orion_wr != null ? `${(q.orion_wr * 100).toFixed(0)}%` : '\u2014'}
                            </td>
                            <td className="text-right px-2 py-2 text-[var(--text-secondary)]">
                              {q.orion_sharpe != null ? fmt(q.orion_sharpe) : '\u2014'}
                            </td>
                            <td className="text-right px-2 py-2 font-semibold" style={{ color: valColor(q.combined_avg) }}>
                              {q.combined_avg != null ? `${fmtSign(q.combined_avg)}%` : '\u2014'}
                            </td>
                            <td className="text-right px-2 py-2 text-[var(--text-secondary)]">
                              {q.combined_sharpe != null ? fmt(q.combined_sharpe) : '\u2014'}
                            </td>
                            <td className="text-right px-2 py-2 font-semibold" style={{ color: q.al_pct > 20 ? 'var(--signal-negative)' : q.al_pct > 10 ? 'var(--signal-warning)' : 'var(--text-secondary)' }}>
                              {fmtPct(q.al_pct)}
                            </td>
                            <td className="text-right px-2 py-2 font-semibold" style={{ color: q.aw_pct > 20 ? 'var(--signal-positive)' : 'var(--text-secondary)' }}>
                              {fmtPct(q.aw_pct)}
                            </td>
                          </tr>
                        ))}
                        {/* Spread row (Q5 - Q1) */}
                        {q1 && q5 && (
                          <tr className="border-t-2 border-[var(--border)] bg-[var(--bg-elevated)]">
                            <td className="px-3 py-2 text-[var(--text-dim)] font-semibold">Spread</td>
                            <td className="px-2 py-2 text-[var(--text-dim)] text-[9px]">Q5\u2212Q1</td>
                            <td className="text-right px-2 py-2" />
                            <td className="text-right px-2 py-2 font-bold" style={{ color: valColor(q5.dm_avg != null && q1.dm_avg != null ? q5.dm_avg - q1.dm_avg : null) }}>
                              {q5.dm_avg != null && q1.dm_avg != null ? `${fmtSign(q5.dm_avg - q1.dm_avg)}%` : '\u2014'}
                            </td>
                            <td className="text-right px-2 py-2" />
                            <td className="text-right px-2 py-2" />
                            <td className="text-right px-2 py-2 font-bold" style={{ color: valColor(q5.wc_avg != null && q1.wc_avg != null ? q5.wc_avg - q1.wc_avg : null) }}>
                              {q5.wc_avg != null && q1.wc_avg != null ? `${fmtSign(q5.wc_avg - q1.wc_avg)}%` : '\u2014'}
                            </td>
                            <td className="text-right px-2 py-2" />
                            <td className="text-right px-2 py-2" />
                            <td className="text-right px-2 py-2 font-bold" style={{ color: valColor(q5.orion_avg != null && q1.orion_avg != null ? q5.orion_avg - q1.orion_avg : null) }}>
                              {q5.orion_avg != null && q1.orion_avg != null ? `${fmtSign(q5.orion_avg - q1.orion_avg)}%` : '\u2014'}
                            </td>
                            <td className="text-right px-2 py-2" />
                            <td className="text-right px-2 py-2" />
                            <td className="text-right px-2 py-2 font-bold" style={{ color: valColor(q5.combined_avg != null && q1.combined_avg != null ? q5.combined_avg - q1.combined_avg : null) }}>
                              {q5.combined_avg != null && q1.combined_avg != null ? `${fmtSign(q5.combined_avg - q1.combined_avg)}%` : '\u2014'}
                            </td>
                            <td className="text-right px-2 py-2" />
                            <td className="text-right px-2 py-2 font-bold" style={{ color: valColor(q5.al_pct - q1.al_pct) }}>
                              {fmtSign(q5.al_pct - q1.al_pct)}%
                            </td>
                            <td className="text-right px-2 py-2 font-bold" style={{ color: valColor(q5.aw_pct - q1.aw_pct) }}>
                              {fmtSign(q5.aw_pct - q1.aw_pct)}%
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                )}

                {/* Per-feature observations */}
                {isExpanded && quintiles.length > 0 && (() => {
                  const obs = generateFeatureObservations(f)
                  if (obs.length === 0) return null
                  return (
                    <div className="border-t border-[var(--border)] px-4 py-3" style={{ borderLeft: '3px solid rgba(168,85,247,0.3)' }}>
                      <div className="space-y-1">
                        {obs.map((o, idx) => (
                          <p key={idx} className="text-[10px] font-mono text-[var(--text-muted)] leading-relaxed">
                            <span className="text-[var(--text-dim)] mr-1">&bull;</span>{o}
                          </p>
                        ))}
                      </div>
                    </div>
                  )
                })()}

                {/* DTE AL Spread mini-table (inside expanded section) */}
                {isExpanded && f.dte_al_spread && Object.keys(f.dte_al_spread).length > 0 && (
                  <div className="border-t border-[var(--border)] px-4 py-3">
                    <div className="text-[9px] font-mono text-[var(--text-dim)] uppercase tracking-wider mb-2">AL Spread by DTE (days to expiry)</div>
                    <div className="flex gap-2 flex-wrap">
                      {Object.entries(f.dte_al_spread)
                        .sort(([a], [b]) => Number(a) - Number(b))
                        .map(([dte, spread]) => (
                          <div
                            key={dte}
                            className="rounded-lg border border-[var(--border-subtle)] px-3 py-1.5 text-center"
                            style={{ backgroundColor: 'var(--bg-elevated)' }}
                          >
                            <div className="text-[9px] font-mono text-[var(--text-dim)]">DTE {dte}</div>
                            <div
                              className="text-[11px] font-mono font-bold"
                              style={{ color: spread > 5 ? 'var(--signal-positive)' : spread < -5 ? 'var(--signal-negative)' : 'var(--text-secondary)' }}
                            >
                              {spread > 0 ? '+' : ''}{spread.toFixed(1)}pp
                            </div>
                          </div>
                        ))}
                      <div
                        className="rounded-lg border border-[var(--border)] px-3 py-1.5 text-center"
                        style={{ backgroundColor: 'rgba(168,85,247,0.06)' }}
                      >
                        <div className="text-[9px] font-mono text-[#c084fc]">Overall</div>
                        <div className="text-[11px] font-mono font-bold text-[#c084fc]">
                          {f.al_spread > 0 ? '+' : ''}{f.al_spread.toFixed(1)}pp
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>

      {/* ═══ 5. Discarded Features (Section 2.3) ═══ */}
      {discarded.length > 0 && (
        <div className="rounded-xl border border-[var(--border)] overflow-hidden" style={{ backgroundColor: 'var(--bg-card)' }}>
          <div className="px-4 py-2.5 border-b border-[var(--border)] bg-[var(--bg-elevated)]">
            <span className="text-xs font-mono uppercase tracking-wider text-[var(--text-dim)]">Also Tested and Discarded</span>
          </div>
          <div className="p-4 space-y-2">
            {discarded.map(d => (
              <div key={d.feature} className="flex items-start gap-3 text-[10px] font-mono">
                <span className="text-[var(--signal-negative)] font-semibold shrink-0">DROP</span>
                <span className="text-[var(--text-primary)] font-semibold shrink-0">{d.label}</span>
                <span className="text-[var(--text-dim)]">{d.reason}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
