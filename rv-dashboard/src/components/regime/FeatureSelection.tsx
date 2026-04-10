'use client'

import { useState, useEffect } from 'react'
import { fetchFeatureSelection } from '@/lib/api'
import type { FeatureSelectionData, ACTableRow } from '@/lib/types'

interface Props {
  startDate: string
  endDate: string
  snapshot: string
  dte: number | null
}

function acColor(v: number | null): string {
  if (v == null) return 'var(--text-dim)'
  if (v < 0) return '#ef5350'
  if (v < 0.1) return '#ef5350'
  if (v < 0.4) return '#ffee58'
  if (v < 0.7) return '#80cbc4'
  return '#69f0ae'
}

function verdictColor(v: string): string {
  const low = v.toLowerCase()
  if (low.includes('best') || low.includes('backbone')) return '#69f0ae'
  if (low.includes('unpredictable') || low.includes('redundant')) return '#ef5350'
  return 'var(--text-secondary)'
}

function fmtAC(v: number | null): string {
  if (v == null) return '—'
  return v.toFixed(2)
}

const FEATURE_COLORS: Record<string, string> = {
  iv_level: '#c084fc',
  pk_iv_ratio: '#ffd740',
  iv_direction: '#64ffda',
}

function featureColor(name: string): string {
  const low = name.toLowerCase()
  if (low.includes('iv') && low.includes('level')) return FEATURE_COLORS.iv_level
  if (low.includes('pk') || low.includes('ratio')) return FEATURE_COLORS.pk_iv_ratio
  if (low.includes('direction') || low.includes('chg')) return FEATURE_COLORS.iv_direction
  return '#c084fc'
}

export function FeatureSelection({ startDate, endDate, snapshot, dte }: Props) {
  const [data, setData] = useState<FeatureSelectionData | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    fetchFeatureSelection(startDate, endDate, snapshot, dte)
      .then(setData)
      .finally(() => setLoading(false))
  }, [startDate, endDate, snapshot, dte])

  if (loading) return <div className="text-center py-20 text-[var(--text-dim)] text-sm font-mono">Loading feature selection...</div>
  if (!data) return <div className="text-center py-20 text-[var(--text-dim)] text-sm font-mono">No data available.</div>

  return (
    <div className="space-y-6">
      {/* ═══ Intro Text ═══ */}
      <div className="rounded-xl border border-[var(--border)] p-5" style={{ backgroundColor: 'var(--bg-card)' }}>
        <h2 className="text-xs font-mono uppercase tracking-wider text-[#c084fc] font-semibold mb-3">Step 3: Feature Selection — The Lagging Problem</h2>
        <p className="text-[11px] font-mono text-[var(--text-secondary)] leading-relaxed mb-2">
          We need features known before 9:15 AM open (1-day lag). Most features from Step 1 don&apos;t persist day-to-day.
        </p>
        <p className="text-[11px] font-mono text-[var(--text-secondary)] leading-relaxed">
          IV level has the strongest autocorrelation (0.78) — this is why it becomes the regime backbone.
        </p>
      </div>

      {/* ═══ Section 3.1: Autocorrelation Table ═══ */}
      <div className="rounded-xl border border-[var(--border)] overflow-hidden" style={{ backgroundColor: 'var(--bg-card)' }}>
        <div className="px-4 py-2.5 border-b border-[var(--border)] bg-[var(--bg-elevated)]">
          <span className="text-[11px] font-mono font-semibold text-[var(--text-secondary)]">
            Autocorrelation: Raw vs Averaged (all features, all windows)
          </span>
          <div className="text-[9px] font-mono text-[var(--text-dim)] mt-0.5">
            Does the feature today predict the same feature tomorrow? Higher AC = more persistent = more usable.
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-[10px] font-mono">
            <thead>
              <tr className="border-b border-[var(--border)]">
                <th className="text-left px-3 py-2 text-[var(--text-dim)] font-normal">Feature</th>
                <th className="text-right px-2 py-2 text-[var(--text-dim)] font-normal">Raw 1d</th>
                <th className="text-right px-2 py-2 text-[var(--text-dim)] font-normal">3d avg</th>
                <th className="text-right px-2 py-2 font-normal" style={{ color: '#c084fc', borderBottom: '2px solid #c084fc' }}>5d avg</th>
                <th className="text-right px-2 py-2 text-[var(--text-dim)] font-normal">7d avg</th>
                <th className="text-right px-2 py-2 text-[var(--text-dim)] font-normal">10d avg</th>
                <th className="text-left px-3 py-2 text-[var(--text-dim)] font-normal">Verdict</th>
              </tr>
            </thead>
            <tbody>
              {data.ac_table.map((row: ACTableRow, i: number) => (
                <tr key={i} className="border-b border-[var(--border-subtle)] hover:bg-[var(--bg-hover)]">
                  <td className="px-3 py-2 text-[var(--text-primary)] font-semibold">{row.label}</td>
                  <td className="text-right px-2 py-2 font-semibold" style={{ color: acColor(row.raw_1d) }}>
                    {fmtAC(row.raw_1d)}
                  </td>
                  <td className="text-right px-2 py-2 font-semibold" style={{ color: acColor(row.avg_3d) }}>
                    {fmtAC(row.avg_3d)}
                  </td>
                  <td className="text-right px-2 py-2 font-bold" style={{ color: acColor(row.avg_5d), backgroundColor: 'rgba(192, 132, 252, 0.06)' }}>
                    {fmtAC(row.avg_5d)}
                  </td>
                  <td className="text-right px-2 py-2 font-semibold" style={{ color: acColor(row.avg_7d) }}>
                    {fmtAC(row.avg_7d)}
                  </td>
                  <td className="text-right px-2 py-2 font-semibold" style={{ color: acColor(row.avg_10d) }}>
                    {fmtAC(row.avg_10d)}
                  </td>
                  <td className="px-3 py-2 text-[10px]" style={{ color: verdictColor(row.verdict) }}>
                    {row.verdict}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ═══ Key Finding Callout ═══ */}
      <div
        className="rounded-xl border border-[var(--border)] p-4"
        style={{ backgroundColor: 'var(--bg-card)', borderLeft: '3px solid #c084fc' }}
      >
        <p className="text-[11px] font-mono font-bold text-[#c084fc] mb-2">
          Key Finding: 5-day averaging makes PK and VRP MORE persistent than IV level itself
        </p>
        <p className="text-[10px] font-mono text-[var(--text-secondary)] leading-relaxed mb-2">
          {data.key_finding}
        </p>
        <p className="text-[10px] font-mono text-[var(--text-muted)] leading-relaxed">
          Raw PK (0.27) and raw VRP (0.19) are useless — but 5d averaging transforms them.
        </p>
      </div>

      {/* ═══ Section 3.2: Features Going Forward ═══ */}
      <div className="rounded-xl border border-[var(--border)] p-5" style={{ backgroundColor: 'var(--bg-card)' }}>
        <h3 className="text-xs font-mono uppercase tracking-wider text-[var(--text-dim)] font-semibold mb-1">
          Features Going Forward
        </h3>
        <p className="text-[10px] font-mono text-[var(--text-muted)] mb-4">
          3 inputs, all 5-day averages, all lagged (known before 9:15 AM):
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {data.selected_features.map((feat, i) => {
            const color = featureColor(feat.feature)
            return (
              <div
                key={i}
                className="rounded-xl border border-[var(--border)] p-4"
                style={{ backgroundColor: 'var(--bg-card)', borderLeft: `3px solid ${color}` }}
              >
                <div className="text-[12px] font-mono font-bold mb-2" style={{ color }}>
                  {feat.feature}
                </div>
                <div className="text-[9px] font-mono text-[var(--text-muted)] mb-2 bg-[var(--bg-elevated)] rounded px-2 py-1.5">
                  {feat.formula}
                </div>
                <div className="text-[10px] font-mono text-[var(--text-secondary)] leading-relaxed">
                  {feat.role}
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
