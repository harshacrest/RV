'use client'

import { useState, useEffect, useMemo } from 'react'
import { fetchBoundarySensitivity } from '@/lib/api'
import type { BoundarySensitivity as BSData } from '@/lib/types'

function scoreColor(score: number, best: number): string {
  if (score <= 0) return 'rgba(255,82,82,0.4)'
  const ratio = score / Math.max(best, 0.01)
  if (ratio > 0.95) return 'rgba(192,132,252,0.6)'
  if (ratio > 0.85) return 'rgba(100,255,218,0.4)'
  if (ratio > 0.7) return 'rgba(68,138,255,0.3)'
  if (ratio > 0.5) return 'rgba(255,171,64,0.2)'
  return 'rgba(255,82,82,0.15)'
}

export function BoundarySensitivity() {
  const [data, setData] = useState<BSData | null>(null)

  useEffect(() => { fetchBoundarySensitivity().then(setData) }, [])

  const { l1Values, l2Values, gridMap } = useMemo(() => {
    if (!data?.grid) return { l1Values: [], l2Values: [], gridMap: new Map<string, number>() }

    const l1s = new Set<number>()
    const l2s = new Set<number>()
    const gm = new Map<string, number>()

    for (const p of data.grid) {
      l1s.add(p.l1)
      l2s.add(p.l2)
      gm.set(`${p.l1}_${p.l2}`, p.score)
    }

    return {
      l1Values: [...l1s].sort((a, b) => a - b),
      l2Values: [...l2s].sort((a, b) => a - b),
      gridMap: gm,
    }
  }, [data])

  if (!data) return <p style={{ color: 'var(--text-dim)' }}>Loading...</p>

  return (
    <div className="space-y-6">
      {/* Summary cards */}
      <div className="grid grid-cols-3 gap-4">
        <div className="stat-card">
          <p className="text-[10px] uppercase tracking-wider" style={{ color: 'var(--text-dim)' }}>Current</p>
          <p className="text-xl font-bold mt-1" style={{ color: 'var(--text-primary)' }}>L1=8.5, L2=11.0</p>
          <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>composite=0.891</p>
        </div>
        <div className="stat-card">
          <p className="text-[10px] uppercase tracking-wider" style={{ color: 'var(--text-dim)' }}>Optimal</p>
          <p className="text-xl font-bold mt-1" style={{ color: 'var(--accent-primary)' }}>
            L1={data.best_l1}, L2={data.best_l2}
          </p>
          <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
            composite={data.best_score?.toFixed(4)}
          </p>
        </div>
        <div className="stat-card">
          <p className="text-[10px] uppercase tracking-wider" style={{ color: 'var(--text-dim)' }}>Sweet Spot</p>
          <p className="text-xl font-bold mt-1" style={{ color: 'var(--signal-positive)' }}>L1: 8-9.5</p>
          <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>L2: 10.5-12.0</p>
        </div>
      </div>

      {/* Heatmap */}
      <div className="stat-card">
        <p className="text-[11px] font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
          Composite Score Heatmap (L1 rows x L2 columns)
        </p>

        <div className="overflow-x-auto">
          <table className="text-[10px]" style={{ borderCollapse: 'separate', borderSpacing: 2 }}>
            <thead>
              <tr>
                <th className="px-2 py-1 text-right" style={{ color: 'var(--text-dim)' }}>L1\L2</th>
                {l2Values.map(l2 => (
                  <th key={l2} className="px-2 py-1 text-center" style={{ color: 'var(--text-dim)', minWidth: 55 }}>
                    {l2.toFixed(1)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {l1Values.map(l1 => (
                <tr key={l1}>
                  <td className="px-2 py-1 text-right font-medium" style={{ color: 'var(--text-secondary)' }}>
                    {l1.toFixed(1)}
                  </td>
                  {l2Values.map(l2 => {
                    const key = `${l1}_${l2}`
                    const score = gridMap.get(key)
                    const isCurrent = l1 === 8.5 && l2 === 11.0
                    const isBest = l1 === data.best_l1 && l2 === data.best_l2

                    if (score === undefined) {
                      return <td key={l2} className="px-2 py-2 text-center" style={{ color: 'var(--text-dim)' }}>&mdash;</td>
                    }

                    return (
                      <td
                        key={l2}
                        className="px-2 py-2 text-center font-mono rounded"
                        style={{
                          background: scoreColor(score, data.best_score),
                          color: score > 0.5 ? 'var(--text-primary)' : 'var(--text-dim)',
                          border: isBest ? '2px solid var(--accent-primary)' : isCurrent ? '2px solid var(--signal-warning)' : '1px solid transparent',
                          fontWeight: (isBest || isCurrent) ? 700 : 400,
                        }}
                      >
                        {score.toFixed(2)}
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="flex gap-4 mt-3 text-[10px]">
          <span style={{ color: 'var(--accent-primary)' }}>Purple border = optimal</span>
          <span style={{ color: 'var(--signal-warning)' }}>Orange border = current</span>
          <span style={{ color: 'var(--text-dim)' }}>Brighter = higher score</span>
        </div>
      </div>

      {/* Key insight */}
      <div className="stat-card">
        <p className="text-[11px] font-semibold mb-2" style={{ color: 'var(--signal-info)' }}>
          Boundary Insights
        </p>
        <ul className="space-y-1 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
          <li>Boundaries above L2=14 collapse performance dramatically</li>
          <li>The L1 boundary matters less than L2 (L3 dominates with 77% of training data)</li>
          <li>Moving L1 from 8.5 to 9.0 gains +0.016 composite without changing L2</li>
          <li>Very tight boundaries [8,10] hurt because L2 has too few days</li>
        </ul>
      </div>
    </div>
  )
}
