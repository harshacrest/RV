'use client'

import type { DisplayMode } from '@/lib/formatters'

const MODES: { key: DisplayMode; label: string }[] = [
  { key: 'pct', label: '%' },
  { key: 'sharpe', label: 'Sharpe' },
  { key: 'annualized', label: 'Ann.' },
]

export function ModeToggle({
  mode, setMode, showSharpe = true,
}: {
  mode: DisplayMode
  setMode: (m: DisplayMode) => void
  showSharpe?: boolean
}) {
  const visible = showSharpe ? MODES : MODES.filter(m => m.key !== 'sharpe')
  return (
    <div className="inline-flex rounded-lg border border-[var(--border)] overflow-hidden text-[11px] font-mono">
      {visible.map((m, i) => (
        <button
          key={m.key}
          onClick={() => setMode(m.key)}
          className="px-3 py-1.5 transition-colors"
          style={{
            backgroundColor: mode === m.key ? 'var(--bg-elevated)' : 'transparent',
            color: mode === m.key ? 'var(--text-primary)' : 'var(--text-muted)',
            borderLeft: i > 0 ? '1px solid var(--border)' : 'none',
          }}
        >
          {m.label}
        </button>
      ))}
    </div>
  )
}
