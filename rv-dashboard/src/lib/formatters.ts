export type DisplayMode = 'pct' | 'sharpe' | 'annualized'

export function formatPct(v: number): string {
  return `${v.toFixed(2)}%`
}

export function formatSharpe(v: number | null): string {
  return v != null ? v.toFixed(2) : '—'
}

export function formatVal(v: number, mode: DisplayMode): string {
  if (mode === 'sharpe') return formatSharpe(v)
  return formatPct(v)
}

export function annualize(dailyPct: number): number {
  return (Math.pow(1 + dailyPct / 100, 252) - 1) * 100
}
