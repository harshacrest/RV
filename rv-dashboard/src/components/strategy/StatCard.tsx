'use client'

export function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="stat-card">
      <div className="text-[10px] font-mono text-[var(--text-dim)] uppercase tracking-wider mb-1">{label}</div>
      <div className="text-lg font-bold text-[var(--text-primary)] font-mono">{value}</div>
      {sub && <div className="text-[10px] text-[var(--text-muted)] mt-0.5 font-mono">{sub}</div>}
    </div>
  )
}
