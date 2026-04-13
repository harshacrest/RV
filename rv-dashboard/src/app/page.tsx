'use client'

import { useState } from 'react'
import { RegimeOverview } from '@/components/regime/RegimeOverview'
import { DataExploration } from '@/components/regime/DataExploration'
import { FeatureRanking } from '@/components/regime/FeatureRanking'
import { FeatureSelection } from '@/components/regime/FeatureSelection'
import { RegimeConstruction } from '@/components/regime/RegimeConstruction'
import { SnapshotComparison } from '@/components/regime/SnapshotComparison'
import { AdaptiveOOS } from '@/components/regime/AdaptiveOOS'
import { Framework } from '@/components/regime/Framework'
import { AlphaInsights } from '@/components/regime/AlphaInsights'

type RegimeTab = 'alpha-insights' | 'data-exploration' | 'framework' | 'feature-ranking' | 'feature-selection' | 'regime-construction' | 'regime-overview' | 'snapshot-comparison' | 'test-data'

const REGIME_TABS: { key: RegimeTab; label: string }[] = [
  
  { key: 'framework', label: 'Framework' },
  { key: 'data-exploration', label: 'Data Exploration' },
  { key: 'feature-ranking', label: 'Feature Ranking' },
  { key: 'feature-selection', label: 'Feature Selection' },
  { key: 'regime-construction', label: 'Regime Construction' },
  { key: 'regime-overview', label: 'Regime Overview' },
  { key: 'snapshot-comparison', label: 'Snapshot Comparison' },
  { key: 'test-data', label: 'Test Data' },
  { key: 'alpha-insights', label: 'Alpha Insights' },
]

const SNAPSHOT_OPTIONS = [
  { value: '1529', label: '3:29 T-1' },
  { value: '1530', label: '3:30 T-1' },
  { value: '0915', label: '9:15 T0' },
  { value: '0916', label: '9:16 T0' },
] as const

const DTE_OPTIONS: { value: number | null; label: string }[] = [
  { value: null, label: 'All' },
  { value: 4, label: '4d' },
  { value: 3, label: '3d' },
  { value: 2, label: '2d' },
  { value: 1, label: '1d' },
  { value: 0, label: '0d' },
]

export default function Home() {
  const [regimeTab, setRegimeTab] = useState<RegimeTab>('framework')
  const [startDate, setStartDate] = useState('2023-01-01')
  const [endDate, setEndDate] = useState('2026-01-30')
  const [snapshot, setSnapshot] = useState('1530')
  const [dte, setDte] = useState<number | null>(null)

  return (
    <div className="min-h-screen" style={{ backgroundColor: 'var(--bg-void)' }}>
      {/* Top Nav */}
      <header className="sticky top-0 z-50 backdrop-blur-md border-b border-[var(--border)]" style={{ backgroundColor: 'rgba(6, 6, 9, 0.92)' }}>
        <div className="max-w-[1400px] mx-auto px-6 py-3 flex items-center justify-between flex-wrap gap-y-2">
          <div className="flex items-center gap-4">
            <h1 className="text-base font-bold text-[var(--text-primary)] tracking-tight">RV Dashboard</h1>
          </div>

          <div className="flex items-center gap-3">
            {/* Date Range */}
            <div className="flex items-center gap-2">
              <input
                type="date"
                value={startDate}
                onChange={e => setStartDate(e.target.value)}
                className="text-[11px] font-mono px-2 py-1 rounded-md border border-[var(--border)] bg-[var(--bg-card)] text-[var(--text-primary)] outline-none"
              />
              <span className="text-[10px] text-[var(--text-dim)]">to</span>
              <input
                type="date"
                value={endDate}
                onChange={e => setEndDate(e.target.value)}
                className="text-[11px] font-mono px-2 py-1 rounded-md border border-[var(--border)] bg-[var(--bg-card)] text-[var(--text-primary)] outline-none"
              />
            </div>

            <div className="h-5 w-px bg-[var(--border)]" />

            {/* Snapshot Toggle */}
            <div className="flex items-center rounded-lg border border-[var(--border)] bg-[var(--bg-card)] overflow-hidden">
              {SNAPSHOT_OPTIONS.map(opt => (
                <button
                  key={opt.value}
                  onClick={() => setSnapshot(opt.value)}
                  className="px-2.5 py-1 text-[10px] font-mono font-semibold transition-all"
                  style={{
                    backgroundColor: snapshot === opt.value ? 'rgba(168,85,247,0.15)' : 'transparent',
                    color: snapshot === opt.value ? '#c084fc' : 'var(--text-muted)',
                    borderRight: '1px solid var(--border)',
                  }}
                >
                  {opt.label}
                </button>
              ))}
            </div>

            <div className="h-5 w-px bg-[var(--border)]" />

            {/* DTE Toggle */}
            <div className="flex items-center gap-1.5">
              <span className="text-[10px] font-mono uppercase tracking-wider text-[var(--text-dim)]">DTE</span>
              <div className="flex items-center rounded-lg border border-[var(--border)] bg-[var(--bg-card)] overflow-hidden">
                {DTE_OPTIONS.map(opt => (
                  <button
                    key={opt.label}
                    onClick={() => setDte(opt.value)}
                    className="px-2.5 py-1 text-[10px] font-mono font-semibold transition-all"
                    style={{
                      backgroundColor: dte === opt.value ? 'rgba(251,191,36,0.15)' : 'transparent',
                      color: dte === opt.value ? '#fbbf24' : 'var(--text-muted)',
                      borderRight: '1px solid var(--border)',
                    }}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Tab bar */}
        <div className="max-w-[1400px] mx-auto px-6">
          <div className="flex gap-1">
            {REGIME_TABS.map(t => (
              <button
                key={t.key}
                onClick={() => setRegimeTab(t.key)}
                className="px-4 py-2 text-[11px] font-mono uppercase tracking-wider transition-colors relative"
                style={{
                  color: regimeTab === t.key ? 'var(--text-primary)' : 'var(--text-muted)',
                }}
              >
                {t.label}
                {regimeTab === t.key && (
                  <div
                    className="absolute bottom-0 left-0 right-0 h-[2px] rounded-full"
                    style={{ backgroundColor: '#c084fc' }}
                  />
                )}
              </button>
            ))}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-[1400px] mx-auto px-6 py-6">
        {regimeTab === 'alpha-insights' && <AlphaInsights snapshot={snapshot} dte={dte} />}
        {regimeTab === 'framework' && <Framework />}
        {regimeTab === 'data-exploration' && <DataExploration startDate={startDate} endDate={endDate} snapshot={snapshot} dte={dte} />}
        {regimeTab === 'feature-ranking' && <FeatureRanking startDate={startDate} endDate={endDate} snapshot={snapshot} dte={dte} />}
        {regimeTab === 'feature-selection' && <FeatureSelection startDate={startDate} endDate={endDate} snapshot={snapshot} dte={dte} />}
        {regimeTab === 'regime-construction' && <RegimeConstruction startDate={startDate} endDate={endDate} snapshot={snapshot} dte={dte} />}
        {regimeTab === 'regime-overview' && <RegimeOverview startDate={startDate} endDate={endDate} snapshot={snapshot} dte={dte} />}
        {regimeTab === 'snapshot-comparison' && <SnapshotComparison startDate={startDate} endDate={endDate} dte={dte} />}
        {regimeTab === 'test-data' && <AdaptiveOOS snapshot={snapshot} dte={dte} />}
      </main>
    </div>
  )
}
