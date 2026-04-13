'use client'

import { useState } from 'react'
import type { TabKey } from '@/lib/types'
import { BaselineOverview } from '@/components/BaselineOverview'
import { RegimeDistribution } from '@/components/RegimeDistribution'
import { FeatureAnalysis } from '@/components/FeatureAnalysis'
import { AblationStudy } from '@/components/AblationStudy'
import { RollingValidation } from '@/components/RollingValidation'
import { BoundarySensitivity } from '@/components/BoundarySensitivity'

const TABS: { key: TabKey; label: string }[] = [
  { key: 'baseline', label: 'Baseline' },
  { key: 'distribution', label: 'Distribution' },
  { key: 'features', label: 'Features' },
  { key: 'ablation', label: 'Ablation' },
  { key: 'rolling', label: 'Walk-Forward' },
  { key: 'boundary', label: 'Boundaries' },
]

export default function Page() {
  const [tab, setTab] = useState<TabKey>('baseline')

  return (
    <div className="min-h-screen" style={{ background: 'var(--bg-void)' }}>
      {/* Header */}
      <header
        className="sticky top-0 z-50 px-6 py-4 flex items-center justify-between"
        style={{ background: 'var(--bg-primary)', borderBottom: '1px solid var(--border)' }}
      >
        <div>
          <h1 className="text-lg font-bold" style={{ color: 'var(--accent-primary)' }}>
            Autoresearch Dashboard
          </h1>
          <p className="text-[10px] mt-0.5" style={{ color: 'var(--text-dim)' }}>
            Vol regime framework diagnostics &amp; insights
          </p>
        </div>
      </header>

      {/* Tab nav */}
      <nav
        className="sticky top-[57px] z-40 px-6 flex gap-1 overflow-x-auto"
        style={{ background: 'var(--bg-primary)', borderBottom: '1px solid var(--border)' }}
      >
        {TABS.map(t => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className="px-4 py-2.5 text-[11px] font-medium whitespace-nowrap transition-colors"
            style={{
              color: tab === t.key ? 'var(--accent-primary)' : 'var(--text-muted)',
              borderBottom: tab === t.key ? '2px solid var(--accent-primary)' : '2px solid transparent',
            }}
          >
            {t.label}
          </button>
        ))}
      </nav>

      {/* Tab content */}
      <main className="px-6 py-6 max-w-[1400px] mx-auto">
        {tab === 'baseline' && <BaselineOverview />}
        {tab === 'distribution' && <RegimeDistribution />}
        {tab === 'features' && <FeatureAnalysis />}
        {tab === 'ablation' && <AblationStudy />}
        {tab === 'rolling' && <RollingValidation />}
        {tab === 'boundary' && <BoundarySensitivity />}
      </main>
    </div>
  )
}
