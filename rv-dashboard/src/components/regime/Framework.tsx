'use client'

import { useRef, useCallback, useState } from 'react'

export function Framework() {
  const contentRef = useRef<HTMLDivElement>(null)
  const [exporting, setExporting] = useState(false)

  const handleDownload = useCallback(async () => {
    if (!contentRef.current) return
    setExporting(true)
    try {
      const html2canvas = (await import('html2canvas-pro')).default
      const { jsPDF } = await import('jspdf')

      const canvas = await html2canvas(contentRef.current, {
        backgroundColor: '#060609',
        scale: 2,
        useCORS: true,
      })

      const imgWidth = 297
      const imgHeight = (canvas.height * imgWidth) / canvas.width
      const pdf = new jsPDF({ orientation: 'landscape', unit: 'mm', format: 'a4' })
      const pageHeight = 210

      if (imgHeight > pageHeight) {
        // Multi-page: slice canvas into pages
        const pageCanvas = document.createElement('canvas')
        pageCanvas.width = canvas.width
        const pxPerPage = Math.floor((canvas.width * pageHeight) / imgWidth)
        let yOffset = 0
        let pageNum = 0

        while (yOffset < canvas.height) {
          const sliceHeight = Math.min(pxPerPage, canvas.height - yOffset)
          pageCanvas.height = sliceHeight
          const ctx = pageCanvas.getContext('2d')!
          ctx.drawImage(canvas, 0, yOffset, canvas.width, sliceHeight, 0, 0, canvas.width, sliceHeight)

          const sliceImgHeight = (sliceHeight * imgWidth) / canvas.width
          if (pageNum > 0) pdf.addPage()
          pdf.addImage(pageCanvas.toDataURL('image/png'), 'PNG', 0, 0, imgWidth, sliceImgHeight)

          yOffset += sliceHeight
          pageNum++
        }
      } else {
        const yOff = (pageHeight - imgHeight) / 2
        pdf.addImage(canvas.toDataURL('image/png'), 'PNG', 0, yOff, imgWidth, imgHeight)
      }

      pdf.save('regime_classification_framework.pdf')
    } catch (err) {
      console.error('PDF export failed:', err)
    } finally {
      setExporting(false)
    }
  }, [])

  return (
    <div className="space-y-6">
      <div ref={contentRef} className="space-y-6" style={{ padding: 4 }}>

        {/* Title + Download */}
        <div className="rounded-2xl border border-[var(--border)] p-8" style={{ backgroundColor: 'var(--bg-card)' }}>
          <div className="flex items-center justify-between mb-5">
            <h2 className="text-lg font-bold text-[var(--text-primary)] tracking-tight uppercase">
              Volatility Regime Classification Framework
            </h2>
            <button
              onClick={handleDownload}
              disabled={exporting}
              className="flex items-center gap-2 px-4 py-2 rounded-lg text-[11px] font-mono font-semibold uppercase tracking-wider transition-all hover:brightness-110"
              style={{
                backgroundColor: 'rgba(168,85,247,0.15)',
                color: '#c084fc',
                border: '1px solid rgba(168,85,247,0.3)',
                opacity: exporting ? 0.5 : 1,
                cursor: exporting ? 'wait' : 'pointer',
              }}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="7 10 12 15 17 10" />
                <line x1="12" y1="15" x2="12" y2="3" />
              </svg>
              {exporting ? 'Exporting...' : 'Download PDF'}
            </button>
          </div>

          {/* The Question */}
          <div className="mb-6 p-4 rounded-xl" style={{ backgroundColor: 'rgba(168,85,247,0.06)', border: '1px solid rgba(168,85,247,0.15)' }}>
            <p className="text-[12px] text-[var(--text-secondary)] leading-relaxed">
              We sell Nifty weekly options intraday using three strategies — DM (delta neutral), WC (directional mid-day), and Orion (trend rider).
              Most days are profitable, but on some days ALL three strategies lose money together. These &quot;all-lose&quot; days are what we want to predict
              and avoid. This framework labels every trading day as one of 8 states — from very safe to very dangerous — using only information
              available before market open.
            </p>
          </div>

          {/* How we built it — 4 steps */}
          <h3 className="text-[13px] font-bold text-[var(--text-primary)] uppercase tracking-wider mb-4">
            How we built it — 4 Steps
          </h3>

          <div className="space-y-4">
            {/* Step 1 */}
            <div className="flex gap-4">
              <div className="flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-[11px] font-bold" style={{ backgroundColor: 'rgba(96,165,250,0.15)', color: '#60a5fa' }}>1</div>
              <div className="flex-1">
                <p className="text-[12px] font-semibold text-[var(--text-primary)] mb-1">Data Exploration — Which numbers matter?</p>
                <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed">
                  We tested 6 volatility features (IV level, Parkinson vol, VRP, IV change, Efficiency, Yang-Zhang RV)
                  by splitting 724 days into 5 equal bins for each feature and checking: which bins have fewer all-lose days?
                  <span className="text-[var(--text-muted)]"> Key finding: DM and Orion are natural mirrors (correlation -0.23) — when one loses, the other usually wins.
                  All-lose days happen in the &quot;boring middle&quot; where moves are big enough to hurt DM but lack direction for Orion/WC.</span>
                </p>
              </div>
            </div>

            {/* Step 2 */}
            <div className="flex gap-4">
              <div className="flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-[11px] font-bold" style={{ backgroundColor: 'rgba(96,165,250,0.15)', color: '#60a5fa' }}>2</div>
              <div className="flex-1">
                <p className="text-[12px] font-semibold text-[var(--text-primary)] mb-1">Feature Ranking — Rank by usefulness, remove noise</p>
                <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed">
                  Each feature was scored on three criteria: (1) all-lose spread between best and worst bins, (2) ability to separate individual strategy returns,
                  and (3) combined portfolio return spread. Result: IV level wins on spread (+12pp), Parkinson vol on strategy discrimination, and IV change
                  on universal direction. Efficiency was dropped (unpredictable, zero persistence). Yang-Zhang was dropped (redundant with Parkinson, r=0.69).
                  VRP was kept as a sub-signal but has 0pp all-lose spread — it tells you which strategy wins, not whether you should trade.
                </p>
              </div>
            </div>

            {/* Step 3 */}
            <div className="flex gap-4">
              <div className="flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-[11px] font-bold" style={{ backgroundColor: 'rgba(96,165,250,0.15)', color: '#60a5fa' }}>3</div>
              <div className="flex-1">
                <p className="text-[12px] font-semibold text-[var(--text-primary)] mb-1">Feature Selection — The Lagging Problem</p>
                <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed">
                  We need inputs known <em>before</em> 9:15 AM. Most raw features don&apos;t persist day-to-day (today&apos;s value ≠ tomorrow&apos;s), so they
                  can&apos;t predict anything. Solution: 5-day averaging. This transforms noisy daily numbers into stable, predictive signals.
                  Parkinson vol goes from AC 0.27 (useless) to AC 0.90 (highly persistent) with 5d averaging.
                  <span className="text-[var(--text-muted)]"> Final 3 inputs: IV_5d (level backbone, AC 0.78), PK_5d (realized movement, AC 0.90), IV_chg_5d (direction, AC 0.64).
                  All are 5-day averages, all lagged by one day.</span>
                </p>
              </div>
            </div>

            {/* Step 4 */}
            <div className="flex gap-4">
              <div className="flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-[11px] font-bold" style={{ backgroundColor: 'rgba(96,165,250,0.15)', color: '#60a5fa' }}>4</div>
              <div className="flex-1">
                <p className="text-[12px] font-semibold text-[var(--text-primary)] mb-1">Regime Construction — Build the 8 states</p>
                <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed">
                  IV level creates 3 tiers using boundaries at 12 and 17 (tested 9 configs, chosen for best spread + streak balance).
                  Within each tier, different signals dominate:
                  at <span style={{ color: '#4ade80' }}>L1</span> (IV&lt;12) and <span style={{ color: '#f87171' }}>L3</span> (IV&gt;17), PK/IV ratio alone splits days into Safe vs Exposed.
                  At <span style={{ color: '#60a5fa' }}>L2</span> (IV 12-17), both PK/IV and IV direction matter — giving 4 sub-states (Safe, Caution-A, Caution-B, Risky).
                  This isn&apos;t curve-fitting: each level genuinely has different dynamics discovered by testing the same features everywhere.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* 3 Inputs */}
        <div className="grid grid-cols-3 gap-4">
          {INPUTS.map((input) => (
            <div key={input.name} className="rounded-2xl border border-[var(--border)] p-5" style={{ backgroundColor: 'var(--bg-card)' }}>
              <div className="flex items-center gap-2 mb-2">
                <div className="w-6 h-6 rounded flex items-center justify-center text-[10px] font-bold" style={{ backgroundColor: input.bg, color: input.color }}>
                  {input.num}
                </div>
                <span className="text-[13px] font-bold font-mono text-[var(--text-primary)]">{input.name}</span>
              </div>
              <p className="text-[11px] text-[var(--text-muted)] mb-2">{input.formula}</p>
              <p className="text-[11px] leading-relaxed text-[var(--text-secondary)]">{input.role}</p>
              <div className="mt-3 pt-2 border-t border-[var(--border)] flex items-center gap-2">
                <span className="text-[10px] font-mono uppercase tracking-wider text-[var(--text-dim)]">Key stat</span>
                <span className="text-[11px] font-mono font-semibold" style={{ color: input.color }}>{input.stat}</span>
              </div>
            </div>
          ))}
        </div>

        {/* The Unified Story */}
        <div className="rounded-2xl border border-[var(--border)] p-6" style={{ backgroundColor: 'var(--bg-card)' }}>
          <h3 className="text-[13px] font-bold text-[var(--text-primary)] uppercase tracking-wider mb-3">
            The Unified Story
          </h3>
          <p className="text-[12px] leading-relaxed text-[var(--text-secondary)]" style={{ maxWidth: 950 }}>
            Risk at every level boils down to one thing: <span className="text-[var(--text-primary)] font-semibold">realized movement eating into premium cushion</span> (PK/IV ratio).
            When PK/IV is low, the market moved less than what was priced in — you keep the premium safely.
            When PK/IV is high, the market ate through most of the cushion — losses are likely.
            At L2 (moderate IV), IV direction adds a second layer: falling IV means the cushion is widening (safer), rising IV means it&apos;s shrinking (riskier).
            At L1 and L3, IV direction is noise — at L1 because changes are too small, at L3 because IV is almost always rising (86% of days).
          </p>
        </div>

        {/* 8 States Table */}
        <div className="rounded-2xl border border-[var(--border)] p-6" style={{ backgroundColor: 'var(--bg-card)' }}>
          <h3 className="text-[13px] font-bold text-[var(--text-primary)] uppercase tracking-wider mb-4">
            The 8 Regime States
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-[11px] font-mono">
              <thead>
                <tr className="text-[var(--text-dim)] text-[10px] uppercase tracking-wider">
                  <th className="text-left py-2 px-3">State</th>
                  <th className="text-left py-2 px-3">Rule</th>
                  <th className="text-right py-2 px-3">Days</th>
                  <th className="text-right py-2 px-3">%</th>
                  <th className="text-right py-2 px-3">AL%</th>
                  <th className="text-right py-2 px-3">AW%</th>
                  <th className="text-right py-2 px-3">Port Avg</th>
                  <th className="text-right py-2 px-3">Sharpe</th>
                </tr>
              </thead>
              <tbody>
                {STATES.map((s) => (
                  <tr key={s.name} className="border-t border-[var(--border)]" style={{ backgroundColor: s.name === 'Overall' ? 'rgba(168,85,247,0.06)' : undefined }}>
                    <td className="py-2 px-3 font-semibold" style={{ color: s.color }}>{s.name}</td>
                    <td className="py-2 px-3 text-[var(--text-muted)]">{s.rule}</td>
                    <td className="py-2 px-3 text-right text-[var(--text-secondary)]">{s.days}</td>
                    <td className="py-2 px-3 text-right text-[var(--text-secondary)]">{s.pct}</td>
                    <td className="py-2 px-3 text-right font-semibold" style={{ color: s.alColor }}>{s.al}</td>
                    <td className="py-2 px-3 text-right text-[var(--text-secondary)]">{s.aw}</td>
                    <td className="py-2 px-3 text-right text-[var(--text-secondary)]">{s.portAvg}</td>
                    <td className="py-2 px-3 text-right font-semibold text-[var(--text-primary)]">{s.sharpe}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Adaptive Boundaries */}
        <div className="rounded-2xl border border-[var(--border)] p-6" style={{ backgroundColor: 'var(--bg-card)' }}>
          <h3 className="text-[13px] font-bold text-[var(--text-primary)] uppercase tracking-wider mb-3">
            Adaptive IV Boundaries
          </h3>
          <p className="text-[12px] leading-relaxed text-[var(--text-secondary)]" style={{ maxWidth: 950 }}>
            The fixed boundaries [12, 17] were calibrated on Jan 2023 – Jan 2026 data. But IV regimes shift over time — 2021-22 had
            persistently elevated IV where &quot;normal&quot; was 17-22 instead of 10-15. The adaptive layer handles this: when more than 50% of the
            trailing 45 days have IV above 17, boundaries shift from [12, 17] to [17, 22]. This ensures L1/L2/L3 always represent
            &quot;low / moderate / high&quot; relative to the current environment, not just absolute numbers.
          </p>
        </div>

        {/* Roadmap */}
        <div className="rounded-2xl border-2 border-[var(--border)] overflow-hidden" style={{ background: 'linear-gradient(135deg, rgba(88,166,255,0.03), var(--bg-card))' }}>
          <div className="p-6">
            <h3 className="text-[13px] font-bold text-[#58a6ff] uppercase tracking-wider mb-1">Roadmap</h3>
            <p className="text-[11px] text-[var(--text-muted)] mb-5">Where we are headed — building a complete system that tells you when to sell options, how much to allocate, and how to execute.</p>

            {/* The three end goals */}
            <div className="grid grid-cols-3 gap-4 mb-6">
              <div className="rounded-xl border border-[#60a5fa] p-4" style={{ backgroundColor: 'rgba(96,165,250,0.05)' }}>
                <div className="text-[10px] uppercase tracking-wider mb-1" style={{ color: '#60a5fa' }}>Goal 1</div>
                <div className="text-[13px] font-bold text-[var(--text-primary)] mb-1">Vol Score</div>
                <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed">
                  A single number that rates how favorable today is for option selling. Think of it like a weather forecast — green means safe to sell, red means stay away.
                </p>
              </div>
              <div className="rounded-xl border border-[#c084fc] p-4" style={{ backgroundColor: 'rgba(168,85,247,0.05)' }}>
                <div className="text-[10px] uppercase tracking-wider mb-1" style={{ color: '#c084fc' }}>Goal 2</div>
                <div className="text-[13px] font-bold text-[var(--text-primary)] mb-1">Strategy Allocation</div>
                <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed">
                  Based on the regime, decide which strategies to run and how much capital each one gets. Safe days get full allocation, risky days scale down or sit out.
                </p>
              </div>
              <div className="rounded-xl border border-[#f59e0b] p-4" style={{ backgroundColor: 'rgba(245,158,11,0.05)' }}>
                <div className="text-[10px] uppercase tracking-wider mb-1" style={{ color: '#f59e0b' }}>Goal 3</div>
                <div className="text-[13px] font-bold text-[var(--text-primary)] mb-1">Micro Structure</div>
                <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed">
                  Fine-tune execution: when exactly to enter trades intraday, and detect when the regime is shifting mid-day so you can react before end-of-day signals catch up.
                </p>
              </div>
            </div>

            {/* Next steps */}
            <h4 className="text-[11px] font-bold text-[var(--text-dim)] uppercase tracking-wider mb-3">Next Steps</h4>
            <div className="space-y-3">

              <div className="flex gap-4 items-start">
                <div className="flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-[11px] font-bold mt-0.5" style={{ backgroundColor: 'rgba(96,165,250,0.15)', color: '#60a5fa' }}>1</div>
                <div>
                  <p className="text-[12px] font-semibold text-[var(--text-primary)] mb-0.5">DTE-wise Regime Analysis</p>
                  <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed">
                    Break down how each regime state performs on different days-to-expiry. A regime might be safe on DTE 3 but risky on DTE 0 when gamma is highest. Knowing this lets us skip dangerous expiry days in bad regimes.
                  </p>
                </div>
              </div>

              <div className="flex gap-4 items-start">
                <div className="flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-[11px] font-bold mt-0.5" style={{ backgroundColor: 'rgba(96,165,250,0.15)', color: '#60a5fa' }}>2</div>
                <div>
                  <p className="text-[12px] font-semibold text-[var(--text-primary)] mb-0.5">Morning-to-Evening IV Change</p>
                  <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed">
                    Track how IV moves from market open to close. If IV drops during the day, sellers win (theta decay is working). If IV spikes intraday, it signals stress. This gives an early warning layer on top of daily regime classification.
                  </p>
                </div>
              </div>

              <div className="flex gap-4 items-start">
                <div className="flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-[11px] font-bold mt-0.5" style={{ backgroundColor: 'rgba(96,165,250,0.15)', color: '#60a5fa' }}>3</div>
                <div>
                  <p className="text-[12px] font-semibold text-[var(--text-primary)] mb-0.5">Outlier Effect Analysis</p>
                  <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed">
                    Remove extreme one-off moves (budget days, flash crashes) and see what strategy returns look like without them. This separates &quot;the regime is genuinely risky&quot; from &quot;one bad day ruined the stats.&quot; Helps us size positions better by understanding tail risk vs. normal risk.
                  </p>
                </div>
              </div>

              <div className="flex gap-4 items-start">
                <div className="flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-[11px] font-bold mt-0.5" style={{ backgroundColor: 'rgba(168,85,247,0.15)', color: '#c084fc' }}>4</div>
                <div>
                  <p className="text-[12px] font-semibold text-[var(--text-primary)] mb-0.5">New Feature Validation Framework</p>
                  <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed">
                    Build a pipeline that lets us test new signals (like order flow, OI change, skew) in a disciplined way. Before any new feature gets added, it must prove it carries information the existing features don&apos;t already capture — no redundancy, no overfitting.
                  </p>
                </div>
              </div>

              <div className="flex gap-4 items-start">
                <div className="flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-[11px] font-bold mt-0.5" style={{ backgroundColor: 'rgba(168,85,247,0.15)', color: '#c084fc' }}>5</div>
                <div>
                  <p className="text-[12px] font-semibold text-[var(--text-primary)] mb-0.5">ML-Driven Feature &amp; Boundary Selection</p>
                  <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed">
                    Replace hand-picked thresholds (like IV = 12, 17) with data-driven choices. Use machine learning to find the best features, the best split points, and the best state definitions — while keeping it interpretable so we understand why the model makes each decision.
                  </p>
                </div>
              </div>

              <div className="flex gap-4 items-start">
                <div className="flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-[11px] font-bold mt-0.5" style={{ backgroundColor: 'rgba(168,85,247,0.15)', color: '#c084fc' }}>6</div>
                <div>
                  <p className="text-[12px] font-semibold text-[var(--text-primary)] mb-0.5">Automatic OOS Validation</p>
                  <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed">
                    Every time a new feature or boundary is proposed, the system automatically runs it through out-of-sample tests — rank correlation, safe-beats-exposed checks, and sample size requirements. No feature gets promoted without passing these gates.
                  </p>
                </div>
              </div>

              <div className="flex gap-4 items-start">
                <div className="flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-[11px] font-bold mt-0.5" style={{ backgroundColor: 'rgba(245,158,11,0.15)', color: '#f59e0b' }}>7</div>
                <div>
                  <p className="text-[12px] font-semibold text-[var(--text-primary)] mb-0.5">Intraday Dynamic Vol Regimes</p>
                  <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed">
                    Move beyond daily classification. Use intraday features and methods like Hidden Markov Models to detect regime shifts as they happen during the trading day. Track transition probabilities — if the market is in &quot;calm&quot; at 10 AM, what&apos;s the chance it flips to &quot;stressed&quot; by 2 PM? This powers real-time execution decisions.
                  </p>
                </div>
              </div>

            </div>
          </div>
        </div>

      </div>{/* close contentRef */}
    </div>
  )
}

const INPUTS = [
  {
    num: '1',
    name: 'IV_5d',
    formula: '5-day average of 7-DTE ATM implied volatility, lagged T-1',
    role: 'The regime backbone. Tells you how much premium the market is pricing in. Most persistent feature (AC 0.78) — today\'s IV strongly predicts tomorrow\'s. Creates the 3 tiers: L1 (low, <12), L2 (moderate, 12-17), L3 (high, >17). Also the denominator in PK/IV ratio.',
    stat: 'AC: 0.78 · AL spread: 12pp',
    color: '#c084fc',
    bg: 'rgba(168,85,247,0.15)',
  },
  {
    num: '2',
    name: 'PK_5d',
    formula: '5-day Parkinson volatility (high-low range), lagged T-1',
    role: 'How much the market actually moved (realized vol). The numerator in PK/IV ratio — when PK is high relative to IV, the premium cushion is thin and losses are likely. Strongest single strategy discriminator: DM correlation -0.52, all-lose spread 8pp.',
    stat: 'DM corr: -0.52 · AL spread: 8pp',
    color: '#fbbf24',
    bg: 'rgba(251,191,36,0.15)',
  },
  {
    num: '3',
    name: 'IV_chg_5d',
    formula: '5-day average of daily IV change (direction), lagged T-1',
    role: 'Is volatility rising or falling? The only feature where ALL strategies move together — falling IV is good for everyone. Only used at L2 where it has enough persistence (AC 0.49). At L1 changes are too small; at L3, IV is almost always rising (86% of days).',
    stat: 'AL spread: 6pp · L2 direction signal',
    color: '#4ade80',
    bg: 'rgba(74,222,128,0.15)',
  },
]

const STATES = [
  { name: 'L1 Safe', rule: 'IV<12, PK/IV≤0.63', days: 162, pct: '22%', al: '4%', alColor: '#4ade80', aw: '17%', portAvg: '+0.053%', sharpe: '3.13', color: '#4ade80' },
  { name: 'L1 Exposed', rule: 'IV<12, PK/IV>0.63', days: 161, pct: '22%', al: '9%', alColor: '#fbbf24', aw: '14%', portAvg: '+0.035%', sharpe: '1.24', color: '#fbbf24' },
  { name: 'L2 Safe', rule: 'IV 12-17, Lo PK/IV, IV fall', days: 92, pct: '13%', al: '3%', alColor: '#4ade80', aw: '20%', portAvg: '+0.111%', sharpe: '6.71', color: '#4ade80' },
  { name: 'L2 Caution-A', rule: 'IV 12-17, Hi PK/IV, IV fall', days: 67, pct: '9%', al: '9%', alColor: '#fbbf24', aw: '16%', portAvg: '+0.046%', sharpe: '1.76', color: '#60a5fa' },
  { name: 'L2 Caution-B', rule: 'IV 12-17, Lo PK/IV, IV rise', days: 69, pct: '10%', al: '14%', alColor: '#f87171', aw: '13%', portAvg: '+0.031%', sharpe: '0.68', color: '#60a5fa' },
  { name: 'L2 Risky', rule: 'IV 12-17, Hi PK/IV, IV rise', days: 94, pct: '13%', al: '13%', alColor: '#f87171', aw: '14%', portAvg: '+0.042%', sharpe: '1.32', color: '#f87171' },
  { name: 'L3 Safe', rule: 'IV>17, PK/IV≤0.67', days: 40, pct: '6%', al: '5%', alColor: '#4ade80', aw: '20%', portAvg: '+0.135%', sharpe: '5.80', color: '#4ade80' },
  { name: 'L3 Exposed', rule: 'IV>17, PK/IV>0.67', days: 39, pct: '5%', al: '21%', alColor: '#f87171', aw: '13%', portAvg: '+0.023%', sharpe: '0.04', color: '#f87171' },
  { name: 'Overall', rule: '', days: 724, pct: '100%', al: '9%', alColor: 'var(--text-secondary)', aw: '15%', portAvg: '+0.055%', sharpe: '2.52', color: '#c084fc' },
]
