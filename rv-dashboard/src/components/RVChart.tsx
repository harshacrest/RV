'use client'

import { useEffect, useRef, useState, useCallback } from 'react'
import { createChart, ColorType, LineSeries, CandlestickSeries, type IChartApi } from 'lightweight-charts'
import { fetchRVTimeseries } from '@/lib/api'
import type { RVDataPoint } from '@/lib/types'

const CHART_OPTIONS = {
  layout: {
    background: { type: ColorType.Solid, color: '#0e0e16' } as const,
    textColor: '#9898ab',
    fontFamily: 'ui-monospace, monospace',
    fontSize: 11,
  },
  grid: {
    vertLines: { color: '#121218' },
    horzLines: { color: '#121218' },
  },
  height: 420,
  rightPriceScale: { borderColor: '#1a1a26' },
  timeScale: { borderColor: '#1a1a26' },
  crosshair: {
    horzLine: { color: '#3a3a4e', style: 2 as const },
    vertLine: { color: '#3a3a4e', style: 2 as const },
  },
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      onClick={() => onChange(!checked)}
      className="flex items-center gap-1.5 px-3 py-1 rounded-md text-[11px] font-mono transition-all"
      style={{
        backgroundColor: checked ? 'rgba(0,230,118,0.1)' : 'transparent',
        color: checked ? '#00e676' : 'var(--text-dim)',
        border: `1px solid ${checked ? 'rgba(0,230,118,0.25)' : 'var(--border)'}`,
      }}
    >
      <span className="inline-block w-1.5 h-1.5 rounded-full" style={{ backgroundColor: checked ? '#00e676' : 'var(--text-dim)' }} />
      {label}
    </button>
  )
}

function RVLevelChart({ data }: { data: RVDataPoint[] }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const [showSpot, setShowSpot] = useState(true)
  const [showIV, setShowIV] = useState(true)

  const buildChart = useCallback(() => {
    if (!containerRef.current || data.length === 0) return
    if (chartRef.current) chartRef.current.remove()

    const container = containerRef.current
    const chart = createChart(container, { ...CHART_OPTIONS, width: container.clientWidth })
    chartRef.current = chart

    if (showSpot) {
      const candles = chart.addSeries(CandlestickSeries, {
        upColor: '#00e676', downColor: '#ff5252',
        borderUpColor: '#00e676', borderDownColor: '#ff5252',
        wickUpColor: '#00e676', wickDownColor: '#ff5252',
        priceScaleId: 'left',
      })
      candles.setData(data.map(d => ({ time: d.date as string, open: d.open, high: d.high, low: d.low, close: d.close })))
      chart.priceScale('left').applyOptions({ borderColor: '#1a1a26', scaleMargins: { top: 0.05, bottom: 0.05 } })
    }

    const rvSeries = chart.addSeries(LineSeries, {
      color: '#00d4ff', lineWidth: 2, priceScaleId: 'right', title: 'RV Today',
    })
    rvSeries.setData(data.filter(d => d.RV_today != null).map(d => ({ time: d.date as string, value: d.RV_today! })))

    if (showIV) {
      const ivSeries = chart.addSeries(LineSeries, {
        color: '#ffc107', lineWidth: 2, priceScaleId: 'right', title: 'IV 7d',
        lineStyle: 2,
      })
      ivSeries.setData(data.filter(d => d.IV_7d != null).map(d => ({ time: d.date as string, value: d.IV_7d! })))
    }

    chart.priceScale('right').applyOptions({ borderColor: '#1a1a26', scaleMargins: { top: 0.1, bottom: 0.1 } })
    chart.timeScale().fitContent()

    const ro = new ResizeObserver(entries => {
      for (const e of entries) chart.applyOptions({ width: e.contentRect.width })
    })
    ro.observe(container)
    return () => { ro.disconnect(); chart.remove(); chartRef.current = null }
  }, [data, showSpot, showIV])

  useEffect(() => {
    const cleanup = buildChart()
    return () => cleanup?.()
  }, [buildChart])

  return (
    <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-5">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-sm font-semibold text-[var(--text-primary)] uppercase tracking-wider">RV Today (Yang-Zhang)</h2>
          <div className="flex items-center gap-4 text-[11px] text-[var(--text-muted)] mt-1">
            <span className="flex items-center gap-1"><span className="inline-block w-2.5 h-0.5 rounded" style={{ backgroundColor: '#00d4ff' }} />RV Today (%)</span>
            {showIV && <span className="flex items-center gap-1"><span className="inline-block w-2.5 h-0.5 rounded" style={{ backgroundColor: '#ffc107' }} />IV 7d Forward (%)</span>}
            {showSpot && <span className="flex items-center gap-1"><span className="inline-block w-2.5 h-2.5 rounded-full" style={{ backgroundColor: '#00e676' }} />Spot (LHS)</span>}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Toggle label="Show IV" checked={showIV} onChange={setShowIV} />
          <Toggle label="Show Spot" checked={showSpot} onChange={setShowSpot} />
        </div>
      </div>
      <div ref={containerRef} />
    </div>
  )
}

export function RVPlotsSection({ startDate, endDate }: { startDate: string; endDate: string }) {
  const [data, setData] = useState<RVDataPoint[]>([])

  useEffect(() => {
    fetchRVTimeseries(startDate, endDate).then(setData)
  }, [startDate, endDate])

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-40 text-[var(--text-dim)] text-sm font-mono">
        Loading RV data…
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-5">
      <RVLevelChart data={data} />
    </div>
  )
}
