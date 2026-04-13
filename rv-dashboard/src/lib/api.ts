import { API_BASE } from './config'
import type {
  PlainReturnsData, FeatureBucketData, CompositeData, RVDataPoint,
  RegimeStatesData, RegimeTimeseriesPoint, RegimeStrategyData,
  RegimeTransitionsData, RegimeFeatureInput, AllLoseData,
  ExplorationFeatureMeta, DataExplorationResult,
  FeatureRankingData, FeatureSelectionData, RegimeConstructionData,
  SnapshotComparisonData, AdaptiveOOSData, AlphaInsightsData,
} from './types'

function buildParams(opts: Record<string, string | undefined>): string {
  const params = new URLSearchParams()
  for (const [k, v] of Object.entries(opts)) {
    if (v) params.set(k, v)
  }
  const str = params.toString()
  return str ? `?${str}` : ''
}

/** Convert dte number|null to string|undefined for buildParams */
function dteStr(dte?: number | null): string | undefined {
  return dte != null ? String(dte) : undefined
}

async function fetchJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`)
  if (!res.ok) throw new Error(`API error: ${res.status}`)
  return res.json()
}

export function fetchPlainReturns(strategy: string, startDate?: string, endDate?: string, snapshot?: string) {
  return fetchJSON<PlainReturnsData>(`/api/plain-returns/${strategy}${buildParams({ start_date: startDate, end_date: endDate, snapshot })}`)
}

export function fetchFeatureBuckets(strategy: string, feature: string, startDate?: string, endDate?: string, snapshot?: string) {
  return fetchJSON<FeatureBucketData>(`/api/feature-buckets/${strategy}/${feature}${buildParams({ start_date: startDate, end_date: endDate, snapshot })}`)
}

export function fetchComposite(strategy: string, rowFeature: string, colFeature: string, startDate?: string, endDate?: string, snapshot?: string) {
  return fetchJSON<CompositeData>(`/api/composite/${strategy}${buildParams({ row_feature: rowFeature, col_feature: colFeature, start_date: startDate, end_date: endDate, snapshot })}`)
}

export function fetchRVTimeseries(startDate?: string, endDate?: string) {
  return fetchJSON<RVDataPoint[]>(`/api/rv-timeseries${buildParams({ start_date: startDate, end_date: endDate })}`)
}

// ── Regime Classification ──

export function fetchRegimeStates(startDate?: string, endDate?: string, snapshot?: string, dte?: number | null) {
  return fetchJSON<RegimeStatesData>(`/api/regime/states${buildParams({ start_date: startDate, end_date: endDate, snapshot, dte: dteStr(dte) })}`)
}

export function fetchRegimeTimeseries(startDate?: string, endDate?: string, snapshot?: string, dte?: number | null) {
  return fetchJSON<RegimeTimeseriesPoint[]>(`/api/regime/timeseries${buildParams({ start_date: startDate, end_date: endDate, snapshot, dte: dteStr(dte) })}`)
}

export function fetchRegimeStrategy(strategy: string, startDate?: string, endDate?: string, snapshot?: string, dte?: number | null) {
  return fetchJSON<RegimeStrategyData>(`/api/regime/strategy/${strategy}${buildParams({ start_date: startDate, end_date: endDate, snapshot, dte: dteStr(dte) })}`)
}

export function fetchRegimeTransitions(startDate?: string, endDate?: string, snapshot?: string, dte?: number | null) {
  return fetchJSON<RegimeTransitionsData>(`/api/regime/transitions${buildParams({ start_date: startDate, end_date: endDate, snapshot, dte: dteStr(dte) })}`)
}

export function fetchRegimeFeatureInputs(startDate?: string, endDate?: string, snapshot?: string, dte?: number | null) {
  return fetchJSON<RegimeFeatureInput[]>(`/api/regime/feature-inputs${buildParams({ start_date: startDate, end_date: endDate, snapshot, dte: dteStr(dte) })}`)
}

export function fetchRegimeAllLose(startDate?: string, endDate?: string, snapshot?: string, dte?: number | null) {
  return fetchJSON<AllLoseData>(`/api/regime/all-lose${buildParams({ start_date: startDate, end_date: endDate, snapshot, dte: dteStr(dte) })}`)
}

// ── Steps 2-4: Feature Ranking, Selection, Construction ──

export function fetchFeatureRanking(startDate?: string, endDate?: string, snapshot?: string, dte?: number | null) {
  return fetchJSON<FeatureRankingData>(`/api/regime/feature-ranking${buildParams({ start_date: startDate, end_date: endDate, snapshot, dte: dteStr(dte) })}`)
}

export function fetchFeatureSelection(startDate?: string, endDate?: string, snapshot?: string, dte?: number | null) {
  return fetchJSON<FeatureSelectionData>(`/api/regime/feature-selection${buildParams({ start_date: startDate, end_date: endDate, snapshot, dte: dteStr(dte) })}`)
}

export function fetchRegimeConstruction(startDate?: string, endDate?: string, snapshot?: string, dte?: number | null) {
  return fetchJSON<RegimeConstructionData>(`/api/regime/regime-construction${buildParams({ start_date: startDate, end_date: endDate, snapshot, dte: dteStr(dte) })}`)
}

// ── Data Exploration ──

export function fetchExplorationFeatures() {
  return fetchJSON<ExplorationFeatureMeta[]>('/api/data-exploration/features')
}

export function fetchDataExploration(feature: string, startDate?: string, endDate?: string, snapshot?: string, dte?: number | null) {
  return fetchJSON<DataExplorationResult>(`/api/data-exploration/${feature}${buildParams({ start_date: startDate, end_date: endDate, snapshot, dte: dteStr(dte) })}`)
}

// ── Snapshot Comparison ──

export function fetchSnapshotComparison(startDate?: string, endDate?: string, dte?: number | null) {
  return fetchJSON<SnapshotComparisonData>(`/api/regime/snapshot-comparison${buildParams({ start_date: startDate, end_date: endDate, dte: dteStr(dte) })}`)
}

// ── Adaptive OOS Validation ──

export function fetchAdaptiveOOS(testPeriod: string, snapshot?: string, dte?: number | null) {
  return fetchJSON<AdaptiveOOSData>(`/api/regime/adaptive-oos${buildParams({ test_period: testPeriod, snapshot, dte: dteStr(dte) })}`)
}

// ── Alpha Insights ──

export function fetchAlphaInsights(snapshot?: string, dte?: number | null) {
  return fetchJSON<AlphaInsightsData>(`/api/alpha-insights${buildParams({ snapshot, dte: dteStr(dte) })}`)
}
