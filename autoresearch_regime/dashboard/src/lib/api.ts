const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5502'

async function fetchJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`)
  if (!res.ok) throw new Error(`API error: ${res.status}`)
  return res.json()
}

import type {
  Baseline, FeatureCorrelations, RankStability,
  BoundarySensitivity, AblationData, RollingValidation, NormSensitivity,
  RegimeDistEntry,
} from './types'

export const fetchBaseline = () => fetchJSON<Baseline>('/api/baseline')
export const fetchRegimeDistribution = () => fetchJSON<Record<string, RegimeDistEntry>>('/api/regime-distribution')
export const fetchFeatureCorrelations = () => fetchJSON<FeatureCorrelations>('/api/feature-correlations')
export const fetchStrategyCorrelations = () => fetchJSON<Record<string, unknown>>('/api/strategy-correlations')
export const fetchNormalization = () => fetchJSON<NormSensitivity>('/api/normalization')
export const fetchRankStability = () => fetchJSON<RankStability>('/api/rank-stability')
export const fetchBoundarySensitivity = () => fetchJSON<BoundarySensitivity>('/api/boundary-sensitivity')
export const fetchAblation = () => fetchJSON<AblationData>('/api/ablation')
export const fetchRollingValidation = () => fetchJSON<RollingValidation>('/api/rolling-validation')
