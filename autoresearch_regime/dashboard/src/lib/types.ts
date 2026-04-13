export interface Baseline {
  composite_score: number
  val_sharpe: number
  safe_separation: number
  rank_stability: number
  state_coverage: number
  val_days: number
  train_days: number
  n_states_used: number
  min_state_days: number
}

export interface RegimeDistEntry {
  state: string
  Train: number
  Val: number
  OOS1: number
  OOS2: number
  Total: number
}

export interface CorrelationPair {
  0: string  // feature 1
  1: string  // feature 2
  2: number  // correlation
}

export interface FeatureCorrelations {
  high_corr_pairs: CorrelationPair[]
  feature_ics: Record<string, number>
}

export interface RankStability {
  rho: number
  pval: number
  perm_pval: number
  train_means: Record<string, number>
  val_means: Record<string, number>
}

export interface BoundaryPoint {
  l1: number
  l2: number
  score: number
}

export interface BoundarySensitivity {
  best_l1: number
  best_l2: number
  best_score: number
  grid: BoundaryPoint[]
}

export interface AblationEntry {
  name: string
  score: number
  delta: number
  impact: string
}

export interface AblationData {
  baseline_score: number
  ablations: AblationEntry[]
}

export interface RollingWindow {
  val_start: string
  val_end: string
  n_days: number
  regime_sharpe: number
  equal_sharpe: number
  delta: number
  winner: string
}

export interface RollingValidation {
  windows: RollingWindow[]
  summary: {
    total_windows: number
    regime_wins: number
    equal_wins: number
    avg_delta: number
  }
}

export interface NormSensitivity {
  sharpe_norm_sweep: Record<string, { 'cap_1.5': number; 'cap_1.0': number }>
  safe_sep_norm_sweep: Record<string, { term: number }>
}

export type TabKey = 'baseline' | 'distribution' | 'features' | 'ablation' | 'rolling' | 'boundary'
