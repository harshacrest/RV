// ── Data Exploration Types ──

export interface ExplorationFeatureMeta {
  key: string
  label: string
  source: string
}

export interface EDADescriptiveStats {
  count: number
  mean: number
  median: number
  std: number
  min: number
  max: number
  skew: number
  kurtosis: number
  p5: number
  p10: number
  p25: number
  p75: number
  p90: number
  p95: number
  iqr: number
  jb_stat?: number
  jb_pvalue?: number
  is_normal?: boolean
}

export interface EDAHistogramBin {
  bin_start: number
  bin_end: number
  count: number
  bin_label: string
}

export interface EDATimeseriesPoint {
  date: string
  value: number
  roll_20_mean: number | null
  roll_20_std: number | null
  roll_50_mean: number | null
}

export interface EDAAutocorrelation {
  lag: number
  acf: number
  significant: boolean
}

export interface EDAStationarity {
  adf_statistic: number
  p_value: number
  lags_used: number
  n_obs: number
  critical_1pct: number
  critical_5pct: number
  critical_10pct: number
  is_stationary_5pct: boolean
}

export interface EDAScatterPoint {
  x: number
  y: number
}

export interface EDAFeatureVsPnl {
  pearson_r: number
  pearson_p: number
  spearman_r: number
  spearman_p: number
  scatter: EDAScatterPoint[]
}

export interface EDAQuintileBucket {
  quintile: string
  count: number
  count_with_pnl: number
  feature_mean: number
  feature_range: string
  pnl_mean: number | null
  pnl_median: number | null
  pnl_std: number | null
  win_rate: number | null
  pnl_sum: number | null
  sharpe: number | null
  al_pct: number
}

export interface EDARegimeDistribution {
  state: string
  color: string
  count: number
  mean: number | null
  median: number | null
  std: number | null
  min: number | null
  max: number | null
  p25?: number
  p75?: number
}

export interface EDAOutlierDay {
  date: string
  value: number
  regime_state: string | null
  z_score: number | null
}

export interface EDAOutliers {
  sigma_threshold: number
  sigma_count: number
  iqr_count: number
  sigma_bounds: { lower: number; upper: number }
  iqr_bounds: { lower: number; upper: number }
  days: EDAOutlierDay[]
}

export interface EDARollingStats {
  current_20d_mean: number | null
  current_20d_std: number | null
  current_50d_mean: number | null
  current_z_20d: number | null
  percentile_rank: number
}

export interface DataExplorationResult {
  feature: string
  label: string
  source: string
  descriptive_stats: EDADescriptiveStats
  histogram: EDAHistogramBin[]
  timeseries: EDATimeseriesPoint[]
  autocorrelation: EDAAutocorrelation[]
  acf_confidence_bound: number | null
  stationarity: EDAStationarity | null
  feature_vs_pnl: Record<string, EDAFeatureVsPnl>
  dte_feature_vs_pnl: Record<string, Record<string, EDAFeatureVsPnl>>
  quintile_analysis: Record<string, { buckets: EDAQuintileBucket[]; n_total: number; n_used: number; n_dropped_missing_pnl: number }>
  dte_quintile_analysis: Record<string, Record<string, { buckets: EDAQuintileBucket[]; n_total: number; n_used: number; n_dropped_missing_pnl: number }>>
  regime_distributions: EDARegimeDistribution[]
  outliers: EDAOutliers
  rolling_stats: EDARollingStats
}

// ── Strategy Analysis Types ──

export interface DrawdownPoint {
  date: string
  dd_pct: number
}

export interface StrategySummary {
  total_pct: number
  mean_daily_pct: number
  median_daily_pct: number
  std_daily_pct: number
  win_rate: number
  sharpe: number | null
  max_win_pct: number
  max_loss_pct: number
  max_drawdown_pct: number
  total_days: number
  positive_days: number
  negative_days: number
  flat_days: number
  // Extended metrics
  cagr_pct: number | null
  ann_return_pct: number
  ann_vol_pct: number | null
  sortino: number | null
  calmar: number | null
  profit_factor: number | null
  avg_win_pct: number
  avg_loss_pct: number
  payoff_ratio: number | null
  expectancy_pct: number
  skewness: number | null
  kurtosis: number | null
  max_dd_duration_days: number
  recovery_factor: number | null
  max_consec_wins: number
  max_consec_losses: number
  var_95_pct: number | null
  p95_pct: number | null
  p5_pct: number | null
  tail_ratio: number | null
  gross_profit_pct: number
  gross_loss_pct: number
  dd_series: DrawdownPoint[]
}

export interface DailyPoint {
  date: string
  pnl_pct: number
  cumulative_pct: number
}

export interface YearlyReturn {
  year: number
  total_pct: number
  mean_daily_pct: number
  std_daily_pct: number
  win_rate: number
  sharpe_pct: number | null
  max_win_pct: number
  max_loss_pct: number
  trading_days: number
}

export interface MonthlyReturn {
  year: number
  month: number
  total_pct: number
  mean_daily_pct: number
  trading_days: number
  win_rate: number
}

export interface PlainReturnsData {
  strategy: string
  date_range: [string, string]
  summary: StrategySummary
  yearly: YearlyReturn[]
  monthly: MonthlyReturn[]
  daily_timeseries: DailyPoint[]
}

// ── Feature Bucket Types ──

export interface BucketMetrics {
  label: string
  range: [number | null, number | null]
  trading_days: number
  total_pct: number
  avg_daily_pct: number
  win_rate: number
  loss_rate: number
  sharpe: number | null
  sharpe_pct: number | null
  ann_vol_pct: number | null
  max_win_pct: number
  max_loss_pct: number
  max_drawdown_pct: number
  feat_mean: number | null
  feat_max: number | null
  feat_min: number | null
  streak_mean: number | null
  streak_median: number | null
  streak_min: number | null
  streak_max: number | null
}

export interface DTECross {
  dte_labels: string[]
  feature_labels: string[]
  grid: (BucketMetrics | null)[][]
}

export interface FeatureBucketData {
  strategy: string
  feature: string
  raw_buckets: BucketMetrics[]
  percentile_buckets: BucketMetrics[]
  dte_cross?: DTECross
}

// ── Composite Cross-tab Types ──

export interface GenericCross {
  feature_labels: string[]
  static_labels: string[]
  grid: (BucketMetrics | null)[][]
  pct_feature_labels: string[]
  pct_grid: (BucketMetrics | null)[][]
}

export interface CompositeDTEComboLabel {
  row_label: string
  col_label: string
  combo_label: string
}

export interface CompositeDTECross {
  combo_labels: CompositeDTEComboLabel[]
  dte_labels: string[]
  grid: (BucketMetrics | null)[][]
}

export interface CompositeData extends GenericCross {
  strategy: string
  row_feature: string
  col_feature: string
  composite_dte_cross?: CompositeDTECross
}

// ── RV Timeseries ──

export interface RVDataPoint {
  date: string
  open: number
  high: number
  low: number
  close: number
  RV_today: number | null
  IV_7d: number | null
  IV_change_1d: number | null
  VRP_today: number | null
}

// ── Regime Classification Types ──

export interface RegimeStateMetrics {
  state: string
  color: string
  description: string
  days: number
  pct_of_total: number
  al_pct: number | null
  aw_pct: number | null
  port_avg: number | null
  sharpe: number | null
  iv_lag_mean: number | null
  pk_iv_mean: number | null
  iv_chg_5d_mean: number | null
}

export interface RegimeCurrentState {
  state: string | null
  date: string | null
  iv_lag: number | null
  pk_iv_ratio: number | null
  iv_chg_5d: number | null
  pk_5d: number | null
  iv_5d: number | null
  color: string | null
  description: string | null
}

export interface RegimeOverallMetrics {
  days: number
  port_avg: number
  sharpe: number | null
  al_pct: number | null
  aw_pct: number | null
}

export interface RegimeStatesData {
  states: RegimeStateMetrics[]
  current: RegimeCurrentState
  overall: RegimeOverallMetrics
}

export interface RegimeTimeseriesPoint {
  date: string
  regime_state: string
  color: string
  iv_lag: number | null
  pk_iv_ratio: number | null
  iv_chg_5d: number | null
  pk_5d: number | null
  iv_5d: number | null
  pk_today: number | null
  close: number
  pnl_dm: number | null
  pnl_wc: number | null
  pnl_orion: number | null
  pnl_dmo: number | null
  pnl_combined: number | null
}

export interface RegimeStrategyState {
  state: string
  color: string
  days: number
  avg_pnl: number | null
  sharpe: number | null
  win_rate: number | null
  total_pct: number | null
  max_win: number | null
  max_loss: number | null
}

export interface RegimeStrategyData {
  strategy: string
  states: RegimeStrategyState[]
}

export interface RegimeStreakStats {
  mean: number | null
  median: number | null
  min: number | null
  max: number | null
  count: number
}

export interface RegimeTransitionsData {
  states: string[]
  transition_counts: Record<string, Record<string, number>>
  transition_probs: Record<string, Record<string, number | null>>
  streak_stats: Record<string, RegimeStreakStats>
  self_transition_rate: Record<string, number | null>
}

export interface RegimeFeatureInput {
  date: string
  iv_lag: number | null
  IV_5d: number | null
  PK_5d: number | null
  PK_today: number | null
  IV_chg_5d: number | null
  PK_IV_ratio: number | null
  iv_level: string
}

// ── All-Lose Spot Analysis ──

export interface AllLoseStateSummary {
  state: string
  color: string
  total_days: number
  al_days: number
  al_pct: number
  spot_chg_mean: number | null
  spot_chg_median: number | null
  spot_chg_std: number | null
  spot_chg_min: number | null
  spot_chg_max: number | null
  spot_chg_p25: number | null
  spot_chg_p75: number | null
  intraday_range_mean: number | null
  gap_mean: number | null
  pnl_combined_mean: number | null
  pnl_dm_mean: number | null
  pnl_wc_mean: number | null
  pnl_orion_mean: number | null
}

export interface AllLoseDayRecord {
  date: string
  regime_state: string
  color: string
  close: number
  spot_chg_pct: number | null
  intraday_range_pct: number | null
  gap_pct: number | null
  pnl_combined: number | null
  pnl_dm: number | null
  pnl_wc: number | null
  pnl_orion: number | null
  iv_lag: number | null
  pk_iv_ratio: number | null
}

export interface AllLoseDistBucket {
  bucket: string
  count: number
}

export interface AllLoseOverall {
  total_al_days: number
  total_trading_days: number
  al_pct: number
  spot_chg_mean: number | null
  spot_chg_median: number | null
  spot_chg_std: number | null
  spot_down_pct: number | null
  spot_up_pct: number | null
}

export interface AllLoseData {
  states: AllLoseStateSummary[]
  days: AllLoseDayRecord[]
  distribution: AllLoseDistBucket[]
  overall: AllLoseOverall
}

// ── Strategy Meta ──

export interface StrategyMeta {
  key: string
  name: string
  accent: string
  accentRgb: string
}

export interface FeatureMeta {
  key: string
  label: string
}

// ── Step 2: Feature Ranking Types ──

export interface FeatureRankingQuintile {
  quintile: string
  range: string
  n: number
  dm_avg: number | null
  dm_wr: number | null
  dm_sharpe: number | null
  wc_avg: number | null
  wc_wr: number | null
  wc_sharpe: number | null
  orion_avg: number | null
  orion_wr: number | null
  orion_sharpe: number | null
  combined_avg: number | null
  combined_sharpe: number | null
  al_pct: number
  aw_pct: number
}

export interface FeatureRankingEntry {
  rank: number
  feature_key: string
  label: string
  dm_corr: number | null
  wc_corr: number | null
  orion_corr: number | null
  al_corr: number | null
  al_spread: number
  best_q: string
  worst_q: string
  dm_orion_gap: number | null
  verdict: string
  verdict_reason: string
  quintiles: FeatureRankingQuintile[]
  dte_quintiles: Record<string, FeatureRankingQuintile[]>
  dte_al_spread: Record<string, number>
}

export interface FrameworkJob {
  title: string
  subtitle: string
  description: string
  features: string[]
}

export interface DiscardedFeature {
  feature: string
  label: string
  reason: string
}

export interface FeatureRankingData {
  features: FeatureRankingEntry[]
  portfolio_stats: {
    days: number
    combined_sharpe: number | null
    al_pct: number
    aw_pct: number
    dm_orion_corr: number | null
    dm_wc_corr: number | null
    wc_orion_corr: number | null
  }
  two_jobs: { job1: FrameworkJob; job2: FrameworkJob }
  discarded: DiscardedFeature[]
}

// ── Step 3: Feature Selection Types ──

export interface ACTableRow {
  feature_key: string
  label: string
  raw_1d: number | null
  avg_3d: number | null
  avg_5d: number | null
  avg_7d: number | null
  avg_10d: number | null
  verdict: string
}

export interface IVDirectionRow {
  window: string
  l1: number | null
  l2: number | null
  l3: number | null
  all: number | null
  note: string
}

export interface SelectedFeatureEntry {
  feature: string
  formula: string
  role: string
}

export interface FeatureSelectionData {
  ac_table: ACTableRow[]
  iv_direction_by_level: IVDirectionRow[]
  selected_features: SelectedFeatureEntry[]
  key_finding: string
}

// ── Step 4: Regime Construction Types ──

export interface BoundaryLevelStats {
  name: string
  rule: string
  days: number
  pct: number
  al_pct: number
  sharpe: number | null
}

export interface BoundaryConfig {
  label: string
  boundaries: number[]
  is_selected: boolean
  levels: BoundaryLevelStats[]
  spread: number
  avg_streak: number
  self_trans_pct: number
}

export interface LevelQuintile {
  quintile: string
  pk_iv_range: string
  days: number
  al_pct: number
  sharpe: number | null
}

export interface LevelFinalState {
  state: string
  rule?: string
  pk_iv?: string
  iv_dir?: string
  days: number
  pct: number
  al_pct: number
  aw_pct: number
  port_avg: number | null
  sharpe: number | null
}

export interface L2StrategyProfile {
  state: string
  dm_avg: number | null
  dm_sharpe: number | null
  wc_avg: number | null
  wc_sharpe: number | null
  orion_avg: number | null
  orion_sharpe: number | null
}

export interface TestedAndFailed {
  approach: string
  result: string
}

export interface PerLevelL1L3 {
  description: string
  pk_iv_quintiles: LevelQuintile[]
  final_states: LevelFinalState[]
  threshold: number
  threshold_ac?: number | null
  tested_and_failed?: TestedAndFailed[]
  strategy_profiles?: L2StrategyProfile[]
}

export interface PerLevelL2 {
  description: string
  final_states: LevelFinalState[]
  strategy_profiles: L2StrategyProfile[]
  pk_iv_threshold: number
  tested_and_failed?: TestedAndFailed[]
}

export interface CompleteTableRow {
  state: string
  color: string
  rule: string
  days: number
  pct: number
  al_pct: number | null
  aw_pct: number | null
  port_avg: number | null
  sharpe: number | null
  dm_avg: number | null
  dm_sharpe: number | null
  wc_avg: number | null
  wc_sharpe: number | null
  orion_avg: number | null
  orion_sharpe: number | null
}

export interface DteBreakdownRow {
  dte: number
  days: number
  al_pct: number
  aw_pct: number
  port_avg: number | null
  sharpe: number | null
  dm_avg: number | null
  dm_sharpe: number | null
  wc_avg: number | null
  wc_sharpe: number | null
  orion_avg: number | null
  orion_sharpe: number | null
}

export interface RegimeConstructionData {
  boundary_configs: BoundaryConfig[]
  per_level: {
    L1: PerLevelL1L3
    L2: PerLevelL2
    L3: PerLevelL1L3
  }
  complete_table: CompleteTableRow[]
  overall: {
    days: number
    al_pct: number
    aw_pct: number
    port_avg: number | null
    sharpe: number | null
  }
  dte_breakdown: Record<string, DteBreakdownRow[]>
  vrp_by_state?: Record<string, number | null>
}

// ── Adaptive OOS Types ──

export interface BoundaryTimelineEntry {
  date: string
  iv_lag: number | null
  pk_iv_ratio: number | null
  iv_chg_5d: number | null
  trailing_45d_high_pct: number
  boundary_shifted: boolean
  l1_upper: number
  l2_upper: number
  regime_fixed: string | null
  regime_adaptive: string | null
  pnl_combined: number | null
}

export interface OOSStateMetric {
  state: string
  color: string
  days: number
  pct: number
  al_pct: number | null
  aw_pct: number | null
  port_avg: number | null
  sharpe: number | null
  pk_iv_mean: number | null
  dm_avg: number | null
  dm_sharpe: number | null
  wc_avg: number | null
  wc_sharpe: number | null
  orion_avg: number | null
  orion_sharpe: number | null
}

export interface LevelComparisonEntry {
  level: string
  safe_avg: number | null
  safe_al: number | null
  safe_sharpe: number | null
  safe_days: number
  exposed_avg: number | null
  exposed_al: number | null
  exposed_sharpe: number | null
  exposed_days: number
}

export interface AdaptiveOOSData {
  training_period: { start: string; end: string; days: number; label: string }
  test_period: { start: string; end: string; days: number; label: string }
  boundary_timeline: BoundaryTimelineEntry[]
  insample_states: OOSStateMetric[]
  fixed_states: OOSStateMetric[]
  adaptive_states: OOSStateMetric[]
  validation: {
    rank_corr_fixed: number | null
    rank_corr_adaptive: number | null
    safe_checks_fixed: Record<string, boolean | null>
    safe_checks_adaptive: Record<string, boolean | null>
    shifted_days: number
    fixed_days: number
    total_test_days: number
    shift_pct: number
    regime_disagreement: number
    disagreement_pct: number
  }
  pkiv_diagnostic: {
    fixed_medians_used: Record<string, number>
    shifted_medians_used: Record<string, number>
    oos_fixed_medians: Record<string, number | null>
    oos_adaptive_medians: Record<string, number | null>
  }
  level_comparison_fixed: LevelComparisonEntry[]
  level_comparison_adaptive: LevelComparisonEntry[]
  level_comparison_insample: LevelComparisonEntry[]
  shift_period_metrics: {
    shifted: { days: number; sharpe: number | null; al_pct: number | null; port_avg: number | null }
    non_shifted: { days: number; sharpe: number | null; al_pct: number | null; port_avg: number | null }
  }
}

// ── Snapshot Comparison Types ──

export interface SnapshotStateMetric {
  state: string
  days: number
  pct: number
  al_pct: number | null
  sharpe: number | null
  port_avg: number | null
  dm_avg: number | null
  dm_sharpe: number | null
  wc_avg: number | null
  wc_sharpe: number | null
  orion_avg: number | null
  orion_sharpe: number | null
}

export interface SnapshotResult {
  label: string
  days: number
  level_distribution: Record<string, { days: number; pct: number }>
  state_metrics: SnapshotStateMetric[]
  vrp_by_state: Record<string, number | null>
  pk_iv_thresholds: Record<string, number>
  overall: {
    days: number
    sharpe: number | null
    al_pct: number | null
    port_avg: number | null
  }
  iv_stats: { mean: number | null; std: number | null }
  error?: string
}

export interface SnapshotAgreement {
  total_days: number
  all_four_agree_pct: number
  close_pair_agree_pct: number
  morning_pair_agree_pct: number
  close_vs_morning_pct: number
}

export interface SnapshotComparisonData {
  snapshots: Record<string, SnapshotResult>
  agreement: SnapshotAgreement
}

// ── Alpha Insights Types ──

export interface MonotonicityDetail {
  better: string
  worse: string
  better_sharpe: number
  worse_sharpe: number
  violated: boolean
}

export interface AlphaBaseline {
  composite_score: number
  val_sharpe: number
  safe_separation: number
  rank_stability: number
  state_coverage: number
  monotonicity: number
  monotonicity_violations: number
  monotonicity_checked: number
  monotonicity_details: MonotonicityDetail[]
  val_days: number
  train_days: number
  n_states_used: number
  score_breakdown: {
    sharpe_term: number
    safe_sep_term: number
    rank_term: number
    coverage_term: number
    monotonicity_term: number
  }
}

export interface AlphaStateEntry {
  state: string
  color: string
  period: string
  days: number
  al_pct: number | null
  aw_pct: number | null
  port_avg: number | null
  port_std: number | null
  sharpe: number | null
  win_rate: number | null
  strategies: Record<string, { avg: number | null; sharpe: number | null; win_rate: number | null }>
}

export interface AlphaFeatureIC {
  feature: string
  train_ic: number
  val_ic: number | null
  abs_ic: number
  direction: string
  signal: string
}

export interface AlphaStrategyWeight {
  dm: number
  wc: number
  orion: number
  signal: string
}

export interface AlphaBoundaryPoint {
  l1: number
  l2: number
  score: number
}

export interface AlphaBestBoundary {
  best_l1: number | null
  best_l2: number | null
  best_score: number | null
  current_l1: number
  current_l2: number
}

export interface AlphaDistributionWarning {
  state: string
  period: string
  days: number
  severity: string
  message: string
}

export interface AlphaOOSSummary {
  days: number
  sharpe: number | null
  al_pct: number | null
}

export interface AlphaRollingSharpePt {
  date: string
  sharpe: number
}

export interface AlphaRankDetail {
  state: string
  color: string
  train_mean: number
  val_mean: number
}

export interface AlphaHighCorrPair {
  feature_1: string
  feature_2: string
  rho: number
  redundant: boolean
}

export interface AlphaStrategyCorr {
  state: string
  dm_wc: number | null
  dm_orion: number | null
  wc_orion: number | null
}

export interface AlphaInsightsData {
  baseline: AlphaBaseline
  state_alpha: AlphaStateEntry[]
  feature_importance: AlphaFeatureIC[]
  strategy_weights: Record<string, AlphaStrategyWeight>
  boundary_grid: AlphaBoundaryPoint[]
  best_boundary: AlphaBestBoundary
  distribution_warnings: AlphaDistributionWarning[]
  oos_summary: Record<string, AlphaOOSSummary>
  rolling_sharpe: AlphaRollingSharpePt[]
  rank_detail: AlphaRankDetail[]
  high_corr_pairs: AlphaHighCorrPair[]
  strategy_correlations: AlphaStrategyCorr[]
}
