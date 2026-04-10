export const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5501'

export const STRATEGY_META: Record<string, { name: string; accent: string; accentRgb: string }> = {
  dm: { name: 'DM Strategy', accent: 'var(--dm-accent)', accentRgb: '255,215,64' },
  wc: { name: 'WC Strategy', accent: 'var(--wc-accent)', accentRgb: '68,138,255' },
  orion: { name: 'Orion Strategy', accent: 'var(--orion-accent)', accentRgb: '100,255,218' },
  dmo: { name: 'DMO Strategy', accent: 'var(--dmo-accent)', accentRgb: '255,128,171' },
}

export const FEATURE_LABELS: Record<string, string> = {
  RV_today: 'RV Today (Yang-Zhang)',
  IV_7d: 'IV 7d Forward',
  IV_change_1d: 'IV Change 1d',
  VRP_today: 'VRP (IV−RV)',
  IV_intraday_change: 'IV Intraday Change (Open−Close)',
}

export const FEATURES = ['RV_today', 'IV_7d', 'IV_change_1d', 'VRP_today', 'IV_intraday_change'] as const
export const STRATEGIES = ['dm', 'wc', 'orion', 'dmo'] as const
