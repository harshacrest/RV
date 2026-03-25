"""
RV Dashboard API Server
Serves pre-computed strategy returns bucketed by RV features.
"""

import math
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="RV Dashboard API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE = Path(__file__).resolve().parent

STRATEGY_FILES = {
    "dm": BASE / "strategy_returns_DM_per_trade_both_max_100.xlsx",
    "wc": BASE / "strategy_returns_90_0_both_itm.xlsx",
    "orion": BASE / "strategy_returns_orion_index_kd_60_40_sl10_max90_min20.xlsx",
}

FEATURES = ["RV_today", "IV_7d", "IV_change_1d", "VRP_today", "IV_intraday_change"]

RISK_FREE_PCT = 5.5  # annual risk-free rate in % (same units as Net_Daily_PnL_PerCent)


def _clean(v):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    return v


TRADING_DATES_PATH = BASE.parent / "DATA" / "NSE" / "trading_dates.csv"


def _load():
    rv = pd.read_parquet(BASE / "rv_daily.parquet")
    rv["timestamp"] = pd.to_datetime(rv["timestamp"])
    rv["date"] = rv["timestamp"].dt.date

    strats = {}
    for key, path in STRATEGY_FILES.items():
        df = pd.read_excel(path, sheet_name="returns")
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        strats[key] = df

    # Load DTE from trading_dates.csv
    dte_df = pd.read_csv(TRADING_DATES_PATH, usecols=["t_date", "DTE"])
    dte_df["t_date"] = pd.to_datetime(dte_df["t_date"]).dt.date
    dte_df["DTE"] = pd.to_numeric(dte_df["DTE"], errors="coerce")
    dte_df = dte_df.dropna(subset=["DTE"])
    dte_df["DTE"] = dte_df["DTE"].astype(int)

    return rv, strats, dte_df


RV_DATA, STRAT_DATA, DTE_DATA = _load()


def _filter_dates(df: pd.DataFrame, date_col: str, start: str | None, end: str | None) -> pd.DataFrame:
    if start:
        df = df[df[date_col] >= pd.Timestamp(start).date()]
    if end:
        df = df[df[date_col] <= pd.Timestamp(end).date()]
    return df


VALID_SNAPSHOTS = {"0915", "0916", "1529", "1530"}

# Mapping: snapshot → which pair to use for IV_intraday_change
# 0915/1530 (edge times) → full day 0915-1530
# 0916/1529 (inner times) → inner day 0916-1529
INTRADAY_PAIRS = {
    "0915": ("0915", "1530"),
    "0916": ("0916", "1529"),
    "1529": ("0916", "1529"),
    "1530": ("0915", "1530"),
}


def _merge(strategy: str, start: str | None = None, end: str | None = None,
           snapshot: str = "1530") -> pd.DataFrame:
    rv = RV_DATA.copy()
    st = STRAT_DATA[strategy].copy()
    merged = rv.merge(st, left_on="date", right_on="Date", how="inner")
    merged = merged.merge(DTE_DATA, left_on="date", right_on="t_date", how="left")
    merged = _filter_dates(merged, "date", start, end)

    # Recompute IV-dependent features based on snapshot
    if snapshot in VALID_SNAPSHOTS:
        iv_col = f"IV_7d_{snapshot}"
        if iv_col in merged.columns:
            merged["IV_7d"] = merged[iv_col]
            merged["IV_change_1d"] = merged["IV_7d"] - merged["IV_7d"].shift(1)
            merged["VRP_today"] = merged["IV_7d"] - merged["RV_today"]

        # IV_intraday_change: use the appropriate pair
        open_snap, close_snap = INTRADAY_PAIRS[snapshot]
        open_col = f"IV_7d_{open_snap}"
        close_col = f"IV_7d_{close_snap}"
        if open_col in merged.columns and close_col in merged.columns:
            merged["IV_intraday_change"] = merged[open_col] - merged[close_col]

    return merged


def _summary(df: pd.DataFrame) -> dict:
    pnl = df["Net_Daily_PnL_PerCent"]
    total_pct = df["Net_Equity_Curve"].iloc[-1] if len(df) > 0 else 0
    pos = (pnl > 0).sum()
    neg = (pnl < 0).sum()
    std = pnl.std()
    mean = pnl.mean()
    sharpe = ((mean * 252 - RISK_FREE_PCT) / (std * math.sqrt(252))) if std > 0 else None
    cum = pnl.cumsum()
    running_max = cum.cummax()
    dd = cum - running_max
    max_dd = dd.min()
    return {
        "total_pct": round(float(total_pct), 2),
        "mean_daily_pct": round(float(mean), 4),
        "median_daily_pct": round(float(pnl.median()), 4),
        "std_daily_pct": round(float(std), 4),
        "win_rate": round(float(pos / max(pos + neg, 1)), 4),
        "sharpe": round(float(sharpe), 2) if sharpe else None,
        "max_win_pct": round(float(pnl.max()), 4),
        "max_loss_pct": round(float(pnl.min()), 4),
        "max_drawdown_pct": round(float(max_dd), 4),
        "total_days": int(len(df)),
        "positive_days": int(pos),
        "negative_days": int(neg),
    }


def _bucket_metrics(sub: pd.DataFrame, label: str, rng: list, feature: str = None) -> dict:
    pnl = sub["Net_Daily_PnL_PerCent"]
    days = len(sub)
    if days == 0:
        return {
            "label": label, "range": rng, "trading_days": 0,
            "total_pct": 0, "avg_daily_pct": 0, "win_rate": 0, "loss_rate": 0,
            "sharpe": None, "sharpe_pct": None, "ann_vol_pct": None,
            "max_win_pct": 0, "max_loss_pct": 0, "max_drawdown_pct": 0,
            "feat_mean": None, "feat_max": None, "feat_min": None,
            "streak_mean": None, "streak_median": None, "streak_min": None, "streak_max": None,
        }
    pos = (pnl > 0).sum()
    neg = (pnl < 0).sum()
    std = pnl.std()
    mean = pnl.mean()
    sharpe = ((mean * 252 - RISK_FREE_PCT) / (std * math.sqrt(252))) if std > 0 else None
    ann_vol = float(std * math.sqrt(252)) if std > 0 else None
    win_rate = float(pos / max(pos + neg, 1))
    cum = pnl.cumsum()
    dd = cum - cum.cummax()
    return {
        "label": label,
        "range": [_clean(rng[0]), _clean(rng[1])],
        "trading_days": int(days),
        "total_pct": round(float(pnl.sum()), 4),
        "avg_daily_pct": round(float(mean), 4),
        "win_rate": round(win_rate, 4),
        "loss_rate": round(1 - win_rate, 4),
        "sharpe": round(float(sharpe), 2) if sharpe else None,
        "sharpe_pct": round(float(sharpe), 2) if sharpe else None,
        "ann_vol_pct": round(ann_vol, 4) if ann_vol else None,
        "max_win_pct": round(float(pnl.max()), 4) if days > 0 else 0,
        "max_loss_pct": round(float(pnl.min()), 4) if days > 0 else 0,
        "max_drawdown_pct": round(float(dd.min()), 4) if days > 0 else 0,
        "feat_mean": round(float(sub[feature].mean()), 6) if feature and feature in sub.columns and days > 0 else None,
        "feat_max": round(float(sub[feature].max()), 6) if feature and feature in sub.columns and days > 0 else None,
        "feat_min": round(float(sub[feature].min()), 6) if feature and feature in sub.columns and days > 0 else None,
        "streak_mean": None, "streak_median": None, "streak_min": None, "streak_max": None,
    }


def _compute_streaks(bucket_labels: np.ndarray) -> dict[int, list[int]]:
    """Given an array of bucket indices (one per day, time-ordered), compute consecutive streaks per bucket."""
    streaks: dict[int, list[int]] = {}
    if len(bucket_labels) == 0:
        return streaks
    cur_label = bucket_labels[0]
    cur_len = 1
    for j in range(1, len(bucket_labels)):
        if bucket_labels[j] == cur_label:
            cur_len += 1
        else:
            streaks.setdefault(cur_label, []).append(cur_len)
            cur_label = bucket_labels[j]
            cur_len = 1
    streaks.setdefault(cur_label, []).append(cur_len)
    return streaks


def _inject_streaks(buckets: list[dict], streaks: dict[int, list[int]]):
    for i, b in enumerate(buckets):
        s = streaks.get(i, [])
        if s:
            b["streak_mean"] = round(float(np.mean(s)), 1)
            b["streak_median"] = round(float(np.median(s)), 1)
            b["streak_min"] = int(np.min(s))
            b["streak_max"] = int(np.max(s))


def _compute_buckets(df: pd.DataFrame, feature: str) -> list[dict]:
    valid = df.dropna(subset=[feature]).sort_values("date")
    if len(valid) == 0:
        return []

    # 3 equal-count (tercile) buckets
    edges = np.unique(np.quantile(valid[feature].values, [0, 1/3, 2/3, 1.0])).tolist()

    buckets = []
    bucket_labels = np.full(len(valid), -1, dtype=int)
    vals = valid[feature].values
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            mask = (vals >= lo) & (vals <= hi)
        else:
            mask = (vals >= lo) & (vals < hi)
        label = f"{lo:.2f} – {hi:.2f}"
        bucket_labels[mask] = i
        sub = valid[mask]
        buckets.append(_bucket_metrics(sub, label, [_clean(float(lo)), _clean(float(hi))], feature))

    streaks = _compute_streaks(bucket_labels)
    _inject_streaks(buckets, streaks)
    return buckets


def _compute_percentile_buckets(df: pd.DataFrame, feature: str) -> list[dict]:
    valid = df.dropna(subset=[feature]).sort_values("date")
    if len(valid) == 0:
        return []
    valid = valid.copy()
    valid["_pct"] = valid[feature].rank(pct=True)
    labels = ["P0–P33", "P33–P67", "P67–P100"]
    p_edges = [0, 1/3, 2/3, 1.01]
    buckets = []
    bucket_labels = np.full(len(valid), -1, dtype=int)
    pct_vals = valid["_pct"].values
    for i in range(3):
        mask = (pct_vals >= p_edges[i]) & (pct_vals < p_edges[i + 1])
        bucket_labels[mask] = i
        sub = valid[mask]
        buckets.append(_bucket_metrics(sub, labels[i], [p_edges[i], p_edges[i + 1]], feature))

    streaks = _compute_streaks(bucket_labels)
    _inject_streaks(buckets, streaks)
    return buckets


def _compute_dte_cross(df: pd.DataFrame, feature: str) -> dict:
    """Cross-tab: feature buckets (rows) x DTE values (columns)."""
    valid = df.dropna(subset=[feature, "DTE"]).sort_values("date")
    if len(valid) == 0:
        return {"dte_labels": [], "feature_labels": [], "grid": []}

    dte_values = sorted(valid["DTE"].dropna().unique().astype(int).tolist())
    dte_labels = [str(d) for d in dte_values]

    edges = np.unique(np.quantile(valid[feature].values, [0, 1/3, 2/3, 1.0])).tolist()

    vals = valid[feature].values
    feature_labels = []
    grid = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        fmt = ".2f" if any(e != int(e) for e in edges if e != float("inf")) else ".0f"
        if hi == float("inf"):
            fmask = vals >= lo
            label = f"≥ {lo:{fmt}}"
        else:
            fmask = (vals >= lo) & (vals < hi)
            label = f"{lo:{fmt}} – {hi:{fmt}}"
        feature_labels.append(label)

        row = []
        for dte_val in dte_values:
            dmask = valid["DTE"].values == dte_val
            sub = valid[fmask & dmask]
            row.append(_bucket_metrics(sub, f"{label} DTE={dte_val}", [_clean(float(lo)), _clean(float(hi))], feature))
        grid.append(row)

    return {"dte_labels": dte_labels, "feature_labels": feature_labels, "grid": grid}


def _compute_composite_dte_cross(df: pd.DataFrame, row_feature: str, col_feature: str) -> dict:
    """For each of the 9 (row×col) combos, break down by DTE."""
    valid = df.dropna(subset=[row_feature, col_feature, "DTE"]).sort_values("date")
    if len(valid) == 0:
        return {"combo_labels": [], "dte_labels": [], "grid": []}

    terciles = [0, 1/3, 2/3, 1.0]
    row_edges = np.unique(np.quantile(valid[row_feature].values, terciles))
    col_edges = np.unique(np.quantile(valid[col_feature].values, terciles))
    dte_values = sorted(valid["DTE"].dropna().unique().astype(int).tolist())
    dte_labels = [str(d) for d in dte_values]

    combo_labels = []
    grid = []  # each row = one combo, each col = one DTE

    for i in range(len(row_edges) - 1):
        rlo, rhi = row_edges[i], row_edges[i + 1]
        if i == len(row_edges) - 2:
            rmask = (valid[row_feature] >= rlo) & (valid[row_feature] <= rhi)
        else:
            rmask = (valid[row_feature] >= rlo) & (valid[row_feature] < rhi)
        r_label = f"{rlo:.2f}–{rhi:.2f}"

        for j in range(len(col_edges) - 1):
            clo, chi = col_edges[j], col_edges[j + 1]
            if j == len(col_edges) - 2:
                cmask = (valid[col_feature] >= clo) & (valid[col_feature] <= chi)
            else:
                cmask = (valid[col_feature] >= clo) & (valid[col_feature] < chi)
            c_label = f"{clo:.2f}–{chi:.2f}"

            combo_mask = rmask & cmask
            combo_label = f"{r_label} × {c_label}"
            combo_labels.append({"row_label": r_label, "col_label": c_label, "combo_label": combo_label})

            row = []
            for dte_val in dte_values:
                dmask = valid["DTE"].values == dte_val
                sub = valid[combo_mask & dmask]
                row.append(_bucket_metrics(sub, f"{combo_label} DTE={dte_val}", [float(rlo), float(rhi)]))
            grid.append(row)

    return {"combo_labels": combo_labels, "dte_labels": dte_labels, "grid": grid}


def _compute_cross(df: pd.DataFrame, row_feature: str, col_feature: str) -> dict:
    valid = df.dropna(subset=[row_feature, col_feature])
    if len(valid) == 0:
        return {"feature_labels": [], "static_labels": [], "grid": [], "pct_feature_labels": [], "pct_grid": []}

    # 3 equal-count (tercile) buckets for both axes
    terciles = [0, 1/3, 2/3, 1.0]
    row_edges = np.unique(np.quantile(valid[row_feature].values, terciles))
    col_edges = np.unique(np.quantile(valid[col_feature].values, terciles))

    def make_grid(r_edges, c_edges, feat_r, feat_c):
        f_labels = []
        s_labels = []
        for i in range(len(c_edges) - 1):
            s_labels.append(f"{c_edges[i]:.2f}–{c_edges[i+1]:.2f}")
        grid = []
        for i in range(len(r_edges) - 1):
            rlo, rhi = r_edges[i], r_edges[i + 1]
            f_labels.append(f"{rlo:.2f}–{rhi:.2f}")
            row = []
            for j in range(len(c_edges) - 1):
                clo, chi = c_edges[j], c_edges[j + 1]
                if i == len(r_edges) - 2:
                    rmask = (valid[feat_r] >= rlo) & (valid[feat_r] <= rhi)
                else:
                    rmask = (valid[feat_r] >= rlo) & (valid[feat_r] < rhi)
                if j == len(c_edges) - 2:
                    cmask = (valid[feat_c] >= clo) & (valid[feat_c] <= chi)
                else:
                    cmask = (valid[feat_c] >= clo) & (valid[feat_c] < chi)
                sub = valid[rmask & cmask]
                row.append(_bucket_metrics(sub, f"{f_labels[-1]} × {s_labels[j]}", [float(rlo), float(rhi)]))
            grid.append(row)
        return f_labels, s_labels, grid

    f_labels, s_labels, grid = make_grid(row_edges, col_edges, row_feature, col_feature)

    # Percentile grid
    valid_pct = valid.copy()
    valid_pct["_rpct"] = valid_pct[row_feature].rank(pct=True)
    pct_labels = ["P0–P33", "P33–P67", "P67–P100"]
    pct_edges = [0, 1/3, 2/3, 1.01]
    pct_grid = []
    for i in range(3):
        row = []
        rmask = (valid_pct["_rpct"] >= pct_edges[i]) & (valid_pct["_rpct"] < pct_edges[i + 1])
        for j in range(len(col_edges) - 1):
            clo, chi = col_edges[j], col_edges[j + 1]
            if j == len(col_edges) - 2:
                cmask = (valid_pct[col_feature] >= clo) & (valid_pct[col_feature] <= chi)
            else:
                cmask = (valid_pct[col_feature] >= clo) & (valid_pct[col_feature] < chi)
            sub = valid_pct[rmask & cmask]
            row.append(_bucket_metrics(sub, f"{pct_labels[i]} × {s_labels[j] if j < len(s_labels) else ''}", [pct_edges[i], pct_edges[i + 1]]))
        pct_grid.append(row)

    return {
        "feature_labels": f_labels,
        "static_labels": s_labels,
        "grid": grid,
        "pct_feature_labels": pct_labels,
        "pct_grid": pct_grid,
    }


# ── Endpoints ──

@app.get("/api/strategies")
def get_strategies():
    return [
        {"key": "dm", "name": "DM Strategy", "accent": "#ffd740", "accentRgb": "255,215,64"},
        {"key": "wc", "name": "WC Strategy", "accent": "#448aff", "accentRgb": "68,138,255"},
        {"key": "orion", "name": "Orion Strategy", "accent": "#64ffda", "accentRgb": "100,255,218"},
    ]


@app.get("/api/features")
def get_features():
    return [
        {"key": "RV_today", "label": "RV Today (Yang-Zhang)"},
        {"key": "IV_7d", "label": "IV 7d Forward"},
        {"key": "IV_change_1d", "label": "IV Change 1d"},
        {"key": "VRP_today", "label": "VRP (IV−RV)"},
        {"key": "IV_intraday_change", "label": "IV Intraday Change (Open−Close)"},
    ]


@app.get("/api/plain-returns/{strategy}")
def get_plain_returns(strategy: str, start_date: str | None = None, end_date: str | None = None, snapshot: str = "1530"):
    merged = _merge(strategy, start_date, end_date, snapshot)
    merged = merged.sort_values("date")
    s = _summary(merged)
    pnl = merged["Net_Daily_PnL_PerCent"].values
    cum = np.cumsum(pnl)

    daily = [
        {"date": str(r["date"]), "pnl_pct": round(float(r["Net_Daily_PnL_PerCent"]), 4), "cumulative_pct": round(float(c), 4)}
        for (_, r), c in zip(merged.iterrows(), cum)
    ]

    # Yearly
    merged["year"] = pd.to_datetime(merged["date"]).dt.year if not pd.api.types.is_datetime64_any_dtype(merged["date"]) else merged["date"].apply(lambda d: d.year)
    yearly = []
    for yr, grp in merged.groupby("year"):
        pnl_yr = grp["Net_Daily_PnL_PerCent"]
        std_yr = pnl_yr.std()
        mean_yr = pnl_yr.mean()
        sh = ((mean_yr * 252 - RISK_FREE_PCT) / (std_yr * math.sqrt(252))) if std_yr > 0 else None
        yearly.append({
            "year": int(yr),
            "total_pct": round(float(pnl_yr.sum()), 2),
            "mean_daily_pct": round(float(mean_yr), 4),
            "std_daily_pct": round(float(std_yr), 4),
            "win_rate": round(float((pnl_yr > 0).sum() / max(len(pnl_yr), 1)), 4),
            "sharpe_pct": round(float(sh), 2) if sh else None,
            "max_win_pct": round(float(pnl_yr.max()), 4),
            "max_loss_pct": round(float(pnl_yr.min()), 4),
            "trading_days": int(len(grp)),
        })

    # Monthly
    merged["month"] = pd.to_datetime(merged["date"]).dt.month if not pd.api.types.is_datetime64_any_dtype(merged["date"]) else merged["date"].apply(lambda d: d.month)
    monthly = []
    for (yr, mo), grp in merged.groupby(["year", "month"]):
        pnl_m = grp["Net_Daily_PnL_PerCent"]
        monthly.append({
            "year": int(yr), "month": int(mo),
            "total_pct": round(float(pnl_m.sum()), 2),
            "mean_daily_pct": round(float(pnl_m.mean()), 4),
            "trading_days": int(len(grp)),
            "win_rate": round(float((pnl_m > 0).sum() / max(len(pnl_m), 1)), 4),
        })

    dates = [str(d) for d in merged["date"]]
    return {
        "strategy": strategy,
        "date_range": [dates[0], dates[-1]] if dates else ["", ""],
        "summary": s,
        "yearly": yearly,
        "monthly": monthly,
        "daily_timeseries": daily,
    }


@app.get("/api/feature-buckets/{strategy}/{feature}")
def get_feature_buckets(strategy: str, feature: str, start_date: str | None = None, end_date: str | None = None, snapshot: str = "1530"):
    merged = _merge(strategy, start_date, end_date, snapshot)
    raw = _compute_buckets(merged, feature)
    pct = _compute_percentile_buckets(merged, feature)
    dte_cross = _compute_dte_cross(merged, feature)
    return {
        "strategy": strategy,
        "feature": feature,
        "raw_buckets": raw,
        "percentile_buckets": pct,
        "dte_cross": dte_cross,
    }


@app.get("/api/composite/{strategy}")
def get_composite(
    strategy: str,
    row_feature: str = Query(...),
    col_feature: str = Query(...),
    start_date: str | None = None,
    end_date: str | None = None,
    snapshot: str = "1530",
):
    merged = _merge(strategy, start_date, end_date, snapshot)
    cross = _compute_cross(merged, row_feature, col_feature)
    composite_dte = _compute_composite_dte_cross(merged, row_feature, col_feature)
    return {
        "strategy": strategy,
        "row_feature": row_feature,
        "col_feature": col_feature,
        **cross,
        "composite_dte_cross": composite_dte,
    }


@app.get("/api/rv-timeseries")
def get_rv_timeseries(start_date: str | None = None, end_date: str | None = None):
    rv = RV_DATA.copy()
    rv = _filter_dates(rv, "date", start_date, end_date)
    rv = rv.dropna(subset=["RV_today"])
    records = []
    for _, r in rv.iterrows():
        records.append({
            "date": str(r["date"]),
            "open": round(float(r["open"]), 2),
            "high": round(float(r["high"]), 2),
            "low": round(float(r["low"]), 2),
            "close": round(float(r["close"]), 2),
            "RV_today": _clean(round(float(r["RV_today"]), 6)),
            "IV_7d": _clean(round(float(r["IV_7d"]), 2)) if pd.notna(r.get("IV_7d")) else None,
            "IV_change_1d": _clean(round(float(r["IV_change_1d"]), 2)) if pd.notna(r.get("IV_change_1d")) else None,
            "VRP_today": _clean(round(float(r["VRP_today"]), 2)) if pd.notna(r.get("VRP_today")) else None,
        })
    return records


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5501)
