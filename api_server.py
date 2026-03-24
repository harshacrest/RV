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

FEATURES = ["RV_today", "RV_3d_avg", "RV_ratio"]


def _clean(v):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    return v


def _load():
    rv = pd.read_parquet(BASE / "rv_daily.parquet")
    rv["timestamp"] = pd.to_datetime(rv["timestamp"])
    rv["date"] = rv["timestamp"].dt.date

    strats = {}
    for key, path in STRATEGY_FILES.items():
        df = pd.read_excel(path, sheet_name="returns")
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        strats[key] = df

    return rv, strats


RV_DATA, STRAT_DATA = _load()


def _merge(strategy: str) -> pd.DataFrame:
    rv = RV_DATA.copy()
    st = STRAT_DATA[strategy].copy()
    merged = rv.merge(st, left_on="date", right_on="Date", how="inner")
    return merged


def _summary(df: pd.DataFrame) -> dict:
    pnl = df["Net_Daily_PnL_PerCent"]
    total_pct = df["Net_Equity_Curve"].iloc[-1] if len(df) > 0 else 0
    pos = (pnl > 0).sum()
    neg = (pnl < 0).sum()
    std = pnl.std()
    mean = pnl.mean()
    sharpe = (mean / std * math.sqrt(252)) if std > 0 else None
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


def _bucket_metrics(sub: pd.DataFrame, label: str, rng: list) -> dict:
    pnl = sub["Net_Daily_PnL_PerCent"]
    days = len(sub)
    if days == 0:
        return {
            "label": label, "range": rng, "trading_days": 0,
            "total_pct": 0, "avg_daily_pct": 0, "win_rate": 0,
            "sharpe": None, "sharpe_pct": None,
            "max_win_pct": 0, "max_loss_pct": 0, "max_drawdown_pct": 0,
        }
    pos = (pnl > 0).sum()
    neg = (pnl < 0).sum()
    std = pnl.std()
    mean = pnl.mean()
    sharpe = (mean / std * math.sqrt(252)) if std > 0 else None
    cum = pnl.cumsum()
    dd = cum - cum.cummax()
    return {
        "label": label,
        "range": [_clean(rng[0]), _clean(rng[1])],
        "trading_days": int(days),
        "total_pct": round(float(pnl.sum()), 4),
        "avg_daily_pct": round(float(mean), 4),
        "win_rate": round(float(pos / max(pos + neg, 1)), 4),
        "sharpe": round(float(sharpe), 2) if sharpe else None,
        "sharpe_pct": round(float(sharpe), 2) if sharpe else None,
        "max_win_pct": round(float(pnl.max()), 4) if days > 0 else 0,
        "max_loss_pct": round(float(pnl.min()), 4) if days > 0 else 0,
        "max_drawdown_pct": round(float(dd.min()), 4) if days > 0 else 0,
    }


def _compute_buckets(df: pd.DataFrame, feature: str, n_buckets: int = 5) -> list[dict]:
    valid = df.dropna(subset=[feature])
    if len(valid) == 0:
        return []
    quantiles = np.linspace(0, 1, n_buckets + 1)
    edges = np.quantile(valid[feature].values, quantiles)
    edges = np.unique(edges)
    if len(edges) < 2:
        return [_bucket_metrics(valid, f"{feature}: all", [float(valid[feature].min()), float(valid[feature].max())])]

    buckets = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            mask = (valid[feature] >= lo) & (valid[feature] <= hi)
        else:
            mask = (valid[feature] >= lo) & (valid[feature] < hi)
        sub = valid[mask]
        label = f"{lo:.4f} – {hi:.4f}"
        buckets.append(_bucket_metrics(sub, label, [float(lo), float(hi)]))
    return buckets


def _compute_percentile_buckets(df: pd.DataFrame, feature: str) -> list[dict]:
    valid = df.dropna(subset=[feature])
    if len(valid) == 0:
        return []
    valid = valid.copy()
    valid["_pct"] = valid[feature].rank(pct=True)
    labels = ["P0–P20", "P20–P40", "P40–P60", "P60–P80", "P80–P100"]
    edges = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
    buckets = []
    for i in range(5):
        mask = (valid["_pct"] >= edges[i]) & (valid["_pct"] < edges[i + 1])
        sub = valid[mask]
        buckets.append(_bucket_metrics(sub, labels[i], [edges[i], edges[i + 1]]))
    return buckets


def _compute_cross(df: pd.DataFrame, row_feature: str, col_feature: str, n_buckets: int = 5) -> dict:
    valid = df.dropna(subset=[row_feature, col_feature])
    if len(valid) == 0:
        return {"feature_labels": [], "static_labels": [], "grid": [], "pct_feature_labels": [], "pct_grid": []}

    # Raw buckets for row
    row_q = np.linspace(0, 1, n_buckets + 1)
    row_edges = np.unique(np.quantile(valid[row_feature].values, row_q))
    col_edges = np.unique(np.quantile(valid[col_feature].values, row_q))

    def make_grid(r_edges, c_edges, feat_r, feat_c):
        f_labels = []
        s_labels = []
        for i in range(len(c_edges) - 1):
            s_labels.append(f"{c_edges[i]:.4f}–{c_edges[i+1]:.4f}")
        grid = []
        for i in range(len(r_edges) - 1):
            rlo, rhi = r_edges[i], r_edges[i + 1]
            f_labels.append(f"{rlo:.4f}–{rhi:.4f}")
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
    pct_labels = ["P0–P20", "P20–P40", "P40–P60", "P60–P80", "P80–P100"]
    pct_edges = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
    pct_grid = []
    for i in range(5):
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
        {"key": "RV_3d_avg", "label": "RV 3d Avg"},
        {"key": "RV_ratio", "label": "RV Ratio"},
    ]


@app.get("/api/plain-returns/{strategy}")
def get_plain_returns(strategy: str):
    merged = _merge(strategy)
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
        sh = (mean_yr / std_yr * math.sqrt(252)) if std_yr > 0 else None
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
def get_feature_buckets(strategy: str, feature: str):
    merged = _merge(strategy)
    raw = _compute_buckets(merged, feature)
    pct = _compute_percentile_buckets(merged, feature)
    return {
        "strategy": strategy,
        "feature": feature,
        "raw_buckets": raw,
        "percentile_buckets": pct,
    }


@app.get("/api/composite/{strategy}")
def get_composite(
    strategy: str,
    row_feature: str = Query(...),
    col_feature: str = Query(...),
):
    merged = _merge(strategy)
    cross = _compute_cross(merged, row_feature, col_feature)
    return {
        "strategy": strategy,
        "row_feature": row_feature,
        "col_feature": col_feature,
        **cross,
    }


@app.get("/api/rv-timeseries")
def get_rv_timeseries():
    rv = RV_DATA.copy()
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
            "RV_3d_avg": _clean(round(float(r["RV_3d_avg"]), 6)) if pd.notna(r["RV_3d_avg"]) else None,
            "RV_ratio": _clean(round(float(r["RV_ratio"]), 6)) if pd.notna(r["RV_ratio"]) else None,
        })
    return records


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5501)
