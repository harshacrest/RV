"""
api_autoresearch.py — FastAPI backend for the Autoresearch Dashboard.

Serves diagnostic data from diagnostics_report.json and cached analysis results.

Usage:
    python api_autoresearch.py
    # or: uvicorn api_autoresearch:app --port 5502 --reload
"""

import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

SCRIPT_DIR = Path(__file__).parent

app = FastAPI(title="Autoresearch Dashboard API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load cached data ──

def _load_json(filename: str) -> dict:
    path = SCRIPT_DIR / filename
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

_diag = _load_json("output/diagnostics_report.json")
_ablation = _load_json("output/ablation_results.json")
_rolling = _load_json("output/rolling_validation_results.json")


# ── Endpoints ──

@app.get("/api/baseline")
def get_baseline():
    return _diag.get("baseline", {})


@app.get("/api/regime-distribution")
def get_regime_distribution():
    return _diag.get("regime_distribution", {})


@app.get("/api/feature-correlations")
def get_feature_correlations():
    data = _diag.get("feature_correlations", {})
    return {
        "high_corr_pairs": data.get("high_corr_pairs", []),
        "feature_ics": data.get("feature_ics", {}),
    }


@app.get("/api/strategy-correlations")
def get_strategy_correlations():
    return _diag.get("strategy_correlations", {})


@app.get("/api/normalization")
def get_normalization():
    return _diag.get("normalization", {})


@app.get("/api/rank-stability")
def get_rank_stability():
    return _diag.get("rank_stability", {})


@app.get("/api/oos-adequacy")
def get_oos_adequacy():
    return _diag.get("oos2_adequacy", {})


@app.get("/api/boundary-sensitivity")
def get_boundary_sensitivity():
    bs = _diag.get("boundary_sensitivity", {})
    # Convert grid to structured format for heatmap
    grid = bs.get("grid", {})
    rows = []
    for key, score in grid.items():
        parts = key.split("_")
        if len(parts) == 2:
            rows.append({"l1": float(parts[0]), "l2": float(parts[1]), "score": score})
    return {
        "best_l1": bs.get("best_l1"),
        "best_l2": bs.get("best_l2"),
        "best_score": bs.get("best_score"),
        "grid": rows,
    }


@app.get("/api/ablation")
def get_ablation():
    return _ablation


@app.get("/api/rolling-validation")
def get_rolling_validation():
    return _rolling


@app.get("/api/config")
def get_config():
    return {
        "scoring_weights": {"w_sharpe": 0.40, "w_safe_sep": 0.25, "w_rank_corr": 0.25, "w_coverage": 0.10},
        "normalization": {"sharpe_norm": 5.0, "sharpe_cap": 1.0, "safe_sep_norm": 15.0, "safe_sep_cap": 1.0},
        "coverage": {"min_state_days": 5, "min_states_used": 4},
        "periods": {
            "train": "2023-02-01 to 2025-06-30",
            "val": "2025-07-01 to 2026-01-30",
            "oos1": "2021-01-01 to 2023-01-31",
            "oos2": "2026-02-01 to 2026-03-23",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5502)
