"""
analyze_results.py — Analyze experiment results from autoresearch regime loop.
Run: python analyze_results.py
"""

import pandas as pd
from pathlib import Path

TSV_FILE = Path(__file__).parent.parent / "results.tsv"


def main():
    if not TSV_FILE.exists():
        print("No results.tsv found. Run experiments first.")
        return

    df = pd.read_csv(TSV_FILE, sep="\t")
    print(f"\n{'='*60}")
    print(f"REGIME AUTORESEARCH — EXPERIMENT ANALYSIS")
    print(f"{'='*60}\n")

    total = len(df)
    kept = df[df["status"] == "keep"]
    discarded = df[df["status"] == "discard"]
    crashed = df[df["status"] == "crash"]

    print(f"Total experiments:  {total}")
    print(f"  Kept:            {len(kept)} ({len(kept)/total*100:.0f}%)")
    print(f"  Discarded:       {len(discarded)} ({len(discarded)/total*100:.0f}%)")
    print(f"  Crashed:         {len(crashed)} ({len(crashed)/total*100:.0f}%)")
    print()

    if len(kept) == 0:
        print("No kept experiments yet.")
        return

    # Best result
    best = kept.loc[kept["composite_score"].idxmax()]
    baseline = df.iloc[0] if len(df) > 0 else None

    print(f"BASELINE:  composite={baseline['composite_score']:.6f}  sharpe={baseline['val_sharpe']:.4f}" if baseline is not None else "")
    print(f"BEST:      composite={best['composite_score']:.6f}  sharpe={best['val_sharpe']:.4f}")
    if baseline is not None:
        delta = best["composite_score"] - baseline["composite_score"]
        print(f"IMPROVEMENT: +{delta:.6f} ({delta/baseline['composite_score']*100:.1f}%)")
    print()

    # All kept experiments (chronological)
    print("KEPT EXPERIMENTS (improvements chain):")
    print(f"{'#':>3s}  {'commit':>7s}  {'score':>10s}  {'sharpe':>8s}  {'safe_sep':>8s}  description")
    print("-" * 80)
    for i, (_, row) in enumerate(kept.iterrows()):
        print(f"{i+1:3d}  {str(row['commit']):>7s}  {row['composite_score']:10.6f}  {row['val_sharpe']:8.4f}  {row['safe_sep']:8.2f}  {row['description']}")

    # Top 10 experiments by score (including discarded)
    print(f"\nTOP 10 BY SCORE (all experiments):")
    top = df[df["status"] != "crash"].nlargest(10, "composite_score")
    for _, row in top.iterrows():
        status_marker = "✓" if row["status"] == "keep" else "✗"
        print(f"  {status_marker} {row['composite_score']:.6f}  {row['description']}")


if __name__ == "__main__":
    main()
