"""
strategy_coopt.py — Upgrade 4: Classifier-Strategy Co-Optimization.

Enables the meta-harness to alternate between:
  (a) Classifier iterations: inner loop modifies boundaries, features, weights (normal)
  (b) Strategy iterations: classifier is fixed, meta-harness optimizes strategy
      weight allocations per regime state

In strategy mode, the meta-harness writes a strategy_weights_override into
regime_experiment.py before the inner loop starts, then the inner loop only
optimizes non-weight dimensions (boundaries, features, architecture).

Usage:
    from strategy_coopt import (
        generate_strategy_config,
        inject_strategy_weights,
        STRATEGY_TEMPLATES,
    )

    # Generate a random strategy weight config
    weights = generate_strategy_config(template="wc_dominant")

    # Inject into regime_experiment.py before inner loop
    inject_strategy_weights("regime_experiment.py", weights)

    # CLI
    python strategy_coopt.py --list-templates
    python strategy_coopt.py --generate random
    python strategy_coopt.py --inject wc_dominant
"""

import json
import random
import re
import sys
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).parent.parent  # autoresearch root

# The 8 regime states
REGIME_STATES = [
    "L1 Safe", "L1 Exposed",
    "L2 Safe", "L2 Caution-A", "L2 Caution-B", "L2 Risky",
    "L3 Safe", "L3 Exposed",
]

# ═══════════════════════════════════════════════════════════════
# Strategy Templates — curated starting points
# ═══════════════════════════════════════════════════════════════

STRATEGY_TEMPLATES = {
    "equal": {
        "description": "Equal weight [1,1,1] everywhere",
        "weights": {s: [1, 1, 1] for s in REGIME_STATES},
    },
    "current_best": {
        "description": "Current best from apr6 autoresearch run",
        "weights": {
            "L1 Safe":      [0, 1, 0],
            "L1 Exposed":   [0, 1, 0.2],
            "L2 Safe":      [1, 1, 1],
            "L2 Caution-A": [0.2, 0.4, 0.8],
            "L2 Caution-B": [1, 1, 1],
            "L2 Risky":     [1, 1, 1],
            "L3 Safe":      [1, 1, 1],
            "L3 Exposed":   [0.3, 0.3, 1],
        },
    },
    "wc_dominant": {
        "description": "WC-heavy across all states (the big insight from apr6)",
        "weights": {
            "L1 Safe":      [0, 1, 0],
            "L1 Exposed":   [0, 1, 0],
            "L2 Safe":      [0.3, 1, 0.3],
            "L2 Caution-A": [0.2, 1, 0.5],
            "L2 Caution-B": [0.3, 1, 0.3],
            "L2 Risky":     [0.5, 1, 0.5],
            "L3 Safe":      [0.3, 1, 0.5],
            "L3 Exposed":   [0.3, 1, 0.3],
        },
    },
    "orion_dominant": {
        "description": "Orion-heavy — best in high-IV regimes",
        "weights": {
            "L1 Safe":      [0.5, 0.5, 1],
            "L1 Exposed":   [0.3, 0.3, 1],
            "L2 Safe":      [0.5, 0.5, 1],
            "L2 Caution-A": [0.3, 0.3, 1],
            "L2 Caution-B": [0.5, 0.5, 1],
            "L2 Risky":     [0.3, 0.3, 1],
            "L3 Safe":      [0.5, 0.5, 1],
            "L3 Exposed":   [0.2, 0.2, 1],
        },
    },
    "regime_adaptive": {
        "description": "WC in low-IV, Orion in high-IV, equal in mid",
        "weights": {
            "L1 Safe":      [0, 1, 0],
            "L1 Exposed":   [0, 1, 0.2],
            "L2 Safe":      [1, 1, 1],
            "L2 Caution-A": [1, 1, 1],
            "L2 Caution-B": [1, 1, 1],
            "L2 Risky":     [1, 1, 1],
            "L3 Safe":      [0.5, 0.5, 1],
            "L3 Exposed":   [0.2, 0.2, 1],
        },
    },
    "conservative": {
        "description": "Cut exposure in all risky states, full weight in safe",
        "weights": {
            "L1 Safe":      [1, 1, 1],
            "L1 Exposed":   [0.3, 0.3, 0.3],
            "L2 Safe":      [1, 1, 1],
            "L2 Caution-A": [0.5, 0.5, 0.5],
            "L2 Caution-B": [1, 1, 1],
            "L2 Risky":     [0.2, 0.2, 0.2],
            "L3 Safe":      [1, 1, 1],
            "L3 Exposed":   [0.2, 0.2, 0.2],
        },
    },
}


def generate_strategy_config(
    template: str = "random",
    perturbation: float = 0.15,
) -> dict[str, list[float]]:
    """Generate a strategy weight configuration.

    Args:
        template: Name of a template from STRATEGY_TEMPLATES, or "random"
        perturbation: How much to randomly perturb template weights (0 = exact template)

    Returns:
        Dict mapping state name -> [dm_weight, wc_weight, orion_weight]
    """
    if template == "random":
        # Fully random weights
        weights = {}
        for state in REGIME_STATES:
            w = [round(random.uniform(0, 1), 2) for _ in range(3)]
            # Ensure at least one strategy has non-trivial weight
            if max(w) < 0.3:
                w[random.randint(0, 2)] = round(random.uniform(0.5, 1.0), 2)
            weights[state] = w
        return weights

    if template not in STRATEGY_TEMPLATES:
        raise ValueError(f"Unknown template: {template}. Available: {list(STRATEGY_TEMPLATES.keys())}")

    base = STRATEGY_TEMPLATES[template]["weights"]

    if perturbation == 0:
        return {s: list(w) for s, w in base.items()}

    # Perturb the template
    weights = {}
    for state in REGIME_STATES:
        base_w = base[state]
        perturbed = []
        for w in base_w:
            p = w + random.uniform(-perturbation, perturbation)
            perturbed.append(round(max(0, min(1, p)), 2))
        weights[state] = perturbed

    return weights


def inject_strategy_weights(
    experiment_file: str | Path,
    weights: dict[str, list[float]],
    lock_weights: bool = True,
) -> None:
    """Inject strategy weights into regime_experiment.py.

    If lock_weights=True, also adds a comment telling the inner loop agent
    NOT to modify strategy weights (co-optimization mode: classifier only).

    Args:
        experiment_file: Path to regime_experiment.py
        weights: Dict mapping state name -> [dm, wc, orion]
        lock_weights: If True, add LOCKED comment to prevent inner loop from changing weights
    """
    path = Path(experiment_file)
    content = path.read_text()

    # Build the new STRATEGY_WEIGHTS block
    lock_comment = ""
    if lock_weights:
        lock_comment = (
            "# !! LOCKED BY META-HARNESS (strategy co-optimization mode) !!\n"
            "# The outer loop is optimizing these weights. Do NOT modify them.\n"
            "# Focus on boundaries, features, and classification architecture instead.\n"
        )

    lines = [f"{lock_comment}STRATEGY_WEIGHTS = {{"]
    for state in REGIME_STATES:
        w = weights.get(state, [1, 1, 1])
        lines.append(f'    "{state}": {w},')
    lines.append("}")

    new_block = "\n".join(lines)

    # Replace the existing STRATEGY_WEIGHTS block
    # Match from "STRATEGY_WEIGHTS = {" to the closing "}"
    pattern = r'(?:#[^\n]*\n)*STRATEGY_WEIGHTS\s*=\s*\{[^}]+\}'
    if re.search(pattern, content):
        content = re.sub(pattern, new_block, content)
    else:
        # Fallback: just replace line by line
        raise ValueError("Could not find STRATEGY_WEIGHTS block in file")

    path.write_text(content)


def extract_strategy_weights(experiment_file: str | Path) -> dict[str, list[float]]:
    """Extract current strategy weights from regime_experiment.py."""
    path = Path(experiment_file)
    content = path.read_text()

    # Use regex to extract the dict
    pattern = r'STRATEGY_WEIGHTS\s*=\s*(\{[^}]+\})'
    match = re.search(pattern, content)
    if not match:
        return {s: [1, 1, 1] for s in REGIME_STATES}

    # Parse it safely
    raw = match.group(1)
    # Convert to valid Python dict syntax for eval
    try:
        # Replace state names that might cause issues
        weights = eval(raw)  # Safe here since we control the file
        return weights
    except Exception:
        return {s: [1, 1, 1] for s in REGIME_STATES}


def strategy_distance(w1: dict, w2: dict) -> float:
    """Compute L2 distance between two strategy weight configs."""
    total = 0
    for state in REGIME_STATES:
        a = w1.get(state, [1, 1, 1])
        b = w2.get(state, [1, 1, 1])
        for i in range(3):
            total += (a[i] - b[i]) ** 2
    return total ** 0.5


def format_weights_summary(weights: dict) -> str:
    """Format weights as a compact summary string."""
    parts = []
    for state in REGIME_STATES:
        w = weights.get(state, [1, 1, 1])
        if w == [1, 1, 1]:
            continue  # Skip equal-weight states
        parts.append(f"{state}: [{w[0]},{w[1]},{w[2]}]")
    return "; ".join(parts) if parts else "all equal"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Strategy Co-Optimization (Upgrade 4)")
    parser.add_argument("--list-templates", action="store_true", help="List available templates")
    parser.add_argument("--generate", type=str, help="Generate weights from template name or 'random'")
    parser.add_argument("--inject", type=str, help="Inject template into regime_experiment.py")
    parser.add_argument("--extract", action="store_true", help="Extract current weights")
    parser.add_argument("--lock", action="store_true", default=True, help="Lock weights (default: True)")
    parser.add_argument("--no-lock", action="store_true", help="Don't lock weights")
    args = parser.parse_args()

    if args.list_templates:
        print("Available strategy templates:\n")
        for name, tmpl in STRATEGY_TEMPLATES.items():
            print(f"  {name:20s} — {tmpl['description']}")
            for state in REGIME_STATES:
                w = tmpl["weights"][state]
                if w != [1, 1, 1]:
                    print(f"    {state:20s}: [DM={w[0]}, WC={w[1]}, Orion={w[2]}]")
            print()

    elif args.generate:
        weights = generate_strategy_config(template=args.generate)
        print(f"Generated weights (template={args.generate}):\n")
        for state, w in weights.items():
            print(f"  {state:20s}: [DM={w[0]:.2f}, WC={w[1]:.2f}, Orion={w[2]:.2f}]")

    elif args.inject:
        weights = generate_strategy_config(template=args.inject, perturbation=0)
        lock = not args.no_lock
        inject_strategy_weights(SCRIPT_DIR / "regime_experiment.py", weights, lock_weights=lock)
        print(f"Injected template '{args.inject}' into regime_experiment.py (locked={lock})")

    elif args.extract:
        weights = extract_strategy_weights(SCRIPT_DIR / "regime_experiment.py")
        print("Current strategy weights:\n")
        for state, w in weights.items():
            print(f"  {state:20s}: [DM={w[0]}, WC={w[1]}, Orion={w[2]}]")

    else:
        parser.print_help()
