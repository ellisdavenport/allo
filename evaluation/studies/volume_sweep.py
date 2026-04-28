"""
volume_sweep.py — measure allo's actual output volume per strategy as --n scales.

Full-grid design:
    117 seeds (from evaluation/allo_seed_set.csv)
    × 6 values of n_per_strategy (5, 10, 20, 50, 80, 100)
    = 702 allo runs total

Goals:
    1. Replace the hand-waved volume estimates in generate.py's docstring
       and main.py's docstring with measured per-strategy scaling behavior.
    2. Identify where each strategy hits its structural ceiling:
       - LLM paraphrasing should scale linearly with n
       - Constrained rewriting should cap at 3 × 15 = 45 when n ≥ ~45
       - MLM should plateau around 25–30 regardless of n
       - Expansion should scale linearly via max(5, n//4)
    3. Characterize variance across linguistic dimensions (length,
       syntactic type, lexical complexity) for each strategy.

Outputs:
    - Per-seed-per-n allo CSVs under runs/volume_sweep_*/n_X/seed_YYY/
    - run_config.json capturing the exact parameters + git SHA
    - volume_sweep_aggregate.csv: long-format table, one row per
      (seed_id, n, strategy) tuple, for analysis

Usage:
    From the project root:
        python -m evaluation.studies.volume_sweep

    To resume after a partial run (e.g. a specific --n failed mid-way):
        python -m evaluation.studies.volume_sweep --resume RUN_DIR --n-values 80 100
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from allo.generate import LLMClient, generate_variants
from allo.evaluate import score_variants
from allo.output import write_csv


# ─── Configuration ─────────────────────────────────────────────────────
N_VALUES = [5, 10, 20, 50, 80, 100]
TEMPERATURE = 0.9
INCLUDE_EXPANSION = True
SEED_SET_PATH = Path("evaluation/allo_seed_set.csv")
OUTPUT_ROOT = Path("evaluation/runs")


# ─── Helpers ───────────────────────────────────────────────────────────

def get_git_sha() -> str:
    """Capture the commit the study was run against, for reproducibility."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def load_seeds() -> list[dict]:
    """Load the seed set CSV, preserving all linguistic dimension tags."""
    with open(SEED_SET_PATH) as f:
        return list(csv.DictReader(f))


def count_strategies(scored: list[dict]) -> dict[str, int]:
    """
    Collapse strategy tags into families for top-level counting.

    'constrained:polar_question', 'constrained:hedged_request', etc.
    all collapse to 'constrained'. Individual transform counts are
    captured separately in count_constrained_by_transform().
    """
    counts = {"llm_paraphrase": 0, "constrained": 0, "mlm_substitution": 0, "expansion": 0}
    for v in scored:
        strat = v["strategy"]
        family = strat.split(":")[0] if ":" in strat else strat
        if family in counts:
            counts[family] += 1
    return counts


def count_constrained_by_transform(scored: list[dict]) -> dict[str, int]:
    """Break down constrained rewriting counts by specific transform."""
    counts = {}
    for v in scored:
        strat = v["strategy"]
        if strat.startswith("constrained:"):
            transform = strat.split(":", 1)[1]
            counts[transform] = counts.get(transform, 0) + 1
    return counts


def run_one(
    seed_text: str,
    n_per_strategy: int,
    client: LLMClient,
    output_dir: Path,
) -> tuple[list[dict], dict] | None:
    """Run the full allo pipeline for a single seed at a single --n."""
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        variants = generate_variants(
            seed=seed_text,
            n_per_strategy=n_per_strategy,
            client=client,
            temperature=TEMPERATURE,
            include_expansion=INCLUDE_EXPANSION,
        )
        scored, summary = score_variants(seed=seed_text, variants=variants)
        write_csv(
            scored_variants=scored,
            summary=summary,
            output_dir=str(output_dir),
        )
        return scored, summary
    except Exception as e:
        print(f"    FAILED: {e}")
        return None


# ─── Main runner ───────────────────────────────────────────────────────

def run_sweep(resume_dir: Path | None = None, n_values: list[int] = None):
    n_values = n_values or N_VALUES

    # Set up run directory
    if resume_dir:
        run_dir = resume_dir
        print(f"Resuming run at: {run_dir}")
        with open(run_dir / "run_config.json") as f:
            config = json.load(f)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = OUTPUT_ROOT / f"volume_sweep_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "n_values": N_VALUES,
            "temperature": TEMPERATURE,
            "include_expansion": INCLUDE_EXPANSION,
            "seed_set": str(SEED_SET_PATH),
            "timestamp": timestamp,
            "git_sha": get_git_sha(),
        }
        with open(run_dir / "run_config.json", "w") as f:
            json.dump(config, f, indent=2)

    seeds = load_seeds()
    client = LLMClient()
    print(f"Provider: {client.provider} | Model: {client.model}")
    print(f"Seeds: {len(seeds)} | n values: {n_values}")
    print(f"Output: {run_dir}\n")

    # Aggregate results collector — long format (seed_id, n, strategy, count)
    aggregate_path = run_dir / "volume_sweep_aggregate.csv"
    aggregate_exists = aggregate_path.exists()

    # When resuming, read existing rows to avoid duplicate work
    completed = set()
    if aggregate_exists:
        with open(aggregate_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add((row["seed_id"], int(row["n"])))

    total_combos = len(seeds) * len(n_values)
    combos_done = 0
    start = time.time()

    with open(aggregate_path, "a", newline="") as agg_f:
        agg_fieldnames = [
            "seed_id", "seed", "domain", "syntactic_type", "register",
            "utterance_length", "lexical_complexity", "speech_phenomena",
            "n", "strategy", "count", "total_for_this_n",
            "constrained_by_transform",
        ]
        writer = csv.DictWriter(agg_f, fieldnames=agg_fieldnames)
        if not aggregate_exists:
            writer.writeheader()

        for n in n_values:
            n_dir = run_dir / f"n_{n}"
            n_dir.mkdir(exist_ok=True)
            print(f"\n{'=' * 60}\n  n = {n}\n{'=' * 60}")

            for seed_row in seeds:
                seed_id = seed_row["id"]
                seed_text = seed_row["seed"]

                if (seed_id, n) in completed:
                    print(f"  [{seed_id}] SKIP (already complete)")
                    combos_done += 1
                    continue

                elapsed = time.time() - start
                pct = combos_done / total_combos * 100
                eta = (elapsed / max(combos_done, 1)) * (total_combos - combos_done) / 60
                print(
                    f"  [{seed_id}/{len(seeds)}] n={n}  "
                    f"({combos_done}/{total_combos}, {pct:.1f}%, ETA {eta:.0f} min)  "
                    f"→ {seed_text[:50]}"
                )

                seed_output_dir = n_dir / f"seed_{int(seed_id):03d}"
                result = run_one(seed_text, n, client, seed_output_dir)

                if result is None:
                    # Log failure as null counts so we can detect them in analysis
                    for strategy in ["llm_paraphrase", "constrained", "mlm_substitution", "expansion"]:
                        writer.writerow({
                            "seed_id": seed_id,
                            "seed": seed_text,
                            "domain": seed_row.get("domain", ""),
                            "syntactic_type": seed_row.get("syntactic_type", ""),
                            "register": seed_row.get("register", ""),
                            "utterance_length": seed_row.get("utterance_length", ""),
                            "lexical_complexity": seed_row.get("lexical_complexity", ""),
                            "speech_phenomena": seed_row.get("speech_phenomena", ""),
                            "n": n,
                            "strategy": strategy,
                            "count": "",
                            "total_for_this_n": "",
                            "constrained_by_transform": "",
                        })
                    agg_f.flush()
                    combos_done += 1
                    continue

                scored, _ = result
                strategy_counts = count_strategies(scored)
                constrained_transforms = count_constrained_by_transform(scored)
                total = sum(strategy_counts.values())

                for strategy, count in strategy_counts.items():
                    writer.writerow({
                        "seed_id": seed_id,
                        "seed": seed_text,
                        "domain": seed_row.get("domain", ""),
                        "syntactic_type": seed_row.get("syntactic_type", ""),
                        "register": seed_row.get("register", ""),
                        "utterance_length": seed_row.get("utterance_length", ""),
                        "lexical_complexity": seed_row.get("lexical_complexity", ""),
                        "speech_phenomena": seed_row.get("speech_phenomena", ""),
                        "n": n,
                        "strategy": strategy,
                        "count": count,
                        "total_for_this_n": total,
                        "constrained_by_transform": (
                            json.dumps(constrained_transforms)
                            if strategy == "constrained" else ""
                        ),
                    })
                agg_f.flush()  # write as we go, so a crash doesn't lose everything
                combos_done += 1

    total_min = (time.time() - start) / 60
    print(f"\n{'=' * 60}")
    print(f"Sweep complete in {total_min:.1f} minutes")
    print(f"Aggregate results: {aggregate_path}")
    print(f"{'=' * 60}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Path to a prior run directory. Skips combos already in volume_sweep_aggregate.csv.",
    )
    parser.add_argument(
        "--n-values", type=int, nargs="+", default=None,
        help="Override the list of --n values to run (e.g. --n-values 80 100).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sweep(resume_dir=args.resume, n_values=args.n_values)