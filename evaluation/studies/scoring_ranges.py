"""
scoring_ranges.py — aggregate variant-level scores from a completed volume sweep.

This is NOT a runner — no API calls, no model loading. It reads the raw allo
output CSVs already produced by volume_sweep.py and assembles a long-format
table where each row is one variant with its seed dimension tags, strategy,
utterance, semantic similarity, and perplexity.

Input:
    A completed volume sweep run directory, typically at
        evaluation/runs/volume_sweep_YYYYMMDD_HHMMSS/
    containing per-seed output CSVs under n_50/seed_NNN/allo_*.csv.
    Plus the seed set at evaluation/allo_seed_set.csv for dimension tags.

Output:
    evaluation/results/scoring_ranges/scoring_ranges_aggregate.csv

    One row per variant, with columns:
        seed_id, seed, domain, syntactic_type, register, utterance_length,
        lexical_complexity, speech_phenomena, strategy, strategy_family,
        constrained_transform, utterance, semantic_similarity, perplexity

    strategy_family is the collapsed 4-category tag (llm_paraphrase,
    constrained, mlm_substitution, expansion).
    constrained_transform is the specific transform name for constrained rows,
    empty for the other strategies.

Usage:
    From the project root:
        python -m evaluation.studies.scoring_ranges \\
            --run-dir evaluation/runs/volume_sweep_YYYYMMDD_HHMMSS
"""

import argparse
import csv
from pathlib import Path


# ─── Configuration ─────────────────────────────────────────────────────
# Restricting to --n=50 per the study design: that's where every strategy
# is at meaningful volume but not hitting the short-seed plateau effects
# that distort distributions at higher --n.
TARGET_N = 50
SEED_SET_PATH = Path("evaluation/allo_seed_set.csv")
OUTPUT_DIR = Path("evaluation/results/scoring_ranges")


# ─── Parsing ───────────────────────────────────────────────────────────

def parse_allo_csv(path: Path) -> list[dict]:
    """
    Read variants from an allo output CSV, skipping summary rows.

    The allo CSV format has variant rows followed by a blank separator row
    and a summary block (total_variants, mean_*, lexical_diversity_ttr).
    We identify variant rows by the presence of a non-empty strategy column
    and a parseable semantic_similarity value.
    """
    variants = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            strategy = (row.get("strategy") or "").strip()
            sim_str = (row.get("semantic_similarity") or "").strip()

            # Skip blank rows, summary header rows, and any row that doesn't
            # have both a strategy and a similarity score
            if not strategy or strategy.startswith("#"):
                continue
            if not sim_str:
                continue

            try:
                variants.append({
                    "seed": row["seed"],
                    "utterance": row["utterance"],
                    "strategy": strategy,
                    "semantic_similarity": float(sim_str),
                    "perplexity": float(row["perplexity"]),
                })
            except (ValueError, KeyError):
                # Defensive: if a malformed row slips through, skip rather
                # than crashing a 14K-row aggregation.
                continue

    return variants


def classify_strategy(raw_tag: str) -> tuple[str, str]:
    """
    Split a raw strategy tag into (family, transform).

    - 'llm_paraphrase' -> ('llm_paraphrase', '')
    - 'constrained:polar_question' -> ('constrained', 'polar_question')
    - 'mlm_substitution' -> ('mlm_substitution', '')
    - 'expansion' -> ('expansion', '')
    """
    if raw_tag.startswith("constrained:"):
        return "constrained", raw_tag.split(":", 1)[1]
    return raw_tag, ""


# ─── Seed set loading ──────────────────────────────────────────────────

def load_seed_dimensions(seed_set_path: Path) -> dict[str, dict]:
    """
    Load allo_seed_set.csv into a {seed_id_str: row_dict} mapping.

    Stringified IDs because seed_dir names like 'seed_007' produce str IDs
    and we don't want type mismatches.
    """
    seeds = {}
    with open(seed_set_path) as f:
        for row in csv.DictReader(f):
            seeds[row["id"]] = row
    return seeds


# ─── Aggregation ───────────────────────────────────────────────────────

def aggregate(run_dir: Path, target_n: int = TARGET_N) -> list[dict]:
    """Walk the target-n directory and build the long-format variant table."""
    n_dir = run_dir / f"n_{target_n}"
    if not n_dir.exists():
        raise FileNotFoundError(f"No n_{target_n} subdirectory in {run_dir}")

    seeds = load_seed_dimensions(SEED_SET_PATH)

    rows = []
    seed_dirs = sorted(
        (d for d in n_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")),
        key=lambda d: int(d.name.split("_")[1]),
    )

    for seed_dir in seed_dirs:
        # seed_dir names are zero-padded ('seed_007'); seed_set ids aren't
        seed_id = str(int(seed_dir.name.split("_")[1]))
        seed_row = seeds.get(seed_id)
        if seed_row is None:
            print(f"  WARN: no seed_set row for {seed_dir.name}, skipping")
            continue

        # Find the allo CSV (if multiple, take the most recent by filename)
        csv_matches = sorted(seed_dir.glob("allo_*.csv"))
        if not csv_matches:
            print(f"  WARN: no allo CSV in {seed_dir}, skipping")
            continue
        csv_path = csv_matches[-1]

        variants = parse_allo_csv(csv_path)
        for v in variants:
            family, transform = classify_strategy(v["strategy"])
            rows.append({
                "seed_id": seed_id,
                "seed": seed_row["seed"],
                "domain": seed_row.get("domain", ""),
                "syntactic_type": seed_row.get("syntactic_type", ""),
                "register": seed_row.get("register", ""),
                "utterance_length": seed_row.get("utterance_length", ""),
                "lexical_complexity": seed_row.get("lexical_complexity", ""),
                "speech_phenomena": seed_row.get("speech_phenomena", ""),
                "strategy": v["strategy"],  # full raw tag (constrained:polar_question)
                "strategy_family": family,  # collapsed 4-category
                "constrained_transform": transform,
                "utterance": v["utterance"],
                "semantic_similarity": v["semantic_similarity"],
                "perplexity": v["perplexity"],
            })

    return rows


def write_aggregate(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed_id", "seed", "domain", "syntactic_type", "register",
        "utterance_length", "lexical_complexity", "speech_phenomena",
        "strategy", "strategy_family", "constrained_transform",
        "utterance", "semantic_similarity", "perplexity",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} variant rows to {output_path}")


# ─── Reporting ─────────────────────────────────────────────────────────

def print_summary(rows: list[dict]) -> None:
    """Quick sanity check on the aggregate before the notebook takes over."""
    from collections import Counter

    print(f"\nTotal variants: {len(rows)}")
    print(f"Unique seeds: {len(set(r['seed_id'] for r in rows))}")

    family_counts = Counter(r["strategy_family"] for r in rows)
    print("\nCount by strategy_family:")
    for family, count in sorted(family_counts.items()):
        print(f"  {family}: {count}")

    print("\nCount by utterance_length:")
    length_counts = Counter(r["utterance_length"] for r in rows)
    for length, count in sorted(length_counts.items()):
        print(f"  {length}: {count}")

    # Sanity: score ranges per family (quick eyeball before the real analysis)
    print("\nSemantic similarity — per-family range:")
    for family in sorted(family_counts):
        sims = [r["semantic_similarity"] for r in rows if r["strategy_family"] == family]
        if sims:
            print(f"  {family:20s}  min={min(sims):.3f}  max={max(sims):.3f}  "
                  f"n={len(sims)}")

    print("\nPerplexity — per-family range:")
    for family in sorted(family_counts):
        ppls = [r["perplexity"] for r in rows if r["strategy_family"] == family]
        if ppls:
            print(f"  {family:20s}  min={min(ppls):.1f}  max={max(ppls):.1f}  "
                  f"n={len(ppls)}")


# ─── CLI ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, required=True,
        help="Path to the volume sweep run directory containing n_50/seed_*/*.csv",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_DIR / "scoring_ranges_aggregate.csv",
        help="Path for the aggregated CSV output.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rows = aggregate(args.run_dir, target_n=TARGET_N)
    write_aggregate(rows, args.output)
    print_summary(rows)