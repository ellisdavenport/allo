"""
prompt_revisions.py — generate constrained variants under v1.1 revised prompts.

Runs four revised constrained transforms (negative_framing, modal_should,
passive_voice, time_framed) on the 117-seed test set at n_per_constraint=3
and writes the output to a CSV ready to feed to llm_judge_batch.py.

The output mirrors the column layout of scoring_ranges_aggregate.csv so the
downstream comparison can join on (seed_id, constrained_transform) and
compare judge-score distributions between v1.0 and v1.1.

Skip tracking: the v1.1 transforms include TRANSFORM_NOT_APPLICABLE
applicability gating. (seed, transform) pairs the LLM judged inapplicable
produce zero variants. The runner tracks these and emits a summary JSON
alongside the output CSV.

Usage:
    From the project root:
        python -m evaluation.studies.prompt_revisions \\
            --output evaluation/runs/prompt_revisions_v1_1_variants.csv

After running, score with the judge:
        python -m evaluation.studies.llm_judge_batch \\
            --input evaluation/runs/prompt_revisions_v1_1_variants.csv \\
            --output evaluation/runs/prompt_revisions_v1_1_judged.csv
"""

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from allo.generate import (
    CONSTRAINT_VERSION,
    LLMClient,
    SYNTACTIC_CONSTRAINTS,
    generate_constrained_variants,
)


# ─── Configuration ─────────────────────────────────────────────────────
REVISED_TRANSFORMS = {
    "negative_framing",
    "modal_should",
    "passive_voice",
    "time_framed",
}
N_PER_CONSTRAINT = 3
TEMPERATURE = 0.4
SEED_SET_PATH = Path("evaluation/allo_seed_set.csv")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output CSV path for the generated variants",
    )
    parser.add_argument(
        "--summary", type=Path, default=None,
        help="Optional path for the run summary JSON "
             "(default: <output_path>.summary.json)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary_path = args.summary or args.output.with_suffix(args.output.suffix + ".summary.json")

    # Sanity checks
    if CONSTRAINT_VERSION != "v1.1":
        print(f"WARNING: CONSTRAINT_VERSION is '{CONSTRAINT_VERSION}', expected 'v1.1'. "
              f"Are the revised prompts committed?")

    constraints_to_run = [c for c in SYNTACTIC_CONSTRAINTS if c["name"] in REVISED_TRANSFORMS]
    if len(constraints_to_run) != len(REVISED_TRANSFORMS):
        found = {c["name"] for c in constraints_to_run}
        missing = REVISED_TRANSFORMS - found
        raise RuntimeError(
            f"Could not find all revised transforms in SYNTACTIC_CONSTRAINTS. "
            f"Missing: {missing}"
        )

    # Load seeds
    with open(SEED_SET_PATH) as f:
        seeds = list(csv.DictReader(f))
    print(f"Loaded {len(seeds)} seeds from {SEED_SET_PATH}")
    print(f"Running {len(constraints_to_run)} revised transforms at n_per_constraint={N_PER_CONSTRAINT}")
    print(f"CONSTRAINT_VERSION: {CONSTRAINT_VERSION}")
    print()

    client = LLMClient()
    print(f"Provider: {client.provider} | Model: {client.model}\n")

    # Output schema — mirrors scoring_ranges_aggregate.csv minus the surface
    # metric columns. The judge runner appends its own columns downstream.
    output_fields = [
        "seed_id", "seed", "domain", "syntactic_type", "register",
        "utterance_length", "lexical_complexity", "speech_phenomena",
        "strategy", "strategy_family", "constrained_transform",
        "variant", "constraint_version",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Skip tracking per (transform, syntactic_type) for the summary
    skip_counts = defaultdict(lambda: defaultdict(int))
    success_counts = defaultdict(lambda: defaultdict(int))
    total_variants = 0
    n_seeds_processed = 0

    start = time.time()

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()

        for seed_row in seeds:
            seed_id = seed_row["id"]
            seed = seed_row["seed"]
            syntactic_type = seed_row.get("syntactic_type", "")

            elapsed = time.time() - start
            print(f"  [{seed_id:>3}/{len(seeds)}] ({elapsed:.0f}s) "
                  f"[{syntactic_type:<20}] {seed[:60]}")

            try:
                variants = generate_constrained_variants(
                    seed=seed,
                    n_per_constraint=N_PER_CONSTRAINT,
                    client=client,
                    temperature=TEMPERATURE,
                    constraints=constraints_to_run,
                )
            except Exception as e:
                print(f"    FAILED entirely: {e}")
                continue

            # Bucket variants by transform
            by_transform = defaultdict(list)
            for v in variants:
                by_transform[v["constraint"]].append(v["utterance"])

            # Write rows + track skip/success counts
            for transform_name in REVISED_TRANSFORMS:
                transform_variants = by_transform.get(transform_name, [])
                if not transform_variants:
                    # Either skip via sentinel or upstream failure.
                    # We treat both as skips here; the summary syntactic_type
                    # breakdown will help distinguish intended skips
                    # (wh_question on negative_framing) from anomalies.
                    skip_counts[transform_name][syntactic_type] += 1
                    continue

                success_counts[transform_name][syntactic_type] += 1
                for variant_text in transform_variants:
                    row = {
                        "seed_id": seed_id,
                        "seed": seed,
                        "domain": seed_row.get("domain", ""),
                        "syntactic_type": syntactic_type,
                        "register": seed_row.get("register", ""),
                        "utterance_length": seed_row.get("utterance_length", ""),
                        "lexical_complexity": seed_row.get("lexical_complexity", ""),
                        "speech_phenomena": seed_row.get("speech_phenomena", ""),
                        "strategy": f"constrained:{transform_name}",
                        "strategy_family": "constrained",
                        "constrained_transform": transform_name,
                        "variant": variant_text,
                        "constraint_version": CONSTRAINT_VERSION,
                    }
                    writer.writerow(row)
                    total_variants += 1

            n_seeds_processed += 1

    elapsed = time.time() - start

    # Build summary
    summary = {
        "constraint_version": CONSTRAINT_VERSION,
        "revised_transforms": sorted(REVISED_TRANSFORMS),
        "n_seeds": len(seeds),
        "n_seeds_processed": n_seeds_processed,
        "n_per_constraint": N_PER_CONSTRAINT,
        "temperature": TEMPERATURE,
        "total_variants_generated": total_variants,
        "elapsed_seconds": int(elapsed),
        "provider": client.provider,
        "model": client.model,
        "per_transform": {},
    }
    for transform in sorted(REVISED_TRANSFORMS):
        successes_by_type = dict(success_counts.get(transform, {}))
        skips_by_type = dict(skip_counts.get(transform, {}))
        all_types = sorted(set(successes_by_type) | set(skips_by_type))
        per_type = {}
        for t in all_types:
            ns = successes_by_type.get(t, 0)
            nk = skips_by_type.get(t, 0)
            per_type[t] = {
                "seeds_with_variants": ns,
                "seeds_skipped_or_failed": nk,
                "skip_rate": round(nk / (ns + nk), 3) if (ns + nk) else 0.0,
            }
        summary["per_transform"][transform] = {
            "total_seeds_with_variants": sum(successes_by_type.values()),
            "total_seeds_skipped_or_failed": sum(skips_by_type.values()),
            "by_syntactic_type": per_type,
        }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"Generation complete in {elapsed:.0f}s")
    print(f"{'=' * 70}")
    print(f"  Constraint version:  {CONSTRAINT_VERSION}")
    print(f"  Seeds processed:     {n_seeds_processed}/{len(seeds)}")
    print(f"  Total variants:      {total_variants}")
    print(f"  Output:              {args.output}")
    print(f"  Summary:             {summary_path}")

    print(f"\nSkip rates by (transform, syntactic_type):")
    print(f"  Expected pattern: revised transforms should skip on syntactic types")
    print(f"  they don't compose with (e.g. negative_framing on wh_question).")
    print(f"\n  {'transform':<22s} {'syntactic_type':<22s} {'kept':>6s} {'skipped':>9s} {'skip%':>7s}")
    print(f"  {'-' * 22} {'-' * 22} {'-' * 6} {'-' * 9} {'-' * 7}")
    for transform in sorted(REVISED_TRANSFORMS):
        by_type = summary["per_transform"][transform]["by_syntactic_type"]
        for stype, stats in sorted(by_type.items(), key=lambda x: -x[1]["skip_rate"]):
            ns = stats["seeds_with_variants"]
            nk = stats["seeds_skipped_or_failed"]
            rate = stats["skip_rate"]
            print(f"  {transform:<22s} {stype:<22s} {ns:>6d} {nk:>9d} {rate*100:>6.0f}%")
        print()


if __name__ == "__main__":
    main()