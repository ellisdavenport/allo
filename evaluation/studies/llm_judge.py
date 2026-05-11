"""
llm_judge.py — apply the LLM-as-judge rubric to allo variants.

Reads (seed, variant) pairs from an input CSV, calls Claude Haiku 4.5 with the
versioned rubric prompt, and writes per-variant judgment scores plus rationales
to an output CSV.

Two layers of caching:
  1. Anthropic prompt caching (cache_control on system message). Saves ~90% on
     the rubric portion of input tokens for repeated calls within 5 minutes.
  2. Local result caching (pickle dict keyed by prompt version + seed + variant).
     Re-runs after restart return cached judgments at zero cost. Cache invalidates
     automatically when PROMPT_VERSION changes.

Designed to scale from the 50-variant Day 1 calibration set to the full
~17K-variant scoring_ranges aggregate without changes.

Usage:
    python -m evaluation.studies.llm_judge \\
        --input evaluation/studies/judge_calibration.csv \\
        --output evaluation/runs/llm_judge_calibration_v1.csv

The input CSV must have columns: seed, variant. Other columns are passed through
unchanged so you can keep strategy, similarity, perplexity, etc. alongside the
new judgment columns.
"""

import argparse
import csv
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from evaluation.studies.judge_prompt import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_user_message,
)

load_dotenv()


# ─── Configuration ─────────────────────────────────────────────────────
JUDGE_MODEL = "claude-haiku-4-5"
TEMPERATURE = 0.0  # deterministic measurement, not generation
MAX_TOKENS = 400  # rationales are one-sentence; 400 is a comfortable ceiling
MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 2.0

CACHE_DIR = Path("evaluation/runs/llm_judge_cache")


# ─── Local result cache ────────────────────────────────────────────────

def cache_key(seed: str, variant: str) -> str:
    """
    Deterministic cache key combining prompt version, seed, and variant.

    Including PROMPT_VERSION means a rubric change automatically invalidates
    all previous cached judgments — no manual cache-clearing required.
    """
    payload = f"{PROMPT_VERSION}|{seed}|{variant}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def save_cache(cache: dict, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)


# ─── Judge call ────────────────────────────────────────────────────────

def call_judge(client: anthropic.Anthropic, seed: str, variant: str) -> dict:
    """
    Send a single judgment request to Haiku 4.5 and return the parsed JSON.

    Uses Anthropic prompt caching: the system prompt (rubric + anchors) is
    marked cache_control=ephemeral so it is cached for 5 minutes and reused
    across calls at 10% of base input cost.

    Retries on transient errors (rate limit, server error) with exponential
    backoff. Hard-fails on a malformed JSON response after retries are
    exhausted, since silently scoring 0 would corrupt the aggregate.
    """
    user_message = build_user_message(seed, variant)

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_message}],
            )
            raw = response.content[0].text.strip()

            # Sometimes the model wraps JSON in markdown fences despite
            # the prompt instruction. Strip them defensively.
            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.startswith("json"):
                    raw = raw[4:].strip()

            parsed = json.loads(raw)

            # Validate required fields and score ranges
            for field in [
                "semantic_equivalence",
                "naturalness",
                "semantic_equivalence_rationale",
                "naturalness_rationale",
            ]:
                if field not in parsed:
                    raise ValueError(f"Missing field {field} in judge response")
            for score_field in ["semantic_equivalence", "naturalness"]:
                score = parsed[score_field]
                if not isinstance(score, int) or not 1 <= score <= 5:
                    raise ValueError(
                        f"{score_field} must be integer 1-5, got {score!r}"
                    )

            return parsed

        except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            last_error = e
            wait = RETRY_BACKOFF_SECONDS * (2 ** attempt)
            print(f"    transient error ({type(e).__name__}), retry in {wait:.1f}s: {e}",
                  file=sys.stderr)
            time.sleep(wait)
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            wait = RETRY_BACKOFF_SECONDS * (2 ** attempt)
            print(f"    parse error, retry in {wait:.1f}s: {e}", file=sys.stderr)
            time.sleep(wait)

    raise RuntimeError(
        f"Judge call failed after {MAX_RETRIES} retries for "
        f"seed='{seed[:40]}...' variant='{variant[:40]}...': {last_error}"
    )


# ─── Main loop ─────────────────────────────────────────────────────────

def judge_csv(input_path: Path, output_path: Path, cache_path: Path) -> None:
    """
    Read input CSV, score each row, write enriched CSV to output.

    Pass-through fields: every column in the input CSV is preserved in the
    output. Judgment columns are appended.
    """
    cache = load_cache(cache_path)
    print(f"Loaded {len(cache)} cached judgments from {cache_path}")

    client = anthropic.Anthropic()

    # Read input rows
    with open(input_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        input_fields = reader.fieldnames or []

    output_fields = list(input_fields) + [
        "judge_prompt_version",
        "judge_semantic_equivalence",
        "judge_semantic_equivalence_rationale",
        "judge_naturalness",
        "judge_naturalness_rationale",
        "judge_status",  # 'cached', 'scored', 'failed'
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = len(rows)
    n_cached = 0
    n_scored = 0
    n_failed = 0
    cache_writes_since_save = 0
    CACHE_SAVE_INTERVAL = 25  # persist cache every 25 fresh scores

    start = time.time()

    with open(output_path, "w", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=output_fields)
        writer.writeheader()

        for i, row in enumerate(rows, 1):
            seed = row.get("seed", "").strip()
            variant = row.get("variant", "").strip()

            if not seed or not variant:
                row["judge_status"] = "failed"
                row["judge_prompt_version"] = PROMPT_VERSION
                writer.writerow(row)
                n_failed += 1
                continue

            key = cache_key(seed, variant)

            if key in cache:
                judgment = cache[key]
                status = "cached"
                n_cached += 1
            else:
                try:
                    judgment = call_judge(client, seed, variant)
                    cache[key] = judgment
                    cache_writes_since_save += 1
                    status = "scored"
                    n_scored += 1
                except RuntimeError as e:
                    print(f"  [{i}/{n_total}] FAIL: {e}", file=sys.stderr)
                    row["judge_status"] = "failed"
                    row["judge_prompt_version"] = PROMPT_VERSION
                    writer.writerow(row)
                    n_failed += 1
                    continue

            # Periodic cache flush so a crash doesn't lose work
            if cache_writes_since_save >= CACHE_SAVE_INTERVAL:
                save_cache(cache, cache_path)
                cache_writes_since_save = 0

            # Progress reporting
            if status == "scored":
                elapsed = time.time() - start
                rate = (n_scored + 1e-9) / elapsed
                remaining_scored_estimate = (n_total - i) * (n_scored / max(i, 1))
                eta_min = remaining_scored_estimate / max(rate, 1e-9) / 60
                print(
                    f"  [{i}/{n_total}] {status:6s}  "
                    f"equiv={judgment['semantic_equivalence']} "
                    f"natural={judgment['naturalness']}  "
                    f"ETA ~{eta_min:.0f} min  "
                    f"seed='{seed[:40]}...'"
                )

            row["judge_prompt_version"] = PROMPT_VERSION
            row["judge_semantic_equivalence"] = judgment["semantic_equivalence"]
            row["judge_semantic_equivalence_rationale"] = judgment[
                "semantic_equivalence_rationale"
            ]
            row["judge_naturalness"] = judgment["naturalness"]
            row["judge_naturalness_rationale"] = judgment["naturalness_rationale"]
            row["judge_status"] = status
            writer.writerow(row)

    save_cache(cache, cache_path)

    print(f"\n{'=' * 60}")
    print(f"Judging complete")
    print(f"{'=' * 60}")
    print(f"  Total rows:    {n_total}")
    print(f"  Cache hits:    {n_cached}")
    print(f"  Newly scored:  {n_scored}")
    print(f"  Failed:        {n_failed}")
    print(f"  Output:        {output_path}")
    print(f"  Cache:         {cache_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True,
        help="CSV with at least 'seed' and 'variant' columns",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output CSV path (parent directory created if needed)",
    )
    parser.add_argument(
        "--cache", type=Path, default=CACHE_DIR / f"judgments_{PROMPT_VERSION}.pkl",
        help="Path for the local judgment cache (default: per-version)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    judge_csv(args.input, args.output, args.cache)