"""
llm_judge_batch.py — submit allo variants to the Anthropic Message Batches API.

Companion to llm_judge.py. Same rubric, same prompt, same output schema —
different submission path. Use this for the full scoring_ranges aggregate
run on Day 2. Use llm_judge.py for interactive single-variant runs (Day 1
calibration, prompt revision validation on smaller subsets).

Why batch:
  - 50% discount on input and output tokens vs the standard API
  - Robust to local machine sleep/network — runs server-side
  - 17K variants fit comfortably in one batch (limit is 100K)
  - Most batches complete in well under an hour; hard timeout 24 hours

Architecture:
  Phase 1 (submit): Read input CSV, build batch requests, submit to Anthropic,
                    persist batch_id + custom_id→row mapping to a state file.
  Phase 2 (poll):   Re-read state, poll batch status until processing_status
                    is 'ended'. Cheap and idempotent — safe to interrupt and
                    re-run.
  Phase 3 (collect): Stream results, match by custom_id, validate, write
                    output CSV in input order. Mirror llm_judge.py's output
                    schema so downstream analysis is identical.

State file at <output_path>.state.json holds the batch_id and metadata. If
the script is interrupted between phases, re-running picks up the state
file and resumes. To start a fresh batch, delete the state file.

Cost projection at Haiku 4.5 batch pricing (50% discount + prompt caching
on the rubric):
  ~17K variants × ~$0.0005 per call ≈ $8-10 total

Usage:
    python -m evaluation.studies.llm_judge_batch \\
        --input evaluation/runs/scoring_ranges_for_judge.csv \\
        --output evaluation/runs/llm_judge_aggregate_v1.csv

The input CSV must have 'seed' and 'variant' columns. Other columns are
passed through to the output (strategy, similarity, perplexity, etc.).
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from dotenv import load_dotenv

from evaluation.studies.judge_prompt import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_user_message,
)

load_dotenv()


# ─── Configuration ─────────────────────────────────────────────────────
JUDGE_MODEL = "claude-haiku-4-5"
TEMPERATURE = 0.0
MAX_TOKENS = 400
POLL_INTERVAL_SECONDS = 30  # cheap to check, batch usually < 1 hour


# ─── State file management ─────────────────────────────────────────────

def state_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".state.json")


def load_state(output_path: Path) -> dict | None:
    p = state_path(output_path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def save_state(output_path: Path, state: dict) -> None:
    p = state_path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(state, f, indent=2)


# ─── Phase 1: build and submit batch ───────────────────────────────────

def build_batch_requests(input_rows: list[dict]) -> tuple[list[Request], dict]:
    """
    Convert input rows into batch Request objects.

    Returns the request list and a custom_id → input row mapping so we can
    reassemble the output in input order after results come back.

    custom_id format: "v_{index:06d}" — keeps it short (Anthropic limits
    custom_id length) and deterministic from input order.
    """
    requests = []
    id_to_row = {}

    for i, row in enumerate(input_rows):
        seed = row.get("seed", "").strip()
        variant = row.get("variant", "").strip()
        if not seed or not variant:
            # Skip blank rows entirely — they'll be marked failed at output time
            continue

        custom_id = f"v_{i:06d}"
        id_to_row[custom_id] = i

        requests.append(
            Request(
                custom_id=custom_id,
                params=MessageCreateParamsNonStreaming(
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
                    messages=[
                        {"role": "user", "content": build_user_message(seed, variant)}
                    ],
                ),
            )
        )

    return requests, id_to_row


def submit_batch(client: anthropic.Anthropic, requests: list[Request]) -> str:
    """Submit batch and return batch_id."""
    print(f"Submitting {len(requests)} requests as a single batch...")
    batch = client.messages.batches.create(requests=requests)
    print(f"  batch_id: {batch.id}")
    print(f"  status: {batch.processing_status}")
    return batch.id


# ─── Phase 2: poll for completion ──────────────────────────────────────

def poll_until_done(client: anthropic.Anthropic, batch_id: str) -> dict:
    """
    Poll batch status until processing_status == 'ended'.

    Cheap to call — server-side polling, no per-poll cost. Re-running this
    function after interruption is safe.
    """
    print(f"\nPolling batch {batch_id} (every {POLL_INTERVAL_SECONDS}s)...")
    start = time.time()

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        elapsed = time.time() - start

        # request_counts breaks down per-status totals (processing, succeeded,
        # errored, canceled, expired)
        counts = batch.request_counts
        total = sum([counts.processing, counts.succeeded, counts.errored,
                     counts.canceled, counts.expired])

        print(
            f"  [{int(elapsed)}s] status={batch.processing_status}  "
            f"processing={counts.processing}  succeeded={counts.succeeded}  "
            f"errored={counts.errored}  total={total}"
        )

        if batch.processing_status == "ended":
            print(f"\nBatch ended after {int(elapsed)}s "
                  f"({counts.succeeded} succeeded, {counts.errored} errored)")
            return {
                "succeeded": counts.succeeded,
                "errored": counts.errored,
                "canceled": counts.canceled,
                "expired": counts.expired,
            }

        time.sleep(POLL_INTERVAL_SECONDS)


# ─── Phase 3: collect results and write output ─────────────────────────

def parse_judge_response(raw_text: str) -> dict:
    """Parse the judge's JSON response and validate score fields."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()

    parsed = json.loads(text)

    for field in [
        "semantic_equivalence",
        "naturalness",
        "semantic_equivalence_rationale",
        "naturalness_rationale",
    ]:
        if field not in parsed:
            raise ValueError(f"Missing field: {field}")

    for score_field in ["semantic_equivalence", "naturalness"]:
        score = parsed[score_field]
        if not isinstance(score, int) or not 1 <= score <= 5:
            raise ValueError(f"{score_field} must be integer 1-5, got {score!r}")

    return parsed


def collect_results(
    client: anthropic.Anthropic,
    batch_id: str,
    input_rows: list[dict],
    id_to_row_idx: dict,
    output_path: Path,
) -> dict:
    """
    Stream batch results, match by custom_id, write output CSV.

    Output schema mirrors llm_judge.py:
      - All input columns preserved
      - Appended: judge_prompt_version, judge_semantic_equivalence,
        judge_semantic_equivalence_rationale, judge_naturalness,
        judge_naturalness_rationale, judge_status

    Output CSV is written in input row order (the order of input_rows),
    not result arrival order.
    """
    print(f"\nCollecting results from batch {batch_id}...")

    # Map: input row index → judgment dict (or failure info)
    judgments_by_idx = {}

    n_succeeded = 0
    n_errored = 0
    n_parse_failed = 0

    for entry in client.messages.batches.results(batch_id):
        custom_id = entry.custom_id
        if custom_id not in id_to_row_idx:
            print(f"  WARN: unexpected custom_id {custom_id}, skipping",
                  file=sys.stderr)
            continue
        row_idx = id_to_row_idx[custom_id]

        match entry.result.type:
            case "succeeded":
                raw = entry.result.message.content[0].text
                try:
                    parsed = parse_judge_response(raw)
                    judgments_by_idx[row_idx] = {
                        "status": "scored",
                        "judgment": parsed,
                    }
                    n_succeeded += 1
                except (json.JSONDecodeError, ValueError) as e:
                    judgments_by_idx[row_idx] = {
                        "status": "parse_failed",
                        "error": f"{type(e).__name__}: {e}",
                        "raw": raw[:200],
                    }
                    n_parse_failed += 1

            case "errored":
                judgments_by_idx[row_idx] = {
                    "status": "api_error",
                    "error": str(entry.result.error),
                }
                n_errored += 1

            case "canceled" | "expired":
                judgments_by_idx[row_idx] = {
                    "status": entry.result.type,
                    "error": f"request {entry.result.type}",
                }
                n_errored += 1

    print(f"  Collected: {n_succeeded} succeeded, {n_errored} errored, "
          f"{n_parse_failed} parse failures")

    # Write output CSV in input order
    input_fields = list(input_rows[0].keys()) if input_rows else []
    output_fields = input_fields + [
        "judge_prompt_version",
        "judge_semantic_equivalence",
        "judge_semantic_equivalence_rationale",
        "judge_naturalness",
        "judge_naturalness_rationale",
        "judge_status",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        for i, row in enumerate(input_rows):
            row_out = dict(row)
            row_out["judge_prompt_version"] = PROMPT_VERSION

            j = judgments_by_idx.get(i)
            if j is None:
                # Blank row that we skipped during submission, or otherwise
                # missing from the batch results
                row_out["judge_status"] = "skipped"
                row_out["judge_semantic_equivalence"] = ""
                row_out["judge_semantic_equivalence_rationale"] = ""
                row_out["judge_naturalness"] = ""
                row_out["judge_naturalness_rationale"] = ""
            elif j["status"] == "scored":
                row_out["judge_status"] = "scored"
                row_out["judge_semantic_equivalence"] = j["judgment"]["semantic_equivalence"]
                row_out["judge_semantic_equivalence_rationale"] = j["judgment"]["semantic_equivalence_rationale"]
                row_out["judge_naturalness"] = j["judgment"]["naturalness"]
                row_out["judge_naturalness_rationale"] = j["judgment"]["naturalness_rationale"]
            else:
                # Failure of any kind — write status and leave score fields empty
                row_out["judge_status"] = j["status"]
                row_out["judge_semantic_equivalence"] = ""
                row_out["judge_semantic_equivalence_rationale"] = j.get("error", "")[:300]
                row_out["judge_naturalness"] = ""
                row_out["judge_naturalness_rationale"] = ""

            writer.writerow(row_out)

    return {
        "succeeded": n_succeeded,
        "errored": n_errored,
        "parse_failed": n_parse_failed,
    }


# ─── Orchestration ─────────────────────────────────────────────────────

def run(input_path: Path, output_path: Path) -> None:
    # Read input
    with open(input_path) as f:
        input_rows = list(csv.DictReader(f))
    print(f"Read {len(input_rows)} rows from {input_path}")

    client = anthropic.Anthropic()
    state = load_state(output_path)

    if state is None:
        # Phase 1: submit
        print("\n=== Phase 1: submitting batch ===")
        requests, id_to_row_idx = build_batch_requests(input_rows)
        print(f"Built {len(requests)} valid requests "
              f"(skipped {len(input_rows) - len(requests)} blank rows)")

        if not requests:
            print("No valid requests to submit. Exiting.")
            return

        batch_id = submit_batch(client, requests)
        state = {
            "batch_id": batch_id,
            "submitted_at": time.time(),
            "input_path": str(input_path),
            "n_requests": len(requests),
            "id_to_row_idx": id_to_row_idx,
            "prompt_version": PROMPT_VERSION,
        }
        save_state(output_path, state)
        print(f"State saved to {state_path(output_path)}")
    else:
        print(f"Resuming from state file: batch_id={state['batch_id']}")
        # Convert keys back to strings (JSON loads them as strings already
        # but row_idx values are ints in our dict)
        state["id_to_row_idx"] = {k: int(v) for k, v in state["id_to_row_idx"].items()}

    # Phase 2: poll
    print("\n=== Phase 2: polling for completion ===")
    poll_until_done(client, state["batch_id"])

    # Phase 3: collect
    print("\n=== Phase 3: collecting results ===")
    summary = collect_results(
        client, state["batch_id"], input_rows,
        state["id_to_row_idx"], output_path
    )

    print(f"\n{'=' * 60}")
    print(f"Batch processing complete")
    print(f"{'=' * 60}")
    print(f"  Input rows:     {len(input_rows)}")
    print(f"  Succeeded:      {summary['succeeded']}")
    print(f"  API errors:     {summary['errored']}")
    print(f"  Parse failures: {summary['parse_failed']}")
    print(f"  Output:         {output_path}")
    print(f"  State file:     {state_path(output_path)}")
    print()
    print(f"If there were parse failures, the rows are in the output CSV with")
    print(f"judge_status='parse_failed'. Re-run those via llm_judge.py against")
    print(f"a filtered input CSV to mop them up sequentially.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="CSV with 'seed' and 'variant' columns")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output CSV path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input, args.output)