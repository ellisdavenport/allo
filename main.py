"""
main.py — command-line entry point for allo

Usage:
    python main.py --seed "turn off the lights" --n 10

    python main.py \\
        --seed "turn off the lights" \\
        --n 50 \\
        --temperature 0.9

    python main.py \\
        --seed "turn off the lights" \\
        --n 50 \\
        --no-expansion \\
        --filter-min-similarity 0.70 \\
        --filter-max-perplexity 300

Arguments:
    --seed                  The input utterance to generate variants from (required)
    --n                     Variants per strategy (default: 10). Controls volume:
                            LLM and expansion scale linearly; constrained rewriting
                            is capped at 3 per transform for quality; MLM is bounded
                            by utterance length and plateaus around 30.
    --temperature           LLM generation temperature, 0.0–1.0 (default: 0.9).
                            Applied to LLM paraphrasing and semantic expansion only.
                            Constrained rewriting always uses 0.4 for compliance.
    --output-dir            Directory to write CSV output (default: output)
    --no-expansion          Disable semantic expansion. Use when you want strictly
                            synonymous paraphrases rather than adjacent-intent variants.
    --filter-min-similarity Optional: discard variants below this similarity score.
    --filter-max-perplexity Optional: discard variants above this perplexity score.

Expected output volumes at common --n values (before dedup and filtering):
    n=10:  ~10 llm + ~15-30 constrained + ~15 mlm + ~3 expansion  = ~40-60
    n=50:  ~50 llm + ~45 constrained    + ~30 mlm + ~13 expansion = ~140
    n=100: ~100 llm + ~75 constrained   + ~30 mlm + ~25 expansion = ~230
"""

import argparse
import sys

from allo.generate import LLMClient, generate_variants
from allo.evaluate import score_variants
from allo.output import write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="allo",
        description=(
            "Generate, score, and export natural language utterance variants "
            "from a seed utterance."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--seed",
        type=str,
        required=True,
        help="The input utterance to generate variants from.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Variants per strategy (default: 10). See module docstring for volume estimates.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help=(
            "LLM generation temperature between 0.0 and 1.0 (default: 0.9). "
            "Applied to paraphrasing and expansion. Constrained rewriting "
            "always uses 0.4 regardless of this setting."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to write CSV output into (default: output).",
    )
    parser.add_argument(
        "--no-expansion",
        action="store_true",
        default=False,
        help=(
            "Disable semantic expansion. Use when you want strictly synonymous "
            "paraphrases rather than adjacent-intent variants in the same domain."
        ),
    )
    parser.add_argument(
        "--filter-min-similarity",
        type=float,
        default=None,
        help=(
            "Optional. Discard variants below this semantic similarity score. "
        ),
    )
    parser.add_argument(
        "--filter-max-perplexity",
        type=float,
        default=None,
        help="Optional. Discard variants above this perplexity score.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.seed.strip():
        print("Error: --seed cannot be empty.", file=sys.stderr)
        sys.exit(1)

    if args.n < 1:
        print("Error: --n must be at least 1.", file=sys.stderr)
        sys.exit(1)

    if not 0.0 <= args.temperature <= 1.0:
        print("Error: --temperature must be between 0.0 and 1.0.", file=sys.stderr)
        sys.exit(1)

    include_expansion = not args.no_expansion

    print(f"\nSeed: '{args.seed}'")
    print(f"Variants per strategy: {args.n} | Temperature: {args.temperature}")
    print(f"Semantic expansion: {'enabled' if include_expansion else 'disabled'}")
    print("─" * 60)

    client = LLMClient()
    print(f"Provider: {client.provider} | Model: {client.model}\n")

    variants = generate_variants(
        seed=args.seed,
        n_per_strategy=args.n,
        client=client,
        temperature=args.temperature,
        include_expansion=include_expansion,
    )

    scored, summary = score_variants(
        seed=args.seed,
        variants=variants,
        filter_min_similarity=args.filter_min_similarity,
        filter_max_perplexity=args.filter_max_perplexity,
    )

    write_csv(
        scored_variants=scored,
        summary=summary,
        output_dir=args.output_dir,
    )

    print("\nScored variants:")
    print("─" * 60)
    for v in scored:
        print(f"  [{v['strategy']}] sim={v['semantic_similarity']} | ppl={v['perplexity']}")
        print(f"    \"{v['utterance']}\"")

    print("\nBatch summary:")
    print("─" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()