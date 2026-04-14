"""
output.py writes scored variants and batch summary to CSV

Responsibility: take the scored variant list and summary dict produced
by evaluate.py and write them to a clean, well-structured CSV file.

CSV structure:
    - One row per variant with seed, utterance, strategy, and scores
    - A blank separator row followed by a summary section at the bottom
      containing batch-level stats (total variants, mean scores, TTR)

The output file is self-contained. Every row includes the seed so the
file is interpretable without any external context.
"""

import csv
import os
from datetime import datetime


def write_csv(
    scored_variants: list[dict],
    summary: dict,
    output_dir: str = "output",
    filename: str = None,
) -> str:
    """
    Write scored variants and batch summary to a CSV file.

    Parameters:
        scored_variants: list of dicts from evaluate.py, each containing
                         'utterance', 'strategy', 'semantic_similarity', 'perplexity'
        summary: batch-level stats dict from evaluate.py
        output_dir: directory to write the file into (created if it doesn't exist)
        filename: optional filename override. If not provided, a timestamped
                  filename is generated automatically.

    Returns:
        The full path to the written CSV file as a string.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate a timestamped filename if none provided
    # Format: allo_YYYYMMDD_HHMMSS.csv
    # This ensures repeated runs don't overwrite previous output,
    # which matters when comparing runs at different temperatures or strategies.
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"allo_{timestamp}.csv"

    filepath = os.path.join(output_dir, filename)

    # Define the column order for the variant rows
    fieldnames = [
        "seed",
        "utterance",
        "strategy",
        "semantic_similarity",
        "perplexity",
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Header row
        writer.writeheader()

        # One row per scored variant
        # extrasaction="ignore" means any extra keys in the dict
        # (e.g. if generate.py adds fields later) are silently ignored
        # rather than raising an error.
        for variant in scored_variants:
            writer.writerow({
                "seed": summary["seed"],
                "utterance": variant["utterance"],
                "strategy": variant["strategy"],
                "semantic_similarity": variant["semantic_similarity"],
                "perplexity": variant["perplexity"],
            })

        # Blank separator row for readability
        writer.writerow({field: "" for field in fieldnames})

        # Summary section — written as key/value pairs in the first two columns
        # This keeps the summary in the same file without requiring a separate output.
        summary_rows = [
            ("# BATCH SUMMARY", ""),
            ("total_variants", summary["total_variants"]),
            ("mean_semantic_similarity", summary["mean_semantic_similarity"]),
            ("mean_perplexity", summary["mean_perplexity"]),
            ("lexical_diversity_ttr", summary["lexical_diversity_ttr"]),
        ]

        summary_writer = csv.writer(f)
        for row in summary_rows:
            summary_writer.writerow(row)

    print(f"\nOutput written to: {filepath}")
    return filepath