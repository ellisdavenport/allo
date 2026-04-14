"""
Smoke test for allo/evaluate.py

Run with:
    python tests/test_evaluate.py

Expected output: a scored, deduplicated list of variants
with per-variant semantic similarity and perplexity scores, plus
batch-level summary stats.
"""

from allo.generate import LLMClient, generate_variants
from allo.evaluate import score_variants
from allo.output import write_csv

client = LLMClient()
seed = "remind me to call the doctor tomorrow morning"

print(f"\nSeed: '{seed}'")
print("─" * 60)

# Generate variants using all four strategies
variants = generate_variants(seed, n_per_strategy=3, client=client, temperature=0.9)

# Score and filter
scored, summary = score_variants(seed, variants)

# Display results
print("\nScored variants:")
print("─" * 60)
for v in scored:
    print(f"  [{v['strategy']}] sim={v['semantic_similarity']} | ppl={v['perplexity']}")
    print(f"    \"{v['utterance']}\"")

print("\nBatch summary:")
print("─" * 60)
for k, v in summary.items():
    print(f"  {k}: {v}")

# Write to CSV
filepath = write_csv(scored, summary)
print(f"\nOpen the file to inspect: {filepath}")