"""
Smoke test for allo/generate.py

Run with:
    python test_generate.py

Expected output: a list of generated variants grouped by strategy,
each prefixed with its strategy name. Back-translation requires a
valid DEEPL_API_KEY in your .env file; if absent those variants
will be skipped with a warning rather than crashing.
"""

from allo.generate import LLMClient, generate_variants

client = LLMClient()

seed = "remind me to call the doctor tomorrow morning"

print(f"\nSeed utterance: '{seed}'")
print(f"Provider: {client.provider} | Model: {client.model}\n")
print("─" * 60)

# Test at default temperature
print("\n[DEFAULT TEMPERATURE: 0.9]")
results = generate_variants(seed, n_per_strategy=3, client=client, temperature=0.9)
for r in results:
    print(f"  [{r['strategy']}] {r['utterance']}")

# Test at lower temperature to observe more conservative output
print("\n[LOW TEMPERATURE: 0.3]")
results_low = generate_variants(seed, n_per_strategy=3, client=client, temperature=0.3)
for r in results_low:
    print(f"  [{r['strategy']}] {r['utterance']}")

print("\n" + "─" * 60)
print("Smoke test complete.")