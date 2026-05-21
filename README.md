# allo

**allo** is a CLI and web interface for generating and scoring natural language utterance variants from a seed utterance. It was built originally for NLU training data augmentation and coverage repair, but it can apply to any task that requires semantically equivalent surface variation, such as robustness testing, query expansion, prompt sensitivity analysis, or paraphrase dataset construction.

The name derives from the prefix *allo-*, a linguistic term which denotes a variant realization of an underlying form. For example, an *allophone* is a variant of a phoneme and an *allomorph* is a variant of a morpheme.

---

## What it does

Given a seed utterance like:

```
turn off the lights
```

allo generates natural variants using four complementary strategies, scores each variant for semantic similarity and fluency, and exports the results to a scored CSV ready for human review.

The project includes a systematic evaluation framework with a stratified 117-seed test set, a volume sweep study, and a scoring ranges study. See [Evaluation](#evaluation) below.

---

## Why four strategies?

Each strategy produces a structurally different kind of variation. The four are designed to complement rather than duplicate each other.

**LLM-based paraphrasing** uses a large language model (OpenAI or Anthropic, user choice) to generate natural paraphrases. This is the primary volume source. Temperature-based sampling produces diverse output at scale and handles idiomatic language well. Large requests are automatically batched into groups of 10 independent API calls rather than a single prompt, which maintains diversity by sampling fresh from the model's distribution each time.

**Constrained syntactic rewriting** applies explicit structural transforms to the seed via the LLM. Each of 15 named constraints is passed as a focused instruction at a fixed low temperature (0.4) to enforce structural compliance rather than producing generic paraphrases. The constraints are grouped across four linguistic axes:

*Illocutionary force* — polar question, polite imperative, indirect need

*Modality and conditionality* — hedged request, conditional, modal would, modal should

*Voice and framing* — passive voice, negative framing, progressive desire

*Pragmatic and contextual variation* — time framed, reason added, abbreviated spoken, emphatic, confirmation seeking

Four of the fifteen transforms (`negative_framing`, `modal_should`, `passive_voice`, `time_framed`) were revised in v1.1 to add applicability gating and address linguistic failure modes identified by the LLM-as-judge diagnostic. See `prompt_revisions.md`.

**Contextual MLM substitution** generates lexically diverse variants by masking each content word in the seed and collecting DistilBERT's top predictions for that position. Because the model conditions on the full surrounding context, it respects phrasal structure; *turn* in *turn off the lights* yields candidates like *switch*, *flip*, and *shut*, not *rotate* or *bend* as a dictionary lookup would.

**Semantic expansion** generates utterances that are related but meaningfully distinct, such as a different scope, degree, or action within the same domain as the seed. Expansion deliberately trades semantic fidelity for coverage breadth, producing neighbors like *dim the lights*, *turn off the bedroom light*, or *close the blinds* from a *turn off the lights* seed. These variants will score lower on semantic similarity than pure paraphrases because they are not paraphrases. Review them separately before ingesting into training data.

---

## Evaluation metrics

Every generated variant is scored on two per-variant metrics and one batch-level metric:

**Semantic similarity** represents the cosine similarity between the seed and variant embeddings using `all-MiniLM-L6-v2`. The two texts are encoded as dense vectors and the cosine of the angle between them is taken as a similarity score, ranging from 0 to 1. Higher scores indicate closer meaning. Expansion variants will score lower than pure paraphrases by design.

**Perplexity** is a fluency proxy computed via GPT-2. The model scores how surprised it is by the variant sequence, where lower perplexity means the text reads more like natural language. Note that perplexity is sensitive to text length, so shorter utterances tend to score higher than longer ones even when equally fluent. Scores are best interpreted relative to other variants of the same seed rather than as absolute thresholds.

**Lexical diversity (TTR)** is a type-token ratio calculated across the full generated batch: unique word types divided by total word tokens. A score of 1.0 means every word in the batch appeared exactly once, whereas a score close to 0.0 means the batch is clustering around the same vocabulary. Because variants share core vocabulary with the seed by definition, TTR will naturally be moderate even for a well-diversified batch. Best to interpret it relative to other runs rather than against an external baseline.

No filtering is applied by default. Scores are presented as-is so the practitioner can apply whatever threshold makes sense for their use case. Optional hard filtering is available via CLI flags and UI controls.

### Metric limitations

The scoring metrics in allo have specific considerations that users should be aware of when setting filter thresholds:

- **Similarity drops sharply on named entity substitutions.** MLM variants that substitute one named entity for another (e.g. *what's the capital of australia* → *what's the capital of babylon*) score very low similarity because the referent has changed substantially. The metric is correctly reporting meaning change, but users running MLM on seeds with named entities should expect a low-similarity tail and threshold accordingly.
- **Similarity can undershoot on synonymy.** LLM paraphrases that preserve meaning through substantial vocabulary change (e.g. *play something relaxing* → *select a laid back track*) may score low similarity despite being high-quality paraphrases.
- **Perplexity can spike on short, disfluent, or repetitive seeds.** GPT-2 scores very short utterances (≤3 words) and utterances with filled pauses (uh, um) disproportionately high even when they're natural English. Users applying `--filter-max-perplexity` aggressively may remove valid variants from these seed types.

See `evaluation/results/scoring_ranges/scoring_ranges.md` for detailed examples and data.

---

## Applying similarity and perplexity filters

Filters are off by default. When enabled, `--filter-min-similarity` discards variants below a similarity score and `--filter-max-perplexity` discards variants above a perplexity score. Both are available as CLI flags and UI controls.

Before setting a threshold, it is worth knowing what it will actually remove. Based on a 117-seed study at `--n=50` (detailed in `evaluation/results/scoring_ranges/scoring_ranges.md`):

**Similarity thresholds by strategy — percent of variants discarded:**

| threshold | llm_paraphrase | constrained | mlm_substitution | expansion |
|:---|---:|---:|---:|---:|
| 0.55 | 4.5% | 1.1% | 5.0% | 28.9% |
| 0.65 | 11.5% | 3.6% | 11.2% | 51.2% |
| 0.70 | 17.0% | 6.9% | 16.2% | 63.7% |

A floor of 0.55 with expansion enabled sits between the expansion and paraphrase-class distributions, removing roughly a third of expansion output while preserving 95–99% of paraphrase-class variants. A floor of 0.70 without expansion discards approximately one in six paraphrase-class variants.

**Perplexity thresholds behave differently by seed type.** On clean, medium-to-long seeds, a ceiling of 200–300 removes genuinely awkward output without significant collateral loss. On short seeds (≤4 words) or seeds containing disfluencies such as filled pauses or repeated tokens, perplexity scores are unreliable, and valid variants routinely exceed 10,000 on these seed types. Applying a perplexity ceiling to a batch containing these seeds may remove valid output.

**The practical recommendation is to inspect before committing.** Run allo without filters first, sort the output CSV by similarity ascending and perplexity descending, review what falls at your intended threshold, then apply the filter.

---

## Installation

**Requirements:** Python 3.12, pip

```bash
git clone https://github.com/ellisdavenport/allo.git
cd allo
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your API keys:

```
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

One API key is required — either OpenAI or Anthropic. `gpt-4o-mini` is the recommended default: it follows structural constraints reliably, handles idiomatic language well, and is fast enough for large batches without significant latency. Set `LLM_PROVIDER=anthropic` and `LLM_MODEL=claude-sonnet-4-6` to use Anthropic instead.

On first run, three models will be downloaded and cached locally: `all-MiniLM-L6-v2` (semantic similarity), DistilBERT (MLM substitution), and GPT-2 (perplexity scoring). This takes a few minutes depending on your connection. Subsequent runs load from cache and start immediately.

---

## Usage

**Command line:**

```bash
python main.py --seed "turn off the lights" --n 10
```

**Larger batch:**

```bash
python main.py \
    --seed "turn off the lights" \
    --n 80 \
    --temperature 0.9
```

**Paraphrases only (no semantic expansion):**

```bash
python main.py \
    --seed "turn off the lights" \
    --n 50 \
    --no-expansion
```

**Full argument reference:**

```
--seed                    Input utterance to generate variants from (required)
--n                       Variants per strategy (default: 10)
--temperature             LLM temperature, 0.0–1.0 (default: 0.9). Applied to
                          paraphrasing and expansion only. Constrained rewriting
                          always uses 0.4 for structural compliance.
--output-dir              Output directory (default: output/)
--no-expansion            Disable semantic expansion. Use when you want strictly
                          synonymous paraphrases rather than adjacent-intent variants.
--filter-min-similarity   Discard variants below this similarity score (optional)
--filter-max-perplexity   Discard variants above this perplexity score (optional)
```

**Streamlit UI:**

```bash
streamlit run app.py
```

Opens a browser interface with sliders for all parameters, an expansion toggle, a strategy filter for the results table, and a download button for the scored CSV.

---

## Output format

Results are written to a timestamped CSV in the `output/` directory:

```
seed,utterance,strategy,semantic_similarity,perplexity
turn off the lights,Please switch off the lights.,llm,0.8304,45.24
turn off the lights,Could you turn the lights off?,constrained:polar_question,0.8612,38.91
turn off the lights,Switch off the lights.,mlm,0.8941,52.17
turn off the lights,Dim the lights.,expansion,0.6730,29.44
...

# BATCH SUMMARY
total_variants,47
mean_semantic_similarity,0.7821
mean_perplexity,61.38
lexical_diversity_ttr,0.4102
```

The `strategy` field identifies both the strategy and, for constrained rewrites, the specific structural transform applied (`constrained:polar_question`, `constrained:hedged_request`, etc.). This lets you filter by transform type in downstream processing. Timestamped filenames ensure repeated runs don't overwrite previous output.

---

## Interpreting scores across strategies

The four strategies target different parts of the variation space, so their similarity and perplexity distributions look different. The table below reports measured medians, IQRs, and p95 values across 16,962 variants from the scoring ranges study (full details in `evaluation/results/scoring_ranges/scoring_ranges.md`):

**Semantic similarity:**

| strategy | p25 | median | p75 | p95 |
|:---|---:|---:|---:|---:|
| constrained | 0.82 | 0.89 | 0.93 | 0.98 |
| mlm_substitution | 0.77 | 0.87 | 0.94 | 0.99 |
| llm_paraphrase | 0.74 | 0.83 | 0.89 | 0.94 |
| expansion | 0.52 | 0.64 | 0.75 | 0.85 |

The three paraphrase-class strategies (constrained, MLM, LLM) form a tight band with medians between 0.83 and 0.89. Expansion sits well below at 0.64, consistent with its design intent of producing adjacent-intent variants rather than paraphrases.

**Perplexity:**

| strategy | p25 | median | p75 | p95 |
|:---|---:|---:|---:|---:|
| constrained | 60 | 99 | 186 | 661 |
| llm_paraphrase | 68 | 123 | 254 | 911 |
| mlm_substitution | 89 | 153 | 312 | 1175 |
| expansion | 86 | 154 | 328 | 1188 |

All four medians fall within GPT-2's typical natural-English band. Constrained rewriting produces the most fluent-scoring output by median; MLM substitution and expansion produce the least.

What a low score means depends on the strategy:

| Strategy | What low semantic similarity means |
|:---|:---|
| `llm_paraphrase` | Semantic drift, or deep synonymy that the embedding undershoots |
| `constrained:*` | The structural transform changed surface form substantially |
| `mlm_substitution` | The substitution changed meaning more than expected, often via named entity swap |
| `expansion` | Expected — expansion variants are adjacent intents, not paraphrases |

Perplexity is not strategy-dependent because fluency varies within every strategy. The exception worth noting is MLM substitution, where swapping a single word without adjusting surrounding syntax can produce grammatically odd results that score high perplexity even when the substituted word appears plausible in isolation.

---

## Generation limitations

**MLM substitution and short utterances.** MLM yield is bounded by utterance length × valid candidates per position. The volume sweep study measured ceilings of approximately 48 variants for short seeds (1–4 words), 75 for medium seeds (5–9 words), and effectively unbounded within `--n ≤ 100` for long seeds (10+ words). This is a structural property of the utterance, not a bug. Longer, more lexically rich seeds will produce more MLM variants.

**MLM substitution and semantic drift.** The LLM-as-judge evaluation found that 75% of MLM variants have judge-rated equivalence ≤2 despite a median semantic similarity of 0.87. Single-word DistilBERT substitutions are grammatically natural but frequently change meaning. Practitioners using MLM output should not rely on semantic similarity alone as a quality gate; either inspect MLM variants manually before ingestion or filter on a third quality signal. See `evaluation/results/llm_judge/llm_judge.md` for the full analysis.

**Constrained rewriting and degenerate seeds.** Some constraints interact poorly with certain seed types. Two transforms in particular show meaningful compliance issues at high `--n`: `polar_question` (75% compliance, with failures concentrated on seeds that are already interrogative) and `abbreviated_spoken` (72% compliance, with failures concentrated on already-short seeds). Passive voice is also difficult to apply to seeds that are already maximally simple (*"lights off"*). Semantic similarity and perplexity scores will surface these cases.

**LLM semantic drift.** At higher temperatures, LLM-generated paraphrases occasionally drift into semantically adjacent territory (e.g. *"dim the lights"* from a *"turn off the lights"* seed). This is not inherently wrong, but it is uncontrolled within the paraphrase strategy. As with other limitations mentioned here, semantic similarity scores will surface it.

**Expansion variant quality.** Semantic expansion is the least constrained strategy. The model is instructed to generate adjacent-intent variants rather than paraphrases, but the boundary is fuzzy and the LLM occasionally produces variants that are either too close (effectively paraphrases) or too far (unrelated utterances). The similarity score is your primary tool for identifying the latter, so make sure to review low-scoring expansion variants carefully before ingesting.

**LLM paraphrase content ceiling on short seeds.** Asking for 100 paraphrases of a 3-word utterance like *"turn it up"* will return roughly 70, not 100. There are only so many ways to say exactly the same short thing. Yield approaches 100% on long seeds and drops to ~71% on short seeds at `--n=100`.

---

## Evaluation

allo includes a systematic evaluation framework. The methodology, runner scripts, analysis notebooks, and committed results all live in `evaluation/`.

**Seed set** — A stratified 117-seed test set spanning 12 domains, 7 syntactic types, 3 register levels, 12 lexical complexity axes, and 4 disfluency types. Seeds were selected to ensure deliberate representation across all dimensions of interest rather than drawn from a convenience sample. Schema, design principles, and known limitations are documented in `evaluation/seed_set_methodology.md`.

**Volume sweep study** (`evaluation/results/volume_sweep/volume_sweep.md`) — Per-strategy output volume measured across six values of `--n` (5, 10, 20, 50, 80, 100) on all 117 seeds, producing 2,808 `(seed, n, strategy)` observations. Findings: a length-gated mechanical ceiling for MLM substitution (~48 variants on short seeds, ~75 on medium, unbounded within `--n ≤ 100` on long); a content ceiling for LLM paraphrasing on short seeds (~71% efficiency at `--n=100`); structural compliance issues for the `polar_question` and `abbreviated_spoken` constrained transforms.

**Scoring ranges study** (`evaluation/results/scoring_ranges/scoring_ranges.md`) — Per-strategy similarity and perplexity distributions measured across 16,962 variants at `--n=50`. Findings: clean modal separation between paraphrase-class strategies and expansion in similarity space; length-sensitivity in both metrics that disproportionately affects short seeds when filters are applied flatly; three documented metric limitations (named entity sensitivity in similarity, synonymy gaps in similarity, perplexity pathology on short disfluent seeds).

**LLM-as-judge evaluation** (`evaluation/results/llm_judge/llm_judge.md`) — Claude Haiku 4.5 with a locked rubric scores all ~17K variants on semantic equivalence and naturalness. Quantifies surface-metric blind spots and identifies MLM substitution as having a structural equivalence problem invisible to similarity scores. The prompt-engineering arm (prompt_revisions.md) revised four constrained-rewriting transforms based on judge-surfaced failure modes.

Each study includes a runner script in `evaluation/studies/`, an analysis notebook that produces all figures and summary tables, and a markdown writeup committed under `evaluation/results/`.

---

## Future enhancements

**Parallel LLM calls.** The constrained rewriting strategy makes one API call per constraint sequentially, at 15 calls per run. Parallelizing these with `concurrent.futures` would give a significant latency reduction at larger `--n` values.

**Constraint customisation.** The 15 syntactic constraints in `SYNTACTIC_CONSTRAINTS` are hardcoded in `generate.py`. Exposing these as a configurable list via a YAML file or UI editor would let practitioners add domain-specific transforms (e.g. always generate a voice-assistant wake-word prefix variant) without touching source code.

**Expansion clustering.** At large scale, expansion output benefits from post-hoc clustering to identify which adjacent intents are well-represented and which are sparse. Adding a clustering step to the summary would help practitioners identify coverage gaps in the generated batch before ingestion.

**Local LLM support.** The current `LLMClient` supports OpenAI and Anthropic only. Adding Ollama as a third provider would remove the API cost requirement entirely, and would be especially useful for practitioners working with sensitive data who cannot send utterances to an external API.

**Packaging as a console script.** Adding a `pyproject.toml` with a console script entry point would enable `allo` as a standalone terminal command rather than `python main.py`, and make the project installable as a proper Python package.

---

## Project structure

```
allo/
├── allo/
│   ├── __init__.py
│   ├── generate.py     # generation strategies and LLM abstraction layer
│   ├── evaluate.py     # scoring: semantic similarity, perplexity, lexical diversity
│   └── output.py       # CSV writing
├── evaluation/
│   ├── allo_seed_set.csv           # 117-seed evaluation corpus
│   ├── seed_set_methodology.md     # seed set design and schema
│   ├── studies/                    # runner scripts and analysis notebooks
│   └── results/                    # committed figures, summaries, and study write-ups
├── tests/
│   ├── test_generate.py
│   └── test_evaluate.py
├── app.py              # Streamlit interface
├── main.py             # CLI entry point
├── requirements.txt
├── .env.example        # API key template
├── .env                # API keys (never committed)
└── .gitignore
```

---

## License

MIT