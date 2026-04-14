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

---

## Why four strategies?

Each strategy produces a structurally different kind of variation. The four are designed to complement rather than duplicate each other.

**LLM-based paraphrasing** uses a large language model (OpenAI or Anthropic, user choice) to generate natural paraphrases. This is the primary volume source. Temperature-based sampling produces diverse output at scale and handles idiomatic language well. Large requests are automatically batched into groups of 10 independent API calls rather than a single prompt, which maintains diversity by sampling fresh from the model's distribution each time.

**Constrained syntactic rewriting** applies explicit structural transforms to the seed via the LLM. Each of 15 named constraints is passed as a focused instruction at a fixed low temperature (0.4) to enforce structural compliance rather than producing generic paraphrases. The constraints are grouped across four linguistic axes:

*Illocutionary force* — polar question, polite imperative, indirect need

*Modality and conditionality* — hedged request, conditional, modal would, modal should

*Voice and framing* — passive voice, negative framing, progressive desire

*Pragmatic and contextual variation* — time framed, reason added, abbreviated spoken, emphatic, confirmation seeking

**Contextual MLM substitution** generates lexically diverse variants by masking each content word in the seed and collecting DistilBERT's top predictions for that position. Because the model conditions on the full surrounding context, it respects phrasal structure; *turn* in *turn off the lights* yields candidates like *switch*, *flip*, and *shut*, not *rotate* or *bend* as a dictionary lookup would.

**Semantic expansion** generates utterances that are related but meaningfully distinct, such as a different scope, degree, or action within the same domain as the seed. This is what enables scale to hundreds of variants: pure paraphrasing has a natural ceiling because there are only so many ways to say exactly the same thing. Expansion deliberately trades semantic fidelity for coverage breadth, producing neighbors like *dim the lights*, *turn off the bedroom light*, or *kill the lights* from a *turn off the lights* seed. These variants will score lower on semantic similarity than pure paraphrases because they are not paraphrases; they are neighbors. Review them separately before ingesting into training data.

---

## Evaluation metrics

Every generated variant is scored on two per-variant metrics and one batch-level metric:

**Semantic similarity** represents the cosine similarity between the seed and variant embeddings using `all-MiniLM-L6-v2`. The two texts are encoded as dense vectors and the cosine of the angle between them is taken as a similarity score, ranging from 0 to 1. Higher scores indicate closer meaning. Expansion variants will score lower than pure paraphrases by design.

**Perplexity** is a fluency proxy computed via GPT-2. The model scores how surprised it is by the variant sequence, where lower perplexity means the text reads more like natural language. Note that perplexity is sensitive to text length, so shorter utterances tend to score higher than longer ones even when equally fluent. Scores are best interpreted relative to other variants of the same seed rather than as absolute thresholds.

**Lexical diversity (TTR)** is a type-token ratio calculated across the full generated batch: unique word types divided by total word tokens. A score of 1.0 means every word in the batch appeared exactly once, whereas a score close to 0.0 means the batch is clustering around the same vocabulary. Because variants share core vocabulary with the seed by definition, TTR will naturally be moderate even for a well-diversified batch. Best to interpret it relative to other runs rather than against an external baseline.

No filtering is applied by default. Scores are presented as-is so the practitioner can apply whatever threshold makes sense for their use case. Optional hard filtering is available via CLI flags and UI controls.

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

Semantic similarity scores are not directly comparable across strategies because the strategies target different parts of the variation space:

| Strategy | What low semantic similarity scores mean |
|---|---|
| `llm` | Semantic drift |
| `constrained:*` | Structural transform worked as intended |
| `mlm` | Substitution changed meaning more than expected |
| `expansion` | Expected |

Perplexity is not strategy-dependent because fluency varies within every strategy. The exception worth noting is MLM substitution, where swapping a single word without adjusting surrounding syntax can produce grammatically odd results that score high perplexity even when the substituted word appears plausible in isolation.

If applying `--filter-min-similarity` with expansion enabled, set your threshold lower than you would for paraphrases only, otherwise you will discard the expansion variants that the strategy is explicitly designed to produce.

---

## Known limitations

**MLM substitution and short utterances.** MLM yield is bounded by utterance length. With only three maskable positions, and after filtering invalid candidates, even an optimal run may return a small number of variants regardless of --n. This is a structural ceiling, not a bug. Longer, more lexically rich seeds will produce more MLM variants.

**Constrained rewriting and degenerate seeds.** Some constraints interact poorly with certain seed types. Passive voice, for instance, is difficult to apply to a seed that is already maximally simple (*"lights off"*). The LLM will attempt compliance but may produce variants that are not genuine structural transforms. Semantic similarity and perplexity scores will surface these cases.

**LLM semantic drift.** At higher temperatures, LLM-generated paraphrases occasionally drift into semantically adjacent territory (e.g. *"dim the lights"* from a *"turn off the lights"* seed). This is not inherently wrong, but it is uncontrolled within the paraphrase strategy. As with other limitations mentioned here, semantic similarity scores will surface it.

**Expansion variant quality.** Semantic expansion is the least constrained strategy. The model is instructed to generate adjacent-intent variants rather than paraphrases, but the boundary is fuzzy and the LLM occasionally produces variants that are either too close (effectively paraphrases) or too far (unrelated utterances). The similarity score is your primary tool for identifying the latter, so make sure to review low-scoring expansion variants carefully before ingesting.

---

## Future enhancements

**Parallel LLM calls.** The constrained rewriting strategy makes one API call per constraint sequentially, at 15 calls per run. Parallelizing these with `concurrent.futures` would give a significant latency reduction at larger `--n` values.

**Constraint customisation.** The 15 syntactic constraints in `SYNTACTIC_CONSTRAINTS` are hardcoded in `generate.py`. Exposing these as a configurable list via a YAML file or UI editor would let practitioners add domain-specific transforms (e.g. always generate a voice-assistant wake-word prefix variant) without touching source code.

**Expansion clustering.** At large scale, expansion output benefits from post-hoc clustering to identify which adjacent intents are well-represented and which are sparse. Adding a clustering step to the summary would help practitioners identify coverage gaps in the generated batch before ingestion.

**Local LLM support.** The current `LLMClient` supports OpenAI and Anthropic only. Adding Ollama as a third provider would remove the API cost requirement entirely, and would be especially useful for practitioners working with sensitive data who cannot send utterances to an external API.

**Packaging as a console script.** Adding a `pyproject.toml` with a console script entry point would enable `allo` as a standalone terminal command rather than `python main.py`, and make the project installable as a proper Python package.

**Systematic testing.** Output volume per strategy and typical similarity ranges per strategy have not yet been benchmarked across a representative sample of seed types. Once completed, this testing will restore the scale table to the usage section and the score range guidance to the interpreting scores and known limitations sections.

---

## Project structure

```
allo/
├── allo/
│   ├── __init__.py
│   ├── generate.py     # generation strategies and LLM abstraction layer
│   ├── evaluate.py     # scoring: semantic similarity, perplexity, lexical diversity
│   └── output.py       # CSV writing
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