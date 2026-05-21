# LLM-as-judge evaluation

A study using a large language model as a structured evaluator for the
quality of allo's generated variants. The judge provides per-variant
semantic-equivalence and naturalness scores on a 1-5 Likert scale, grounded
in an anchored rubric and validated against a held-out calibration set.

The study has two arms. This document covers the main arm: rubric design,
calibration, and aggregate scoring across 16,962 variants from the four
generation strategies. The prompt-engineering arm — targeted revisions to
four constrained-rewriting transforms based on judge-surfaced failure modes
— is documented in [`prompt_revisions.md`](./prompt_revisions.md).

**Headline findings:**

1. **Judge and surface metrics measure largely different things.** Spearman
   correlation between semantic similarity and judge equivalence is +0.188
   overall; between log perplexity and judge naturalness, −0.120. Surface
   metrics surface what they measure (vector closeness, GPT-2 unsurprisingness)
   and the judge surfaces what it was asked to measure (intent equivalence,
   standalone fluency). These targets only partially overlap.

2. **MLM substitution has a substantial equivalence problem invisible to
   surface metrics.** MLM variants have a median semantic similarity of 0.87
   (high) and a median judge equivalence of 2 (low). The surface metric is
   measuring vector closeness of single-word substitutions; the judge is
   measuring whether the substituted variant means the same thing. 1,840
   MLM variants have similarity >0.85 and judge equivalence ≤2.

3. **The four generation strategies show distinct quality profiles.**
   `llm_paraphrase` is the strongest performer (88.8% clean rate, both
   dimensions ≥4). `constrained` is solid with per-transform variation
   (see prompt_revisions.md). `mlm_substitution` and `expansion` have
   characteristic dimension-asymmetries: MLM has high naturalness but
   poor equivalence; expansion has poor equivalence by design but the
   highest naturalness of any strategy (99.5% rated ≥4).

## Background

allo scores every generated variant on two surface metrics: semantic
similarity (cosine distance between sentence embeddings via
`all-MiniLM-L6-v2`) and perplexity (GPT-2). The scoring ranges study
(`scoring_ranges.md`) characterized these metrics across ~17K variants
on the 117-seed test set and identified three blind spots that limit
their reliability as quality signals:

1. **Named entity sensitivity.** Sentence embeddings treat numeric tokens,
   times, and named entities as semantically equivalent when they should
   not be. "Set a timer for 8am" and "Set a timer for 9am" have similarity
   ~0.93 by `all-MiniLM-L6-v2`, but they refer to different times.

2. **Synonymy gaps.** Embeddings sometimes score true synonyms low — "kill
   the lights" and "turn off the lights" have similarity ~0.62 despite
   being communicatively equivalent.

3. **Compound perplexity pathology on short disfluent seeds.** GPT-2
   assigns very high perplexity to short utterances with filler words
   ("um can you remind me to") even when they are perfectly fluent
   conversational speech.

These blind spots motivated adding a third evaluation signal that scores
variant quality on different criteria from the surface metrics. An LLM-as-judge
approach is natural for this: the judge can be asked to read the seed and
variant together, evaluate intent equivalence holistically, and reason about
naturalness in a way that doesn't reduce to perplexity. The cost is the
judge's own biases and inconsistencies, which the calibration step addresses.

## Method

### Judge model and prompt

The judge is `claude-haiku-4-5` invoked via the Anthropic batch API at
temperature 0 with prompt caching. Batch mode provides a 50% cost discount
relative to the synchronous API; prompt caching reduces cost further by
amortizing the rubric and worked examples across the 16,962 calls.

The judge prompt is locked at `PROMPT_VERSION="v1.0-locked"` and stored in
`evaluation/studies/judge_prompt.py`. The prompt version is written into
every scored row's `judge_prompt_version` column so downstream analyses can
match scores to the rubric they were produced under. Any change to the
prompt bumps the version and invalidates prior aggregates.

### Two dimensions

The judge scores on two dimensions:

- **Semantic equivalence:** does the variant express the same intent as
  the seed? Scored 1-5 with anchored rubric.
- **Naturalness:** would a native English speaker say this? Scored 1-5
  with anchored rubric.

A third dimension, **intent preservation**, was considered during rubric
design and dropped. Intent preservation overlapped substantially with
semantic equivalence — judge variance within the calibration set was
near-zero between the two, and inspection showed they were measuring the
same underlying construct under different labels. Two dimensions are simpler
to interpret and don't lose signal.

Naturalness is judged **as a standalone utterance**, not in the context of
the seed. This is a deliberate choice: many natural conversational variants
sound stilted when read alongside the seed they were generated from (because
the reader expects a paraphrase, not a stand-alone), and many surface-natural
variants are nonsense when the seed context is added. Standalone framing
removes the context bias and makes the metric stable.

### Anchored rubric with worked examples

Each dimension has a 1-5 anchored rubric with worked examples. The full
rubric is in [`rubric.md`](./rubric.md); briefly:

- **Equivalence 5:** Same intent, no information added or lost.
- **Equivalence 4:** Same intent with minor surface variation or one
  acceptable additional context word.
- **Equivalence 3:** Mostly same intent with some addition/omission.
- **Equivalence 2:** Different intent with overlapping topic.
- **Equivalence 1:** Different intent or opposite meaning.

- **Naturalness 5:** Sounds completely natural; could appear in real speech.
- **Naturalness 4:** Mostly natural with minor awkwardness.
- **Naturalness 3:** Comprehensible but stilted or unusual phrasing.
- **Naturalness 2:** Grammatically odd or clearly machine-generated.
- **Naturalness 1:** Ungrammatical or nonsensical.

The prompt includes three worked examples covering high-, medium-, and
low-quality variants. These examples ground the judge's understanding of
the rubric anchors. The cost is that the judge's behavior on cases close
to the anchors may be biased toward the anchor verdict — a limitation
documented below.

### Cost and scale

Total cost for the aggregate run was $14 across 16,962 variants. Mean cost
per call was approximately $0.0008 using batch API and prompt caching.
This is significantly cheaper than running the same judging synchronously
without caching (~$56 estimated) and made full-aggregate scoring affordable.

## Calibration (Day 1)

Before running the judge on the full aggregate, a 50-variant calibration
set was scored both by Ellis (predicted Likert) and by the judge, with
disagreements analyzed before locking the prompt.

### Calibration set construction

50 variants drawn from a pilot generation run, stratified across:

- All four generation strategies (~12 per strategy)
- The three identified surface-metric blind spots (16 variants targeting
  named-entity sensitivity, synonymy gaps, and short-disfluent-seed
  perplexity pathology)
- A range of expected quality levels (clean successes, clear failures,
  ambiguous middle)

The blind-spot cases are the most important subset: they test whether the
judge correctly disambiguates the surface-metric failure modes the study
exists to address.

### Calibration results

Judge-predictor agreement:

| Dimension | Exact match | Within ±1 |
|:---|---:|---:|
| Equivalence | 56% | 88% |
| Naturalness | 64% | 92% |

All 16 blind-spot cases were judged correctly. Specifically:

- Named-entity cases: the judge correctly distinguished variants with
  changed times, dates, or proper nouns even when surface similarity was
  high.
- Synonymy gap cases: the judge correctly rated "kill the lights" and
  "turn off the lights" as semantically equivalent despite low surface
  similarity.
- Compound-perplexity cases: the judge correctly rated short disfluent
  seeds as natural despite GPT-2's high perplexity.

The within-±1 agreement rates and 16/16 blind-spot accuracy were treated
as a green light to proceed with aggregate scoring.

### What calibration validates and what it doesn't

Two important honesty notes about what this calibration tells us:

**Single-annotator predictions.** All 50 predicted Likert scores were
produced by one annotator (Ellis). What calibration validates is
*predictor-judge alignment*, not *absolute correctness*. If the predictor's
intuitions are systematically biased, the judge can closely match the
predictor and both can be wrong together. The blind-spot cases provide
some protection against this (they target known failure modes with
unambiguous ground truth) but the bulk of the 50-variant set does not.

A meaningful next step is inter-annotator agreement on a 50-100 variant
subset with multiple annotators, computing Cohen's κ or Krippendorff's α
between annotators and between annotators and the judge. This is named in
the future work section.

**Anchor leakage.** The prompt contains three worked examples for grounding.
Row 1 of the calibration set was a near-verbatim variant of the
high-equivalence worked example, and the judge scored it identically. This
confirms anchor leakage: variants close to the worked examples get scored
toward the worked-example verdict. The aggregate scoring is unlikely to
include many variants this close to the anchors, but variants in the
broader neighborhood of an anchor may be biased toward it.

A future revision could replace the worked examples with synthetic seeds
that don't appear in the actual evaluation distribution, eliminating
leakage entirely. This is also named in future work.

## Aggregate scoring (Day 2)

The full Day 2 run scored 16,962 variants from the scoring-ranges aggregate
at `--n=50`. 16,959 were successfully scored (99.98% success rate). The
three failures break down as:

- 1 `api_error` (transient network failure)
- 1 `parse_failed` (judge returned malformed JSON)
- 1 `expired` (batch timeout on a single variant)

Total cost: $14. Wall-clock time (including batch wait): roughly 4 hours.

### Per-strategy headline distributions

| Strategy | n | Mean eq | %eq≤2 | %eq≥4 | Mean nat | %nat≤2 | %nat≥4 | Clean rate |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| llm_paraphrase | 5,154 | 4.38 | 1.5% | 90.8% | 4.67 | 0.3% | 97.1% | **88.8%** |
| constrained | 5,143 | 3.81 | 12.3% | 69.1% | 3.96 | 9.4% | 71.5% | 54.9% |
| mlm_substitution | 5,274 | 2.29 | **75.0%** | 14.9% | 3.34 | 35.9% | 51.7% | 8.8% |
| expansion | 1,388 | 2.70 | 43.5% | 14.4% | **4.83** | 0.1% | **99.5%** | 14.4% |

Clean rate = % of variants rated ≥4 on both dimensions.

The four strategies are striking in how differently they fail:

**`llm_paraphrase`** is the unambiguous winner. 88.8% of variants are clean
on both dimensions, and only 1.5% have equivalence failures. This is
expected — paraphrasing is the strategy most directly aligned with the
judge's equivalence-and-naturalness target. The strategy works.

**`constrained`** is solid baseline quality with substantial per-transform
variation. Mean equivalence 3.81, clean rate 54.9%. The per-transform
breakdown reveals four transforms (negative_framing, modal_should,
passive_voice, time_framed) carrying disproportionate failure mass. These
were addressed by the v1.1 prompt revisions documented in
`prompt_revisions.md`. The eleven other transforms range from
`indirect_need` (mean equiv 4.46) to `progressive_desire` (mean equiv 3.50).

**`mlm_substitution`** has a structural equivalence problem. 75% of MLM
variants have equivalence ≤2. The pattern is consistent across seeds:
single-word lexical substitutions via DistilBERT's fill-mask predictions
produce variants that are *grammatically natural* (naturalness mean 3.34,
51.7% rated ≥4) but *semantically incorrect* relative to the seed.
Substituting a word in "turn off the lights" with one of DistilBERT's
top context-conditioned predictions often produces a different action
or different referent. The judge recognizes this; the surface similarity
metric does not (median sim 0.87 on MLM variants).

This finding is the clearest demonstration that the surface and judge
metrics measure different things, and the most action-relevant finding
for allo's design. See [Surface-metric correlation](#surface-metric-correlation)
below and [Implications](#implications-for-allo) for follow-up.

**`expansion`** is dimension-asymmetric. Equivalence is low by design
(43.5% rated ≤2): expansion is explicitly intended to produce
semantically-adjacent intents, not paraphrases. Naturalness is the
highest of any strategy (99.5% rated ≥4) because expansion variants
are LLM-generated fresh utterances rather than mechanical transformations
of the seed. This is the expected profile for an expansion strategy and
the existing documentation in `allo/generate.py` and the main README
correctly characterizes it. Practitioners using expansion-mode output
should filter on equivalence (or accept that expansion variants are not
paraphrases) before ingesting to training data.

## Surface-metric correlation

The most interesting analytic finding from the aggregate run is how
weakly the judge scores correlate with the surface metrics they
complement.

### Overall Spearman correlations (n=16,959)

- **Similarity vs judge equivalence:** +0.188
- **Log perplexity vs judge naturalness:** −0.120

Both correlations are in the expected direction but small in magnitude.
A Spearman of +0.188 means similarity and judge equivalence share a weak
monotonic relationship but explain very little of each other's variance.

### Per-strategy Spearman correlations

| Strategy | sim ↔ judge_equiv | log(ppl) ↔ judge_natural |
|:---|---:|---:|
| llm_paraphrase | +0.212 | −0.072 |
| constrained | +0.242 | −0.106 |
| mlm_substitution | +0.389 | −0.198 |
| expansion | +0.127 | −0.096 |

The per-strategy view reveals texture the overall number obscures:

- **MLM has the strongest sim↔equiv correlation (+0.389),** which is
  surprising at first glance but makes sense on reflection. Within the MLM
  variant set, similarity does meaningfully track equivalence — variants
  with very high similarity (above ~0.95) tend to be substitutions where
  the model produced the original word or a true synonym; variants with
  lower similarity are substitutions that changed meaning. The correlation
  is real within MLM. The reason similarity is misleading overall is that
  the MLM distribution is shifted high (median 0.87) and the equivalence
  distribution is shifted low (median 2), so most MLM variants live in the
  surface "looks good" region while being judged "doesn't match" by the
  judge.

- **Expansion has the weakest sim↔equiv correlation (+0.127),** which
  reflects that within expansion variants, similarity simply doesn't
  predict whether the judge will call a variant equivalent or not.
  Expansion variants vary in similarity but the judge rates most of them
  low on equivalence regardless.

### Disagreement regions

The most action-relevant view of judge-surface divergence is counting
variants where the two metrics disagree sharply.

**Variants with judge_equiv ≤ 2 AND similarity > 0.85:** 2,178
| Strategy | n |
|:---|---:|
| mlm_substitution | 1,840 |
| constrained | 288 |
| expansion | 27 |
| llm_paraphrase | 23 |

The MLM dominance here (84.5% of all such disagreements) is the
quantitative form of the structural equivalence problem described above.
1,840 MLM variants pass the surface similarity threshold a practitioner
might apply (similarity > 0.85) but fail the judge's equivalence check.
If allo's similarity score is used as the gate for training-data
ingestion, those 1,840 variants are false positives.

**Variants with judge_natural ≤ 2 AND perplexity < 200:** 1,291
| Strategy | n |
|:---|---:|
| mlm_substitution | 946 |
| constrained | 334 |
| llm_paraphrase | 11 |

The MLM dominance here (73%) reflects that low-perplexity MLM variants
can be locally fluent in the bigram/trigram sense GPT-2 captures while
being globally awkward — substituting a word may produce a sentence
that's locally probable but semantically forced.

## The prompt-engineering arm

The constrained-rewriting per-transform diagnostic (Day 4) identified four
transforms with single linguistically diagnosable failure mechanisms.
Revised v1.1 prompts addressed each, validated on the same locked judge
in Day 5. Full results in [`prompt_revisions.md`](./prompt_revisions.md).
Summary: all four revisions improved their targeted dimensions, none
degraded their non-targeted dimensions, and the v1.1 prompts are now
committed in `allo/generate.py` at `CONSTRAINT_VERSION = "v1.1"`.

The prompt-engineering arm demonstrates that the judge is useful for more
than aggregate measurement: it surfaces specific actionable failure modes
that drive targeted improvements to allo's generation strategies.

## Implications for allo

### Three known surface-metric limitations are confirmed

The named-entity sensitivity, synonymy-gap, and compound-perplexity-pathology
limitations identified in `scoring_ranges.md` are confirmed by the judge
data on the full aggregate. These are documented in `allo/README.md` under
"Known metric limitations" so practitioners are aware before applying
similarity- or perplexity-based filtering.

### MLM strategy-level redesign deferred to future work

The judge data shows MLM substitution has a structural equivalence problem
beyond the scope of prompt engineering. The strategy generates lexically
diverse variants that are largely not equivalent to the seed. Three paths
forward are conceivable:

1. **Restrict MLM to content words with strong contextual constraints.**
   The current implementation operates over every content word position.
   Limiting it to positions where the surrounding context strongly
   determines the lexical choice (e.g., where DistilBERT's top prediction
   has high confidence and the top 5 predictions are close synonyms)
   might filter out the bulk of the failure cases.

2. **Post-hoc equivalence filtering using the judge.** Score every MLM
   variant with the judge and filter on equivalence ≥3. This adds cost
   but would salvage the strategy as a generator of high-confidence
   substitutions.

3. **Replace MLM with a different lexical-variation mechanism.** E.g.,
   a constrained LLM call that produces single-word substitutions with
   explicit synonym constraints. This loses the "no-API-needed" property
   that motivated MLM in the first place.

None of these are pursued in this study. The decision rationale is that
prompt-engineering against MLM doesn't address the underlying mechanism;
the strategy itself needs redesign, which is a larger scope than this
study's close-out.

### Judge as a third metric behind a flag

A natural future addition is `--use-judge` on `allo` CLI and a corresponding
toggle in the Streamlit UI that scores variants with the judge in addition
to similarity and perplexity. This would be off by default (judge calls
cost money and add latency) and on when the practitioner explicitly opts
in. Implementation is straightforward — the judge runner already exists
at `evaluation/studies/llm_judge_batch.py`. This is named in Day 7.

### Constrained rewriting v1.1 ships

Already committed. See `prompt_revisions.md`.

## Limitations of this study

Listed in priority order — the most important come first.

**Single-annotator calibration.** All 50 calibration predictions were
produced by one annotator (Ellis). The calibration validates predictor-judge
alignment, not absolute correctness. The blind-spot cases provide some
protection against systematic predictor bias but most of the calibration
set does not.

**Anchor leakage from worked examples.** The judge prompt contains three
worked examples. Variants close to these examples may be biased toward
the worked-example verdict. The aggregate scoring is unlikely to contain
many variants this close, but variants in the broader anchor neighborhood
may be biased.

**No inter-judge agreement.** The aggregate scoring used a single judge
model (`claude-haiku-4-5`). Running the same calibration set through
multiple judge models (different sizes or providers) would surface judge
biases that single-model scoring cannot detect.

**No human gold standard on the aggregate.** The 16,959 aggregate scores
have no human ground truth. Spot-checking suggests the judge tracks well
with manual inspection, but this is not systematic.

**Naturalness as standalone framing tradeoff.** Judging naturalness on
the variant in isolation (rather than alongside the seed) is a deliberate
design choice that has tradeoffs. Variants whose naturalness depends on
context — e.g., elliptical responses that only make sense after the seed
— may be scored lower than they would be in a contextual rubric. This
mostly affects expansion variants.

**Single seed set.** All scoring was done on the 117-seed test set from
`seed_set_methodology.md`. The findings are specific to this distribution
of intent types, syntactic structures, and domains. Generalization to
different seed distributions (e.g., longer utterances, different domains)
is not validated.

## Future work

In rough order of value relative to effort:

1. **Human inter-annotator agreement on a 50-100 variant subset.** Recruit
   2-3 linguistically-trained annotators, score a subset of the aggregate,
   compute Krippendorff's α between annotators and between annotators and
   judge. This is the single highest-value follow-up: it would
   substantially strengthen the calibration claim and provide a real
   ground truth for evaluating the judge's biases.

2. **MLM strategy-level redesign.** Choose one of the three paths from
   the implications section and prototype it. The post-hoc judge filtering
   option is the cheapest to test; the restrict-by-confidence option is
   the most architecturally aligned with allo's existing design.

3. **Inter-judge agreement.** Run the calibration set through one or two
   additional judge models (e.g., a smaller Anthropic model and a non-Anthropic
   model). Differences between judges indicate which findings are
   judge-specific vs. robust across judges.

4. **Synthetic worked examples in the rubric.** Replace the current worked
   examples with synthetic seeds that don't appear in any evaluation
   distribution. Eliminates anchor leakage. Requires re-running the
   calibration set under the new prompt and verifying agreement is maintained.

5. **Judge as a third metric in `evaluate.py`.** `--use-judge` flag, off
   by default, that adds judge scoring to the CSV output. Documentation
   updated to describe when judge scoring is worth the cost.

6. **Cross-lingual judge calibration.** A natural next study, separate
   from allo but reusing the rubric infrastructure: evaluate whether the
   judge's calibration holds for variants generated in Spanish, Portuguese,
   or Italian. This is a placeholder for a separate project.

## Reproducibility

**Code:**
- Judge runner: `evaluation/studies/llm_judge_batch.py`
- Locked judge prompt: `evaluation/studies/judge_prompt.py` (`PROMPT_VERSION="v1.0-locked"`)
- Rubric: `evaluation/results/llm_judge/rubric.md`

**Data:**
- Aggregate input: `evaluation/results/scoring_ranges/scoring_ranges_aggregate.csv`
- Aggregate output: `evaluation/runs/llm_judge_aggregate_v1.csv` (16,962 rows, 16,959 scored)
- Calibration: `evaluation/runs/judge_calibration_audited.csv`

**To regenerate aggregate scoring:**

```bash
python -m evaluation.studies.llm_judge_batch \
    --input evaluation/results/scoring_ranges/scoring_ranges_aggregate.csv \
    --output evaluation/runs/llm_judge_aggregate_v1.csv
```

Cost: ~$14 batch API + caching. Wall-clock: ~4 hours including batch wait.

**Configuration:**
- Judge model: `claude-haiku-4-5`
- Temperature: 0
- Batch API: enabled
- Prompt caching: enabled

## See also

- [`prompt_revisions.md`](./prompt_revisions.md) — the prompt-engineering arm
- [`rubric.md`](./rubric.md) — full anchored rubric with worked examples
- [`scoring_ranges.md`](../scoring_ranges/scoring_ranges.md) — surface-metric
  characterization that motivated this study
- [`seed_set_methodology.md`](../../seed_set_methodology.md) — the 117-seed
  test set used throughout
- [`volume_sweep.md`](../volume_sweep/volume_sweep.md) — earlier scaling study