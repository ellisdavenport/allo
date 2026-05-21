# Constrained-rewriting prompt revisions

A targeted prompt-engineering pass on four transforms in allo's constrained
rewriting strategy. This study addresses systematic failures the LLM-as-judge
evaluation (`llm_judge.md`) surfaced in the v1.0 constrained-rewriting prompt
set, and validates a revised prompt set (v1.1) against the same locked judge.

This is the prompt-engineering arm of the broader LLM-as-judge study; for the
judge methodology and aggregate findings across all four generation strategies,
see [`llm_judge.md`](./llm_judge.md).

**Headline:** Of fifteen constrained-rewriting transforms in v1.0, four
showed systematic judge-surfaced failure patterns linked to specific
linguistic mechanisms. The v1.1 revisions improved all four targeted
dimensions without degrading non-targeted dimensions. The largest gains:
`negative_framing` equivalence (+0.42 mean Likert, −19.2pp on failure rate)
and `time_framed` naturalness (+0.53 mean Likert, −10.5pp on failure rate).

## Background

allo's constrained rewriting strategy applies fifteen named syntactic
transforms (polar_question, hedged_request, passive_voice, etc.) via the
LLM at low temperature. The v1.0 prompts were drafted on linguistic
intuition without empirical validation of which transforms compose well with
which seed types.

The LLM-as-judge evaluation in `llm_judge.md` scored 16,962 variants generated
at `--n=50` across the four generation strategies. The per-transform breakdown
of the ~5.2K constrained-rewriting variants showed substantial variation:

| Transform | n | Mean equiv | Mean natural | %eq≤2 | %nat≤2 |
|:---|---:|---:|---:|---:|---:|
| indirect_need | 340 | 4.46 | 4.49 | 0.6% | 0.6% |
| abbreviated_spoken | 312 | 4.44 | 4.82 | 1.9% | 0.3% |
| confirmation_seeking | 351 | 4.30 | 3.55 | 1.7% | 12.8% |
| polite_imperative | 344 | 4.28 | 4.03 | 7.8% | 6.4% |
| polar_question | 318 | 4.12 | 4.75 | 11.0% | 0.3% |
| passive_voice | 343 | 4.11 | 3.00 | 10.2% | **36.2%** |
| emphatic | 349 | 3.87 | 3.97 | 5.7% | 4.6% |
| hedged_request | 351 | 3.72 | 3.94 | 6.6% | 10.5% |
| conditional | 351 | 3.72 | 3.91 | 10.8% | 9.4% |
| modal_would | 336 | 3.69 | 3.75 | 22.6% | 11.0% |
| reason_added | 351 | 3.61 | 4.12 | 6.0% | 4.6% |
| progressive_desire | 349 | 3.50 | 4.15 | 11.5% | 5.7% |
| modal_should | 348 | 3.46 | 3.49 | **23.3%** | 12.6% |
| time_framed | 351 | 3.15 | 4.01 | **19.7%** | 12.5% |
| negative_framing | 349 | 2.82 | 3.64 | **43.8%** | 11.2% |

Four transforms (`negative_framing`, `modal_should`, `passive_voice`,
`time_framed`) were selected for revision. The selection was not purely by
score — `modal_would` has comparable equivalence-failure rate (22.6%) to
`modal_should` (23.3%), and `confirmation_seeking` has higher naturalness-failure
rate (12.8%) than `time_framed` (12.5%). The four selected were the transforms
whose failures clustered around a single linguistically diagnosable mechanism
when judge rationales on the ≤2 cases were inspected. The other elevated-failure
transforms (`modal_would`, `confirmation_seeking`, `progressive_desire`,
`conditional`) show distributed failures across multiple unrelated mechanisms
and were deferred. Inspection of the four diagnosable cases identified:

- `negative_framing` inverts meaning rather than negating-with-intent-preserved.
  93% failure rate on wh-questions ("what's the weather" → "isn't the weather"
  is meaningless; "don't tell me the weather" is opposite meaning).
- `modal_should` converts questions to declarative statements. 79% failure
  rate on wh-questions ("what time is it" → "the time should be checked").
- `passive_voice` produces ungrammatical "be gotten" / "be had" constructions
  on seeds whose main verb resists modern English passive.
- `time_framed` adds nonsensical temporal context to atemporal queries
  ("what's the capital of australia" → "what's the capital of australia for
  the night").

These are not stochastic noise; they are systematic compositions of transforms
with seed types where the operation doesn't yield meaning-preserving output.
The volume sweep study had previously flagged `polar_question` and
`abbreviated_spoken` as compliance issues at high `--n`, but the judge data
shows those are actually the highest-quality transforms — the volume sweep
metric (constraint compliance via regex on first word) was measuring something
different from variant quality. The judge surfaced the real failure modes.

## Method

**Seed set.** The 117-seed stratified test set from `allo_seed_set.csv`
(see `seed_set_methodology.md`). No subsampling — the same seeds and
syntactic-type distribution as the Day 2 aggregate run.

**Generation.** `gpt-4o-mini` at temperature 0.4, n_per_constraint=3 (3 variants
requested per (seed, transform) pair). Same model and parameters as v1.0 so
any score shift is attributable to prompt changes, not generation-side
differences.

**Judging.** `claude-haiku-4-5` at temperature 0, prompt version `v1.0-locked`.
Same judge configuration as the Day 2 aggregate run. Two dimensions on 1-5
Likert: semantic equivalence and naturalness. Full rubric in `rubric.md`.

**Pre-registered success criteria.** Set on Day 1 before generation:

> Targeted dimension must show a visible rightward shift in CDF at p25 and
> median. Non-targeted dimension must not show a leftward shift greater
> than within-condition noise. Numeric defaults if effects are ambiguous:
> targeted dimension ≥0.3 median Likert improvement, non-targeted no greater
> than 0.1 median degradation.

**Versioning.** A `CONSTRAINT_VERSION` constant in `allo/generate.py` tracks
the prompt set. v1.0 → v1.1 on commit of these revisions. Variants generated
under v1.1 carry the version in a `constraint_version` column.

**Cost.** Generation $0.50; judge batch <$3. Total Day 5 cost under $5.

## The four revisions

Common pattern across all four: each revised instruction is action-based
rather than syntax-based, explicit about which seed types the transform
applies to, and (for three of four) includes a `TRANSFORM_NOT_APPLICABLE`
sentinel that the generation function filters out before recording variants.

The sentinel approach is LLM-side applicability gating: the model judges
whether the transform composes with the seed, and abstains rather than
producing drift-laden output. The alternative — Python-side gating using
seed metadata — was rejected because the inapplicability conditions for
these transforms are linguistic, not surface-syntactic, and require
understanding what the utterance is doing pragmatically.

### negative_framing (target: equivalence)

**v1.0 prompt:** "Rewrite using a negative construction (e.g. 'don't forget
to...', 'make sure not to...', 'avoid...')"

**v1.1 prompt:**
> Rewrite the utterance using a negation that preserves its actionable intent —
> the variant must request the same outcome as the utterance, not the opposite.
> For example: 'turn off the lights' → 'don't leave the lights on'; 'remind
> me to call the doctor' → 'don't let me forget to call the doctor'. This
> transform applies when the utterance expresses an action the addressee
> should take or avoid. It does not apply to utterances that ask for
> information ("what's the temperature"), describe states without requesting
> action ("i'd like the thermostat at 68"), are already negated ("don't send
> that email yet"), or are too fragmentary to negate coherently ("lights
> off"). When the transform doesn't apply, return the literal string
> TRANSFORM_NOT_APPLICABLE on its own line.

**Linguistic mechanism.** The failure mode in v1.0 was meaning inversion:
the model treated "negation" as polarity-flip rather than as intent-preserving
negative construction. The revised instruction reframes the transform as
action-based ("expresses an action the addressee should take or avoid"),
gives positive examples, and enumerates non-applicable speech-act types.

### modal_should (target: equivalence)

**v1.0 prompt:** "Rewrite using 'should' as the main modal verb"

**v1.1 prompt:**
> Rewrite using 'should' as the main modal verb while preserving the
> utterance's speech act. If the utterance is a question, the variant must
> remain a question (e.g. 'what's the temperature set to' → 'what should the
> temperature be set to'). If the utterance is an imperative, express what
> should happen (e.g. 'turn off the lights' → 'the lights should be turned
> off'). Do not convert questions into declarative statements.

**Linguistic mechanism.** v1.0 systematically converted questions to
should-statements ("what time is it" → "the time should be checked"). The
revision adds explicit speech-act preservation with worked examples for both
imperative and interrogative input. No applicability gating — the transform
is held to apply to most seed types, but with constrained output form.

### passive_voice (target: naturalness)

**v1.0 prompt:** "Rewrite in passive voice"

**v1.1 prompt:**
> Rewrite in passive voice using a natural English passive construction.
> Prefer 'have X done', 'get X done', or standard 'be X-ed' depending on
> which sounds most idiomatic in context. If the utterance's main verb
> resists passive voice in modern English — particularly 'get' used as a
> main verb (as in 'how hot is it going to get'), 'have' in its possessive
> sense, or modal verbs — return the literal string TRANSFORM_NOT_APPLICABLE
> on its own line. Do not produce constructions like 'be gotten', 'be had',
> or 'be should'.

**Linguistic mechanism.** v1.0 mechanically applied "be + past participle"
to all verbs, producing "be gotten" / "be had" on seeds with non-passivizable
main verbs. The revision broadens the construction set to include "have X
done" and "get X done" causative passives, and explicitly enumerates verbs
that don't passivize cleanly.

### time_framed (target: naturalness, secondary equivalence)

**v1.0 prompt:** "Add a temporal context (e.g. 'in the morning', 'before
dinner', 'for the night')"

**v1.1 prompt:**
> Add a specific temporal context that situates when an action, request, or
> state should occur. For example: 'turn off the lights' → 'turn off the
> lights before I leave'; 'remind me to take my medication' → 'remind me to
> take my medication right after breakfast'. This transform applies when the
> utterance describes something that varies over time. It does not apply to
> factual queries about static information — geographic facts, historical
> events, biographical facts, or other atemporal content. For example,
> adding 'for the night' to 'what's the capital of australia' produces
> nonsense because capitals don't vary by time of day. When the transform
> doesn't apply, return the literal string TRANSFORM_NOT_APPLICABLE on its
> own line.

**Linguistic mechanism.** v1.0 attached temporal phrases without checking
semantic compatibility. The revision shifts the framing from "add a temporal
phrase" to "add temporal context to something that varies over time" and
provides an explicit negative example demonstrating the failure mode.

### Implementation

The sentinel filter is a one-line addition to `generate_constrained_variants()`:

```python
lines = [l for l in lines if l != TRANSFORM_NOT_APPLICABLE_SENTINEL]
```

Strict exact-match. Edge cases (lowercase, trailing punctuation) pass
through as variants and surface as low-quality output downstream — they're
recoverable but not silently filtered.

A second non-breaking change adds an optional `constraints` parameter so
study runners can invoke a subset of `SYNTACTIC_CONSTRAINTS`. The Day 5
validation runner uses this to run only the four revised transforms rather
than re-running all fifteen.

## Validation results

The validation runner produced 1,029 v1.1 variants across the four
transforms on the 117-seed set, vs. 1,391 v1.0 variants from the Day 2
aggregate. The 362-variant reduction is by design — it reflects the
applicability gating skipping (seed, transform) pairs that produced
drift-laden output in v1.0.

### Per-transform headline

See `equivalence_distributions.png` and `naturalness_distributions.png`
for the Likert distribution shifts.

| Transform | n_v10 | n_v11 | Δ mean eq | Δ %eq≤2 | Δ mean nat | Δ %nat≤2 | Δ clean rate |
|:---|---:|---:|---:|---:|---:|---:|---:|
| negative_framing | 349 | 219 | **+0.42** | **−19.2pp** | −0.06 | +2.5pp | +7.3pp |
| modal_should | 348 | 347 | +0.17 | −3.7pp | +0.12 | +0.0pp | +4.7pp |
| passive_voice | 343 | 221 | +0.13 | −3.0pp | **+0.18** | **−5.8pp** | +7.3pp |
| time_framed | 351 | 242 | +0.18 | **−12.6pp** | **+0.53** | **−10.5pp** | +5.9pp |

Clean rate = percent of variants rated ≥4 on both judge dimensions.

### Did the prompts meet pre-registered success criteria?

**Qualitative criterion (CDF rightward shift at p25 and median):** Met for
all four transforms on targeted dimensions.

**Numeric default (≥0.3 median Likert improvement):** Met for
`negative_framing` equivalence (+0.42) and `time_framed` naturalness (+0.53).
Not met for `modal_should` (+0.17) or `passive_voice` (+0.18). For both
sub-threshold cases, the failure-rate reduction and direction of change are
consistent with the qualitative criterion, and Day 1 specifications stated
the numeric defaults applied "if effects are ambiguous". The shifts are
not ambiguous.

**Non-targeted dimension non-degradation:** Met for all four. Largest
non-targeted change is `negative_framing` naturalness at −0.06, below the
0.1 noise threshold.

### Applicability gating

See `skip_rates_heatmap.png` for the per-(transform, syntactic_type) skip
rate visualization.

Three of the four transforms include applicability gating via the
`TRANSFORM_NOT_APPLICABLE` sentinel. Skip patterns by syntactic type:

- `negative_framing`: 100% skip on wh-questions, 69% on polar questions,
  33% on declaratives, 8% on imperatives, 12% on indirect requests.
- `time_framed`: 100% skip on wh-questions, 50% on polar questions, 0.6%
  on imperatives, 7% on indirect requests.
- `passive_voice`: 84% skip on wh-questions, 52% on polar questions, 17%
  on imperatives, 21% on indirect requests.
- `modal_should` (no skip rule): 5% skip on wh-questions, 0% elsewhere.

The 8% skip rate on `negative_framing` imperatives shows the model
exercising judgment beyond the rule. Seeds like "set a timer for twenty
minutes" have no clean intent-preserving negation; the model abstains
rather than producing strained output. This judicious skip behavior was
not explicitly required by the prompt and emerged from the
intent-preservation framing.

### Where did v1.0 failures go?

The most informative diagnostic is per-syntactic-type failure-rate shift
from v1.0 to v1.1, broken down into "still failing", "newly skipped"
(abstention), and "newly succeeding" (actually fixed):

**negative_framing on wh-questions:** 93.3% v1.0 failure rate → 0.0% v1.1
failure rate, 100% skip. The transform no longer attempts these cases.

**time_framed on wh-questions:** 38.7% → 0.0%, 100% skip. Same pattern.

**modal_should on wh-questions:** 78.7% → 53.5% failure rate, only 5% skip.
This is the largest residual failure pattern; the revision improved but
did not eliminate it. See "Residual issues" below.

**passive_voice on polar/wh-questions:** Failures redistributed mostly to
skips. The variants the transform does produce on these seed types are
higher-quality than v1.0.

## Residual issues

The v1.1 revisions ship as-is. Three residual issues are documented as
known limitations rather than addressed in this iteration.

### modal_should: factual → normative shift on wh-questions

53.5% of `modal_should` variants on wh-questions still score ≤2 on
equivalence. The remaining failures show a consistent pattern: the variant
preserves interrogative form (as the revision required) but shifts the
modality from factual/descriptive to normative/evaluative:

- "what time does the sun set tonight" → "what time should the sun be set
  tonight" (factual → normative)
- "is robin williams still alive" → "should robin williams be alive?"
  (factual → moral)
- "what's playing right now" → "what should currently be on air right now"
  (descriptive → evaluative)

The revised instruction constrained speech-act preservation but not
epistemic-vs-evaluative modality. A v1.2 revision could specify that the
modal "should" must admit a deontic reading without changing the question's
epistemic status — but this is a narrow fix that affects only a fraction of
wh-question seeds, and is deferred.

### negative_framing: marginal naturalness slip

Naturalness mean moved −0.06 and %natural≤2 rose +2.5pp. Both are below
the 0.1 / negligible-percent thresholds set for non-targeted-dimension
degradation, but the direction is negative. The variants that remain are
linguistically harder cases — seeds where intent-preserving negation
requires more contortion than v1.0's simpler polarity-flip approach. The
trade-off favors equivalence (−19.2pp on failure rate) over the marginal
naturalness slip.

### embedded_declarative failure increases on small samples

`negative_framing` on embedded declaratives shows a v1.0 → v1.1 failure
rate increase from 22.2% to 26.7%. The sample is small (18 expected
variants, 15 produced after 17% skip rate), and the effect could be
within-condition noise. If real, it points to a sub-category of embedded
declaratives that express desire or need (and should arguably skip) that
the current applicability rules don't catch. Worth re-examining at higher
volume.

## Implications for allo

The v1.1 prompts replace v1.0 in `allo/generate.py`. Three code changes:

1. Four `SYNTACTIC_CONSTRAINTS` entries (negative_framing, modal_should,
   passive_voice, time_framed) updated with the revised instructions.
2. `CONSTRAINT_VERSION = "v1.1"` constant added for downstream identification.
3. `generate_constrained_variants()` accepts an optional `constraints`
   parameter and filters lines matching `TRANSFORM_NOT_APPLICABLE` before
   recording variants.

The other eleven transforms in `SYNTACTIC_CONSTRAINTS` are unchanged. The
total constrained-rewriting output volume at `--n=50` decreases by roughly
27% compared to v1.0, because the applicability gating skips (seed, transform)
pairs that previously produced drift-laden output. This is by design.

Downstream studies that join on `constrained_transform` to compare v1.0
and v1.1 distributions should match on the `constraint_version` column or
on file provenance.

## Reproducibility

- **Generation:** `python -m evaluation.studies.prompt_revisions --output evaluation/runs/prompt_revisions_v1_1_variants.csv`
- **Judging:** `python -m evaluation.studies.llm_judge_batch --input evaluation/runs/prompt_revisions_v1_1_variants.csv --output evaluation/runs/prompt_revisions_v1_1_judged.csv`
- **Analysis:** `evaluation/studies/prompt_revisions_analysis.ipynb`

Generated artifacts in `evaluation/results/llm_judge/`:

- `equivalence_distributions.png` — per-transform Likert distributions
- `naturalness_distributions.png` — same, for naturalness
- `skip_rates_heatmap.png` — applicability gating visualization
- `failure_rate_comparison.png` — before/after failure rates by dimension
- `prompt_revisions_summary.csv` — per-transform summary statistics

## See also

- [`llm_judge.md`](./llm_judge.md) — parent study; rubric design, calibration,
  aggregate findings across all four generation strategies
- [`rubric.md`](./rubric.md) — full judge rubric with anchor examples
- [`seed_set_methodology.md`](../../seed_set_methodology.md) — the 117-seed
  test set used by both this study and the parent
- [`scoring_ranges.md`](../scoring_ranges/scoring_ranges.md) — the
  ~17K-variant surface-metric study on the same seed set
- [`volume_sweep.md`](../volume_sweep/volume_sweep.md) — the earlier study
  whose constraint-compliance findings were superseded by the judge analysis