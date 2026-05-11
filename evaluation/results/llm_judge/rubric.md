# LLM-as-judge rubric for allo

This document describes the design of the rubric used for LLM-as-judge evaluation of allo-generated variants. The rubric is operationalized in `judge_prompt.py` and applied via `llm_judge.py`. This document is the human-readable design rationale for the writeup.

**Status:** locked at `PROMPT_VERSION = v1.0-locked` following Day 1 calibration. Subsequent rubric changes require a fresh version bump and recalibration; the locked status is recorded in the prompt version string so cached judgments under earlier versions are mechanically invalidated.

---

## Design goals

The rubric is designed to do three things:

1. **Score variants on dimensions the existing surface metrics measure imperfectly.** Cosine similarity from `all-MiniLM-L6-v2` and GPT-2 perplexity have documented blind spots identified in `evaluation/results/scoring_ranges/scoring_ranges.md`: similarity drops sharply on named entity substitutions and undershoots on deep synonymy; perplexity is unreliable on short and disfluent seeds. The rubric dimensions need to be capable of producing scores that diverge from the surface metrics on exactly these cases.

2. **Be genuinely orthogonal.** Two correlated dimensions waste judge calls and weaken the integration commit's case for the judge as a third metric. The two dimensions chosen — semantic equivalence and naturalness — are naturally orthogonal: a fluent variant can shift meaning, an awkward variant can preserve it.

3. **Be applicable across all four generation strategies.** The same rubric scores LLM paraphrases, constrained rewrites, MLM substitutions, and semantic expansion variants. This requires the dimension definitions to work on both close paraphrases (where most variants will be high equivalence) and adjacent-intent expansion variants (where most variants will be mid equivalence by design).

The rubric does not score "intent preservation" as a separate dimension. That dimension was considered and rejected because (a) it correlates heavily with semantic equivalence on most variants, (b) the substitutability anchoring needed to dissociate it produces awkward judgments on cross-domain pragmatic variants, and (c) two clean dimensions outperform three muddled ones in practice.

---

## Dimension 1: Semantic equivalence

**Question:** How closely does the variant preserve the seed's meaning?

| Score | Anchor description |
|:-----:|:-------------------|
| **5** | Equivalent. The variant expresses the same proposition, request, or question as the seed. Substituting it for the seed in any reasonable context would produce no functional difference. **Includes paraphrases that share little vocabulary but preserve identical meaning.** |
| **4** | Closely related. The variant expresses the same core meaning with a minor shift in scope, specificity, emphasis, or politeness. The communicative goal is preserved. |
| **3** | Adjacent. The variant expresses related but meaningfully different content. The action, scope, or referent has shifted, but it remains recognizably in the same domain or context as the seed. |
| **2** | Loosely related. The variant shares some structural or topical elements with the seed (a verb, a noun, a question pattern) but expresses substantially different content. **Includes named-entity substitutions where the entity has changed.** |
| **1** | Unrelated. The variant has no meaningful semantic connection to the seed. |

### Why the bolded clauses

The bolded clauses on rows 5 and 2 are the explicit handles for the metric blind spots:

- **Row 5's clause** ("paraphrases that share little vocabulary but preserve identical meaning") cues the judge to score deep-synonymy paraphrases as 5. The cosine similarity metric undershoots these (e.g. "play something uh relaxing" → "can you select a laid back track" scored 0.103). The rubric explicitly tells the judge to recognize them as equivalent.

- **Row 2's clause** ("named-entity substitutions where the entity has changed") cues the judge to score MLM variants where the substituted token is a named entity as 2. The cosine similarity metric scores these very low (e.g. "what's the capital of australia" → "what's the capital of babylon" scored 0.059), which is in the vicinity of correct but framed as if the variant is unrelated. Anchoring at 2 rather than 1 reflects that the variant is structurally and topically related but referentially distinct.

### Expansion variants and the 3-4 band

Semantic expansion is designed to produce variants that are *adjacent intent*, not paraphrases. By design, these variants land in the 3-4 band — they share domain and communicative context with the seed but differ in scope, degree, or action. The rubric does not penalize this; it places expansion variants where they belong on the scale.

---

## Dimension 2: Naturalness

**Question:** How natural and fluent is the variant **as a standalone English utterance**?

| Score | Anchor description |
|:-----:|:-------------------|
| **5** | Fluent. The variant reads as natural English a real speaker would produce. No grammatical or collocational issues. **Short utterances and casual register are fine if they sound natural.** |
| **4** | Slightly marked. The variant is fluent but mildly unidiomatic — a slightly unusual word choice or phrasing that doesn't break grammaticality. |
| **3** | Awkward. The variant is grammatical but reads as stilted, mechanical, or unidiomatic. A native speaker would notice something off. |
| **2** | Strained. The variant has grammatical issues or unnatural collocations that would make a native speaker hesitate, but the meaning is still recoverable. |
| **1** | Broken. The variant is ungrammatical, incoherent, or contains errors that prevent normal interpretation. |

### Why the bolded clause

The bolded clause on row 5 ("Short utterances and casual register are fine if they sound natural") cues the judge to score variants like "call mom" as 5 even though GPT-2 perplexity scores them in the tens of thousands. The perplexity inflation on short and disfluent seeds is mechanical, not a reflection of unnaturalness, and the rubric explicitly tells the judge not to penalize these.

### Standalone-utterance framing

Naturalness is judged of the variant **as a standalone utterance**, not in pragmatic context against the seed. A variant like "avoid the ambiguity" is fluent English in isolation and scores 5 on naturalness even if it would be an odd paraphrase of "avoid the highway" — that contextual oddness belongs on the equivalence axis, not the naturalness axis. This separation keeps the two dimensions orthogonal. Day 1 calibration confirmed the judge applies this framing consistently; predictions that brought in contextual oddness diverged from the judge in a recurring pattern documented in the calibration findings below.

### Awkward MLM substitutions and the 2-3 band

MLM substitution sometimes produces grammatically valid but unidiomatic outputs — for example, swapping a verb's collocational partner for a near-synonym that doesn't compose naturally. The rubric places these in the 2-3 band. This dimension is where MLM's specific failure pattern shows up most clearly.

---

## Output format

The judge produces JSON with rationale-before-score ordering on each dimension:

```json
{
  "semantic_equivalence_rationale": "...",
  "semantic_equivalence": <integer 1-5>,
  "naturalness_rationale": "...",
  "naturalness": <integer 1-5>
}
```

Rationale-before-score is intentional. Producing the rationale first means the score is conditioned on the reasoning; producing it after means the score is generated and then the rationale post-hoc-justifies it. The former pattern produces better-calibrated scores and rationales that drive rather than rationalize the judgment. The rationale field is also the primary input to the diagnostic phase — clustering rationales surfaces recurring quality patterns that the score distributions alone don't reveal.

---

## Calibration anchors in the prompt

Every judgment call includes three worked examples in the prompt:

1. **High equivalence + high naturalness** — A clean paraphrase ("remind me to call the doctor tomorrow morning" → "set a reminder to call the doctor tomorrow morning"). Demonstrates the canonical 5/5 case.

2. **Low equivalence + high naturalness** — A named entity substitution ("what's the capital of australia" → "what's the capital of babylon"). Demonstrates that grammatical fluency does not imply semantic equivalence, and shows how a same-structure-different-referent variant lands at 2 on equivalence.

3. **High equivalence + high naturalness via deep synonymy** — A paraphrase that swaps almost all vocabulary ("turn it up" → "boost the volume would you"). Demonstrates that vocabulary divergence does not imply semantic divergence, addressing the synonymy gap directly.

The fourth corner of the 2D space (high equivalence + low naturalness) is not included as a worked example. The dimension definitions handle this case via Likert interpolation; Day 1 calibration showed no systematic mishandling of this corner that a fourth anchor would address.

### Anchor leakage

The three anchor seeds (seed IDs 12, 35, and 63 in `allo_seed_set.csv`) are present in the scoring_ranges aggregate. This is documented leakage: the judge sees the anchor case-and-answer in the prompt and then encounters the same case in the Day 2 run. Day 1 calibration confirmed concrete leakage on one of the three anchors — row 1 of the calibration set is Example 2 verbatim, and the judge produced word-for-word identical rationales on that row. Row 9 (= Example 3 verbatim) produced a different rationale with the same score, so leakage was partial rather than total across the anchors.

The choice to retain the v1.0 anchors despite the leakage was made on Day 1 in favor of measuring rather than reasoning about the issue. Day 2 analysis excludes anchor-exact-match variants from any agreement, correlation, or independence-dependent statistics. Future iterations of the rubric should use synthetic seeds outside the 117-seed set for anchors.

---

## Day 1 calibration findings

The rubric was applied to a 50-variant calibration set spanning the metric blind spots, expansion mid-range, low/low corner, constrained transform failures, and disfluent seeds beyond the two cited in the scoring ranges writeup. 20 variants were pre-filled from documented examples in the scoring ranges study or constructed for edge-of-scale coverage; 30 were sampled from the aggregate at `--n=50`. Predictions were made by a single annotator (the project author) before judge scores were obtained.

**Agreement:**

| Dimension | Exact | Within ±1 | Within ±2 |
|:---|---:|---:|---:|
| Semantic equivalence | 42% | 88% | 98% |
| Naturalness | 62% | 92% | 98% |

Mean signed deltas: equivalence −0.56, naturalness +0.20 (judge is stricter on equivalence and slightly more lenient on naturalness than predictions).

**Blind-spot handling:**

All 16 cases targeting the three documented metric blind spots were scored consistently with the metric-divergence hypothesis:

- Named entity substitutions: 6/6 scored equivalence ≤2 despite high naturalness
- Deep synonymy paraphrases: 6/6 scored equivalence ≥4 despite low cosine similarity
- Perplexity pathology on short/disfluent seeds: 4/4 scored naturalness ≥4 despite perplexity > 1,000

The single strongest demonstration of the judge adding signal beyond surface metrics is calibration row 42 ("turn it up" → "have it increased"): cosine similarity 0.26, predicted equivalence 5, judge equivalence 5. The constrained:passive_voice variant preserves intent through near-total vocabulary change, which the embedding undershoots and the judge correctly recognizes.

**Interpretive divergences from predictions:**

Ten variants showed divergences of 2+ Likert points between prediction and judge. Two systematic patterns emerged:

1. *Standalone vs in-context naturalness.* The judge consistently applied the rubric's standalone framing; predictions occasionally brought in contextual oddness (variants that read fluent in isolation but odd as paraphrases of the specific seed). This is a rubric-interpretation pattern, not a rubric bug. The standalone framing is intentional and is documented in the dimension definition above.

2. *Strict vs pragmatic equivalence.* The judge scored factual shifts (Wednesday → Thursday, rain → cloudy) as significant equivalence drops; predictions sometimes treated these as preserved intent. On one such case (calibration row 37) the prediction missed a factual change that the judge caught, confirming the judge is doing real reasoning rather than rubber-stamping predictions.

These patterns are documented as findings of the calibration phase, not corrected as rubric defects. The locked rubric proceeds to Day 2 with both readings explicit.

---

## Methodological scope and limitations

The judge's outputs are not validated against human annotation. The study is comparative — it asks whether the judge produces score patterns that diverge from the surface metrics in ways consistent with the metric blind spots identified by the prior scoring ranges study, and whether prompt revisions designed against judge-surfaced quality patterns shift the judge-score distribution as expected. Both questions are answerable with judge data alone.

Predictions in the Day 1 calibration set were produced by a single annotator (the project author). Predictions are themselves judgments, not ground truth; the calibration measures predictor-judge alignment, not judge correctness in any absolute sense. At least one prediction (calibration row 37) was demonstrably incorrect in a way the judge surfaced.

Calibration against human judgment on a sampled subset (50-100 variants scored by a second annotator) is named as future work. Inter-judge agreement using a second LLM judge is also reserved for future work. The current study uses a single judge model (`claude-haiku-4-5` at temperature 0) for cost and reproducibility.

The judge was chosen from the Claude family rather than OpenAI to avoid self-preference bias — allo's primary backend in the scoring ranges aggregate is `gpt-4o-mini`, and OpenAI judges have measured self-preference bias on OpenAI-generated text. Claude Haiku 4.5 has no such relationship to the variants under judgment.