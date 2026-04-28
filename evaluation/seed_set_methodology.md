# allo seed set — methodology

This document records the design decisions made in constructing `allo_seed_set.csv`, the evaluation seed set used as the basis for allo systemic testing experiments. It is intended to make the seed set interpretable and reproducible without external context, and to support future validation and extension work.

---

## Purpose

The seed set is the foundation for all systematic evaluation of allo's generation and scoring pipeline. Its primary purpose is to enable:

- Empirical measurement of output volume per strategy across a range of seed types
- Prompt evaluation for LLM paraphrasing, constrained rewriting, and semantic expansion
- Batch size and temperature optimization experiments
- Similarity and perplexity range documentation per strategy
- Perplexity-fluency correlation study (planned)

All evaluation work should use this seed set as the input corpus to ensure findings are comparable across experiments.

---

## Design principles

**Stratified sampling over convenience sampling.** Seeds were selected to ensure deliberate representation across all dimensions of interest — domain, syntactic type, register, lexical complexity, utterance length, and speech disfluencies — rather than drawn from a convenience sample. Stratification allows findings to be framed as representative across utterance types rather than incidental to a particular sample.

**Proportional syntactic distribution with minimum floor.** Syntactic types are distributed proportionally to their approximate natural occurrence in voice assistant and conversational AI contexts, based on general patterns reported in the conversational AI literature. Imperatives and indirect requests dominate as they do in real-world usage. There is a minimum floor of 3 seeds per syntactic type.

**Domain weighting by real-world frequency.** Domains that account for the majority of real-world voice assistant interactions such as smart home, calendar and reminders, communication, media control all receive more seeds proportionally. Narrow or specialized domains receive fewer seeds.

**Naturalistic over forced combinations.** Domain × syntactic type combinations that would produce unnatural-sounding utterances were avoided. Seeds that required forced categorization are flagged in the notes column for review.

**Ecological validity prioritized.** The set is designed to reflect the kinds of utterances that realistically appear in NLU training pipelines, including a small number of disfluent seeds to represent natural spoken language disfluencies.

---

## Dimensions and schema

Each seed is tagged with the following metadata:

| Column | Description |
|---|---|
| `id` | Unique integer identifier (1–117) |
| `seed` | The utterance text |
| `domain` | Thematic domain (see below) |
| `syntactic_type` | Primary syntactic structure (see below) |
| `register` | Formality level: `formal`, `neutral`, `colloquial` |
| `lexical_complexity` | One or more complexity axes (see below), semicolon-separated |
| `utterance_length` | `short` (1–4 words), `medium` (5–9 words), `long` (10+ words) |
| `notes` | Free text for ambiguous categorizations, pragmatic properties, or flags for review |
| `speech_phenomena` | Disfluency type if applicable (see below), empty for clean speech |

---

## Domains

| Domain | Seeds | Notes |
|---|---|---|
| `smart_home` | 11 | Wide action space, high real-world frequency |
| `calendar` | 11 | Includes timers and alarms; rich temporal reference variation |
| `communication` | 10 | Covers calls, messages, email, voicemail |
| `media` | 10 | Music, video, podcasts; broad action space |
| `shopping` | 9 | Browse, track, return, subscribe, cancel |
| `navigation` | 8 | Directions, routing, points of interest |
| `weather_info` | 8 | Weather queries and general information seeking |
| `health` | 8 | Logging, tracking, appointments, wellness queries |
| `finance` | 7 | Balance, transfers, transactions, disputes |
| `travel` | 7 | Flights, hotels, transport schedules |
| `contact_center` | 6 | Generic contact center intents across verticals |
| `entertainment_trivia` | 5 | Presuppositional information seeking; biographical facts |

**Note on timers and alarms:** Utterances relating to `timers and alarms` fit under the domain `calendars`.

**Note on contact center:** Seeds represent generic contact center intents ("I want to cancel my account", "my order hasn't arrived") rather than any specific vertical. Domain-specific contact center evaluation (pharmacy, insurance, banking, etc.) is not considered for this version.

**Note on entertainment trivia:** Included for its distinct pragmatic profile rather than domain breadth. Findings from this domain should be treated as exploratory given the small sample size.

---

## Syntactic types

| Type | Approximate proportion | Description |
|---|---|---|
| `imperative` | ~35% | Direct commands and requests |
| `indirect_request` | ~20% | Statements of need or desire ("I need...", "I'd like...") |
| `polar_question` | ~12% | Yes/no questions |
| `wh_question` | ~12% | Information-seeking questions |
| `declarative` | ~8% | Statements that imply an action or intent |
| `embedded_declarative` | ~8% | Complex structures with subordinate clauses ("I was hoping you could...") |
| `fragment` | ~5% | Reduced or telegraphic utterances ("lights off") |
| `multi_intent` | distributed | Utterances containing two distinct intents; distributed across syntactic types |

**Note on proportions:** These proportions reflect the author's domain judgment based on production experience with NLU pipelines and voice assistant systems, not a specific corpus or published frequency study. They are intended to be ecologically plausible rather than empirically derived.

**Note on multi-intent:** Multi-intent utterances are not a separate syntactic type but a property that can apply across types. They are tagged in `lexical_complexity` as `multi_intent` and noted where present.

---

## Lexical complexity axes

One or more of the following tags may appear in the `lexical_complexity` column, semicolon-separated:

| Tag | Description |
|---|---|
| `simple` | Common, high-frequency vocabulary with no notable complexity |
| `phrasal_verb` | Contains a phrasal verb (e.g. "turn off", "pick up") |
| `morphological_complexity` | Contains derived or morphologically complex forms (e.g. "reschedule", "discontinue") |
| `syntactic_embedding` | Contains an embedded or subordinate clause |
| `referential_complexity` | Contains pronouns or referring expressions requiring contextual resolution |
| `named_entity` | Contains a proper noun (person, place, product, organization) |
| `low_freq_vocab` | Contains lower-frequency but natural vocabulary (e.g. "dispute", "aisle", "voicemail") |
| `temporal_reference` | Contains a temporal expression (e.g. "tomorrow", "next Friday", "when I get home") |
| `multi_intent` | Utterance expresses two distinct intents |
| `negation` | Contains a negation that is semantically significant |
| `conditional` | Contains a conditional or hypothetical construction |
| `pragmatic_indirect` | Intent is implied rather than stated (e.g. "it's cold in here" implying thermostat adjustment) |

---

## Register

Three levels:

- `formal` — polite, complete sentences, appropriate for professional or unfamiliar interlocutors
- `neutral` — standard everyday speech, neither marked formal nor colloquial
- `colloquial` — casual, reduced, or informal; includes telegraphic/fragment-style utterances

Highly reduced or telegraphic speech (e.g. "lights off", "call call mom") is categorized as `colloquial` rather than as a separate register level.

---

## Utterance length

| Band | Word count | Notes |
|---|---|---|
| `short` | 1–4 words | Relevant to MLM substitution ceiling analysis |
| `medium` | 5–9 words | Most common band in the set |
| `long` | 10+ words | Tests syntactic embedding and constrained rewriting compliance |

---

## Speech disfluencies

Ten seeds contain disfluency phenomena representing natural spoken language. These seeds were selected by replacing existing neutral seeds in high-frequency domains with disfluent equivalents.

| Type | Count | Description |
|---|---|---|
| `filled_pause` | 3 | "um" or "uh" inserted at utterance-initial or mid-utterance position |
| `self_correction` | 2 | Speaker repairs a word, phrase, or destination mid-utterance |
| `repetition` | 2 | A word or phrase is repeated |
| `false_start` | 3 | Speaker begins a syntactic structure and restarts with a different one |

**Methodological note:** Disfluency coverage is intentionally limited in this version. Ten seeds is sufficient to identify whether allo's normalization pipeline handles disfluent input gracefully, but too sparse to draw strong conclusions about strategy-level performance on disfluent speech.

---

## Known limitations

**Clean speech dominance.** 94 of 117 seeds are clean, well-formed utterances. The set does not represent the full range of disfluency, non-native speaker patterns, or code-switching phenomena that appear in real-world voice assistant deployments.

**No code-switching.** Multilingual utterances are absent from this version. Future testing will examine performance across mixed-language seed utterances.

**Register imbalance.** The formal register is concentrated in a small number of domains (finance, travel, contact center) rather than distributed across the full set.

**Entertainment trivia and contact center are sparse.** With 5 and 6 seeds respectively, findings from these domains should be treated as exploratory rather than representative.

**Validation pending.** Dimension labels (syntactic type, register, lexical complexity) have been assigned by a single author and have not yet been validated by a second annotator. Inter-annotator agreement measurement is planned as a subsequent step. Seeds flagged in the notes column should be prioritized for validation.

---

## Version history

| Version | Date | Changes |
|---|---|---|
| v1 | April 2026 | Initial 117-seed set with 9 metadata dimensions. Long-seed bucket expanded from 3 to 20 seeds after initial 100-seed set was identified as inadequate for length-gated analysis. |

---

## Planned future work

- Second annotator validation with inter-annotator agreement measurement (Cohen's κ or Krippendorff's α)
- Disfluency subcorpus expansion
- Code-switching seeds
- Corpus-derived syntactic proportion verification
- Extension to additional domains (accessibility, smart office, automotive)