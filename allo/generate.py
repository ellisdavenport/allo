"""
generate.py — utterance variant generation strategies

Four strategies, designed to complement rather than duplicate each other:

    1. LLM paraphrasing
       High-volume diverse paraphrases via API. Primary volume source.
       Batches requests in groups of 10 to maintain diversity at scale.
       a single prompt for 100 variants produces clustering around the
       same surface forms, where 10 independent calls of 10 each do not.

    2. Constrained syntactic rewriting
       Applies explicit structural transforms via the LLM: polar questions,
       hedged requests, passive voice, conditionals, etc.

    3. Contextual MLM substitution
       Uses DistilBERT's masked language model to suggest lexical
       substitutions that are contextually appropriate. The model
       understands position and surrounding context, so it won't substitute
       'twist' for 'turn' in 'turn off the lights'.
       Operates over every content word position independently.

    4. Semantic expansion
       Generates semantically adjacent utterances that are related but meaningfully
       distinct intents in the same domain. This is what enables scale to
       hundreds of variants: pure paraphrasing plateaus because there are
       only so many ways to say exactly the same thing. Expansion
       deliberately trades semantic fidelity for coverage breadth.
"""

import os
import re
import random
import nltk
from dotenv import load_dotenv
from transformers import pipeline as hf_pipeline

load_dotenv()

nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# Version marker for the constrained rewriting prompt set.
# Bumped when SYNTACTIC_CONSTRAINTS instruction strings change so downstream
# analysis can identify which prompt set produced any given variant.
#
# v1.0: original 15 transforms
# v1.1: revised negative_framing, modal_should, passive_voice, time_framed
#       after Day 4 diagnostic findings. Added TRANSFORM_NOT_APPLICABLE
#       sentinel for LLM-side applicability gating.
CONSTRAINT_VERSION = "v1.1"


# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────

# Loaded at module level so initialization happens once per session.
# DistilBERT is a lightweight BERT distillation — fast enough for
# per-token masking across a batch of seeds without GPU requirement.
# top_k=15 gives a wide candidate pool before validity filtering.

print("Loading MLM model for contextual substitution...")
_mlm = hf_pipeline("fill-mask", model="distilbert-base-uncased", top_k=15)


# ─────────────────────────────────────────────
# LLM ABSTRACTION LAYER
# ─────────────────────────────────────────────

class LLMClient:
    """
    A provider-agnostic LLM client.
    Reads provider and model from environment variables,
    so the rest of the code never needs to know which API is being used.
    """

    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "openai").lower()
        self.model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.api_key = self._get_api_key()

    def _get_api_key(self) -> str:
        if self.provider == "openai":
            return os.getenv("OPENAI_API_KEY", "")
        elif self.provider == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY", "")
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def generate(self, prompt: str, temperature: float = 0.9) -> str:
        """
        Send a prompt to the configured LLM provider and return the
        response as a plain string.
        """
        if self.provider == "openai":
            return self._call_openai(prompt, temperature)
        elif self.provider == "anthropic":
            return self._call_anthropic(prompt, temperature)

    def _call_openai(self, prompt: str, temperature: float) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    def _call_anthropic(self, prompt: str, temperature: float) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()


# ─────────────────────────────────────────────
# STRATEGY 1: LLM-BASED PARAPHRASING
# ─────────────────────────────────────────────

# Batch size for LLM paraphrase calls. Requesting more than ~15 variants
# in a single prompt causes the model to start repeating itself and
# clustering around a few surface forms. Multiple independent calls with
# the same temperature produce genuine diversity because each call samples
# fresh from the model's distribution.
_LLM_BATCH_SIZE = 10


def generate_llm_variants(
    seed: str,
    n: int,
    client: LLMClient,
    temperature: float = 0.9,
) -> list[str]:
    """
    Generate n LLM paraphrases of the seed utterance.

    Splits large requests into batches of _LLM_BATCH_SIZE to maintain
    diversity. Each batch is an independent API call with fresh sampling,
    which avoids the within-prompt repetition that emerges when asking
    for many variants in a single prompt.
    """
    results = []
    remaining = n

    while remaining > 0:
        batch_n = min(_LLM_BATCH_SIZE, remaining)
        prompt = (
            f"Generate {batch_n} natural, varied paraphrases of the following utterance.\n"
            f"Each should sound like something a real person would say.\n"
            f"Return only the paraphrases, one per line, with no numbering or extra text.\n\n"
            f"Utterance: {seed}"
        )
        try:
            raw = client.generate(prompt, temperature=temperature)
            batch = [line.strip() for line in raw.splitlines() if line.strip()]
            results.extend(batch[:batch_n])
        except Exception as e:
            print(f"  LLM paraphrase batch failed: {e}")
        remaining -= batch_n

    return results[:n]


# ─────────────────────────────────────────────
# STRATEGY 2: CONSTRAINED SYNTACTIC REWRITING
# ─────────────────────────────────────────────

# Sentinel string the LLM is instructed to return when a transform doesn't
# apply to a given seed. Filtered out before variants are recorded.
TRANSFORM_NOT_APPLICABLE_SENTINEL = "TRANSFORM_NOT_APPLICABLE"


# Each entry defines a structural transform by name and a prompt instruction.
# The LLM is asked to follow the instruction exactly, at low temperature,
# which focuses it on structural compliance rather than lexical creativity.
#
# The set is designed to cover several main syntactic variation axes:
# illocutionary force (imperative → question → indirect request),
# modality (can/could/would/should), politeness register, temporal framing,
# and aspectual variation.
#
# Four transforms (negative_framing, modal_should, passive_voice, time_framed)
# were revised in v1.1 after Day 4 LLM-as-judge diagnostic identified specific
# linguistic failure modes. These four include TRANSFORM_NOT_APPLICABLE
# guardrails for seed types they don't compose with.

SYNTACTIC_CONSTRAINTS = [
    {
        "name": "polar_question",
        "instruction": "Rewrite as a yes/no question (e.g. 'Could you...', 'Can you...', 'Would you mind...')",
    },
    {
        "name": "polite_imperative",
        "instruction": "Keep the imperative form but add a politeness marker such as 'please', 'kindly', or 'go ahead and'",
    },
    {
        "name": "indirect_need",
        "instruction": "Rewrite as an indirect statement of need or desire (e.g. 'I need...', 'I want...', 'I'd like...')",
    },
    {
        "name": "hedged_request",
        "instruction": "Add a hedging phrase that softens the urgency (e.g. 'when you get a chance', 'if possible', 'whenever you can', 'at some point')",
    },
    {
        "name": "conditional",
        "instruction": "Rewrite as a conditional or hypothetical request (e.g. 'if you could...', 'would you be able to...', 'is it possible to...')",
    },
    {
        "name": "modal_would",
        "instruction": "Rewrite using 'would' as the main modal verb",
    },
    {
        "name": "modal_should",
        # Revised in v1.1 to preserve speech act.
        # Day 4 diagnostic: 79% failure rate on wh-questions because the
        # original instruction converted questions to statements.
        "instruction": (
            "Rewrite using 'should' as the main modal verb while preserving "
            "the utterance's speech act. If the utterance is a question, the "
            "variant must remain a question (e.g. 'what's the temperature set "
            "to' → 'what should the temperature be set to'). If the utterance "
            "is an imperative, express what should happen (e.g. 'turn off the "
            "lights' → 'the lights should be turned off'). Do not convert "
            "questions into declarative statements."
        ),
    },
    {
        "name": "passive_voice",
        # Revised in v1.1 to handle verbs without idiomatic passives.
        # Day 4 diagnostic: 36% naturalness failure rate, predominantly
        # "be gotten" type ungrammatical passives.
        "instruction": (
            "Rewrite in passive voice using a natural English passive "
            "construction. Prefer 'have X done', 'get X done', or standard "
            "'be X-ed' depending on which sounds most idiomatic in context. "
            "If the utterance's main verb resists passive voice in modern "
            "English — particularly 'get' used as a main verb (as in 'how "
            "hot is it going to get'), 'have' in its possessive sense, or "
            "modal verbs — return the literal string TRANSFORM_NOT_APPLICABLE "
            "on its own line. Do not produce constructions like 'be gotten', "
            "'be had', or 'be should'."
        ),
    },
    {
        "name": "negative_framing",
        # Revised in v1.1 to prevent meaning inversion.
        # Day 4 diagnostic: 43.8% equivalence failure rate, with 93% failure
        # on wh-questions because the transform doesn't compose with
        # non-imperative speech acts.
        "instruction": (
            "Rewrite the utterance using a negation that preserves its "
            "actionable intent — the variant must request the same outcome "
            "as the utterance, not the opposite. For example: 'turn off the "
            "lights' → 'don't leave the lights on'; 'remind me to call the "
            "doctor' → 'don't let me forget to call the doctor'. This "
            "transform applies when the utterance expresses an action the "
            "addressee should take or avoid. It does not apply to utterances "
            "that ask for information (\"what's the temperature\"), describe "
            "states without requesting action (\"i'd like the thermostat at "
            "68\"), are already negated (\"don't send that email yet\"), or "
            "are too fragmentary to negate coherently (\"lights off\"). When "
            "the transform doesn't apply, return the literal string "
            "TRANSFORM_NOT_APPLICABLE on its own line."
        ),
    },
    {
        "name": "progressive_desire",
        "instruction": "Express the request as an ongoing desire using progressive aspect (e.g. 'I was hoping you could...', 'I've been meaning to ask you to...')",
    },
    {
        "name": "time_framed",
        # Revised in v1.1 to require semantic compatibility with temporal context.
        # Day 4 diagnostic: 39% failure rate on wh-questions due to adding
        # temporal context to atemporal facts ("capital of australia for the night").
        "instruction": (
            "Add a specific temporal context that situates when an action, "
            "request, or state should occur. For example: 'turn off the "
            "lights' → 'turn off the lights before I leave'; 'remind me to "
            "take my medication' → 'remind me to take my medication right "
            "after breakfast'. This transform applies when the utterance "
            "describes something that varies over time. It does not apply to "
            "factual queries about static information — geographic facts, "
            "historical events, biographical facts, or other atemporal "
            "content. For example, adding 'for the night' to 'what's the "
            "capital of australia' produces nonsense because capitals don't "
            "vary by time of day. When the transform doesn't apply, return "
            "the literal string TRANSFORM_NOT_APPLICABLE on its own line."
        ),
    },
    {
        "name": "reason_added",
        "instruction": "Add a brief stated reason or context for the request (e.g. 'I'm going to sleep', 'I'm heading out', 'it's getting late')",
    },
    {
        "name": "abbreviated_spoken",
        "instruction": "Rewrite in a shorter, more casual form as someone might say it quickly in natural speech",
    },
    {
        "name": "emphatic",
        "instruction": "Rewrite with added emphasis or urgency (e.g. using 'really', 'definitely', 'make sure to', 'be sure to')",
    },
    {
        "name": "confirmation_seeking",
        "instruction": "Rewrite as a request that also seeks confirmation the listener understood (e.g. ending with 'okay?', 'alright?', 'can you do that?')",
    },
]


def generate_constrained_variants(
    seed: str,
    n_per_constraint: int,
    client: LLMClient,
    temperature: float = 0.4,
    constraints: list[dict] = None,
) -> list[dict]:
    """
    Apply each syntactic constraint to the seed and collect the results.

    Temperature is set low (default 0.4) because the goal is structural
    compliance, not lexical creativity. At high temperatures the model
    tends to ignore the constraint instruction and produce generic paraphrases.

    n_per_constraint controls how many variants to request per transform.
    At n_per_constraint=1 (default at low n_per_strategy) this produces
    one structurally distinct variant per constraint — ~15 high-quality
    structural variants total. At n_per_constraint=2-3 the model produces
    alternative realizations of the same structural pattern.

    constraints (optional) lets the caller pass a subset of SYNTACTIC_CONSTRAINTS
    to run rather than all 15. Used by the prompt-revision validation runner
    in evaluation/studies/prompt_revisions.py to evaluate only the four
    transforms revised in v1.1. If None, runs all SYNTACTIC_CONSTRAINTS.

    Variants returned as TRANSFORM_NOT_APPLICABLE (the sentinel returned by
    revised v1.1 transforms when they judge the seed unsuitable) are filtered
    out before recording. This means (seed, transform) pairs the LLM judges
    inapplicable produce zero variants rather than off-task output.

    Returns a list of dicts with 'utterance' and 'constraint' keys so
    generate_variants() can tag each result with its specific transform.
    """
    if constraints is None:
        constraints = SYNTACTIC_CONSTRAINTS

    results = []
    for constraint in constraints:
        count_phrase = (
            f"{n_per_constraint} versions"
            if n_per_constraint > 1
            else "one version"
        )
        prompt = (
            f"Rewrite the following utterance {count_phrase} by following this structural instruction exactly.\n"
            f"Focus on applying the structure — do not just paraphrase freely.\n"
            f"Return only the rewritten utterance(s), one per line, no numbering or extra text.\n\n"
            f"Instruction: {constraint['instruction']}\n"
            f"Utterance: {seed}"
        )
        try:
            raw = client.generate(prompt, temperature=temperature)
            lines = [l.strip() for l in raw.splitlines() if l.strip()]
            # Filter sentinel — TRANSFORM_NOT_APPLICABLE indicates the LLM
            # judged this constraint inapplicable to this seed. Strict
            # exact-match check; edge-case variations on the sentinel
            # (e.g. lowercase, trailing punctuation) pass through as
            # variants and will surface as low-quality output downstream.
            lines = [l for l in lines if l != TRANSFORM_NOT_APPLICABLE_SENTINEL]
            for line in lines[:n_per_constraint]:
                results.append({"utterance": line, "constraint": constraint["name"]})
        except Exception as e:
            print(f"  Constrained rewriting failed for '{constraint['name']}': {e}")

    return results


# ─────────────────────────────────────────────
# STRATEGY 3: CONTEXTUAL MLM SUBSTITUTION
# ─────────────────────────────────────────────

# Penn Treebank POS tags that identify substitutable content words.
# Function words (determiners, prepositions, conjunctions, auxiliaries)
# are intentionally excluded — substituting 'the' or 'off' produces
# ungrammatical output with no semantic value.
_CONTENT_POS_TAGS = {
    "NN", "NNS", "NNP", "NNPS",        # nouns
    "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",  # verbs
    "JJ", "JJR", "JJS",                  # adjectives
    "RB", "RBR", "RBS",                  # adverbs
}


def _is_valid_mlm_candidate(original: str, candidate: str) -> bool:
    """
    Filter out candidates that are not useful substitutions.

    Rejects:
    - Same word as original (case-insensitive)
    - BERT subword tokens (prefixed with ##)
    - Tokens containing non-alphabetic characters (punctuation artifacts,
      numbers, special characters that appear in BERT's vocabulary)
    """
    if candidate.lower() == original.lower():
        return False
    if candidate.startswith("##"):
        return False
    if not re.match(r"^[a-zA-Z][a-zA-Z'-]*$", candidate):
        return False
    return True


def _fix_tokenization_spacing(text: str) -> str:
    """
    Repair spacing artifacts from nltk word_tokenize reconstruction.

    nltk's tokenizer splits contractions and punctuation into separate
    tokens, so rejoining with spaces produces "I 'm going" or "lights ."
    This corrects the most common cases.
    """
    # Contractions: "I 'm" → "I'm", "do n't" → "don't"
    text = re.sub(r"\s+(n't|'s|'re|'ve|'d|'ll|'m)", r"\1", text)
    # Punctuation: "lights ." → "lights."
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    return text


def generate_mlm_variants(seed: str, n: int) -> list[str]:
    """
    Generate variants by masking each content word and collecting
    DistilBERT's top predictions for that position.

    For each content word in the seed:
        1. Replace it with [MASK]
        2. Run the fill-mask pipeline
        3. Filter invalid candidates (subwords, punctuation, same word)
        4. Reconstruct the full utterance with the substitution

    This produces contextually grounded substitutions. The model
    conditions on the full sentence context, not just the word in
    isolation, so it respects phrasal structure.
    'Turn' in 'turn off the lights' will yield
    candidates like 'switch', 'flip', 'shut', not 'rotate' or 'bend'.

    MLM yield is bounded by utterance length × valid candidates per
    position. Short utterances (3-4 content words) with a narrow
    semantic field produce fewer variants than longer ones. If the
    seed is very short, n may not be fully satisfied. This is
    expected and score-filtering will further reduce the set.

    Returns up to n deduplicated variants in randomised order.
    """
    tokens = nltk.word_tokenize(seed)
    pos_tags = nltk.pos_tag(tokens)

    candidates = []

    for i, (word, tag) in enumerate(pos_tags):
        if tag not in _CONTENT_POS_TAGS:
            continue

        masked_tokens = tokens.copy()
        masked_tokens[i] = "[MASK]"
        # BERT expects natural sentence spacing, not tokenizer artifacts
        masked_sentence = _fix_tokenization_spacing(" ".join(masked_tokens))

        try:
            predictions = _mlm(masked_sentence)
            for pred in predictions:
                candidate_word = pred["token_str"].strip()
                if _is_valid_mlm_candidate(word, candidate_word):
                    new_tokens = tokens.copy()
                    new_tokens[i] = candidate_word
                    variant = _fix_tokenization_spacing(" ".join(new_tokens))
                    if variant.lower() != seed.lower():
                        candidates.append(variant)
        except Exception as e:
            print(f"  MLM substitution failed at position {i} ('{word}'): {e}")

    # Deduplicate preserving insertion order, then shuffle to avoid
    # position-order bias (earlier positions would otherwise dominate)
    seen = set()
    unique = []
    for c in candidates:
        if c.lower() not in seen:
            seen.add(c.lower())
            unique.append(c)

    random.shuffle(unique)
    return unique[:n]


# ─────────────────────────────────────────────
# STRATEGY 4: SEMANTIC EXPANSION
# ─────────────────────────────────────────────

def generate_semantic_expansion(
    seed: str,
    n: int,
    client: LLMClient,
    temperature: float = 0.85,
) -> list[str]:
    """
    Generate semantically adjacent utterances — related but meaningfully
    distinct intents in the same domain as the seed.

    This strategy exists because pure paraphrasing has a natural ceiling.
    There are only so many ways to say exactly the same thing, and beyond
    ~50 paraphrases the output starts clustering. To reach hundreds of
    potentially useful training examples, we need utterances that vary
    not just in surface form but in intent scope and action.

    Examples from 'turn off the lights':
        - 'dim the lights'           (related action, different degree)
        - 'switch off the lamp'      (narrowed scope, synonym action)
        - 'turn off all the lights'  (broadened scope)
        - 'kill the lights'          (colloquial register variation)
        - 'can you make it darker'   (indirect, same communicative goal)

    These variants will have lower semantic similarity scores than pure
    paraphrases. This is expected and correct.
    Practitioners should review expansion variants more carefully before
    ingesting them into training data, and may want to apply a tighter
    similarity filter or keep them in a separate column.

    Temperature is slightly higher here than for paraphrasing since we want
    the model to explore the adjacent space and not converge on near-synonyms.
    """
    prompt = (
        f"Generate {n} utterances that are semantically adjacent to the following utterance.\n"
        f"These should be things a real user might say in the same context, "
        f"but with a meaningfully different scope, degree, or action.\n"
        f"Good adjacency examples: changing degree ('dim' vs 'turn off'), "
        f"narrowing or expanding scope ('the bedroom light' vs 'the lights'), "
        f"related action in the same domain ('mute the TV' near 'turn off the TV'), "
        f"colloquial register ('kill the lights' vs 'turn off the lights').\n"
        f"Do NOT generate synonymous paraphrases — every line should have a distinct intent.\n"
        f"Return only the utterances, one per line, no numbering or extra text.\n\n"
        f"Utterance: {seed}"
    )
    try:
        raw = client.generate(prompt, temperature=temperature)
        variants = [line.strip() for line in raw.splitlines() if line.strip()]
        return variants[:n]
    except Exception as e:
        print(f"  Semantic expansion failed: {e}")
        return []


# ─────────────────────────────────────────────
# UNIFIED GENERATION FUNCTION
# ─────────────────────────────────────────────

def generate_variants(
    seed: str,
    n_per_strategy: int,
    client: LLMClient,
    temperature: float = 0.9,
    include_expansion: bool = True,
) -> list[dict]:
    """
    Run all generation strategies and return a combined list of variant dicts.

    Each dict contains 'utterance' and 'strategy'. The strategy field is
    specific enough to be informative in the output:
        'llm_paraphrase'             — LLM paraphrase
        'constrained:polar_question' — constrained rewrite with named transform
        'mlm_substitution'           — contextual MLM substitution
        'expansion'                  — semantic expansion

    Parameters:
        seed             — the input utterance
        n_per_strategy   — variants to generate per strategy. For constrained
                           rewriting this is variants *per constraint*, so
                           total constrained output is n_per_constraint ×
                           len(SYNTACTIC_CONSTRAINTS). Set to 1 for one clean
                           variant per transform; 2-3 for richer structural
                           coverage.
        temperature      — passed to LLM paraphrasing and expansion. Constrained
                           rewriting always uses a lower fixed temperature (0.4)
                           to enforce structural compliance.
        include_expansion — whether to run semantic expansion. Disable if you
                            want strictly synonymous paraphrases only.

    Measured output at common n_per_strategy values (mean across the 117-seed
    test set, before deduplication; full study at
    evaluation/results/volume_sweep/volume_sweep.md):
        n=5:   ~5 llm   + ~15 constrained + ~5 mlm   + ~5 expansion  = ~30
        n=10:  ~10 llm  + ~15 constrained + ~10 mlm  + ~5 expansion  = ~40
        n=20:  ~19 llm  + ~15 constrained + ~20 mlm  + ~5 expansion  = ~58
        n=50:  ~44 llm  + ~44 constrained + ~45 mlm  + ~12 expansion = ~145
        n=80:  ~67 llm  + ~44 constrained + ~59 mlm  + ~20 expansion = ~190
        n=100: ~80 llm  + ~44 constrained + ~65 mlm  + ~25 expansion = ~213

    MLM output is bounded by utterance length × valid candidates per position.
    Measured ceilings: ~48 variants for short seeds (1-4 words), ~75 for
    medium (5-9 words), effectively unbounded within --n <= 100 for long
    (10+ words).

    LLM paraphrasing has a content ceiling on short seeds: ~71% efficiency
    at --n=100 on short seeds vs ~97% on long seeds.

    Constrained rewriting caps at 3 variants per transform x 15 transforms
    = 45 total when --n >= 50.

    Note: in CONSTRAINT_VERSION v1.1, four transforms (negative_framing,
    modal_should, passive_voice, time_framed) include applicability gating
    via the TRANSFORM_NOT_APPLICABLE sentinel. Total constrained output
    for v1.1 may be lower than v1.0 because these four transforms now
    correctly skip on incompatible seed types instead of producing
    drift-laden variants.
    """
    results = []

    # Strategy 1: LLM paraphrasing
    print(f"Generating LLM paraphrases (n={n_per_strategy}, batched in {_LLM_BATCH_SIZE}s)...")
    for utt in generate_llm_variants(seed, n_per_strategy, client, temperature=temperature):
        results.append({"utterance": utt, "strategy": "llm_paraphrase"})
    print(f"  → {sum(1 for r in results if r['strategy'] == 'llm_paraphrase')} variants")

    # Strategy 2: Constrained syntactic rewriting
    # n_per_constraint is capped at 3, beyond which variants within a single
    # constraint start to converge. Total output scales with constraint count.
    n_per_constraint = min(3, max(1, n_per_strategy // len(SYNTACTIC_CONSTRAINTS)))
    print(f"Generating constrained rewrites ({len(SYNTACTIC_CONSTRAINTS)} constraints × {n_per_constraint})...")
    before = len(results)
    for item in generate_constrained_variants(seed, n_per_constraint, client):
        results.append({
            "utterance": item["utterance"],
            "strategy": f"constrained:{item['constraint']}",
        })
    print(f"  → {len(results) - before} variants")

    # Strategy 3: MLM substitution
    print(f"Generating MLM substitution variants (n={n_per_strategy})...")
    before = len(results)
    for utt in generate_mlm_variants(seed, n_per_strategy):
        results.append({"utterance": utt, "strategy": "mlm_substitution"})
    print(f"  → {len(results) - before} variants")

    # Strategy 4: Semantic expansion (optional)
    if include_expansion:
        n_expansion = max(5, n_per_strategy // 4)
        print(f"Generating semantic expansion variants (n={n_expansion})...")
        before = len(results)
        for utt in generate_semantic_expansion(seed, n_expansion, client, temperature=temperature):
            results.append({"utterance": utt, "strategy": "expansion"})
        print(f"  → {len(results) - before} variants")

    print(f"\nTotal before deduplication and scoring: {len(results)}")
    return results