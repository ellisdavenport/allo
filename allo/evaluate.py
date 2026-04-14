"""
evaluate.py — scoring and deduplication for generated utterance variants

Metrics:
    - Semantic similarity: cosine similarity between seed and variant embeddings
    - Perplexity: how natural/fluent the variant sounds (lower = more natural)
    - Lexical diversity: type-token ratio across the full generated batch
    - Normalization: punctuation stripping and whitespace normalization
    - Deduplication: exact duplicate removal before output

Scores are presented as-is in the output. Exact duplicates are the only variants silently removed,
as they have no value as distinct training examples.
Optional hard filtering is available via parameters on score_variants() for users who want it.
"""

import math
import re
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────

# These models are downloaded once on first use and cached locally.
# SentenceTransformer: produces dense vector embeddings for semantic comparison.
# GPT2: used as a proxy for naturalness via perplexity scoring.
# Both are loaded at module level so they're only initialized once per session,
# not once per function call.

print("Loading embedding model...")
_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading language model for perplexity scoring...")
_lm_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
_lm_model = GPT2LMHeadModel.from_pretrained("gpt2")
_lm_model.eval()  # set to inference mode — disables dropout, saves compute

# ─────────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────────

def normalize_utterance(text: str) -> str:
    """
    Normalize a generated utterance for NLU training data consistency.

    Steps:
        1. Strip leading numbering artifacts (e.g. "1.", "1)", "- ")
           that LLMs may occasionally produce
        2. Strip leading and trailing quotation marks
        3. Strip all hyphens, en dashes, and em dashes (–, —, -)
           These are generation artifacts in short conversational utterances.
           Dashed constructions like "turn off the lights — it's getting late"
           are reduced to "turn off the lights  it's getting late" and then
           whitespace-normalized in the final step.
        4. Strip terminal punctuation: periods, question marks, exclamation points
        5. Strip all commas: punctuation carries no semantic value in NLU
           utterances and creates inconsistent tokenization
        6. Normalize whitespace: collapse multiple spaces produced by
           dash removal and strip leading/trailing whitespace

    Apostrophes are intentionally preserved; they are morphologically
    significant in contractions (don't, I'd, I've, can't) which are
    common in conversational NLU utterances.
    """
    # Step 1: strip leading numbering artifacts
    # Matches: "1. ", "1) ", "- ", "* "
    text = re.sub(r"^\s*(\d+[\.\)]\s+|[-\*]\s+)", "", text)

    # Step 2: strip leading and trailing quotation marks
    # Covers straight quotes, curly quotes, and double variants
    text = re.sub(r'^["\u201c\u201d\u2018\u2019]+|["\u201c\u201d\u2018\u2019]+$', "", text)

    # Step 3: strip hyphens, en dashes, em dashes
    text = re.sub(r"[-\u2013\u2014]", " ", text)

    # Step 4: strip terminal punctuation
    text = re.sub(r"[.?!]+$", "", text)

    # Step 5: strip commas
    text = re.sub(r",", "", text)

    # Step 6: normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

# ─────────────────────────────────────────────
# METRIC 1: SEMANTIC SIMILARITY
# ─────────────────────────────────────────────

def semantic_similarity(seed: str, variant: str) -> float:
    """
    Compute cosine similarity between the sentence embeddings of seed and variant.

    Cosine similarity measures the angle between two vectors in embedding space.
    A score of 1.0 means identical meaning; 0.0 means completely unrelated.

    We use the all-MiniLM-L6-v2 model for its lightweight and accurate sentence transformer.
    """
    embeddings = _embedding_model.encode([seed, variant], convert_to_tensor=True)
    score = util.cos_sim(embeddings[0], embeddings[1]).item()
    return round(score, 4)


# ─────────────────────────────────────────────
# METRIC 2: PERPLEXITY
# ─────────────────────────────────────────────

def perplexity(text: str) -> float:
    """
    Compute perplexity of a text string using GPT-2.

    Perplexity measures how "surprised" a language model is by a sequence.
    A well-trained model will assign low perplexity to natural, fluent text
    and high perplexity to awkward, ungrammatical, or unnatural text.

    Lower perplexity = more natural sounding.

    GPT-2 is used here because it is small, fast, open-weight, and does
    not require an API call. Perplexity scoring runs locally.

    Note: perplexity is sensitive to text length. Shorter utterances tend
    to score higher than longer ones even when equally natural, so scores
    are best interpreted relative to other variants of the same seed rather
    than as absolute thresholds.
    """
    encodings = _lm_tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids

    with torch.no_grad():
        outputs = _lm_model(input_ids, labels=input_ids)
        # outputs.loss is the mean negative log-likelihood per token
        loss = outputs.loss

    # Perplexity is the exponent of the mean NLL loss
    ppl = math.exp(loss.item())
    return round(ppl, 2)


# ─────────────────────────────────────────────
# METRIC 3: LEXICAL DIVERSITY
# ─────────────────────────────────────────────

def lexical_diversity(utterances: list[str]) -> float:
    """
    Compute the type-token ratio (TTR) across a list of utterances.

    TTR = unique words (types) / total words (tokens)

    A TTR of 1.0 means every word in the batch is unique — maximum diversity.
    A TTR close to 0.0 means the batch is highly repetitive.

    This is a batch-level metric, not a per-variant metric. It tells you
    whether the generation strategies are collectively producing varied
    output or clustering around the same vocabulary.

    Note: TTR is sensitive to batch size — larger batches naturally produce
    lower TTR scores even when output is varied. Interpret relative to batch
    size rather than as an absolute quality threshold.
    """
    all_tokens = []
    for utt in utterances:
        # Lowercase and tokenize by whitespace for a simple, fast word count
        tokens = re.findall(r"\b\w+\b", utt.lower())
        all_tokens.extend(tokens)

    if not all_tokens:
        return 0.0

    ttr = len(set(all_tokens)) / len(all_tokens)
    return round(ttr, 4)


# ─────────────────────────────────────────────
# METRIC 4: DEDUPLICATION
# ─────────────────────────────────────────────

def deduplicate(seed: str, variants: list[dict]) -> list[dict]:
    """
    Remove exact duplicates from the variant list.

    The seed itself is included in the seen set,
    so variants that reproduce the seed verbatim are removed.

    Near-duplicates are intentionally kept, since their semantic similarity
    scores in the output let the user decide whether to keep or discard them
    for their specific use case.

    Returns a deduplicated list of variant dicts.
    """
    seen_texts = {seed.strip().lower()}
    filtered = []

    for variant in variants:
        normalized = variant["utterance"].strip()
        if normalized not in seen_texts:
            seen_texts.add(normalized)
            filtered.append(variant)

    return filtered


# ─────────────────────────────────────────────
# UNIFIED SCORING FUNCTION
# ─────────────────────────────────────────────

def score_variants(
    seed: str,
    variants: list[dict],
    filter_min_similarity: float = None,
    filter_max_perplexity: float = None,
) -> tuple[list[dict], dict]:
    """
    Score and annotate a list of generated variants.

    Pipeline:
        1. Lowercase and normalize utterances
        2. Deduplicate (exact duplicates only)
        3. Score each variant for semantic similarity and perplexity,
           optionally discarding variants that fall outside thresholds
        4. Compute batch-level lexical diversity
        5. Return annotated variants and a summary dict

    By default no filtering is applied — all variants are returned with
    their scores so the user can decide what to keep. Optional hard
    filtering is available via filter_min_similarity and filter_max_perplexity
    for users who want the tool to pre-screen output.

    Parameters:
        seed: the original input utterance
        variants: list of dicts from generate.py, each with 'utterance' and 'strategy'
        filter_min_similarity: if set, discard variants below this semantic similarity score
        filter_max_perplexity: if set, discard variants above this perplexity score

    Returns:
        scored_variants: list of dicts with added 'semantic_similarity' and 'perplexity' fields
        summary: dict with batch-level stats (count, mean scores, lexical diversity)
    """
    print(f"\nScoring {len(variants)} variants...")

    # Step 1: lowercase and normalize all utterances before deduplication
    # so that casing and punctuation differences don't produce false negatives.
    for variant in variants:
        variant["utterance"] = normalize_utterance(variant["utterance"].lower())

    # Step 2: exact deduplication
    variants = deduplicate(seed, variants)
    print(f"  After deduplication: {len(variants)} variants remaining")

    # Step 3: score each variant, with optional filtering
    scored = []
    for variant in variants:
        text = variant["utterance"]
        sim = semantic_similarity(seed, text)
        ppl = perplexity(text)

        # Optional hard filtering — only applied if thresholds are explicitly set
        if filter_min_similarity is not None and sim < filter_min_similarity:
            print(f"  FILTERED (similarity {sim} < {filter_min_similarity}): '{text}'")
            continue
        if filter_max_perplexity is not None and ppl > filter_max_perplexity:
            print(f"  FILTERED (perplexity {ppl} > {filter_max_perplexity}): '{text}'")
            continue

        scored.append({
            **variant,
            "semantic_similarity": sim,
            "perplexity": ppl,
        })

    # Step 4: batch-level lexical diversity
    utterance_texts = [v["utterance"] for v in scored]
    ttr = lexical_diversity(utterance_texts) if utterance_texts else 0.0

    # Step 5: summary stats
    if scored:
        mean_similarity = round(
            sum(v["semantic_similarity"] for v in scored) / len(scored), 4
        )
        mean_perplexity = round(
            sum(v["perplexity"] for v in scored) / len(scored), 2
        )
    else:
        mean_similarity = 0.0
        mean_perplexity = 0.0

    summary = {
        "seed": seed,
        "total_variants": len(scored),
        "mean_semantic_similarity": mean_similarity,
        "mean_perplexity": mean_perplexity,
        "lexical_diversity_ttr": ttr,
    }

    print(f"  Scoring complete: {len(scored)} variants")
    print(f"  Mean similarity: {mean_similarity} | Mean perplexity: {mean_perplexity} | TTR: {ttr}")

    return scored, summary