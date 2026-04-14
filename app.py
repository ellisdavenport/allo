"""
app.py — Streamlit interface for allo

Run with:
    streamlit run app.py
"""

import pandas as pd
import streamlit as st

from allo.generate import LLMClient, generate_variants
from allo.evaluate import score_variants
from allo.output import write_csv


st.set_page_config(
    page_title="allo",
    page_icon="🌿",
    layout="wide",
)

st.title("allo")
st.markdown(
    "allo generates natural language utterance variants from a seed utterance using four complementary strategies " \
    "(LLM paraphrasing, constrained syntactic rewriting, contextual MLM substitution, and semantic expansion),"
    "and scores each variant for semantic similarity and fluency." \
    "Results are exported as a scored CSV ready for human review and ingestion into a training pipeline."
)
st.markdown("---")


with st.sidebar:
    st.header("Settings")

    seed = st.text_input(
        label="Seed utterance",
        placeholder="e.g. turn off the lights",
        help="The utterance you want to generate variants from.",
    )

    n_per_strategy = st.slider(
        label="Variants per strategy",
        min_value=1,
        max_value=100,
        value=10,
        help=(
            "Controls output volume per strategy. "
            "LLM paraphrasing and semantic expansion scale linearly with this value. "
            "Constrained rewriting generates up to 3 variants per structural transform "
            "regardless of this setting (capped for quality). "
            "MLM substitution is bounded by utterance length and plateaus around 30. "
            "See the README for expected totals at common values."
        ),
    )

    temperature = st.slider(
        label="Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.9,
        step=0.05,
        help=(
            "Controls LLM output diversity for paraphrasing and expansion. "
            "Constrained rewriting always uses a lower fixed temperature (0.4) "
            "to enforce structural compliance regardless of this setting."
        ),
    )

    include_expansion = st.toggle(
        label="Include semantic expansion",
        value=True,
        help=(
            "Semantic expansion generates utterances that are related but meaningfully "
            "distinct (e.g. 'dim the lights' from a 'turn off the lights' seed). "
            "Enable for large batches. Disable if you want strictly synonymous paraphrases only. "
            "Expansion variants will have lower semantic similarity scores "
            "than pure paraphrases. Review them carefully before ingesting into training data."
        ),
    )

    with st.expander("Advanced filters (optional)"):
        st.markdown(
            "By default all variants are returned with their scores. "
            "Enable filters below to discard variants that fall outside your thresholds. "
            "Note: expansion variants have lower similarity by design- "
            "if using expansion, set your minimum similarity threshold accordingly."
        )

        use_sim_filter = st.checkbox("Filter by minimum semantic similarity")
        filter_min_similarity = None
        if use_sim_filter:
            filter_min_similarity = st.slider(
                "Minimum semantic similarity",
                min_value=0.0,
                max_value=1.0,
                value=0.55 if include_expansion else 0.7,
                step=0.05,
                help=(
                    "With expansion enabled, 0.55–0.65 is a reasonable floor. "
                    "Without expansion, 0.70–0.80 is typical."
                ),
            )

        use_ppl_filter = st.checkbox("Filter by maximum perplexity")
        filter_max_perplexity = None
        if use_ppl_filter:
            filter_max_perplexity = st.slider(
                "Maximum perplexity",
                min_value=50,
                max_value=1000,
                value=300,
                step=50,
            )

    st.markdown("---")
    generate_clicked = st.button(
        "Generate",
        type="primary",
        width='stretch',
        disabled=not seed.strip(),
    )


if generate_clicked:
    if not seed.strip():
        st.error("Please enter a seed utterance before generating.")
    else:
        with st.status("Running allo pipeline...", expanded=True) as status:
            try:
                st.write("Initialising LLM client...")
                client = LLMClient()

                st.write("Generating variants...")
                variants = generate_variants(
                    seed=seed.strip(),
                    n_per_strategy=n_per_strategy,
                    client=client,
                    temperature=temperature,
                    include_expansion=include_expansion,
                )

                st.write("Scoring variants...")
                scored, summary = score_variants(
                    seed=seed.strip(),
                    variants=variants,
                    filter_min_similarity=filter_min_similarity,
                    filter_max_perplexity=filter_max_perplexity,
                )

                st.write("Writing CSV output...")
                filepath = write_csv(
                    scored_variants=scored,
                    summary=summary,
                )

                st.session_state["scored"] = scored
                st.session_state["summary"] = summary
                st.session_state["filepath"] = filepath
                st.session_state["seed"] = seed.strip()

                status.update(label="Pipeline complete.", state="complete")

            except Exception as e:
                status.update(label="Pipeline failed.", state="error")
                st.error(f"An error occurred: {e}")
                st.stop()


if "scored" in st.session_state:
    scored = st.session_state["scored"]
    summary = st.session_state["summary"]
    filepath = st.session_state["filepath"]
    seed_display = st.session_state["seed"]

    st.subheader(f"Results for: \"{seed_display}\"")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Variants", summary["total_variants"])
    col2.metric("Mean Similarity", summary["mean_semantic_similarity"])
    col3.metric("Mean Perplexity", summary["mean_perplexity"])
    col4.metric("Lexical Diversity (TTR)", summary["lexical_diversity_ttr"])

    st.markdown("---")

    df = pd.DataFrame(scored)
    df = df[["utterance", "strategy", "semantic_similarity", "perplexity"]]

    # ── Strategy filter ───────────────────────────────────────────────────
    # At scale the results table gets long. Let the user filter by strategy
    # so they can review expansion variants separately from paraphrases.
    all_strategies = sorted(df["strategy"].unique().tolist())
    selected_strategies = st.multiselect(
        "Filter by strategy",
        options=all_strategies,
        default=all_strategies,
        help=(
            "Filter the table to specific strategies. "
            "'constrained:*' entries show the specific structural transform applied. "
            "'expansion' variants have lower similarity by design — review separately."
        ),
    )
    if selected_strategies:
        df = df[df["strategy"].isin(selected_strategies)]

    st.dataframe(
        df,
        width='stretch',
        column_config={
            "utterance": st.column_config.TextColumn("Utterance", width="large"),
            "strategy": st.column_config.TextColumn("Strategy", width="medium"),
            "semantic_similarity": st.column_config.NumberColumn(
                "Semantic Similarity",
                format="%.4f",
                width="small",
                help="Cosine similarity to seed (0–1). Higher = closer in meaning. Expansion variants intentionally score lower.",
            ),
            "perplexity": st.column_config.NumberColumn(
                "Perplexity",
                format="%.2f",
                width="small",
                help="GPT-2 perplexity. Lower = more natural sounding.",
            ),
        },
        hide_index=True,
    )

    st.markdown("---")

    with open(filepath, "rb") as f:
        csv_bytes = f.read()

    st.download_button(
        label="Download CSV",
        data=csv_bytes,
        file_name=filepath.split("/")[-1],
        mime="text/csv",
        type="primary",
    )