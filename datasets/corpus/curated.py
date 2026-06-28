"""Curated fallacy and bias records for the LOGOS corpus."""

from __future__ import annotations

from datasets.corpus.schema import CorpusRecord, new_uuid_id
from datasets.corpus.split import estimate_token_count

FALLACY_CASES: list[dict[str, str]] = [
    {"name": "straw_man", "premise": "Atheists claim science proves God does not exist.", "explanation": "Misrepresents the atheist position; most argue lack of evidence, not scientific proof of absence."},
    {"name": "ad_hominem", "premise": "You cannot trust his argument about morality because he is divorced.", "explanation": "Attacks the person rather than the argument."},
    {"name": "appeal_to_authority", "premise": "God exists because Einstein believed in God.", "explanation": "Einstein's physics authority does not settle theological questions."},
    {"name": "false_dilemma", "premise": "Either the Bible is literally true or morality is impossible.", "explanation": "Ignores non-literal theist and secular moral frameworks."},
    {"name": "slippery_slope", "premise": "If we allow doubt about one miracle, all faith will collapse.", "explanation": "Assumes inevitable chain without justification."},
    {"name": "begging_the_question", "premise": "The Qur'an is true because it says it is from God.", "explanation": "Assumes what it tries to prove."},
    {"name": "appeal_to_emotion", "premise": "How dare you question God after your mother prayed for you?", "explanation": "Uses guilt instead of evidence."},
    {"name": "hasty_generalization", "premise": "One cruel priest proves religion corrupts everyone.", "explanation": "Generalizes from a single case."},
    {"name": "red_herring", "premise": "The problem of evil is irrelevant; look at church charity work.", "explanation": "Diverts from the logical challenge."},
    {"name": "tu_quoque", "premise": "You cannot criticize religious violence; atheists killed millions under communism.", "explanation": "Deflects by accusing the critic, not answering the point."},
    {"name": "appeal_to_ignorance", "premise": "No one has disproved my god, so my god exists.", "explanation": "Absence of disproof is not proof."},
    {"name": "circular_reasoning", "premise": "Scripture is reliable because God inspired it; we know God inspired it from scripture.", "explanation": "Premise and conclusion support each other without independent evidence."},
    {"name": "equivocation", "premise": "Love is a chemical reaction; therefore spiritual love is an illusion.", "explanation": "Shifts the meaning of 'love' mid-argument."},
    {"name": "composition", "premise": "Each part of the universe is contingent; therefore the whole must have a cause like parts do.", "explanation": "What holds for parts may not hold for the whole."},
    {"name": "division", "premise": "Society is religious; therefore every individual citizen is religious.", "explanation": "Attributes group property to each member."},
    {"name": "affirming_consequent", "premise": "If God exists, there is order; there is order; therefore God exists.", "explanation": "Order could have other explanations."},
    {"name": "denying_antecedent", "premise": "If you have faith, you will be healed; you were not healed; therefore you lacked faith.", "explanation": "Ignores other causes of failure to heal."},
    {"name": "post_hoc", "premise": "I prayed and recovered; therefore prayer caused recovery.", "explanation": "Confuses sequence with causation."},
    {"name": "false_cause", "premise": "Crime rose after church attendance fell; therefore secularism causes crime.", "explanation": "Correlation without controlled causation."},
    {"name": "appeal_to_tradition", "premise": "We have always worshipped this way; therefore it is true.", "explanation": "Longevity does not establish truth."},
    {"name": "appeal_to_popularity", "premise": "Billions believe in God; therefore God exists.", "explanation": "Belief count is not evidence of truth."},
    {"name": "no_true_scotsman", "premise": "No true Christian would commit violence.", "explanation": "Redefines the group to exclude counterexamples."},
    {"name": "genetic_fallacy", "premise": "The idea of karma came from ancient India; therefore it is false.", "explanation": "Origin does not determine truth value."},
    {"name": "loaded_question", "premise": "When did you stop ignoring God's obvious design?", "explanation": "Presupposes contested claims."},
    {"name": "burden_of_proof_shift", "premise": "You cannot disprove my invisible dragon; therefore it is rational to believe.", "explanation": "Shifts burden to the skeptic improperly."},
    {"name": "appeal_to_nature", "premise": "Homosexuality is unnatural; therefore it is wrong.", "explanation": "Naturalness does not entail moral status."},
    {"name": "appeal_to_consequences", "premise": "Free will must exist because otherwise we could not blame criminals.", "explanation": "Undesirable consequences do not prove a claim."},
    {"name": "false_analogy", "premise": "Universe is like a watch; watches have makers; therefore universe has a maker.", "explanation": "Analogy may not hold for cosmological scale."},
    {"name": "special_pleading", "premise": "God is uncaused, but nothing else can be uncaused.", "explanation": "Exempts preferred claim from own rules."},
    {"name": "moving_goalposts", "premise": "Every miracle claim refuted was not the right kind of evidence.", "explanation": "Standards change to avoid falsification."},
    {"name": "cherry_picking", "premise": "Only fulfilled prophecies count; failed ones are metaphor.", "explanation": "Selects confirming data only."},
    {"name": "anecdotal", "premise": "My uncle saw a vision; therefore visions prove God.", "explanation": "Single story cannot establish general truth."},
    {"name": "middle_ground", "premise": "Theism and atheism are both extreme; therefore agnosticism is correct.", "explanation": "Truth is not always between extremes."},
    {"name": "personal_incredulity", "premise": "I cannot imagine how mind arises from matter; therefore dualism is true.", "explanation": "Limits of imagination are not limits of reality."},
    {"name": "texas_sharpshooter", "premise": "Fine-tuning fits life so precisely someone aimed at it.", "explanation": "Selects targets after seeing results."},
    {"name": "appeal_to_mystery", "premise": "Evil is a mystery we must not question.", "explanation": "Labels difficulty as answer."},
    {"name": "complex_question", "premise": "Why does your faith make you hate science?", "explanation": "Bundles contested assumptions."},
    {"name": "faulty_analogy_god", "premise": "Father creates baby; Father in heaven creates universe.", "explanation": "Human fatherhood may not map to cosmology."},
    {"name": "straw_man_science", "premise": "Religion says God is a bearded man in the sky.", "explanation": "Caricatures sophisticated theism."},
    {"name": "appeal_to_fear", "premise": "Without God you will have no meaning and despair.", "explanation": "Uses fear instead of argument."},
    {"name": "bandwagon", "premise": "Everyone at my mosque agrees; therefore it is true.", "explanation": "Local consensus is not proof."},
    {"name": "false_equivalence", "premise": "Belief in gravity equals belief in resurrection.", "explanation": "Different evidence bases treated as identical."},
    {"name": "non_sequitur", "premise": "Quantum mechanics is weird; therefore consciousness creates reality.", "explanation": "Conclusion does not follow."},
    {"name": "accent_fallacy", "premise": "Scripture says 'thou shalt not kill' except in war.", "explanation": "Selective emphasis changes meaning."},
    {"name": "amphiboly", "premise": "The Bible says God is light; therefore physics proves God.", "explanation": "Ambiguous term exploited."},
    {"name": "fallacy_of_single_cause", "premise": "Decline in religion alone caused social decay.", "explanation": "Complex phenomena have many causes."},
    {"name": "retrospective_determinism", "premise": "History had to unfold this way because God willed it.", "explanation": "Reads inevitability backward."},
    {"name": "appeal_to_spite", "premise": "Reject God to spite religious hypocrites.", "explanation": "Spite is not a truth procedure."},
    {"name": "proof_by_verbosity", "premise": "A hundred pages of theology therefore proves God.", "explanation": "Length is not validity."},
    {"name": "gish_gallop", "premise": "Twenty quick objections; answer all in five minutes or I win.", "explanation": "Quantity overwhelms careful reply."},
    {"name": "etymological_fallacy", "premise": "Atman means breath; therefore soul is just breathing.", "explanation": "Original meaning may not fix current concept."},
    {"name": "moralistic_fallacy", "premise": "Karma must be true because otherwise injustice would go unpunished.", "explanation": "Wish does not make fact."},
    {"name": "naturalistic_fallacy", "premise": "Survival of fittest means might makes right.", "explanation": "Derives ought from is."},
]

BIAS_CASES: list[dict[str, str]] = [
    {"name": "confirmation_bias", "example": "Noticing only prayers that seem answered.", "remedy": "Track misses; pre-register predictions."},
    {"name": "anchoring", "example": "First cosmological argument heard shapes all later evaluation.", "remedy": "Steel opposing best case before scoring."},
    {"name": "availability_heuristic", "example": "Recent terrorist attack inflates religion-violence estimates.", "remedy": "Use base rates and long-run data."},
    {"name": "ingroup_bias", "example": "Trusting co-religionist testimony over outsider critique.", "remedy": "Blind source review where possible."},
    {"name": "outgroup_homogeneity", "example": "Treating all atheists as Dawkins clones.", "remedy": "Sample multiple schools within tradition."},
    {"name": "belief_perseverance", "example": "Keeping faith after decisive counter-evidence.", "remedy": "State what would change your mind."},
    {"name": "backfire_effect", "example": "Correcting myth hardens original belief.", "remedy": "Affirm identity before fact correction."},
    {"name": "motivated_reasoning", "example": "Finding flaws only in disliked traditions.", "remedy": "Apply same standards to own side."},
    {"name": "sunk_cost_fallacy", "example": "Staying in religion because years invested.", "remedy": "Evaluate forward, not past investment."},
    {"name": "halo_effect", "example": "Charismatic guru assumed correct on metaphysics.", "remedy": "Separate charisma from argument validity."},
    {"name": "authority_bias", "example": "Accepting fatwa without examining reasons.", "remedy": "Demand explicit argument chain."},
    {"name": "affect_heuristic", "example": "God feels real during music; therefore exists.", "remedy": "Distinguish emotion from evidence."},
    {"name": "illusory_correlation", "example": "Linking personal virtue to prosperity.", "remedy": "Look for disconfirming cases."},
    {"name": "fundamental_attribution_error", "example": "Blaming sin for others' suffering, situation for yours.", "remedy": "Consider structural causes."},
    {"name": "self_serving_bias", "example": "Credit God for success, blame self for failure only.", "remedy": "Symmetric attribution rules."},
    {"name": "just_world_hypothesis", "example": "Assuming victims must have deserved karma.", "remedy": "Study unjust outcomes explicitly."},
    {"name": "curse_of_knowledge", "example": "Apologist assumes listeners share theological vocabulary.", "remedy": "Define terms; check comprehension."},
    {"name": "dunning_kruger", "example": "Novice debater overconfident on cosmology.", "remedy": "Calibrate confidence; cite experts."},
    {"name": "optimism_bias", "example": "Assuming afterlife will surely be pleasant.", "remedy": "Stress-test worst-case doctrines."},
    {"name": "pessimism_bias", "example": "Assuming religion always leads to war.", "remedy": "Compare peacemaking traditions too."},
    {"name": "status_quo_bias", "example": "Keeping childhood faith unexamined.", "remedy": "Periodic structured re-evaluation."},
    {"name": "normalcy_bias", "example": "Ignoring rising extremism in own community.", "remedy": "Seek disconfirming local data."},
    {"name": "bandwagon_effect", "example": "Converting because peers did.", "remedy": "Separate social proof from truth."},
    {"name": "false_consensus", "example": "Assuming everyone shares your prayer habit.", "remedy": "Survey diverse samples."},
    {"name": "hostile_media_effect", "example": "Neutral report seen as attack on faith.", "remedy": "Rate bias blind on text alone."},
    {"name": "reactance", "example": "Doubling faith when challenged.", "remedy": "Invite self-paced inquiry."},
    {"name": "naive_realism", "example": "Assuming you see facts, they see ideology.", "remedy": "Model own interpretive frame."},
    {"name": "hindsight_bias", "example": "Saying prophecy obviously meant X after event.", "remedy": "Record predictions before outcomes."},
    {"name": "survivorship_bias", "example": "Only successful converts testimonies counted.", "remedy": "Include deconversion narratives."},
    {"name": "framing_effect", "example": "Pro-life vs pro-choice labels shift moral intuitions.", "remedy": "Rephrase in neutral language."},
    {"name": "omission_bias", "example": "Preferring harmful inaction to harmful action in trolley cases.", "remedy": "Compare outcomes, not defaults."},
    {"name": "scope_insensitivity", "example": "Same donation for one child vs thousands.", "remedy": "Use magnitude scaling explicitly."},
    {"name": "conjunction_fallacy", "example": "Detailed prophecy story feels more likely than base claim.", "remedy": "Compare probabilities formally."},
]


def build_fallacy_records() -> list[CorpusRecord]:
    """Build curated fallacy corpus rows."""
    records: list[CorpusRecord] = []
    for case in FALLACY_CASES:
        text = (
            f"Premise: {case['premise']}\n\n"
            f"Fallacy: {case['name'].replace('_', ' ').title()}\n\n"
            f"Explanation: {case['explanation']}"
        )
        records.append(
            CorpusRecord(
                id=new_uuid_id(),
                text=text,
                split="train",
                title="LOGOS Fallacy Corpus",
                author="LOGOS",
                year=2026,
                language="en",
                religion="none",
                tradition="logic",
                school="logic",
                topic=["fallacy", "logic"],
                stance="survey",
                source_type="essay",
                source_id="logos_fallacies",
                license="PD-US",
                token_count=estimate_token_count(text),
                mix_bucket="ethics_logic_psych",
            )
        )
    return records


def build_bias_records() -> list[CorpusRecord]:
    """Build curated moral-psychology bias rows."""
    records: list[CorpusRecord] = []
    for case in BIAS_CASES:
        text = (
            f"Bias: {case['name'].replace('_', ' ').title()}\n\n"
            f"Example: {case['example']}\n\n"
            f"Remedy: {case['remedy']}"
        )
        records.append(
            CorpusRecord(
                id=new_uuid_id(),
                text=text,
                split="train",
                title="LOGOS Bias Corpus",
                author="LOGOS",
                year=2026,
                language="en",
                religion="none",
                tradition="moral_psychology",
                school="moral_psychology",
                topic=["epistemology"],
                stance="survey",
                source_type="essay",
                source_id="logos_biases",
                license="PD-US",
                token_count=estimate_token_count(text),
                mix_bucket="ethics_logic_psych",
            )
        )
    return records
