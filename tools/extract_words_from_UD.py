#!/usr/bin/env python3
"""
Extract function words from the UD treebank and emit Nayiri-compatible JSON.

This script:
1. Identifies lemmas that are missing from the Nayiri lexicon but frequent
   in the treebank (function words, auxiliaries, pronouns, postpositions, etc.)
2. Collects every observed surface form + UD morphological features for each
3. Converts UD feature bundles into Nayiri-style inflection objects
4. Outputs a JSON file in the same schema as the Nayiri lexicon

Usage:
    python -m hyw_augment.extract_function_words \
        --conllu data/*.conllu \
        --nayiri data/nayiri-armenian-lexicon-2026-02-15-v1.json \
        --output data/function-words.json \
        --min-freq 5

    # Then merge it:
    python -m hyw_augment.cli \
        --conllu data/*.conllu \
        --nayiri data/nayiri-armenian-lexicon-2026-02-15-v1.json \
                 data/function-words.json \
        --coverage
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from hyw_augment.conllu import Treebank, Token
from hyw_augment.nayiri import Lexicon


# ── UD features → Nayiri inflection mapping ───────────────────────────────────

UD_TO_NAYIRI_POS = {
    "NOUN": "NOUN",
    "PROPN": "NOUN",
    "VERB": "VERB",
    "AUX": "AUX",
    "ADJ": "ADJECTIVE",
    "ADV": "ADVERB",
    "ADP": "ADPOSITION",
    "DET": "DETERMINER",
    "PRON": "PRONOUN",
    "SCONJ": "CONJUNCTION",
    "CCONJ": "CONJUNCTION",
    "PART": "PARTICLE",
    "INTJ": "INTERJECTION",
    "NUM": "NUMERAL",
}

UD_TO_NAYIRI_CASE = {
    "Nom": "NOMINATIVE",
    "Acc": "ACCUSATIVE",
    "Gen": "GENITIVE",
    "Dat": "DATIVE",
    "Abl": "ABLATIVE",
    "Ins": "INSTRUMENTAL",
    "Loc": "LOCATIVE",
}

UD_TO_NAYIRI_NUMBER = {
    "Sing": "SINGULAR",
    "Plur": "PLURAL",
    "Coll": "COLLECTIVE",
}

UD_TO_NAYIRI_PERSON = {
    "1": "FIRST",
    "2": "SECOND",
    "3": "THIRD",
}

UD_TO_NAYIRI_TENSE = {
    "Pres": "SIMPLE_PRESENT",
    "Past": "SIMPLE_PAST",
    "Imp": "IMPERFECT",
}

UD_TO_NAYIRI_MOOD = {
    "Ind": "INDICATIVE",
    "Sub": "SUBJUNCTIVE",
    "Imp": "IMPERATIVE",
    "Cnd": "CONDITIONAL",
}

UD_TO_NAYIRI_POLARITY = {
    "Pos": "POSITIVE",
    "Neg": "NEGATIVE",
}


def _feats_to_inflection_key(tok: Token) -> str:
    """
    Build a stable string key from a token's UD features.
    Used to deduplicate inflections.
    """
    parts = []
    for feat_name in sorted(tok.feats.keys()):
        parts.append(f"{feat_name}={tok.feats[feat_name]}")
    # Include UPOS because same features on different POS = different inflection
    return f"{tok.upos}|{'|'.join(parts)}" if parts else f"{tok.upos}|_"


def _feats_to_display_name(tok: Token) -> dict[str, str]:
    """Build a human-readable display name from UD features."""
    parts_en = []

    polarity = tok.feat("Polarity")
    if polarity == "Neg":
        parts_en.append("(Negative)")

    tense = tok.feat("Tense")
    if tense:
        tense_map = {"Pres": "Present Tense", "Past": "Past Tense", "Imp": "Imperfect"}
        parts_en.append(tense_map.get(tense, tense))

    aspect = tok.feat("Aspect")
    if aspect:
        aspect_map = {"Imp": "Imperfective", "Perf": "Perfective", "Prosp": "Prospective"}
        parts_en.append(aspect_map.get(aspect, aspect))

    mood = tok.feat("Mood")
    if mood:
        mood_map = {"Ind": "Indicative", "Sub": "Subjunctive", "Imp": "Imperative", "Cnd": "Conditional"}
        parts_en.append(mood_map.get(mood, mood))

    verb_form = tok.feat("VerbForm")
    if verb_form:
        vf_map = {"Fin": "Finite", "Inf": "Infinitive", "Part": "Participle", "Conv": "Converb", "Gdv": "Gerundive"}
        parts_en.append(vf_map.get(verb_form, verb_form))

    person = tok.feat("Person")
    if person:
        p_map = {"1": "First Person", "2": "Second Person", "3": "Third Person"}
        parts_en.append(p_map.get(person, f"Person {person}"))

    number = tok.feat("Number")
    if number:
        n_map = {"Sing": "Singular", "Plur": "Plural", "Coll": "Collective"}
        parts_en.append(n_map.get(number, number))

    case = tok.feat("Case")
    if case:
        c_map = {
            "Nom": "Nominative", "Acc": "Accusative", "Gen": "Genitive",
            "Dat": "Dative", "Abl": "Ablative", "Ins": "Instrumental", "Loc": "Locative",
        }
        parts_en.append(c_map.get(case, case) + " case")

    definite = tok.feat("Definite")
    if definite:
        d_map = {"Def": "Definite", "Ind": "Indefinite"}
        parts_en.append(d_map.get(definite, definite))

    pron_type = tok.feat("PronType")
    if pron_type:
        pt_map = {
            "Prs": "Personal", "Dem": "Demonstrative", "Int": "Interrogative",
            "Rel": "Relative", "Ind": "Indefinite", "Tot": "Totalizing",
            "Art": "Article", "Neg": "Negative", "Rcp": "Reciprocal",
            "Exc": "Exclamative",
        }
        parts_en.append(pt_map.get(pron_type, pron_type))

    adp_type = tok.feat("AdpType")
    if adp_type:
        at_map = {"Post": "Postposition", "Prep": "Preposition"}
        parts_en.append(at_map.get(adp_type, adp_type))

    if not parts_en:
        parts_en.append(tok.upos)

    return {
        "hy": "",  # leave Armenian display names empty for now
        "en": " • ".join(parts_en),
    }


def _make_inflection_id(index: int) -> str:
    """Generate a FW-prefixed inflection ID."""
    return f"FW{index:04d}"


def _make_lexeme_id(lemma: str, pos: str) -> str:
    """Generate a stable lexeme ID from lemma + POS."""
    # Use a simple hash-based approach for stability
    key = f"{lemma}_{pos}"
    h = hash(key) % 0xFFFF
    return f"FW-{h:04X}"


def _make_lemma_id(lexeme_id: str) -> str:
    return f"{lexeme_id}-L"


def extract_function_words(
    treebank: Treebank,
    lexicon: Lexicon,
    *,
    min_freq: int = 3,
) -> dict:
    """
    Extract words missing from the lexicon and return Nayiri-format JSON.

    Args:
        treebank: Parsed UD treebank (all splits)
        lexicon: Loaded Nayiri lexicon
        min_freq: Minimum lemma frequency to include (default 3)
    """

    # Step 1: Find missing lemmas and their tokens
    # Group tokens by (lemma, upos) for tokens not found in lexicon
    missing_tokens: dict[tuple[str, str], list[Token]] = defaultdict(list)

    for sent in treebank:
        for tok in sent.real_tokens:
            if tok.upos in ("PUNCT", "SYM", "X"):
                continue
            analyses = lexicon.analyze(tok.form)
            if not analyses:
                analyses = lexicon.analyze_insensitive(tok.form)
            if not analyses:
                missing_tokens[(tok.lemma, tok.upos)].append(tok)

    # Step 2: Filter by frequency
    frequent_missing = {
        key: tokens
        for key, tokens in missing_tokens.items()
        if len(tokens) >= min_freq
    }

    print(
        f"Found {len(frequent_missing)} missing lemma+POS pairs "
        f"with >= {min_freq} occurrences",
        file=sys.stderr,
    )

    # Step 3: For each missing lemma, collect unique (form, feature-bundle) pairs
    inflection_registry: dict[str, dict] = {}  # feat_key → inflection dict
    inflection_key_to_id: dict[str, str] = {}  # feat_key → inflection ID
    inf_counter = 0

    lexemes = []

    for (lemma, upos), tokens in sorted(
        frequent_missing.items(), key=lambda x: -len(x[1])
    ):
        lexeme_id = _make_lexeme_id(lemma, upos)
        lemma_id = _make_lemma_id(lexeme_id)
        nayiri_pos = UD_TO_NAYIRI_POS.get(upos, upos)

        # Collect unique form → inflection mappings
        seen_form_inf: set[tuple[str, str]] = set()
        word_forms = []

        for tok in tokens:
            feat_key = _feats_to_inflection_key(tok)

            # Register inflection if new
            if feat_key not in inflection_key_to_id:
                inf_id = _make_inflection_id(inf_counter)
                inf_counter += 1
                inflection_key_to_id[feat_key] = inf_id

                # Determine lemma type for inflection
                if upos in ("VERB", "AUX"):
                    lemma_type = "VERBAL"
                elif upos in ("NOUN", "PROPN", "PRON", "DET", "NUM"):
                    lemma_type = "NOMINAL"
                else:
                    lemma_type = "UNINFLECTED"

                inflection_registry[feat_key] = {
                    "inflectionId": inf_id,
                    "lemmaType": lemma_type,
                    "displayName": _feats_to_display_name(tok),
                }

                # Add mapped fields
                case = tok.feat("Case")
                if case and case in UD_TO_NAYIRI_CASE:
                    inflection_registry[feat_key]["grammaticalCase"] = UD_TO_NAYIRI_CASE[case]

                number = tok.feat("Number")
                if number and number in UD_TO_NAYIRI_NUMBER:
                    inflection_registry[feat_key]["grammaticalNumber"] = UD_TO_NAYIRI_NUMBER[number]

                person = tok.feat("Person")
                if person and person in UD_TO_NAYIRI_PERSON:
                    inflection_registry[feat_key]["grammaticalPerson"] = UD_TO_NAYIRI_PERSON[person]

                tense = tok.feat("Tense")
                if tense and tense in UD_TO_NAYIRI_TENSE:
                    inflection_registry[feat_key]["verbTense"] = UD_TO_NAYIRI_TENSE[tense]

                mood = tok.feat("Mood")
                if mood and mood in UD_TO_NAYIRI_MOOD:
                    inflection_registry[feat_key]["verbMood"] = UD_TO_NAYIRI_MOOD[mood]

                polarity = tok.feat("Polarity")
                if polarity and polarity in UD_TO_NAYIRI_POLARITY:
                    inflection_registry[feat_key]["verbPolarity"] = UD_TO_NAYIRI_POLARITY[polarity]

            inf_id = inflection_key_to_id[feat_key]

            # Deduplicate (same form + same inflection)
            pair = (tok.form, inf_id)
            if pair not in seen_form_inf:
                seen_form_inf.add(pair)
                word_forms.append({"s": tok.form, "i": inf_id})

        lexeme = {
            "lexemeId": lexeme_id,
            "lemmas": [
                {
                    "lemmaId": lemma_id,
                    "lemmaString": lemma,
                    "partOfSpeech": nayiri_pos,
                    "numWordForms": len(word_forms),
                    "wordForms": word_forms,
                }
            ],
            "description": f"Extracted from UD treebank ({len(tokens)} occurrences, {upos})",
            "lemmaType": "FUNCTION_WORD",
        }
        lexemes.append(lexeme)

    # Step 4: Build the output JSON
    output = {
        "metadata": {
            "description": (
                "Supplementary function words extracted from the UD Western Armenian "
                "ArmTDP treebank. Covers items not present in the Nayiri Armenian Lexicon "
                "(pronouns, copula, auxiliaries, postpositions, determiners, etc.)."
            ),
            "source": "UD_Western_Armenian-ArmTDP",
            "generatedBy": "hyw_augment.extract_words_from_UD.py",
            "minFrequency": min_freq,
            "numLexemes": len(lexemes),
            "numInflections": len(inflection_registry),
        },
        "lexemes": lexemes,
        "inflections": list(inflection_registry.values()),
    }

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Extract words from UD treebank into Nayiri-format JSON"
    )
    parser.add_argument(
        "--conllu",
        nargs="+",
        required=True,
        help="Path(s) to .conllu treebank files",
    )
    parser.add_argument(
        "--nayiri",
        required=True,
        help="Path to Nayiri lexicon JSON",
    )
    parser.add_argument(
        "--output",
        default="data/function-words.json",
        help="Output path (default: data/function-words.json)",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=3,
        help="Minimum lemma frequency to include (default: 3)",
    )
    parser.add_argument(
        "--indent",
        action="store_true",
        help="Pretty-print the JSON output",
    )
    args = parser.parse_args()

    # Load data
    tb = Treebank.from_files(*[Path(p) for p in args.conllu])
    lex = Lexicon.from_file(args.nayiri)

    print(f"Treebank: {len(tb)} sentences, {tb.token_count} tokens", file=sys.stderr)
    print(f"Lexicon:  {lex.num_lexemes} lexemes, {lex.num_word_forms} forms", file=sys.stderr)

    # Extract
    result = extract_function_words(tb, lex, min_freq=args.min_freq)

    # Write
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2 if args.indent else None)

    total_forms = sum(
        lem["numWordForms"]
        for lexeme in result["lexemes"]
        for lem in lexeme["lemmas"]
    )
    print(
        f"\nWrote {out_path}:",
        f"\n  {result['metadata']['numLexemes']} lexemes",
        f"\n  {result['metadata']['numInflections']} inflections",
        f"\n  {total_forms} word forms",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
