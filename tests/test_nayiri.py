"""Tests for the Nayiri lexicon parser (nayiri.py)."""

from pathlib import Path

import pytest

from hyw_augment.nayiri import Lexicon, MorphAnalysis

# ── In-memory fixture ─────────────────────────────────────────────────────────

# Self-contained minimal lexicon dict (no file I/O needed).
# Uses ASCII placeholder strings for surface forms to keep them portable.
# Inflection IDs are arbitrary; case/number values use the real enum strings.
FIXTURE = {
    "inflections": [
        {
            "inflectionId": "NOM_SG",
            "lemmaType": "NOMINAL",
            "displayName": {"hy": "nom-sg", "en": "Nominative Singular"},
            "grammaticalNumber": "SINGULAR",
            "grammaticalCase": "NOMINATIVE",
        },
        {
            "inflectionId": "ACC_SG",
            "lemmaType": "NOMINAL",
            "displayName": {"hy": "acc-sg", "en": "Accusative Singular"},
            "grammaticalNumber": "SINGULAR",
            "grammaticalCase": "ACCUSATIVE",
        },
        {
            "inflectionId": "ABL_SG",
            "lemmaType": "NOMINAL",
            "displayName": {"hy": "abl-sg", "en": "Ablative Singular"},
            "grammaticalNumber": "SINGULAR",
            "grammaticalCase": "ABLATIVE",
        },
        {
            "inflectionId": "NOM_PL",
            "lemmaType": "NOMINAL",
            "displayName": {"hy": "nom-pl", "en": "Nominative Plural"},
            "grammaticalNumber": "PLURAL",
            "grammaticalCase": "NOMINATIVE",
        },
        {
            "inflectionId": "VERB_PRES_1SG",
            "lemmaType": "VERBAL",
            "displayName": {"hy": "pres-ind-1sg", "en": "Present Tense Indicative 1st Singular"},
            "grammaticalNumber": "SINGULAR",
            "grammaticalPerson": "FIRST",
            "verbTense": "SIMPLE_PRESENT",
            "verbMood": "INDICATIVE",
            "verbPolarity": "POSITIVE",
        },
    ],
    "lexemes": [
        {
            "lexemeId": "LEX1",
            "lemmas": [
                {
                    "lemmaId": "LEX1A",
                    "lemmaString": "noun-alpha",
                    "partOfSpeech": "NOUN",
                    "wordForms": [
                        {"s": "noun-alpha", "i": "NOM_SG"},
                        {"s": "noun-alpha-acc", "i": "ACC_SG"},
                        {"s": "noun-alpha-abl", "i": "ABL_SG"},
                        {"s": "noun-alphas", "i": "NOM_PL"},
                    ],
                }
            ],
        },
        {
            "lexemeId": "LEX2",
            "lemmas": [
                {
                    "lemmaId": "LEX2A",
                    "lemmaString": "verb-beta",
                    "partOfSpeech": "VERB",
                    "wordForms": [
                        {"s": "verb-beta-pres", "i": "VERB_PRES_1SG"},
                    ],
                }
            ],
        },
        {
            "lexemeId": "LEX3",
            "lemmas": [
                {
                    "lemmaId": "LEX3A",
                    "lemmaString": "noun-gamma",
                    "partOfSpeech": "NOUN",
                    "wordForms": [
                        {"s": "noun-gamma", "i": "NOM_SG"},
                    ],
                }
            ],
        },
    ],
}

# A second smaller fixture for merge testing
FIXTURE2 = {
    "inflections": [
        {
            "inflectionId": "NOM_SG",  # same ID, same data — safe to merge
            "lemmaType": "NOMINAL",
            "displayName": {"hy": "nom-sg", "en": "Nominative Singular"},
            "grammaticalNumber": "SINGULAR",
            "grammaticalCase": "NOMINATIVE",
        },
    ],
    "lexemes": [
        {
            "lexemeId": "LEX4",
            "lemmas": [
                {
                    "lemmaId": "LEX4A",
                    "lemmaString": "noun-delta",
                    "partOfSpeech": "NOUN",
                    "wordForms": [
                        {"s": "noun-delta", "i": "NOM_SG"},
                    ],
                }
            ],
        },
    ],
}


def _find_data(filename: str) -> Path | None:
    for base in [Path("data"), Path("../data")]:
        p = base / filename
        if p.exists():
            return p
    return None


# ── Loading ───────────────────────────────────────────────────────────────────

def test_from_dict_num_lexemes():
    lex = Lexicon.from_dict(FIXTURE)
    assert lex.num_lexemes == 3


def test_from_dict_inflections_loaded():
    lex = Lexicon.from_dict(FIXTURE)
    assert len(lex.inflections) == 5
    assert "NOM_SG" in lex.inflections
    assert "VERB_PRES_1SG" in lex.inflections


def test_from_dict_form_index_populated():
    lex = Lexicon.from_dict(FIXTURE)
    assert "noun-alpha" in lex.form_index
    assert "noun-alpha-acc" in lex.form_index
    assert "verb-beta-pres" in lex.form_index
    assert "noun-gamma" in lex.form_index


def test_from_dict_lemma_index_populated():
    lex = Lexicon.from_dict(FIXTURE)
    assert "noun-alpha" in lex.lemma_index
    assert "verb-beta" in lex.lemma_index
    assert "noun-gamma" in lex.lemma_index


def test_from_dict_word_form_count():
    lex = Lexicon.from_dict(FIXTURE)
    assert lex.num_word_forms == 6  # 4 + 1 + 1


# ── analyze ───────────────────────────────────────────────────────────────────

def test_analyze_returns_analysis_for_known_form():
    lex = Lexicon.from_dict(FIXTURE)
    results = lex.analyze("noun-alpha")
    assert len(results) == 1
    a = results[0]
    assert isinstance(a, MorphAnalysis)
    assert a.lemma == "noun-alpha"
    assert a.pos == "NOUN"


def test_analyze_returns_empty_for_unknown():
    lex = Lexicon.from_dict(FIXTURE)
    results = lex.analyze("nonexistent-word")
    assert results == []


def test_analyze_inflected_form():
    lex = Lexicon.from_dict(FIXTURE)
    results = lex.analyze("noun-alpha-abl")
    assert len(results) == 1
    assert results[0].case == "ABLATIVE"
    assert results[0].number == "SINGULAR"


def test_analyze_plural_form():
    lex = Lexicon.from_dict(FIXTURE)
    results = lex.analyze("noun-alphas")
    assert len(results) == 1
    assert results[0].number == "PLURAL"


def test_analyze_verb_form():
    lex = Lexicon.from_dict(FIXTURE)
    results = lex.analyze("verb-beta-pres")
    assert len(results) == 1
    a = results[0]
    assert a.pos == "VERB"
    assert a.tense == "SIMPLE_PRESENT"
    assert a.mood == "INDICATIVE"
    assert a.polarity == "POSITIVE"
    assert a.person == "FIRST"


# ── analyze_insensitive ───────────────────────────────────────────────────────

def test_analyze_case_insensitive_real():
    """Indexer stores lowercase variant for forms that have uppercase."""
    fixture_with_caps = {
        "inflections": FIXTURE["inflections"],
        "lexemes": [
            {
                "lexemeId": "LEXX",
                "lemmas": [{
                    "lemmaId": "LEXXA",
                    "lemmaString": "Noun-Cap",
                    "partOfSpeech": "NOUN",
                    "wordForms": [{"s": "Noun-Cap", "i": "NOM_SG"}],
                }],
            }
        ],
    }
    lex = Lexicon.from_dict(fixture_with_caps)
    # Direct lookup
    assert len(lex.analyze("Noun-Cap")) == 1
    # Lowercase variant also indexed
    assert len(lex.analyze("noun-cap")) == 1


def test_analyze_insensitive_returns_list():
    lex = Lexicon.from_dict(FIXTURE)
    results = lex.analyze_insensitive("NOUN-ALPHA")
    assert isinstance(results, list)


# ── is_valid_form ─────────────────────────────────────────────────────────────

def test_is_valid_form_known():
    lex = Lexicon.from_dict(FIXTURE)
    assert lex.is_valid_form("noun-alpha") is True
    assert lex.is_valid_form("verb-beta-pres") is True


def test_is_valid_form_unknown():
    lex = Lexicon.from_dict(FIXTURE)
    assert lex.is_valid_form("totally-unknown") is False


# ── generate ─────────────────────────────────────────────────────────────────

def test_generate_wildcard_returns_all_forms():
    lex = Lexicon.from_dict(FIXTURE)
    results = lex.generate("noun-alpha")
    assert len(results) == 4
    surfaces = {s for s, _ in results}
    assert "noun-alpha" in surfaces
    assert "noun-alpha-abl" in surfaces
    assert "noun-alphas" in surfaces


def test_generate_filtered_by_case():
    lex = Lexicon.from_dict(FIXTURE)
    results = lex.generate("noun-alpha", case="ABLATIVE")
    assert len(results) == 1
    surface, inf = results[0]
    assert surface == "noun-alpha-abl"
    assert inf.case == "ABLATIVE"


def test_generate_filtered_by_number():
    lex = Lexicon.from_dict(FIXTURE)
    results = lex.generate("noun-alpha", number="PLURAL")
    assert len(results) == 1
    assert results[0][0] == "noun-alphas"


def test_generate_filtered_by_case_and_number():
    lex = Lexicon.from_dict(FIXTURE)
    results = lex.generate("noun-alpha", case="NOMINATIVE", number="SINGULAR")
    assert len(results) == 1
    assert results[0][0] == "noun-alpha"


def test_generate_unknown_lemma_returns_empty():
    lex = Lexicon.from_dict(FIXTURE)
    results = lex.generate("does-not-exist")
    assert results == []


def test_generate_no_match_returns_empty():
    lex = Lexicon.from_dict(FIXTURE)
    # noun-alpha has no DATIVE forms in the fixture
    results = lex.generate("noun-alpha", case="DATIVE")
    assert results == []


# ── MorphAnalysis properties ──────────────────────────────────────────────────

def test_morph_analysis_description_en():
    lex = Lexicon.from_dict(FIXTURE)
    a = lex.analyze("noun-alpha")[0]
    assert "Nominative" in a.description_en


def test_morph_analysis_properties_nominal():
    lex = Lexicon.from_dict(FIXTURE)
    a = lex.analyze("noun-alpha-acc")[0]
    assert a.case == "ACCUSATIVE"
    assert a.number == "SINGULAR"
    assert a.person is None
    assert a.tense is None
    assert a.mood is None
    assert a.polarity is None


def test_morph_analysis_properties_verbal():
    lex = Lexicon.from_dict(FIXTURE)
    a = lex.analyze("verb-beta-pres")[0]
    assert a.case is None
    assert a.person == "FIRST"
    assert a.tense == "SIMPLE_PRESENT"


def test_morph_analysis_repr():
    lex = Lexicon.from_dict(FIXTURE)
    a = lex.analyze("noun-alpha")[0]
    r = repr(a)
    assert "noun-alpha" in r
    assert "NOUN" in r


# ── merge ─────────────────────────────────────────────────────────────────────

def test_merge_combines_forms():
    lex1 = Lexicon.from_dict(FIXTURE)
    lex2 = Lexicon.from_dict(FIXTURE2)
    lex1.merge(lex2)
    # Original forms still present
    assert lex1.is_valid_form("noun-alpha")
    # New forms from lex2
    assert lex1.is_valid_form("noun-delta")


def test_merge_updates_counts():
    lex1 = Lexicon.from_dict(FIXTURE)
    n_lexemes_before = lex1.num_lexemes
    lex2 = Lexicon.from_dict(FIXTURE2)
    lex1.merge(lex2)
    assert lex1.num_lexemes == n_lexemes_before + lex2.num_lexemes


def test_from_files_merges(tmp_path):
    import json
    f1 = tmp_path / "lex1.json"
    f2 = tmp_path / "lex2.json"
    f1.write_text(json.dumps(FIXTURE), encoding="utf-8")
    f2.write_text(json.dumps(FIXTURE2), encoding="utf-8")

    combined = Lexicon.from_files(str(f1), str(f2))
    assert combined.is_valid_form("noun-alpha")
    assert combined.is_valid_form("noun-delta")
    assert combined.num_lexemes == 4


# ── all_forms / all_lemmas / lemmas_for_pos ───────────────────────────────────

def test_all_forms_non_empty():
    lex = Lexicon.from_dict(FIXTURE)
    forms = list(lex.all_forms())
    assert "noun-alpha" in forms
    assert len(forms) >= 6


def test_all_lemmas():
    lex = Lexicon.from_dict(FIXTURE)
    lemmas = set(lex.all_lemmas())
    assert "noun-alpha" in lemmas
    assert "verb-beta" in lemmas


def test_lemmas_for_pos_noun():
    lex = Lexicon.from_dict(FIXTURE)
    nouns = lex.lemmas_for_pos("NOUN")
    assert "noun-alpha" in nouns
    assert "noun-gamma" in nouns
    assert "verb-beta" not in nouns


def test_lemmas_for_pos_verb():
    lex = Lexicon.from_dict(FIXTURE)
    verbs = lex.lemmas_for_pos("VERB")
    assert "verb-beta" in verbs
    assert "noun-alpha" not in verbs


# ── summary ───────────────────────────────────────────────────────────────────

def test_summary_contains_counts():
    lex = Lexicon.from_dict(FIXTURE)
    s = lex.summary()
    assert "Lexemes" in s
    assert "Word forms" in s
    assert "NOUN" in s


# ── Integration tests (skip if data files absent) ────────────────────────────

def test_lexicon_full_file_loads():
    p = _find_data("nayiri-armenian-lexicon-2026-02-15-v1.json")
    if p is None:
        pytest.skip("Full Nayiri lexicon not found")
    lex = Lexicon.from_file(p)
    assert lex.num_lexemes > 7000
    assert lex.num_word_forms > 1_000_000
    assert len(lex.form_index) > 100_000


def test_lexicon_sample_roundtrip():
    p = _find_data("nayiri-armenian-lexicon-2026-02-15-v1-sample.json")
    if p is None:
        pytest.skip("Sample Nayiri lexicon not found")
    lex = Lexicon.from_file(p)
    # Every form in the index should analyze back to at least one result
    for form in list(lex.form_index.keys())[:100]:
        assert lex.analyze(form), f"Form {form!r} should analyze"
