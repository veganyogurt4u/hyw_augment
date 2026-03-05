"""Tests for the Calfa lexical-databases integration (calfa.py)."""

import pytest

from hyw_augment.calfa import _POS_MAP, CaLFAEntry, CaLFALexicon, _primary_pos

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_entry(**kwargs) -> CaLFAEntry:
    defaults = dict(
        headword="word", complement="", pos="NOUN", pos_raw="s.", definition_en="a thing"
    )
    defaults.update(kwargs)
    return CaLFAEntry(**defaults)


def _lexicon_with(entries: dict, synonyms: dict | None = None) -> CaLFALexicon:
    """Build a CaLFALexicon with pre-populated data (no file I/O)."""
    lex = CaLFALexicon()
    lex.entries = entries
    lex._total = sum(len(v) for v in entries.values())
    if synonyms:
        lex._synonyms = synonyms
    return lex


# ── _primary_pos ──────────────────────────────────────────────────────────────

def test_primary_pos_noun():
    assert _primary_pos("s.") == "NOUN"


def test_primary_pos_adjective():
    assert _primary_pos("adj.") == "ADJECTIVE"


def test_primary_pos_verb():
    assert _primary_pos("v.") == "VERB"


def test_primary_pos_adverb():
    assert _primary_pos("adv.") == "ADVERB"


def test_primary_pos_interjection():
    assert _primary_pos("int.") == "INTERJECTION"


def test_primary_pos_compound_uses_first():
    """Compound POS like "s. adv." should use the first recognized token."""
    assert _primary_pos("s. adv.") == "NOUN"


def test_primary_pos_compound_adv_first():
    assert _primary_pos("adv. adj.") == "ADVERB"


def test_primary_pos_unknown_passthrough():
    """An unrecognized POS abbreviation should be returned as-is."""
    assert _primary_pos("xyz.") == "xyz."


# ── _POS_MAP coverage ─────────────────────────────────────────────────────────

def test_pos_map_all_values_are_strings():
    for abbr, pos in _POS_MAP.items():
        assert isinstance(abbr, str)
        assert isinstance(pos, str)
        assert pos.isupper(), f"Expected uppercase POS for {abbr!r}, got {pos!r}"


# ── CaLFAEntry ────────────────────────────────────────────────────────────────

def test_entry_fields():
    e = _make_entry(headword="test", complement="ու", pos="NOUN",
                    pos_raw="s.", definition_en="a test")
    assert e.headword == "test"
    assert e.complement == "ու"
    assert e.pos == "NOUN"
    assert e.pos_raw == "s."
    assert e.definition_en == "a test"


# ── CaLFALexicon.lookup ───────────────────────────────────────────────────────

def test_lookup_found():
    e = _make_entry(headword="alpha")
    lex = _lexicon_with({"alpha": [e]})
    result = lex.lookup("alpha")
    assert result is not None
    assert len(result) == 1
    assert result[0].headword == "alpha"


def test_lookup_not_found_returns_none():
    lex = _lexicon_with({"alpha": [_make_entry()]})
    assert lex.lookup("beta") is None


def test_lookup_case_fallback_lowercase():
    """lookup() falls back to word.lower() if exact key not found."""
    e = _make_entry(headword="word")
    lex = _lexicon_with({"word": [e]})
    result = lex.lookup("WORD")
    assert result is not None


def test_lookup_exact_wins_over_lowercase():
    """If exact key exists, it is returned (no fallback needed)."""
    e = _make_entry(headword="Word")
    lex = _lexicon_with({"Word": [e]})
    result = lex.lookup("Word")
    assert result is not None
    assert result[0].headword == "Word"


def test_lookup_returns_all_entries():
    """A headword may have multiple entries (multiple POS)."""
    e1 = _make_entry(pos="NOUN")
    e2 = _make_entry(pos="VERB")
    lex = _lexicon_with({"word": [e1, e2]})
    result = lex.lookup("word")
    assert result is not None
    assert len(result) == 2
    pos_set = {e.pos for e in result}
    assert "NOUN" in pos_set
    assert "VERB" in pos_set


# ── CaLFALexicon.synonyms_for ─────────────────────────────────────────────────

def test_synonyms_for_found():
    lex = _lexicon_with({}, synonyms={"WORD": ["syn1", "syn2"]})
    result = lex.synonyms_for("WORD")
    assert result == ["syn1", "syn2"]


def test_synonyms_for_uppercase_fallback():
    """synonyms_for() falls back to word.upper() if exact key not found."""
    lex = _lexicon_with({}, synonyms={"WORD": ["syn1"]})
    result = lex.synonyms_for("word")
    assert result == ["syn1"]


def test_synonyms_for_not_found_returns_empty():
    lex = _lexicon_with({})
    assert lex.synonyms_for("unknown") == []


def test_synonyms_for_empty_list_not_stored():
    """Words with no synonyms should not be in _synonyms at all."""
    lex = _lexicon_with({})
    assert lex.synonyms_for("any") == []


# ── CaLFALexicon._total ───────────────────────────────────────────────────────

def test_total_counts_all_entries():
    lex = _lexicon_with({
        "word1": [_make_entry()],
        "word2": [_make_entry(), _make_entry()],
    })
    assert lex._total == 3


# ── CaLFALexicon.summary ──────────────────────────────────────────────────────

def test_summary_contains_headword_count():
    lex = _lexicon_with({"a": [_make_entry()], "b": [_make_entry()]})
    s = lex.summary()
    assert "2" in s


def test_summary_contains_synonym_count():
    lex = _lexicon_with({}, synonyms={"A": ["x"], "B": ["y", "z"]})
    s = lex.summary()
    assert "2" in s


# ── CaLFALexicon._add_definition_row (parsing logic) ─────────────────────────

def test_add_definition_row_single_pos():
    lex = CaLFALexicon()
    # Simulate a parsed TSV row: Title, Complement, POS_1, Def_1, empty...
    row = ["Աբ", "", "s.", "a word", "", "", "", "", "", "", "", "", "", ""]
    lex._add_definition_row(row)
    assert "Աբ" in lex.entries
    assert lex.entries["Աբ"][0].pos == "NOUN"
    assert lex.entries["Աբ"][0].definition_en == "a word"


def test_add_definition_row_multiple_pos():
    lex = CaLFALexicon()
    row = ["word", "ու", "s.", "noun def", "adj.", "adj def", "", "", "", "", "", "", "", ""]
    lex._add_definition_row(row)
    assert len(lex.entries["word"]) == 2
    pos_set = {e.pos for e in lex.entries["word"]}
    assert "NOUN" in pos_set
    assert "ADJECTIVE" in pos_set


def test_add_definition_row_stops_at_empty_pos():
    lex = CaLFALexicon()
    # Only first POS is non-empty; second is empty → stop
    row = ["word", "", "v.", "verb def", "", "should not appear", "", "", "", "", "", "", "", ""]
    lex._add_definition_row(row)
    assert len(lex.entries["word"]) == 1


def test_add_definition_row_complement_stored():
    lex = CaLFALexicon()
    row = ["word", "ու", "s.", "def", "", "", "", "", "", "", "", "", "", ""]
    lex._add_definition_row(row)
    assert lex.entries["word"][0].complement == "ու"


def test_add_definition_row_pos_raw_stored():
    lex = CaLFALexicon()
    row = ["word", "", "s. adv.", "compound def", "", "", "", "", "", "", "", "", "", ""]
    lex._add_definition_row(row)
    assert lex.entries["word"][0].pos_raw == "s. adv."
    assert lex.entries["word"][0].pos == "NOUN"  # first token


# ── Integration test (requires real data) ────────────────────────────────────

@pytest.fixture
def calfa_dir(pytestconfig):
    from pathlib import Path
    candidates = [
        Path("/home/van/ml-work/arm_data/lexical-databases"),
        Path("../arm_data/lexical-databases"),
    ]
    for p in candidates:
        if (p / "definitions" / "en-definitions01.tsv").exists():
            return p
    return None


def test_calfa_from_real_dir(calfa_dir):
    if calfa_dir is None:
        pytest.skip("lexical-databases not found")
    lex = CaLFALexicon.from_dir(calfa_dir)
    assert len(lex.entries) > 10_000
    assert lex._total > 10_000
    assert len(lex._synonyms) > 1_000
