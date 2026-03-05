"""Tests for the glossary module (glossary.py)."""

import pytest

from hyw_augment.glossary import _POS_MAP, Glossary, GlossaryEntry

# Armenian POS abbreviations used below are taken directly from _POS_MAP keys
# and written as \uXXXX escapes to avoid script-mixing bugs in source
# (useful for llms, which seem to do token prediction on two-bit unicode in a way that causes rendering errors) :
#   \u0563.                  (NOUN,        Armenian: գ.)
#   \u0561\u056e.            (ADJECTIVE,   Armenian: ած.)
#   \u0576\u0580\u0563.      (VERB_TR,     Armenian: Նրգ.)
#   \u0579\u0566.            (VERB_INTR,   Armenian: չզ.)
#   \u0589                   (Armenian full stop ։)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _glossary_from_lines(*lines: str) -> Glossary:
    """Build a Glossary from raw text lines without touching the filesystem."""
    g = Glossary()
    for line in lines:
        g._parse_line(line)
    return g


# Convenient constants so tests don't repeat the escape codes
NOUN     = "\u0563."
ADJ      = "\u0561\u056e."
VERB_TR  = "\u0576\u0580\u0563."
VERB_INT = "\u0579\u0566."
ARM_STOP = "\u0589"   # ։ Armenian full stop


# ── GlossaryEntry properties ──────────────────────────────────────────────────

def test_entry_is_transitive_verb_tr():
    e = GlossaryEntry(headword="w", pos="VERB_TR", pos_raw="x", definition="d")
    assert e.is_transitive is True


def test_entry_is_transitive_verb_intr():
    e = GlossaryEntry(headword="w", pos="VERB_INTR", pos_raw="x", definition="d")
    assert e.is_transitive is False


def test_entry_is_transitive_verb_refl():
    e = GlossaryEntry(headword="w", pos="VERB_REFL", pos_raw="x", definition="d")
    assert e.is_transitive is False


def test_entry_is_transitive_verb_mid():
    e = GlossaryEntry(headword="w", pos="VERB_MID", pos_raw="x", definition="d")
    assert e.is_transitive is False


def test_entry_is_transitive_noun_returns_none():
    e = GlossaryEntry(headword="w", pos="NOUN", pos_raw="x", definition="d")
    assert e.is_transitive is None


# ── _parse_line: single POS entries ──────────────────────────────────────────

def test_parse_line_noun():
    g = _glossary_from_lines(f"headword {NOUN} a definition here")
    assert "headword" in g.entries
    entry = g.entries["headword"][0]
    assert entry.pos == "NOUN"
    assert entry.definition == "a definition here"


def test_parse_line_adjective():
    g = _glossary_from_lines(f"word {ADJ} quality of being X")
    assert "word" in g.entries
    assert g.entries["word"][0].pos == "ADJECTIVE"


def test_parse_line_transitive_verb():
    g = _glossary_from_lines(f"word {VERB_TR} to do something")
    entry = g.entries["word"][0]
    assert entry.pos == "VERB_TR"
    assert entry.is_transitive is True


def test_parse_line_intransitive_verb():
    g = _glossary_from_lines(f"word {VERB_INT} to happen")
    entry = g.entries["word"][0]
    assert entry.pos == "VERB_INTR"
    assert entry.is_transitive is False


def test_parse_line_strips_armenian_full_stop():
    g = _glossary_from_lines(f"word {NOUN} a definition{ARM_STOP}")
    entry = g.entries["word"][0]
    assert not entry.definition.endswith(ARM_STOP)
    assert entry.definition == "a definition"


def test_parse_line_pos_raw_preserved():
    g = _glossary_from_lines(f"word {NOUN} a definition")
    entry = g.entries["word"][0]
    assert entry.pos_raw == NOUN


def test_parse_line_headword_preserved():
    g = _glossary_from_lines(f"myword {NOUN} some def")
    entry = g.entries["myword"][0]
    assert entry.headword == "myword"


def test_parse_line_no_pos_skipped():
    g = _glossary_from_lines("justoneword")
    assert "justoneword" not in g.entries


def test_parse_line_empty_skipped():
    g = _glossary_from_lines("")
    assert len(g.entries) == 0


# ── Multi-POS entries (semicolon separator) ───────────────────────────────────

def test_parse_line_multi_pos():
    g = _glossary_from_lines(f"word {ADJ} first meaning; {NOUN} second meaning")
    assert "word" in g.entries
    entries = g.entries["word"]
    assert len(entries) == 2
    pos_set = {e.pos for e in entries}
    assert "ADJECTIVE" in pos_set
    assert "NOUN" in pos_set


def test_parse_line_multi_pos_definitions_separate():
    g = _glossary_from_lines(f"word {NOUN} noun def; {VERB_TR} verb def")
    by_pos = {e.pos: e.definition for e in g.entries["word"]}
    assert by_pos["NOUN"] == "noun def"
    assert by_pos["VERB_TR"] == "verb def"


# ── Glossary.lookup ───────────────────────────────────────────────────────────

def test_lookup_found():
    g = _glossary_from_lines(f"alpha {NOUN} a definition")
    result = g.lookup("alpha")
    assert result is not None
    assert len(result) == 1
    assert result[0].pos == "NOUN"


def test_lookup_not_found_returns_none():
    g = _glossary_from_lines(f"alpha {NOUN} a definition")
    assert g.lookup("beta") is None


def test_lookup_case_insensitive():
    # lookup() lowercases the input as a fallback, so an uppercase input finds
    # a lowercase-keyed headword
    g = _glossary_from_lines(f"alpha {NOUN} a definition")
    result = g.lookup("ALPHA")
    assert result is not None


def test_lookup_returns_all_entries_for_word():
    g = _glossary_from_lines(f"myword {NOUN} noun def; {VERB_TR} verb def")
    result = g.lookup("myword")
    assert result is not None
    assert len(result) == 2


# ── Glossary._total counter ───────────────────────────────────────────────────

def test_total_counter():
    g = _glossary_from_lines(
        f"word1 {NOUN} def",
        f"word2 {NOUN} def",
        f"word3 {NOUN} def1; {ADJ} def2",  # two entries for word3
    )
    assert g._total == 4


# ── POS map coverage ──────────────────────────────────────────────────────────

def test_pos_map_entries_recognized():
    """All POS abbreviations in _POS_MAP should parse without falling through."""
    for abbr, expected_pos in _POS_MAP.items():
        g = _glossary_from_lines(f"testword {abbr} some definition")
        assert "testword" in g.entries, f"POS {abbr!r} should be recognized"
        entries = g.entries.pop("testword")
        assert entries[0].pos == expected_pos


# ── from_file (skip if HySpell not installed) ─────────────────────────────────

def test_glossary_from_real_file(hyspell_dir):
    path = hyspell_dir / "SmallArmDic.txt"
    if not path.exists():
        pytest.skip("SmallArmDic.txt not found in HySpell directory")
    g = Glossary.from_file(path)
    assert len(g.entries) > 10_000
    assert g._total > 10_000
