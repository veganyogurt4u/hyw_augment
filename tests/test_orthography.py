"""Tests for the orthography converter (orthography.py)."""

from hyw_augment.orthography import (
    OrthographyConverter,
    _FlexRule,
    _parse_char_rules,
    _parse_flex_rules,
    _parse_flex_side,
)

# ── _parse_flex_side ──────────────────────────────────────────────────────────

def test_parse_flex_side_plus_prefix():
    """'+ABC' means strip nothing, add 'ABC'."""
    strip, add = _parse_flex_side("+ABC")
    assert strip == ""
    assert add == "ABC"


def test_parse_flex_side_minus_plus():
    """-XY+ABC' means strip 'XY', add 'ABC'."""
    strip, add = _parse_flex_side("-XY+ABC")
    assert strip == "XY"
    assert add == "ABC"


def test_parse_flex_side_pipe_separator():
    """-XY|+ABC' means strip 'XY', add 'ABC'."""
    strip, add = _parse_flex_side("-XY|+ABC")
    assert strip == "XY"
    assert add == "ABC"


def test_parse_flex_side_minus_only():
    """-XY' (no add part) means strip 'XY', add nothing."""
    strip, add = _parse_flex_side("-XY")
    assert strip == "XY"
    assert add == ""


def test_parse_flex_side_bracket_notation():
    """Bracket notation returns empty strings (handled separately)."""
    strip, add = _parse_flex_side("[X]+")
    assert strip == ""
    assert add == ""


def test_parse_flex_side_strips_trailing_digits():
    """Trailing digits (Hunspell flag refs) should be stripped before parsing."""
    strip, add = _parse_flex_side("+ABC123")
    assert strip == ""
    assert add == "ABC"


# ── _parse_flex_rules ─────────────────────────────────────────────────────────

def test_parse_flex_rules_basic():
    lines = ["-suffix+restore:-cls_strip+cls_add\n"]
    rules = _parse_flex_rules(lines)
    assert len(rules) == 1
    r = rules[0]
    assert isinstance(r, _FlexRule)


def test_parse_flex_rules_skips_bracket_lines():
    lines = [
        "[X]+:[Y]+\n",  # char class rule → skip
        "+restore:+add\n",  # valid suffix rule
    ]
    rules = _parse_flex_rules(lines)
    assert len(rules) == 1


def test_parse_flex_rules_skips_lines_without_colon():
    lines = ["no colon here\n", "+a:+b\n"]
    rules = _parse_flex_rules(lines)
    assert len(rules) == 1


def test_parse_flex_rules_skips_empty_lines():
    lines = ["", "\n", "   \n", "+a:+b"]
    rules = _parse_flex_rules(lines)
    assert len(rules) == 1


def test_parse_flex_rules_skips_zero_suffix_and_restore():
    """A rule where both ref_suffix and ref_restore are empty should be skipped."""
    lines = ["[X]+:[Y]+\n"]  # bracket rule → empty parse → skipped
    rules = _parse_flex_rules(lines)
    assert len(rules) == 0


def test_parse_flex_rules_multiple():
    lines = [
        "+a:+b\n",
        "-xy+z:-pq+r\n",
        "[bracket]+:[bracket2]+\n",  # skipped
        "+c:+d\n",
    ]
    rules = _parse_flex_rules(lines)
    assert len(rules) == 3


# ── _parse_char_rules ─────────────────────────────────────────────────────────

def test_parse_char_rules_matches():
    lines = ["[X]+:[Y]+\n", "[a]+:[b]+\n"]
    rules = _parse_char_rules(lines)
    assert len(rules) == 2
    assert rules[0] == ("X", "Y")
    assert rules[1] == ("a", "b")


def test_parse_char_rules_ignores_non_bracket():
    lines = ["+abc:+def\n", "[X]+:[Y]+\n"]
    rules = _parse_char_rules(lines)
    assert len(rules) == 1
    assert rules[0] == ("X", "Y")


# ── OrthographyConverter (with manually injected maps) ───────────────────────

def _make_converter(
    rc_lex_map: dict | None = None,
    rc_flex_rules: list | None = None,
    rc_char_rules: list | None = None,
    rc_exceptions: set | None = None,
) -> OrthographyConverter:
    """Create an OrthographyConverter with test data instead of real files."""
    # Pass a nonexistent dir — _load() checks .exists() before reading,
    # so all maps stay empty, then we override with test data.
    conv = OrthographyConverter("/nonexistent")
    conv.rc_lex_map = rc_lex_map or {}
    conv.rc_flex_rules = rc_flex_rules or []
    conv.rc_char_rules = rc_char_rules or []
    conv.rc_exceptions = rc_exceptions or set()
    return conv


# convert_word ─────────────────────────────────────────────────────────────────

def test_convert_word_direct_lookup():
    conv = _make_converter(rc_lex_map={"ref": "cls"})
    assert conv.convert_word("ref") == "cls"


def test_convert_word_unknown_unchanged():
    conv = _make_converter(rc_lex_map={"ref": "cls"})
    assert conv.convert_word("unknown") == "unknown"


def test_convert_word_empty_string():
    conv = _make_converter()
    assert conv.convert_word("") == ""


def test_convert_word_exception_skipped():
    conv = _make_converter(
        rc_lex_map={"ref": "cls"},
        rc_exceptions={"ref"},
    )
    # Exception words are returned unchanged even if in the lex map
    assert conv.convert_word("ref") == "ref"


def test_convert_word_case_insensitive_lowercase():
    """Lookup falls through to lowercase; capitalization is restored."""
    conv = _make_converter(rc_lex_map={"ref": "cls"})
    result = conv.convert_word("Ref")
    # "Ref".lower() → "ref" → "cls" → "Cls"
    assert result == "Cls"


def test_convert_word_case_insensitive_all_caps():
    conv = _make_converter(rc_lex_map={"ref": "cls"})
    result = conv.convert_word("REF")
    # "REF".lower() → "ref" → "cls" → "Cls"
    assert result == "Cls"


def test_convert_word_flex_rule_applied():
    """Test the suffix rule path: reformed inflected → classical inflected."""
    # reformed base "ref" → classical "cls"
    # flex rule: ref word ending "-ed" → reformed base; classical base has "-s" suffix
    rule = _FlexRule(
        ref_suffix="ed",      # word ends with "ed"
        ref_restore="",       # stripping "ed" gives the base directly
        cls_strip="",         # classical base unchanged
        cls_suffix="s",       # append "s" to get classical inflected form
    )
    conv = _make_converter(
        rc_lex_map={"ref": "cls"},
        rc_flex_rules=[rule],
    )
    result = conv.convert_word("refed")
    # "refed" ends with "ed" → base "ref" → classical "cls" + "s" = "clss"
    assert result == "clss"


def test_convert_word_flex_rule_with_cls_strip():
    """Classical base may need stripping before adding suffix."""
    rule = _FlexRule(
        ref_suffix="ed",
        ref_restore="",
        cls_strip="s",   # strip trailing 's' from classical base
        cls_suffix="z",  # then add 'z'
    )
    conv = _make_converter(
        rc_lex_map={"ref": "clss"},  # classical base ends with 's'
        rc_flex_rules=[rule],
    )
    result = conv.convert_word("refed")
    # base "ref" → cls "clss" → strip "s" → "cls" + "z" → "clsz"
    assert result == "clsz"


def test_convert_word_flex_rule_cls_strip_mismatch_skips():
    """If classical base doesn't end with cls_strip, rule is skipped."""
    rule = _FlexRule(
        ref_suffix="ed",
        ref_restore="",
        cls_strip="XYZ",  # classical base does NOT end with this
        cls_suffix="z",
    )
    conv = _make_converter(
        rc_lex_map={"ref": "cls"},
        rc_flex_rules=[rule],
    )
    result = conv.convert_word("refed")
    # Rule skipped → no flex result → char rules → no char rules → unchanged
    assert result == "refed"


def test_convert_word_char_rule_applied():
    """Character-class replacement is the last resort."""
    conv = _make_converter(rc_char_rules=[("x", "y")])
    # "fox" → "foy" via char replacement
    assert conv.convert_word("fox") == "foy"


def test_convert_word_char_rule_not_applied_if_lex_matches():
    """Char rule not applied if lex map already converts the word."""
    conv = _make_converter(
        rc_lex_map={"fox": "baz"},
        rc_char_rules=[("x", "y")],
    )
    # Direct lex lookup wins
    assert conv.convert_word("fox") == "baz"


# convert_text ─────────────────────────────────────────────────────────────────

def test_convert_text_single_word():
    conv = _make_converter(rc_lex_map={"ref": "cls"})
    assert conv.convert_text("ref") == "cls"


def test_convert_text_multiple_words():
    conv = _make_converter(rc_lex_map={"foo": "bar", "baz": "qux"})
    assert conv.convert_text("foo baz") == "bar qux"


def test_convert_text_mixed_known_unknown():
    conv = _make_converter(rc_lex_map={"ref": "cls"})
    assert conv.convert_text("ref unknown") == "cls unknown"


def test_convert_text_preserves_punctuation():
    conv = _make_converter(rc_lex_map={"ref": "cls"})
    assert conv.convert_text("ref, ref!") == "cls, cls!"


def test_convert_text_empty():
    conv = _make_converter()
    assert conv.convert_text("") == ""


def test_convert_text_preserves_whitespace():
    conv = _make_converter(rc_lex_map={"a": "b"})
    # Tabs and multi-spaces should survive
    result = conv.convert_text("a  a")
    assert result == "b  b"


# is_reformed ──────────────────────────────────────────────────────────────────

def test_is_reformed_true_when_different():
    conv = _make_converter(rc_lex_map={"ref": "cls"})
    assert conv.is_reformed("ref") is True


def test_is_reformed_false_when_same():
    conv = _make_converter()  # no mappings
    assert conv.is_reformed("unchanged") is False


def test_is_reformed_false_when_maps_to_itself():
    conv = _make_converter(rc_lex_map={"same": "same"})
    assert conv.is_reformed("same") is False


# detect_reformed_words ────────────────────────────────────────────────────────

def test_detect_reformed_words_finds_pair():
    conv = _make_converter(rc_lex_map={"ref": "cls"})
    pairs = conv.detect_reformed_words("ref and something")
    assert ("ref", "cls") in pairs


def test_detect_reformed_words_no_duplicates():
    conv = _make_converter(rc_lex_map={"ref": "cls"})
    pairs = conv.detect_reformed_words("ref ref ref")
    # "ref" appears three times but should only be reported once
    assert len(pairs) == 1


def test_detect_reformed_words_empty_text():
    conv = _make_converter(rc_lex_map={"ref": "cls"})
    assert conv.detect_reformed_words("") == []


def test_detect_reformed_words_no_reformed():
    conv = _make_converter()
    assert conv.detect_reformed_words("no reformed words here") == []


# Integration test (requires HySpell data) ────────────────────────────────────

def test_orthography_converter_from_real_files(hyspell_dir):
    conv = OrthographyConverter(hyspell_dir)
    assert len(conv.rc_lex_map) > 100_000
    assert len(conv.rc_flex_rules) > 50
