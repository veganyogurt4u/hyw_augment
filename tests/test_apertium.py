"""Tests for apertium.py — parsing logic only, no hfst-lookup binary needed."""

import pytest

from hyw_augment.apertium import ApertiumAnalysis, ApertiumAnalyzer, _inflection_from_tags

# ── Fixture helper ────────────────────────────────────────────────────────────

def _analyzer() -> ApertiumAnalyzer:
    """Create an ApertiumAnalyzer pointing at a nonexistent dir.

    available=False, but the parsing methods (_parse_lines,
    _parse_batch_output, _parse_analysis_string) don't check availability.
    """
    return ApertiumAnalyzer("/nonexistent")


def _make_analysis(tags: list[str], *, form: str = "form", lemma: str = "lemma") -> ApertiumAnalysis:
    return ApertiumAnalysis(form=form, lemma=lemma, tags=tags, raw="raw")


# ── ApertiumAnalysis properties ───────────────────────────────────────────────

def test_analysis_pos_noun():
    assert _make_analysis(["n", "sg", "nom"]).pos == "NOUN"


def test_analysis_pos_verb():
    assert _make_analysis(["v", "pres", "indc"]).pos == "VERB"


def test_analysis_pos_adjective():
    assert _make_analysis(["adj", "sg"]).pos == "ADJECTIVE"


def test_analysis_pos_proper_noun_maps_to_noun():
    # np (proper noun) maps to NOUN in _APT_POS_MAP
    assert _make_analysis(["np", "ant", "sg"]).pos == "NOUN"


def test_analysis_pos_unknown_when_no_pos_tag():
    # Tags with no POS entry fall back to UNKNOWN
    assert _make_analysis(["sg", "nom"]).pos == "UNKNOWN"


def test_analysis_case_nominative():
    assert _make_analysis(["n", "sg", "nom"]).case == "NOMINATIVE"


def test_analysis_case_ablative():
    assert _make_analysis(["n", "pl", "abl", "def"]).case == "ABLATIVE"


def test_analysis_case_none_for_verb():
    assert _make_analysis(["v", "pres", "indc", "p3", "sg"]).case is None


def test_analysis_number_singular():
    assert _make_analysis(["n", "sg", "nom"]).number == "SINGULAR"


def test_analysis_number_plural():
    assert _make_analysis(["n", "pl", "nom"]).number == "PLURAL"


def test_analysis_number_none_if_absent():
    assert _make_analysis(["v", "pres"]).number is None


def test_analysis_person_first():
    assert _make_analysis(["v", "pres", "p1", "sg"]).person == "FIRST"


def test_analysis_person_third():
    assert _make_analysis(["v", "past", "p3", "pl"]).person == "THIRD"


def test_analysis_person_none_for_noun():
    assert _make_analysis(["n", "sg", "nom"]).person is None


def test_analysis_article_definite():
    assert _make_analysis(["n", "sg", "nom", "def"]).article == "DEFINITE"


def test_analysis_article_indefinite():
    assert _make_analysis(["n", "sg", "nom", "indef"]).article == "INDEFINITE"


def test_analysis_article_none_if_absent():
    assert _make_analysis(["n", "sg", "nom"]).article is None


def test_analysis_is_proper_noun_true():
    assert _make_analysis(["np", "ant", "sg"]).is_proper_noun is True


def test_analysis_is_proper_noun_false():
    assert _make_analysis(["n", "sg", "nom"]).is_proper_noun is False


def test_analysis_description_en_known_tags():
    desc = _make_analysis(["n", "sg"]).description_en
    assert "noun" in desc
    assert "singular" in desc


def test_analysis_description_en_unknown_tag_falls_back_to_tag():
    assert "xyzzy" in _make_analysis(["xyzzy"]).description_en


def test_analysis_description_en_multiple_tags_comma_joined():
    desc = _make_analysis(["n", "sg", "nom"]).description_en
    assert ", " in desc


def test_analysis_repr_contains_form_lemma_and_tags():
    a = _make_analysis(["n", "sg"], form="word", lemma="root")
    r = repr(a)
    assert "word" in r
    assert "root" in r
    assert "<n>" in r


# ── _parse_analysis_string ────────────────────────────────────────────────────

def test_parse_analysis_string_with_tags():
    result = ApertiumAnalyzer._parse_analysis_string("lemma<n><sg><nom>")
    assert result is not None
    lemma, tags = result
    assert lemma == "lemma"
    assert tags == ["n", "sg", "nom"]


def test_parse_analysis_string_no_tags():
    result = ApertiumAnalyzer._parse_analysis_string("lemma")
    assert result is not None
    lemma, tags = result
    assert lemma == "lemma"
    assert tags == []


def test_parse_analysis_string_many_tags():
    result = ApertiumAnalyzer._parse_analysis_string("stem<v><pres><indc><p3><pl>")
    assert result is not None
    assert result[1] == ["v", "pres", "indc", "p3", "pl"]


def test_parse_analysis_string_plus_question_parsed_as_tagless_lemma():
    # _parse_analysis_string is pure regex — it does NOT filter +?.
    # The +? check happens one level up in _parse_lines / _parse_batch_output.
    result = ApertiumAnalyzer._parse_analysis_string("word+?")
    assert result is not None
    lemma, tags = result
    assert lemma == "word+?"
    assert tags == []


def test_parse_analysis_string_empty_returns_none():
    assert ApertiumAnalyzer._parse_analysis_string("") is None


# ── _parse_lines ──────────────────────────────────────────────────────────────

def test_parse_lines_single_analysis():
    apt = _analyzer()
    results = apt._parse_lines("word", ["word\tlemma<n><sg><nom>\t0.0"])
    assert len(results) == 1
    a = results[0]
    assert a.form == "word"
    assert a.lemma == "lemma"
    assert a.tags == ["n", "sg", "nom"]
    assert a.weight == 0.0


def test_parse_lines_multiple_analyses_for_same_form():
    apt = _analyzer()
    lines = [
        "word\tlemma<n><sg><nom>\t0.0",
        "word\tlemma<adj><sg>\t1.0",
    ]
    results = apt._parse_lines("word", lines)
    assert len(results) == 2
    assert results[0].tags == ["n", "sg", "nom"]
    assert results[1].tags == ["adj", "sg"]


def test_parse_lines_unknown_skipped():
    apt = _analyzer()
    # +? means the transducer has no analysis
    assert apt._parse_lines("word", ["word\tword+?\t∞"]) == []


def test_parse_lines_no_tab_skipped():
    apt = _analyzer()
    assert apt._parse_lines("word", ["word"]) == []


def test_parse_lines_weight_parsed():
    apt = _analyzer()
    results = apt._parse_lines("word", ["word\tlemma<n><sg>\t3.14"])
    assert results[0].weight == pytest.approx(3.14)


def test_parse_lines_invalid_weight_defaults_to_zero():
    apt = _analyzer()
    results = apt._parse_lines("word", ["word\tlemma<n><sg>\tnot-a-float"])
    assert results[0].weight == 0.0


def test_parse_lines_empty_input():
    assert _analyzer()._parse_lines("word", []) == []


# ── _parse_batch_output ───────────────────────────────────────────────────────

def test_parse_batch_output_single_form():
    results = _analyzer()._parse_batch_output("word\tlemma<n><sg><nom>\t0.0\n")
    assert "word" in results
    assert len(results["word"]) == 1
    assert results["word"][0].lemma == "lemma"


def test_parse_batch_output_multiple_forms():
    output = "w1\tlemma1<n><sg>\t0.0\nw2\tlemma2<v><pres>\t0.0\n"
    results = _analyzer()._parse_batch_output(output)
    assert results["w1"][0].pos == "NOUN"
    assert results["w2"][0].pos == "VERB"


def test_parse_batch_output_multiple_analyses_same_form():
    output = "word\tlemma<n><sg>\t0.0\nword\tlemma<adj><sg>\t1.0\n"
    assert len(_analyzer()._parse_batch_output(output)["word"]) == 2


def test_parse_batch_output_unknown_form_gives_empty_list():
    # Unknown forms still appear in the dict, but with an empty list
    results = _analyzer()._parse_batch_output("word\tword+?\t∞\n")
    assert "word" in results
    assert results["word"] == []


def test_parse_batch_output_empty_string():
    assert _analyzer()._parse_batch_output("") == {}


def test_parse_batch_output_ignores_blank_lines():
    output = "\n\nword\tlemma<n><sg>\t0.0\n\n"
    results = _analyzer()._parse_batch_output(output)
    assert "word" in results


# ── _inflection_from_tags ────────────────────────────────────────────────────

def test_inflection_from_tags_noun():
    inf = _inflection_from_tags(["n", "pl", "abl", "def"])
    assert inf.lemma_type == "NOMINAL"
    assert inf.case == "ABLATIVE"
    assert inf.number == "PLURAL"
    assert inf.article == "DEFINITE"
    assert inf.raw_tags == ["n", "pl", "abl", "def"]
    assert inf.inflection_id == "apt:n|pl|abl|def"


def test_inflection_from_tags_verb():
    inf = _inflection_from_tags(["v", "pres", "indc", "p1", "sg"])
    assert inf.lemma_type == "VERBAL"
    assert inf.person == "FIRST"
    assert inf.number == "SINGULAR"
    assert inf.case is None
    assert inf.raw_tags == ["v", "pres", "indc", "p1", "sg"]


def test_inflection_from_tags_adjective():
    inf = _inflection_from_tags(["adj", "sg"])
    assert inf.lemma_type == "NOMINAL"
    assert inf.number == "SINGULAR"


def test_inflection_from_tags_uninflected():
    inf = _inflection_from_tags(["adv"])
    assert inf.lemma_type == "UNINFLECTED"


def test_inflection_from_tags_display_name_en():
    inf = _inflection_from_tags(["n", "pl", "abl"])
    assert "noun" in inf.display_name_en
    assert "plural" in inf.display_name_en
    assert "ablative" in inf.display_name_en


def test_inflection_from_tags_unknown_tag_in_display():
    inf = _inflection_from_tags(["n", "xyzzy"])
    assert "xyzzy" in inf.display_name_en


def test_inflection_from_tags_preserves_raw_tags():
    tags = ["v", "caus", "past", "p3", "pl"]
    inf = _inflection_from_tags(tags)
    assert inf.raw_tags == tags
    # caus has no structured field — only preserved in raw_tags
    assert inf.case is None
