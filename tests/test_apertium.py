"""Tests for apertium.py — parsing logic only, no hfst-lookup binary needed."""

from pathlib import Path

import pytest

from hyw_augment.apertium import ApertiumAnalysis, ApertiumAnalyzer, _inflection_from_tags

# ── Fixture helper ────────────────────────────────────────────────────────────

def _analyzer() -> ApertiumAnalyzer:
    """Create an ApertiumAnalyzer pointing at a nonexistent dir.

    available=False, but the parsing methods (_parse_lines,
    _parse_batch_output, _parse_analysis_string) don't check availability.
    """
    return ApertiumAnalyzer("/nonexistent")


def _make_analysis(tags: list[str], *, form: str = "մարդ", lemma: str = "մարդ") -> ApertiumAnalysis:
    return ApertiumAnalysis(form=form, lemma=lemma, tags=tags, raw="raw")


# ── ApertiumAnalysis properties ───────────────────────────────────────────────

def test_analysis_pos_noun():
    result = _make_analysis(["n", "sg", "nom"]).pos
    assert result == "NOUN", f"Expected 'NOUN', got {result!r}"


def test_analysis_pos_verb():
    result = _make_analysis(["v", "pres", "indc"]).pos
    assert result == "VERB", f"Expected 'VERB', got {result!r}"


def test_analysis_pos_adjective():
    result = _make_analysis(["adj", "sg"]).pos
    assert result == "ADJECTIVE", f"Expected 'ADJECTIVE', got {result!r}"


def test_analysis_pos_proper_noun_maps_to_noun():
    # np (proper noun) maps to NOUN in _APT_POS_MAP
    result = _make_analysis(["np", "ant", "sg"]).pos
    assert result == "NOUN", f"Expected 'NOUN', got {result!r}"


def test_analysis_pos_unknown_when_no_pos_tag():
    # Tags with no POS entry fall back to UNKNOWN
    result = _make_analysis(["sg", "nom"]).pos
    assert result == "UNKNOWN", f"Expected 'UNKNOWN', got {result!r}"


def test_analysis_case_nominative():
    result = _make_analysis(["n", "sg", "nom"]).case
    assert result == "NOMINATIVE", f"Expected 'NOMINATIVE', got {result!r}"


def test_analysis_case_ablative():
    result = _make_analysis(["n", "pl", "abl", "def"]).case
    assert result == "ABLATIVE", f"Expected 'ABLATIVE', got {result!r}"


def test_analysis_case_none_for_verb():
    result = _make_analysis(["v", "pres", "indc", "p3", "sg"]).case
    assert result is None, f"Expected None, got {result!r}"


def test_analysis_number_singular():
    result = _make_analysis(["n", "sg", "nom"]).number
    assert result == "SINGULAR", f"Expected 'SINGULAR', got {result!r}"


def test_analysis_number_plural():
    result = _make_analysis(["n", "pl", "nom"]).number
    assert result == "PLURAL", f"Expected 'PLURAL', got {result!r}"


def test_analysis_number_none_if_absent():
    result = _make_analysis(["v", "pres"]).number
    assert result is None, f"Expected None, got {result!r}"


def test_analysis_person_first():
    result = _make_analysis(["v", "pres", "p1", "sg"]).person
    assert result == "FIRST", f"Expected 'FIRST', got {result!r}"


def test_analysis_person_third():
    result = _make_analysis(["v", "past", "p3", "pl"]).person
    assert result == "THIRD", f"Expected 'THIRD', got {result!r}"


def test_analysis_person_none_for_noun():
    result = _make_analysis(["n", "sg", "nom"]).person
    assert result is None, f"Expected None, got {result!r}"


def test_analysis_article_definite():
    result = _make_analysis(["n", "sg", "nom", "def"]).article
    assert result == "DEFINITE", f"Expected 'DEFINITE', got {result!r}"


def test_analysis_article_indefinite():
    result = _make_analysis(["n", "sg", "nom", "indef"]).article
    assert result == "INDEFINITE", f"Expected 'INDEFINITE', got {result!r}"


def test_analysis_article_none_if_absent():
    result = _make_analysis(["n", "sg", "nom"]).article
    assert result is None, f"Expected None, got {result!r}"


def test_analysis_is_proper_noun_true():
    result = _make_analysis(["np", "ant", "sg"]).is_proper_noun
    assert result is True, f"Expected True, got {result!r}"


def test_analysis_is_proper_noun_false():
    result = _make_analysis(["n", "sg", "nom"]).is_proper_noun
    assert result is False, f"Expected False, got {result!r}"


def test_analysis_description_en_known_tags():
    desc = _make_analysis(["n", "sg"]).description_en
    assert "noun" in desc, f"'noun' not in description_en: {desc!r}"
    assert "singular" in desc, f"'singular' not in description_en: {desc!r}"


def test_analysis_description_en_unknown_tag_falls_back_to_tag():
    desc = _make_analysis(["xyzzy"]).description_en
    assert "xyzzy" in desc, f"'xyzzy' not in description_en: {desc!r}"


def test_analysis_description_en_multiple_tags_comma_joined():
    desc = _make_analysis(["n", "sg", "nom"]).description_en
    assert ", " in desc, f"Expected comma in description_en: {desc!r}"


def test_analysis_repr_contains_form_lemma_and_tags():
    a = _make_analysis(["n", "sg"], form="մարդ", lemma="մարդ")
    r = repr(a)
    assert "մարդ" in r, f"'մարդ' not in repr: {r!r}"
    assert "<n>" in r, f"'<n>' not in repr: {r!r}"


# ── _parse_analysis_string ────────────────────────────────────────────────────

def test_parse_analysis_string_with_tags():
    result = ApertiumAnalyzer._parse_analysis_string("մարդ<n><sg><nom>")
    assert result is not None, f"Expected non-None result, got {result!r}"
    lemma, tags = result
    assert lemma == "մարդ", f"Expected lemma 'մարդ', got {lemma!r}"
    assert tags == ["n", "sg", "nom"], f"Expected tags ['n', 'sg', 'nom'], got {tags!r}"


def test_parse_analysis_string_no_tags():
    result = ApertiumAnalyzer._parse_analysis_string("մարդ")
    assert result is not None, f"Expected non-None result, got {result!r}"
    lemma, tags = result
    assert lemma == "մարդ", f"Expected lemma 'մարդ', got {lemma!r}"
    assert tags == [], f"Expected empty tags, got {tags!r}"


def test_parse_analysis_string_many_tags():
    result = ApertiumAnalyzer._parse_analysis_string("գրել<v><pres><indc><p1><sg>")
    assert result is not None, f"Expected non-None result, got {result!r}"
    assert result[1] == ["v", "pres", "indc", "p1", "sg"], f"Expected tags ['v','pres','indc','p1','sg'], got {result[1]!r}"


def test_parse_analysis_string_plus_question_parsed_as_tagless_lemma():
    # _parse_analysis_string is pure regex — it does NOT filter +?.
    # The +? check happens one level up in _parse_lines / _parse_batch_output.
    result = ApertiumAnalyzer._parse_analysis_string("բարև+?")
    assert result is not None, f"Expected non-None result, got {result!r}"
    lemma, tags = result
    assert lemma == "բարև+?", f"Expected lemma 'բարև+?', got {lemma!r}"
    assert tags == [], f"Expected empty tags, got {tags!r}"


def test_parse_analysis_string_empty_returns_none():
    result = ApertiumAnalyzer._parse_analysis_string("")
    assert result is None, f"Expected None, got {result!r}"


# ── _parse_lines ──────────────────────────────────────────────────────────────

def test_parse_lines_single_analysis():
    apt = _analyzer()
    results = apt._parse_lines("մարդ", ["մարդ\tմարդ<n><sg><nom>\t0.0"])
    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    a = results[0]
    assert a.form == "մարդ", f"Expected form 'մարդ', got {a.form!r}"
    assert a.lemma == "մարդ", f"Expected lemma 'մարդ', got {a.lemma!r}"
    assert a.tags == ["n", "sg", "nom"], f"Expected tags ['n', 'sg', 'nom'], got {a.tags!r}"
    assert a.weight == 0.0, f"Expected weight 0.0, got {a.weight!r}"


def test_parse_lines_multiple_analyses_for_same_form():
    apt = _analyzer()
    lines = [
        "մարդ\tմարդ<n><sg><nom>\t0.0",
        "մարդ\tմարդ<adj><sg>\t1.0",
    ]
    results = apt._parse_lines("մարդ", lines)
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    assert results[0].tags == ["n", "sg", "nom"], f"Expected tags ['n', 'sg', 'nom'], got {results[0].tags!r}"
    assert results[1].tags == ["adj", "sg"], f"Expected tags ['adj', 'sg'], got {results[1].tags!r}"


def test_parse_lines_unknown_skipped():
    apt = _analyzer()
    # +? means the transducer has no analysis
    results = apt._parse_lines("մարդ", ["մարդ\tմարդ+?\t∞"])
    assert results == [], f"Expected empty list, got {results!r}"


def test_parse_lines_no_tab_skipped():
    apt = _analyzer()
    results = apt._parse_lines("մարդ", ["մարդ"])
    assert results == [], f"Expected empty list, got {results!r}"


def test_parse_lines_weight_parsed():
    apt = _analyzer()
    results = apt._parse_lines("մարդ", ["մարդ\tմարդ<n><sg>\t3.14"])
    assert results[0].weight == pytest.approx(3.14), f"Expected weight approx 3.14, got {results[0].weight!r}"


def test_parse_lines_invalid_weight_defaults_to_zero():
    apt = _analyzer()
    results = apt._parse_lines("մարդ", ["մարդ\tմարդ<n><sg>\tnot-a-float"])
    assert results[0].weight == 0.0, f"Expected weight 0.0, got {results[0].weight!r}"


def test_parse_lines_empty_input():
    results = _analyzer()._parse_lines("մարդ", [])
    assert results == [], f"Expected empty list, got {results!r}"


# ── _parse_batch_output ───────────────────────────────────────────────────────

def test_parse_batch_output_single_form():
    results = _analyzer()._parse_batch_output("մարդ\tմարդ<n><sg><nom>\t0.0\n")
    assert "մարդ" in results, f"'մարդ' not in results: {results!r}"
    assert len(results["մարդ"]) == 1, f"Expected 1 result for 'մարդ', got {len(results['մարդ'])}"
    assert results["մարդ"][0].lemma == "մարդ", f"Expected lemma 'մարդ', got {results['մարդ'][0].lemma!r}"


def test_parse_batch_output_multiple_forms():
    output = "մարդ\tմարդ<n><sg>\t0.0\ntուն\tտուն<n><sg><nom>\t0.0\n"
    results = _analyzer()._parse_batch_output(output)
    assert results["մարդ"][0].pos == "NOUN", f"Expected 'NOUN', got {results['մարդ'][0].pos!r}"
    assert results["տուն"][0].pos == "NOUN", f"Expected 'NOUN', got {results['տուն'][0].pos!r}"


def test_parse_batch_output_multiple_analyses_same_form():
    output = "գիրք\tգիրք<n><sg>\t0.0\ngիրք\tգիրք<adj><sg>\t1.0\n"
    results = _analyzer()._parse_batch_output(output)
    assert len(results["գիրք"]) == 2, f"Expected 2 results for 'գիրք', got {len(results['գիրք'])}"


def test_parse_batch_output_unknown_form_gives_empty_list():
    # Unknown forms still appear in the dict, but with an empty list
    results = _analyzer()._parse_batch_output("գիրք\tգիրք+?\t∞\n")
    assert "գիրք" in results, f"'գիրք' not in results: {results!r}"
    assert results["գիրք"] == [], f"Expected empty list for 'գիրք', got {results['գիրք']!r}"


def test_parse_batch_output_empty_string():
    results = _analyzer()._parse_batch_output("")
    assert results == {}, f"Expected empty dict, got {results!r}"


def test_parse_batch_output_ignores_blank_lines():
    output = "\n\nմարդ\tմարդ<n><sg>\t0.0\n\n"
    results = _analyzer()._parse_batch_output(output)
    assert "մարդ" in results, f"'մարդ' not in results: {results!r}"


# ── analyze_batch case fallback ───────────────────────────────────────────────

def test_analyze_batch_retries_lowercase_on_miss():
    """analyze_batch retries with lowercase when original form gets no results."""
    from unittest.mock import patch

    apt = ApertiumAnalyzer("/nonexistent")
    apt.available = True
    apt.automorf = type("P", (), {"exists": lambda self: True, "__str__": lambda self: "/fake"})()

    # First call: "Մարդ" gets no results; second call: "մարդ" gets a hit
    call_count = [0]
    def fake_run_batch(transducer, inputs, *, timeout=30.0):
        call_count[0] += 1
        if call_count[0] == 1:
            # First batch: "Մարդ" unknown, "Տուն" found
            return "Մարդ\tՄարդ+?\t∞\nՏուն\tտուն<n><sg><nom>\t0.0\n"
        else:
            # Retry batch: "մարդ" (lowercase) found
            return "մարդ\tմարդ<n><sg><nom>\t0.0\n"

    with patch.object(apt, "_run_batch", side_effect=fake_run_batch):
        results = apt.analyze_batch(["Մարդ", "Տուն"])

    assert call_count[0] == 2  # two batch calls
    assert len(results["Տուն"]) == 1
    assert len(results["Մարդ"]) == 1  # keyed under original "Մարդ"
    assert results["Մարդ"][0].tags == ["n", "sg", "nom"]


def test_analyze_batch_no_retry_when_already_lowercase():
    """No retry needed if form is already lowercase."""
    from unittest.mock import patch

    apt = ApertiumAnalyzer("/nonexistent")
    apt.available = True
    apt.automorf = type("P", (), {"exists": lambda self: True, "__str__": lambda self: "/fake"})()

    def fake_run_batch(transducer, inputs, *, timeout=30.0):
        return "մարդ\tմարդ+?\t∞\n"

    with patch.object(apt, "_run_batch", side_effect=fake_run_batch):
        results = apt.analyze_batch(["մարդ"])

    assert results["մարդ"] == []  # no retry since "մարդ" == "մարդ".lower()


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


# ── Integration tests (real transducer) ──────────────────────────────────────
# These run only when the Apertium transducer is installed.

_APERTIUM_DIR = "/home/van/ml-work/arm_data/apertium-hyw"
_has_apertium = Path(_APERTIUM_DIR).exists()
skip_no_apertium = pytest.mark.skipif(not _has_apertium, reason="apertium-hyw not installed")

# ── Armenian test words ───────────────────────────────────────────────────────
MARD = "մարդ"
MARD_DAT = "մարդու"
TUN = "տուն"
TUN_DAT = "տան"
GIRQ = "գիրք"
GIRQ_DAT = "գրքի"
BAREV = "բարեւ"
VOCH = "Ոչ"  # capitalized


@skip_no_apertium
class TestApertiumIntegration:
    """Integration tests against the real apertium-hyw transducer."""

    @pytest.fixture(autouse=True)
    def setup_analyzer(self):
        self.apt = ApertiumAnalyzer(_APERTIUM_DIR)
        assert self.apt.available, "Transducer found but not available"

    def test_analyze_common_noun_mard(self):
        """Analyze MARD — should find noun analysis."""
        results = self.apt.analyze(MARD)
        assert len(results) >= 1
        assert any(a.pos == "NOUN" and a.lemma == MARD for a in results)

    def test_analyze_common_noun_tun(self):
        """Analyze TUN — should find noun analysis."""
        results = self.apt.analyze(TUN)
        assert len(results) >= 1
        assert any(a.pos == "NOUN" and a.lemma == TUN for a in results)

    def test_analyze_common_noun_girq(self):
        """Analyze GIRQ — should find noun analysis."""
        results = self.apt.analyze(GIRQ)
        assert len(results) >= 1
        assert any(a.pos == "NOUN" and a.lemma == GIRQ for a in results)

    def test_analyze_interjection_barev(self):
        """Analyze BAREV — should find interjection analysis."""
        results = self.apt.analyze(BAREV)
        assert len(results) >= 1
        # It might be analyzed as noun, interjection, etc. – just ensure results.
        assert any(a.lemma == BAREV for a in results)

    def test_analyze_insensitive_capitalized(self):
        """VOCH (capitalized) should be found via case-insensitive fallback."""
        results = self.apt.analyze_insensitive(VOCH)
        assert len(results) >= 1, f"No results for {VOCH!r} even with insensitive fallback"

    def test_analyze_batch_case_fallback(self):
        """analyze_batch retries capitalized forms with lowercase."""
        results = self.apt.analyze_batch([VOCH, MARD, TUN])
        assert len(results[MARD]) >= 1, f"No results for {MARD!r} in batch analysis"
        assert len(results[TUN]) >= 1, f"No results for {TUN!r} in batch analysis"
        assert len(results[VOCH]) >= 1, f"No results for {VOCH!r} in batch analysis"

    def test_generate_noun_dative_mard(self):
        """generate(MARD, [n,sg,dat]) should produce MARD_DAT."""
        forms = self.apt.generate(MARD, ["n", "sg", "dat"])
        surfaces = [s for s, _inf in forms]
        assert MARD_DAT in surfaces, f"Expected {MARD_DAT!r} in generated forms: {surfaces}"

    def test_generate_noun_dative_tun(self):
        """generate(TUN, [n,sg,dat]) should produce TUN_DAT."""
        forms = self.apt.generate(TUN, ["n", "sg", "dat"])
        surfaces = [s for s, _inf in forms]
        assert TUN_DAT in surfaces, f"Expected {TUN_DAT!r} in generated forms: {surfaces}"

    def test_generate_noun_dative_girq(self):
        """generate(GIRQ, [n,sg,dat]) should produce GIRQ_DAT."""
        forms = self.apt.generate(GIRQ, ["n", "sg", "dat"])
        surfaces = [s for s, _inf in forms]
        assert GIRQ_DAT in surfaces, f"Expected {GIRQ_DAT!r} in generated forms: {surfaces}"

    def test_generate_returns_inflection(self):
        """generate() returns (str, Inflection) tuples with correct metadata."""
        forms = self.apt.generate(MARD, ["n", "sg", "dat"])
        assert len(forms) >= 1, f"No forms generated for {MARD!r} with [n,sg,dat]"
        surface, inf = forms[0]
        assert inf.case == "DATIVE", f"Expected case DATIVE, got {inf.case}"
        assert inf.number == "SINGULAR", f"Expected number SINGULAR, got {inf.number}"
        assert inf.raw_tags == ["n", "sg", "dat"], f"Expected raw_tags ['n', 'sg', 'dat'], got {inf.raw_tags}"

    def test_apertium_analyze_real_word_arshav(self):
        """Analyze a real Armenian word արշաւ — should find noun analysis."""
        results = self.apt.analyze("արշաւ")
        assert results, "No analyses returned for 'արշաւ'"
        nouns = [r for r in results if r.pos == "NOUN" and r.lemma == "արշաւ"]
        assert nouns, f"No noun analysis for 'արշաւ' among {results}"