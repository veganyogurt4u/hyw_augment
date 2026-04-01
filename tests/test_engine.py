"""Tests for MorphEngine and AnalysisResult (engine.py).

Unit tests for engine routing logic.  Uses a real Nayiri Lexicon built
from the in-memory FIXTURE (no file I/O) wherever possible; mocks are
reserved for the Apertium backend (which needs a subprocess) and for
component-delegation tests (spellchecker, orthography, glossary).

The FIXTURE lexicon contains:
    noun-alpha   NOUN  — NOM_SG, ACC_SG, ABL_SG, NOM_PL
    verb-beta    VERB  — PRES_1SG
    noun-gamma   NOUN  — NOM_SG

Integration tests with real Apertium/HySpell live in test_integration.py.
"""

from unittest.mock import MagicMock

from hyw_augment.engine import AnalysisResult, MorphEngine
from hyw_augment.nayiri import Lexicon
from tests.test_nayiri import FIXTURE

# ── Helpers ──────────────────────────────────────────────────────────────────

def _nayiri_backend() -> tuple[str, Lexicon]:
    """A real Nayiri Lexicon from the in-memory fixture."""
    return ("nayiri", Lexicon.from_dict(FIXTURE))


def _mock_apertium(form_map: dict[str, list]) -> tuple[str, MagicMock]:
    """Mock Apertium backend for two-backend fallthrough tests.

    Exposes analyze_batch (like real Apertium) so the engine's batch
    path is exercised.  Spec-limited so hasattr checks behave correctly.
    """
    backend = MagicMock(
        spec=["analyze", "analyze_insensitive", "analyze_batch",
              "is_known", "close", "generate", "summary"],
    )
    backend.analyze.side_effect = lambda form: form_map.get(form, [])
    backend.analyze_insensitive.side_effect = lambda form: form_map.get(form, [])
    backend.analyze_batch.side_effect = lambda forms: {
        f: form_map[f] for f in forms if f in form_map
    }
    backend.is_known.side_effect = lambda form: form in form_map
    backend.summary.return_value = "mock apertium"
    return ("apertium", backend)


def _mock_analysis(lemma="lemma", pos="NOUN", description_en="Nom Sg"):
    """Lightweight mock analysis for Apertium-side form_maps."""
    a = MagicMock()
    a.lemma = lemma
    a.pos = pos
    a.description_en = description_en
    a.case = None
    a.number = None
    a.person = None
    a.article = None
    return a


def _engine_with(*backends) -> MorphEngine:
    """Build a MorphEngine with given (name, backend) pairs."""
    engine = MorphEngine()
    engine.backends = list(backends)
    return engine


# ── AnalysisResult ────────────────────────────────────────────────────────────
# Verifies that the AnalysisResult wrapper correctly delegates to a real
# MorphAnalysis object (not a mock).

class TestAnalysisResult:
    """AnalysisResult wraps a backend analysis with a source tag."""

    def test_delegates_noun_properties(self):
        """All noun properties read through to the real MorphAnalysis."""
        lex = Lexicon.from_dict(FIXTURE)
        analysis = lex.analyze("noun-alpha")[0]
        r = AnalysisResult(source="nayiri", analysis=analysis)

        assert r.lemma == "noun-alpha"
        assert r.pos == "NOUN"
        assert r.description_en == "Nominative Singular"
        assert r.case == "NOMINATIVE"
        assert r.number == "SINGULAR"

    def test_delegates_verb_properties(self):
        """Verb analyses expose .person through the wrapper."""
        lex = Lexicon.from_dict(FIXTURE)
        analysis = lex.analyze("verb-beta-pres")[0]
        r = AnalysisResult(source="nayiri", analysis=analysis)

        assert r.lemma == "verb-beta"
        assert r.pos == "VERB"
        assert r.person == "FIRST"

    def test_article_falls_back_to_none(self):
        """Nayiri MorphAnalysis has no .article — getattr fallback returns None."""
        lex = Lexicon.from_dict(FIXTURE)
        analysis = lex.analyze("noun-alpha")[0]
        r = AnalysisResult(source="nayiri", analysis=analysis)
        assert r.article is None

    def test_source_tag(self):
        lex = Lexicon.from_dict(FIXTURE)
        analysis = lex.analyze("noun-alpha")[0]
        r = AnalysisResult(source="apertium", analysis=analysis)
        assert r.source == "apertium"

    def test_repr_contains_source(self):
        lex = Lexicon.from_dict(FIXTURE)
        analysis = lex.analyze("noun-alpha")[0]
        r = AnalysisResult(source="nayiri", analysis=analysis)
        assert "nayiri" in repr(r)


# ── MorphEngine.analyze ──────────────────────────────────────────────────────

class TestAnalyze:
    """Engine.analyze() returns the first backend's results that match."""

    def test_empty_engine(self):
        """No backends → empty list, not an error."""
        engine = MorphEngine()
        assert engine.analyze("anyword") == []

    def test_nayiri_hit(self):
        """When Nayiri knows the form, results come from Nayiri."""
        engine = _engine_with(_nayiri_backend())
        results = engine.analyze("noun-alpha")

        assert len(results) >= 1
        assert results[0].source == "nayiri"
        assert results[0].lemma == "noun-alpha"
        assert results[0].pos == "NOUN"

    def test_nayiri_hit_stops_search(self):
        """If Nayiri hits, Apertium is not queried (first-wins)."""
        apt_analysis = _mock_analysis(lemma="apt-lemma")
        engine = _engine_with(
            _nayiri_backend(),
            _mock_apertium({"noun-alpha": [apt_analysis]}),
        )
        results = engine.analyze("noun-alpha")

        assert results[0].source == "nayiri"

    def test_fallthrough_to_apertium(self):
        """When Nayiri misses, Apertium catches the form."""
        apt_analysis = _mock_analysis(lemma="apt-lemma", pos="VERB")
        engine = _engine_with(
            _nayiri_backend(),
            _mock_apertium({"apt-only-word": [apt_analysis]}),
        )
        results = engine.analyze("apt-only-word")

        assert len(results) == 1
        assert results[0].source == "apertium"
        assert results[0].lemma == "apt-lemma"

    def test_all_miss(self):
        """When no backend knows the form, result is empty."""
        engine = _engine_with(
            _nayiri_backend(),
            _mock_apertium({}),
        )
        assert engine.analyze("unknown-word") == []

    def test_insensitive_fallback(self):
        """Nayiri exact miss → engine retries with analyze_insensitive."""
        engine = _engine_with(_nayiri_backend())
        # FIXTURE has "noun-alpha" (lowercase); "Noun-Alpha" misses exact
        # but hits insensitive (which lowercases the query).
        results = engine.analyze("Noun-Alpha")

        assert len(results) >= 1
        assert results[0].lemma == "noun-alpha"


# ── MorphEngine.analyze_all ──────────────────────────────────────────────────

class TestAnalyzeAll:
    """Engine.analyze_all() queries every backend, returns dict."""

    def test_both_backends_queried(self):
        """Both Nayiri and Apertium results appear when both know the form."""
        apt_analysis = _mock_analysis(lemma="apt-noun")
        engine = _engine_with(
            _nayiri_backend(),
            _mock_apertium({"noun-alpha": [apt_analysis]}),
        )
        all_results = engine.analyze_all("noun-alpha")

        assert "nayiri" in all_results
        assert "apertium" in all_results
        assert all_results["nayiri"][0].lemma == "noun-alpha"
        assert all_results["apertium"][0].lemma == "apt-noun"

    def test_only_hitting_backends_included(self):
        """Backends with no results are excluded from the dict."""
        engine = _engine_with(
            _nayiri_backend(),
            _mock_apertium({}),
        )
        all_results = engine.analyze_all("noun-alpha")

        assert "nayiri" in all_results
        assert "apertium" not in all_results


# ── MorphEngine.analyze_batch ────────────────────────────────────────────────

class TestAnalyzeBatch:
    """Engine.analyze_batch() routes forms through backends with fallthrough."""

    def test_nayiri_covers_known_forms(self):
        """Forms known to Nayiri are resolved without fallthrough."""
        engine = _engine_with(_nayiri_backend())
        results = engine.analyze_batch(["noun-alpha", "noun-gamma"])

        assert "noun-alpha" in results
        assert "noun-gamma" in results
        assert results["noun-alpha"][0].source == "nayiri"

    def test_fallthrough_to_batch_backend(self):
        """Nayiri covers w1; Apertium batch-covers w2."""
        apt_analysis = _mock_analysis(lemma="apt-lemma")
        engine = _engine_with(
            _nayiri_backend(),
            _mock_apertium({"apt-only": [apt_analysis]}),
        )
        results = engine.analyze_batch(["noun-alpha", "apt-only"])

        assert results["noun-alpha"][0].source == "nayiri"
        assert results["apt-only"][0].source == "apertium"

    def test_unknown_forms_absent(self):
        """Forms no backend knows don't appear in results."""
        engine = _engine_with(_nayiri_backend())
        results = engine.analyze_batch(["unknown-word"])
        assert results == {}


# ── MorphEngine.validate ─────────────────────────────────────────────────────

class TestValidate:
    """Engine.validate() checks Nayiri, Apertium, then Hunspell in order."""

    def test_valid_via_nayiri(self):
        """Nayiri's is_valid_form confirms the word."""
        engine = _engine_with(_nayiri_backend())
        assert engine.validate("noun-alpha") is True

    def test_valid_via_apertium(self):
        """Apertium's is_known confirms the word (mock)."""
        engine = _engine_with(_mock_apertium({"apt-word": [_mock_analysis()]}))
        assert engine.validate("apt-word") is True

    def test_false_no_backends(self):
        engine = MorphEngine()
        assert engine.validate("anything") is False

    def test_fallthrough_to_spellchecker(self):
        """When backends miss, the spellchecker gets a chance."""
        engine = _engine_with(_nayiri_backend())
        sc = MagicMock()
        sc.check.return_value = True
        engine.spellchecker = sc

        assert engine.validate("unknown-to-nayiri") is True
        sc.check.assert_called_once_with("unknown-to-nayiri")

    def test_false_all_miss(self):
        """When everything misses, validate returns False."""
        engine = _engine_with(_nayiri_backend())
        sc = MagicMock()
        sc.check.return_value = False
        engine.spellchecker = sc

        assert engine.validate("totally-unknown") is False


# ── Delegation tests (suggest, convert, detect, lookup) ───────────────────────
# These test the "component present → delegate, absent → safe default" pattern.
# Mocks are appropriate here: we're testing the routing, not the components.

class TestSuggest:
    def test_empty_without_spellchecker(self):
        assert MorphEngine().suggest("word") == []

    def test_delegates_to_spellchecker(self):
        engine = MorphEngine()
        sc = MagicMock()
        sc.suggest.return_value = ["fix1", "fix2"]
        engine.spellchecker = sc
        assert engine.suggest("word") == ["fix1", "fix2"]


class TestConvertReformed:
    def test_passthrough_without_orthography(self):
        assert MorphEngine().convert_reformed("text") == "text"

    def test_delegates_to_orthography(self):
        engine = MorphEngine()
        orth = MagicMock()
        orth.convert_text.return_value = "converted"
        engine.orthography = orth
        assert engine.convert_reformed("text") == "converted"


class TestDetectReformed:
    def test_empty_without_orthography(self):
        assert MorphEngine().detect_reformed("text") == []

    def test_delegates_to_orthography(self):
        engine = MorphEngine()
        orth = MagicMock()
        orth.detect_reformed_words.return_value = [("ref", "cls")]
        engine.orthography = orth
        assert engine.detect_reformed("text") == [("ref", "cls")]


class TestLookupDefinition:
    def test_none_without_glossary(self):
        assert MorphEngine().lookup_definition("word") is None

    def test_delegates_to_glossary(self):
        engine = MorphEngine()
        entry = MagicMock()
        engine.glossary = MagicMock()
        engine.glossary.lookup.return_value = [entry]
        assert engine.lookup_definition("word") == [entry]

    def test_not_found_returns_none(self):
        engine = MorphEngine()
        engine.glossary = MagicMock()
        engine.glossary.lookup.return_value = None
        assert engine.lookup_definition("unknown") is None


# ── Context manager / close ──────────────────────────────────────────────────
# These must use mocks to verify that .close() is actually called.

class TestClose:
    def test_context_manager_closes_backend(self):
        """Exiting `with engine:` calls .close() on backends."""
        b = MagicMock()
        b.close = MagicMock()
        b.summary.return_value = ""
        engine = _engine_with(("backend", b))

        with engine:
            pass

        b.close.assert_called_once()

    def test_context_manager_closes_spellchecker(self):
        engine = MorphEngine()
        sc = MagicMock()
        engine.spellchecker = sc

        with engine:
            pass

        sc.close.assert_called_once()

    def test_skips_backend_without_close(self):
        """Backends without .close() don't cause AttributeError."""
        b = MagicMock(spec=["analyze", "summary"])
        b.summary.return_value = ""
        engine = _engine_with(("backend", b))
        engine.close()  # should not raise


# ── MorphEngine.generate ─────────────────────────────────────────────────────

class TestGenerate:
    """Engine.generate() — Apertium first (with tags), Nayiri fallback."""

    def test_nayiri_wildcard(self):
        """Without tags, Nayiri returns all forms for the lemma."""
        engine = _engine_with(_nayiri_backend())
        results = engine.generate("noun-alpha")

        surfaces = [s for s, _inf in results]
        assert "noun-alpha" in surfaces        # NOM_SG
        assert "noun-alpha-acc" in surfaces    # ACC_SG
        assert "noun-alpha-abl" in surfaces    # ABL_SG
        assert "noun-alphas" in surfaces       # NOM_PL

    def test_nayiri_filtered_by_case(self):
        """With tags that translate to Nayiri kwargs, results are filtered."""
        engine = _engine_with(_nayiri_backend())
        # "n" → pos=NOUN, "abl" → case=ABLATIVE
        results = engine.generate("noun-alpha", tags=["n", "abl"])

        assert len(results) == 1
        surface, inf = results[0]
        assert surface == "noun-alpha-abl"
        assert inf.case == "ABLATIVE"

    def test_apertium_first_with_tags(self):
        """With tags, Apertium is tried before Nayiri."""
        mock_inf = MagicMock()
        mock_inf.display_name_en = "Abl Pl"
        apt = _mock_apertium({})
        apt[1].generate.side_effect = lambda lemma, tags: [("apt-form", mock_inf)]

        engine = _engine_with(_nayiri_backend(), apt)
        results = engine.generate("noun-alpha", tags=["n", "pl", "abl"])

        assert results[0][0] == "apt-form"

    def test_apertium_miss_falls_back_to_nayiri(self):
        """When Apertium can't generate, Nayiri handles the request."""
        apt = _mock_apertium({})
        apt[1].generate.side_effect = lambda lemma, tags: []

        engine = _engine_with(_nayiri_backend(), apt)
        results = engine.generate("noun-alpha", tags=["n", "sg", "abl"])

        assert len(results) >= 1
        assert results[0][0] == "noun-alpha-abl"

    def test_no_tags_skips_apertium(self):
        """Without tags, only Nayiri is queried (no Apertium generate call)."""
        apt = _mock_apertium({})
        engine = _engine_with(_nayiri_backend(), apt)

        results = engine.generate("noun-alpha")
        assert len(results) >= 1
        apt[1].generate.assert_not_called()

    def test_empty_no_backends(self):
        assert MorphEngine().generate("anything", tags=["n", "sg"]) == []

    def test_empty_all_miss(self):
        apt = _mock_apertium({})
        apt[1].generate.side_effect = lambda lemma, tags: []
        engine = _engine_with(_nayiri_backend(), apt)

        assert engine.generate("nonexistent-lemma", tags=["n", "sg"]) == []

    def test_unknown_tags_silently_dropped(self):
        """Apertium tags without Nayiri equivalents are ignored, not errors."""
        engine = _engine_with(_nayiri_backend())
        # "caus" has no Nayiri mapping; "n" → NOUN, "abl" → ABLATIVE
        results = engine.generate("noun-alpha", tags=["n", "abl", "caus"])

        assert len(results) == 1
        assert results[0][0] == "noun-alpha-abl"


# ── MorphEngine.generate_all ─────────────────────────────────────────────────

class TestGenerateAll:
    """Engine.generate_all() returns results keyed by backend name."""

    def test_apertium_hit_stops_search(self):
        """With tags, Apertium results exclude Nayiri (unless all_backends)."""
        mock_inf = MagicMock()
        mock_inf.display_name_en = "mock"
        apt = _mock_apertium({})
        apt[1].generate.side_effect = lambda lemma, tags: [("apt-form", mock_inf)]

        engine = _engine_with(_nayiri_backend(), apt)
        all_gen = engine.generate_all("noun-alpha", tags=["n", "sg", "nom"])

        assert "apertium" in all_gen
        assert "nayiri" not in all_gen

    def test_apertium_miss_falls_back_to_nayiri(self):
        apt = _mock_apertium({})
        apt[1].generate.side_effect = lambda lemma, tags: []

        engine = _engine_with(_nayiri_backend(), apt)
        all_gen = engine.generate_all("noun-alpha", tags=["n", "sg", "nom"])

        assert "nayiri" in all_gen
        assert "apertium" not in all_gen

    def test_all_backends_flag(self):
        """all_backends=True queries both even when Apertium succeeds."""
        mock_inf = MagicMock()
        mock_inf.display_name_en = "mock"
        apt = _mock_apertium({})
        apt[1].generate.side_effect = lambda lemma, tags: [("apt-form", mock_inf)]

        engine = _engine_with(_nayiri_backend(), apt)
        all_gen = engine.generate_all(
            "noun-alpha", tags=["n", "sg", "nom"], all_backends=True,
        )

        assert "apertium" in all_gen
        assert "nayiri" in all_gen

    def test_no_tags_nayiri_only(self):
        """Without tags, only Nayiri wildcard is queried."""
        apt = _mock_apertium({})
        engine = _engine_with(_nayiri_backend(), apt)

        all_gen = engine.generate_all("noun-alpha")
        assert "nayiri" in all_gen
        assert "apertium" not in all_gen
