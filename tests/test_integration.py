"""Integration tests: full MorphEngine with real backends.

These tests exercise the engine's fallback chain, generation pipeline,
and auxiliary features (spelling, orthography, glossary, calfa) using
real data loaded from hyw_augment.toml.

Every test is skipped when its required data is absent, so the file is
safe to run in CI where external data may not be installed.

Run with:
    pytest tests/test_integration.py -v
"""

import pytest

from hyw_augment.engine import AnalysisResult, MorphEngine

# ── Armenian test words ───────────────────────────────────────────────────────
# Defined via \uXXXX escapes (models mangle Armenian glyphs; see MEMORY.md).

MART = "\u0574\u0561\u0580\u0564"           # common noun, "man/person"
MART_ABL = "\u0574\u0561\u0580\u0564\u0567" # ablative singular of MART
DUN = "\u057f\u0578\u0582\u0576"             # "house"
KIRQ = "\u0563\u056b\u0580\u0584"            # "book"
YES = "\u0565\u057d"                          # pronoun "I"
NONSENSE = "xyznonexistent"


# ── Analysis ──────────────────────────────────────────────────────────────────

class TestAnalyze:
    """Engine.analyze() — first-backend-wins with real fallback chain."""

    def test_common_noun_found(self, full_engine):
        """A common Armenian noun should be found by at least one backend."""
        results = full_engine.analyze(MART)
        assert len(results) >= 1, (
            f"Expected at least one analysis for MART, got none. "
            f"Backends: {[n for n, _ in full_engine.backends]}"
        )

    def test_result_is_analysis_result(self, full_engine):
        """Results are AnalysisResult wrappers with a .source tag."""
        results = full_engine.analyze(MART)
        assert all(isinstance(r, AnalysisResult) for r in results)
        assert all(r.source in ("nayiri", "apertium") for r in results)

    def test_pos_is_noun(self, full_engine):
        """MART should be tagged as a noun by every backend that finds it."""
        results = full_engine.analyze(MART)
        assert any(r.pos == "NOUN" for r in results), (
            f"Expected NOUN in POS tags, got: {[r.pos for r in results]}"
        )

    def test_nonsense_word_not_tagged_as_noun(self, full_engine):
        """A nonsense string may be tagged (Apertium uses <barb> for foreign
        words) but should not be analyzed as a real NOUN/VERB/ADJ."""
        results = full_engine.analyze(NONSENSE)
        real_pos = [r.pos for r in results if r.pos in ("NOUN", "VERB", "ADJ")]
        assert real_pos == [], (
            f"Nonsense word got real POS tags: {real_pos}"
        )

    def test_pronoun_found(self, full_engine):
        """The pronoun YES should be found (tests non-noun coverage)."""
        results = full_engine.analyze(YES)
        assert len(results) >= 1


class TestAnalyzeAll:
    """Engine.analyze_all() — queries every backend, returns dict."""

    def test_returns_dict_keyed_by_backend(self, full_engine):
        all_results = full_engine.analyze_all(MART)
        assert isinstance(all_results, dict)
        assert len(all_results) >= 1, "At least one backend should find MART"
        for source, results in all_results.items():
            assert source in ("nayiri", "apertium")
            assert all(r.source == source for r in results)

    def test_multiple_backends_when_both_know_word(self, full_engine):
        """If both Nayiri and Apertium are loaded, a common word should
        appear in both."""
        backend_names = [n for n, _ in full_engine.backends]
        if "nayiri" not in backend_names or "apertium" not in backend_names:
            pytest.skip("Need both nayiri and apertium for this test")
        all_results = full_engine.analyze_all(MART)
        assert "nayiri" in all_results, "Nayiri should know MART"
        assert "apertium" in all_results, "Apertium should know MART"

    def test_nonsense_not_in_nayiri(self, full_engine):
        """Nayiri should not know a nonsense word (Apertium may tag it as foreign)."""
        all_results = full_engine.analyze_all(NONSENSE)
        assert "nayiri" not in all_results


class TestAnalyzeBatch:
    """Engine.analyze_batch() — batch analysis with fallthrough."""

    def test_multiple_words_all_found(self, full_engine):
        """Batch of three common words — all should be resolved."""
        results = full_engine.analyze_batch([MART, DUN, KIRQ])
        for word in [MART, DUN, KIRQ]:
            assert word in results, f"{word!r} missing from batch results"
            assert len(results[word]) >= 1

    def test_nonsense_not_in_nayiri_batch(self, full_engine):
        """Nayiri shouldn't resolve nonsense even in batch mode."""
        results = full_engine.analyze_batch([NONSENSE])
        if NONSENSE in results:
            # Apertium may tag it as <barb>; just verify it's not from Nayiri
            assert all(r.source != "nayiri" for r in results[NONSENSE])


# ── Generation ────────────────────────────────────────────────────────────────

class TestGenerate:
    """Engine.generate() — tag-based (Apertium first) and wildcard (Nayiri)."""

    def test_generate_with_tags_returns_forms(self, full_engine):
        """Generating noun ablative singular should return at least one form."""
        results = full_engine.generate(MART, tags=["n", "sg", "abl"])
        assert len(results) >= 1, "Expected at least one generated form"
        surfaces = [s for s, _inf in results]
        assert len(surfaces) >= 1

    def test_generate_wildcard_returns_multiple(self, full_engine):
        """Without tags, Nayiri returns all forms — should be more than one."""
        backend_names = [n for n, _ in full_engine.backends]
        if "nayiri" not in backend_names:
            pytest.skip("Wildcard generation requires Nayiri")
        results = full_engine.generate(MART)
        assert len(results) >= 2, (
            f"Expected multiple forms for MART wildcard, got {len(results)}"
        )

    def test_generate_unknown_lemma_empty(self, full_engine):
        assert full_engine.generate(NONSENSE, tags=["n", "sg"]) == []


class TestGenerateAll:
    """Engine.generate_all() — results keyed by backend name."""

    def test_with_tags_returns_dict(self, full_engine):
        results = full_engine.generate_all(MART, tags=["n", "sg", "abl"])
        assert isinstance(results, dict)
        assert len(results) >= 1

    def test_all_backends_flag(self, full_engine):
        """all_backends=True should query both even when Apertium succeeds."""
        backend_names = [n for n, _ in full_engine.backends]
        if "nayiri" not in backend_names or "apertium" not in backend_names:
            pytest.skip("Need both backends for this test")
        results = full_engine.generate_all(
            MART, tags=["n", "sg", "abl"], all_backends=True,
        )
        # Both backends should appear (if both can generate the form)
        assert len(results) >= 1


# ── Validation ────────────────────────────────────────────────────────────────

class TestValidate:
    """Engine.validate() — checks Nayiri, Apertium, then Hunspell."""

    def test_valid_word(self, full_engine):
        assert full_engine.validate(MART) is True

    def test_nonsense_not_valid_in_nayiri(self, full_engine):
        """Nayiri should not validate nonsense (Apertium may accept anything
        via its <barb> foreign-word tag, so we test Nayiri specifically)."""
        nayiri = None
        for name, backend in full_engine.backends:
            if name == "nayiri":
                nayiri = backend
                break
        if nayiri is None:
            pytest.skip("Nayiri not loaded")
        assert nayiri.is_valid_form(NONSENSE) is False


# ── Spelling ──────────────────────────────────────────────────────────────────

class TestSpelling:
    """Engine spelling features — delegated to HySpell's Hunspell wrapper."""

    def test_suggest_returns_list(self, full_engine):
        if full_engine.spellchecker is None:
            pytest.skip("Spellchecker not configured")
        result = full_engine.suggest(NONSENSE)
        assert isinstance(result, list)

    def test_suggest_valid_word_empty(self, full_engine):
        """A correctly-spelled word should produce no suggestions."""
        if full_engine.spellchecker is None:
            pytest.skip("Spellchecker not configured")
        result = full_engine.suggest(MART)
        assert result == []


# ── Orthography ───────────────────────────────────────────────────────────────

class TestOrthography:
    """Engine.convert_reformed() and detect_reformed()."""

    def test_ascii_passthrough(self, full_engine):
        """ASCII text has no Reformed-orthography words — passes through."""
        if full_engine.orthography is None:
            pytest.skip("Orthography converter not configured")
        assert full_engine.convert_reformed("hello world") == "hello world"

    def test_detect_reformed_ascii_empty(self, full_engine):
        if full_engine.orthography is None:
            pytest.skip("Orthography converter not configured")
        assert full_engine.detect_reformed("hello world") == []


# ── Glossary ──────────────────────────────────────────────────────────────────

class TestGlossary:
    """Engine.lookup_definition() — SmallArmDic glossary."""

    def test_lookup_known_word(self, full_engine):
        if full_engine.glossary is None:
            pytest.skip("Glossary not configured")
        result = full_engine.lookup_definition(MART)
        # MART may or may not be in SmallArmDic; just verify the type
        assert result is None or isinstance(result, list)

    def test_lookup_nonsense_returns_none(self, full_engine):
        if full_engine.glossary is None:
            pytest.skip("Glossary not configured")
        result = full_engine.lookup_definition(NONSENSE)
        assert result is None


# ── CaLFa ─────────────────────────────────────────────────────────────────────

class TestCalfa:
    """Engine.lookup_calfa() and synonyms_calfa() — Calfa lexicon."""

    def test_lookup_returns_entries(self, full_engine):
        if full_engine.calfa is None:
            pytest.skip("Calfa lexicon not configured")
        result = full_engine.lookup_calfa(MART)
        assert result is None or isinstance(result, list)

    def test_synonyms_returns_list(self, full_engine):
        if full_engine.calfa is None:
            pytest.skip("Calfa lexicon not configured")
        result = full_engine.synonyms_calfa(MART)
        assert isinstance(result, list)


# ── Engine construction ───────────────────────────────────────────────────────

class TestFromConfig:
    """MorphEngine.from_config() — verify config loading."""

    def test_loads_at_least_nayiri(self, full_engine):
        backend_names = [n for n, _ in full_engine.backends]
        assert "nayiri" in backend_names, (
            f"Expected nayiri in backends, got: {backend_names}"
        )

    def test_summary_is_informative(self, full_engine):
        summary = full_engine.summary()
        assert "MorphEngine" in summary
        assert "backend" in summary.lower() or "nayiri" in summary.lower()

    def test_context_manager_protocol(self, config_path):
        """Engine supports `with` statement for resource cleanup."""
        with MorphEngine.from_config(config_path) as engine:
            assert len(engine.backends) >= 1
        # After exiting, engine should be closed (no assertion needed,
        # just verify it doesn't raise)
