"""Tests for MorphEngine and AnalysisResult (engine.py)."""

from unittest.mock import MagicMock

from hyw_augment.engine import AnalysisResult, MorphEngine

# ── Helpers / mock backends ───────────────────────────────────────────────────

def _mock_analysis(lemma="lemma", pos="NOUN", description_en="Nom Sg",
                   case="NOMINATIVE", number="SINGULAR", person=None,
                   article=None):
    a = MagicMock()
    a.lemma = lemma
    a.pos = pos
    a.description_en = description_en
    a.case = case
    a.number = number
    a.person = person
    a.article = article
    return a


def _mock_backend(name: str, form_map: dict[str, list]) -> tuple[str, MagicMock]:
    """Create a (name, backend) pair that returns preset analyses.

    Spec is limited to the methods the engine actually uses, so that
    hasattr(backend, "analyze_batch") returns False for plain backends.
    """
    backend = MagicMock(spec=["analyze", "analyze_insensitive", "is_valid_form",
                               "is_known", "close", "summary"])
    backend.analyze.side_effect = lambda form: form_map.get(form, [])
    backend.analyze_insensitive.side_effect = lambda form: form_map.get(form, [])
    backend.is_valid_form.side_effect = lambda form: form in form_map or form.lower() in form_map
    backend.is_known.side_effect = lambda form: form in form_map
    backend.summary.return_value = f"mock {name}"
    return (name, backend)


def _engine_with(*backends) -> MorphEngine:
    """Build a MorphEngine with given (name, backend) pairs directly."""
    engine = MorphEngine()
    engine.backends = list(backends)
    return engine


# ── AnalysisResult ────────────────────────────────────────────────────────────

def test_analysis_result_delegates_lemma():
    inner = _mock_analysis(lemma="mylemma")
    r = AnalysisResult(source="nayiri", analysis=inner)
    assert r.lemma == "mylemma"


def test_analysis_result_delegates_pos():
    inner = _mock_analysis(pos="VERB")
    r = AnalysisResult(source="nayiri", analysis=inner)
    assert r.pos == "VERB"


def test_analysis_result_delegates_description_en():
    inner = _mock_analysis(description_en="Present 1sg")
    r = AnalysisResult(source="apertium", analysis=inner)
    assert r.description_en == "Present 1sg"


def test_analysis_result_delegates_case():
    inner = _mock_analysis(case="ABLATIVE")
    r = AnalysisResult(source="nayiri", analysis=inner)
    assert r.case == "ABLATIVE"


def test_analysis_result_delegates_number():
    inner = _mock_analysis(number="PLURAL")
    r = AnalysisResult(source="nayiri", analysis=inner)
    assert r.number == "PLURAL"


def test_analysis_result_delegates_person():
    inner = _mock_analysis(person="SECOND")
    r = AnalysisResult(source="nayiri", analysis=inner)
    assert r.person == "SECOND"


def test_analysis_result_article_falls_back_to_none():
    """If the inner analysis has no .article attribute, return None."""
    inner = MagicMock(spec=["lemma", "pos", "description_en", "case", "number", "person"])
    inner.lemma = "w"
    inner.pos = "NOUN"
    inner.description_en = "x"
    inner.case = None
    inner.number = None
    inner.person = None
    r = AnalysisResult(source="nayiri", analysis=inner)
    assert r.article is None


def test_analysis_result_source():
    r = AnalysisResult(source="apertium", analysis=_mock_analysis())
    assert r.source == "apertium"


def test_analysis_result_repr():
    inner = _mock_analysis()
    r = AnalysisResult(source="nayiri", analysis=inner)
    assert "nayiri" in repr(r)


# ── MorphEngine.analyze ───────────────────────────────────────────────────────

def test_analyze_returns_empty_no_backends():
    engine = MorphEngine()
    assert engine.analyze("anyword") == []


def test_analyze_hits_first_backend():
    a = _mock_analysis(lemma="thelemma")
    b1 = _mock_backend("nayiri", {"word": [a]})
    b2 = _mock_backend("apertium", {"word": [_mock_analysis(lemma="other")]})
    engine = _engine_with(b1, b2)

    results = engine.analyze("word")
    assert len(results) == 1
    assert results[0].source == "nayiri"
    assert results[0].lemma == "thelemma"


def test_analyze_falls_through_to_second_backend():
    a = _mock_analysis(lemma="apt-lemma")
    b1 = _mock_backend("nayiri", {})  # misses "word"
    b2 = _mock_backend("apertium", {"word": [a]})
    engine = _engine_with(b1, b2)

    results = engine.analyze("word")
    assert len(results) == 1
    assert results[0].source == "apertium"
    assert results[0].lemma == "apt-lemma"


def test_analyze_returns_empty_when_all_miss():
    b1 = _mock_backend("nayiri", {})
    b2 = _mock_backend("apertium", {})
    engine = _engine_with(b1, b2)
    assert engine.analyze("unknown") == []


def test_analyze_nayiri_falls_back_to_insensitive():
    """When nayiri misses exact form, engine retries with analyze_insensitive."""
    a = _mock_analysis(lemma="lemma")
    b = MagicMock(spec=["analyze", "analyze_insensitive", "summary"])
    b.analyze.side_effect = lambda form: []           # exact always misses
    # engine passes the original form to analyze_insensitive, not lowercased
    b.analyze_insensitive.side_effect = lambda form: [a] if form == "Word" else []
    b.summary.return_value = "mock"
    engine = _engine_with(("nayiri", b))

    results = engine.analyze("Word")  # uppercase → exact miss → insensitive hit
    assert len(results) == 1


# ── MorphEngine.analyze_all ───────────────────────────────────────────────────

def test_analyze_all_queries_every_backend():
    a1 = _mock_analysis(lemma="nayiri-lemma")
    a2 = _mock_analysis(lemma="apertium-lemma")
    b1 = _mock_backend("nayiri", {"word": [a1]})
    b2 = _mock_backend("apertium", {"word": [a2]})
    engine = _engine_with(b1, b2)

    all_results = engine.analyze_all("word")
    assert "nayiri" in all_results
    assert "apertium" in all_results
    assert all_results["nayiri"][0].lemma == "nayiri-lemma"
    assert all_results["apertium"][0].lemma == "apertium-lemma"


def test_analyze_all_only_includes_backends_with_results():
    a = _mock_analysis()
    b1 = _mock_backend("nayiri", {"word": [a]})
    b2 = _mock_backend("apertium", {})  # misses
    engine = _engine_with(b1, b2)

    all_results = engine.analyze_all("word")
    assert "nayiri" in all_results
    assert "apertium" not in all_results


# ── MorphEngine.analyze_batch ─────────────────────────────────────────────────

def test_analyze_batch_nayiri_covers_all():
    a1 = _mock_analysis(lemma="l1")
    a2 = _mock_analysis(lemma="l2")
    b = _mock_backend("nayiri", {"w1": [a1], "w2": [a2]})
    engine = _engine_with(b)

    results = engine.analyze_batch(["w1", "w2"])
    assert "w1" in results
    assert "w2" in results


def _mock_batch_backend(name: str, form_map: dict[str, list]) -> tuple[str, MagicMock]:
    """Backend that exposes analyze_batch, as Apertium does."""
    backend = MagicMock(spec=["analyze", "analyze_batch", "is_known", "summary"])
    backend.analyze.side_effect = lambda form: form_map.get(form, [])
    backend.analyze_batch.side_effect = lambda forms: {
        f: form_map[f] for f in forms if f in form_map
    }
    backend.is_known.side_effect = lambda form: form in form_map
    backend.summary.return_value = f"mock {name}"
    return (name, backend)


def test_analyze_batch_fallthrough_to_second():
    a1 = _mock_analysis(lemma="nayiri-l")
    a2 = _mock_analysis(lemma="apt-l")
    b1 = _mock_backend("nayiri", {"w1": [a1]})          # covers w1 only
    b2 = _mock_batch_backend("apertium", {"w2": [a2]})  # covers w2
    engine = _engine_with(b1, b2)

    results = engine.analyze_batch(["w1", "w2"])
    assert results["w1"][0].source == "nayiri"
    assert results["w2"][0].source == "apertium"


def test_analyze_batch_missing_forms_not_in_result():
    b = _mock_backend("nayiri", {})
    engine = _engine_with(b)
    results = engine.analyze_batch(["unknown1", "unknown2"])
    assert results == {}


# ── MorphEngine.validate ──────────────────────────────────────────────────────

def test_validate_true_via_nayiri():
    b = MagicMock()
    b.is_valid_form.return_value = True
    b.summary.return_value = ""
    engine = _engine_with(("nayiri", b))
    assert engine.validate("word") is True


def test_validate_true_via_apertium():
    b = MagicMock()
    b.is_known.return_value = True
    b.summary.return_value = ""
    engine = _engine_with(("apertium", b))
    assert engine.validate("word") is True


def test_validate_false_no_backends_no_spellcheck():
    engine = MorphEngine()
    assert engine.validate("anything") is False


def test_validate_falls_through_to_spellchecker():
    b = MagicMock()
    b.is_valid_form.return_value = False
    b.summary.return_value = ""
    engine = _engine_with(("nayiri", b))
    sc = MagicMock()
    sc.check.return_value = True
    engine.spellchecker = sc

    assert engine.validate("word") is True
    sc.check.assert_called_once_with("word")


def test_validate_false_all_miss():
    b = MagicMock()
    b.is_valid_form.return_value = False
    b.summary.return_value = ""
    engine = _engine_with(("nayiri", b))
    sc = MagicMock()
    sc.check.return_value = False
    engine.spellchecker = sc

    assert engine.validate("word") is False


# ── MorphEngine.suggest ───────────────────────────────────────────────────────

def test_suggest_returns_empty_without_spellchecker():
    engine = MorphEngine()
    assert engine.suggest("misspelled") == []


def test_suggest_delegates_to_spellchecker():
    engine = MorphEngine()
    sc = MagicMock()
    sc.suggest.return_value = ["suggestion1", "suggestion2"]
    engine.spellchecker = sc

    result = engine.suggest("misspelled")
    assert result == ["suggestion1", "suggestion2"]
    sc.suggest.assert_called_once_with("misspelled")


# ── MorphEngine.convert_reformed ─────────────────────────────────────────────

def test_convert_reformed_passthrough_no_orthography():
    engine = MorphEngine()
    assert engine.convert_reformed("some text") == "some text"


def test_convert_reformed_delegates_to_orthography():
    engine = MorphEngine()
    orth = MagicMock()
    orth.convert_text.return_value = "converted text"
    engine.orthography = orth

    result = engine.convert_reformed("input text")
    assert result == "converted text"
    orth.convert_text.assert_called_once_with("input text")


# ── MorphEngine.detect_reformed ──────────────────────────────────────────────

def test_detect_reformed_returns_empty_no_orthography():
    engine = MorphEngine()
    assert engine.detect_reformed("some text") == []


def test_detect_reformed_delegates_to_orthography():
    engine = MorphEngine()
    orth = MagicMock()
    orth.detect_reformed_words.return_value = [("ref", "cls")]
    engine.orthography = orth

    result = engine.detect_reformed("ref text")
    assert result == [("ref", "cls")]


# ── MorphEngine.lookup_definition ────────────────────────────────────────────

def test_lookup_definition_returns_none_no_glossary():
    engine = MorphEngine()
    assert engine.lookup_definition("word") is None


def test_lookup_definition_delegates_to_glossary():
    engine = MorphEngine()
    entry = MagicMock()
    glossary = MagicMock()
    glossary.lookup.return_value = [entry]
    engine.glossary = glossary

    result = engine.lookup_definition("word")
    assert result == [entry]
    glossary.lookup.assert_called_once_with("word")


def test_lookup_definition_not_found_returns_none():
    engine = MorphEngine()
    glossary = MagicMock()
    glossary.lookup.return_value = None
    engine.glossary = glossary

    assert engine.lookup_definition("unknown") is None


# ── MorphEngine context manager + close ──────────────────────────────────────

def test_context_manager_calls_close():
    b = MagicMock()
    b.close = MagicMock()
    b.summary.return_value = ""
    engine = _engine_with(("backend", b))

    with engine:
        pass

    b.close.assert_called_once()


def test_context_manager_closes_spellchecker():
    engine = MorphEngine()
    sc = MagicMock()
    engine.spellchecker = sc

    with engine:
        pass

    sc.close.assert_called_once()


def test_close_skips_backend_without_close():
    """Backends without .close() should not cause AttributeError."""
    b = MagicMock(spec=["analyze", "summary"])  # no .close
    b.summary.return_value = ""
    engine = _engine_with(("backend", b))
    engine.close()  # should not raise


# ── MorphEngine.generate ────────────────────────────────────────────────────

def _mock_inflection(**kw):
    """Minimal Inflection-like object for generation tests."""
    inf = MagicMock()
    inf.display_name_en = kw.get("display_name_en", "mock inflection")
    inf.case = kw.get("case")
    inf.number = kw.get("number")
    inf.person = kw.get("person")
    inf.article = kw.get("article")
    inf.raw_tags = kw.get("raw_tags")
    return inf


def _mock_gen_backend(name, gen_map):
    """Backend with generate() that returns preset (surface, inflection) tuples."""
    spec = ["analyze", "generate", "summary"]
    if name == "apertium":
        spec.append("analyze_batch")
    backend = MagicMock(spec=spec)
    backend.analyze.side_effect = lambda form: []

    if name == "apertium":
        # Apertium generate takes (lemma, tags)
        backend.generate.side_effect = lambda lemma, tags: gen_map.get(lemma, [])
    else:
        # Nayiri generate takes (lemma, **kwargs)
        backend.generate.side_effect = lambda lemma, **kw: gen_map.get(lemma, [])

    backend.summary.return_value = f"mock {name}"
    return (name, backend)


def test_generate_apertium_first_with_tags():
    apt_inf = _mock_inflection(case="ABLATIVE", number="PLURAL", raw_tags=["n", "pl", "abl"])
    nay_inf = _mock_inflection(case="ABLATIVE", number="PLURAL")

    b1 = _mock_gen_backend("nayiri", {"word": [("word-abl-pl", nay_inf)]})
    b2 = _mock_gen_backend("apertium", {"word": [("word-apt", apt_inf)]})
    engine = _engine_with(b1, b2)

    results = engine.generate("word", tags=["n", "pl", "abl"])
    assert len(results) == 1
    assert results[0][0] == "word-apt"


def test_generate_falls_back_to_nayiri_when_apertium_misses():
    nay_inf = _mock_inflection(case="ABLATIVE")
    b1 = _mock_gen_backend("nayiri", {"word": [("word-nayiri", nay_inf)]})
    b2 = _mock_gen_backend("apertium", {})  # misses "word"
    engine = _engine_with(b1, b2)

    results = engine.generate("word", tags=["n", "sg", "abl"])
    assert len(results) == 1
    assert results[0][0] == "word-nayiri"


def test_generate_no_tags_uses_nayiri_only():
    nay_inf = _mock_inflection()
    b1 = _mock_gen_backend("nayiri", {"word": [("w1", nay_inf), ("w2", nay_inf)]})
    b2 = _mock_gen_backend("apertium", {"word": [("w-apt", nay_inf)]})
    engine = _engine_with(b1, b2)

    results = engine.generate("word")
    assert len(results) == 2
    # Apertium should NOT have been called (no tags)
    b2[1].generate.assert_not_called()


def test_generate_returns_empty_no_backends():
    engine = MorphEngine()
    assert engine.generate("anything", tags=["n", "sg"]) == []


def test_generate_returns_empty_all_miss():
    b1 = _mock_gen_backend("nayiri", {})
    b2 = _mock_gen_backend("apertium", {})
    engine = _engine_with(b1, b2)
    assert engine.generate("unknown", tags=["n", "sg"]) == []


def test_generate_tag_degradation_for_nayiri():
    """Tags that Nayiri doesn't understand are silently dropped."""
    nay_inf = _mock_inflection(case="ABLATIVE")
    b1 = _mock_gen_backend("nayiri", {"word": [("word-nay", nay_inf)]})
    engine = _engine_with(b1)  # no apertium

    # "n" maps to pos="NOUN", "abl" to case="ABLATIVE", "caus" silently ignored
    results = engine.generate("word", tags=["n", "abl", "caus"])
    assert len(results) == 1
    b1[1].generate.assert_called_once_with("word", pos="NOUN", case="ABLATIVE")


# ── MorphEngine.generate_all ────────────────────────────────────────────────

def test_generate_all_queries_both_backends():
    apt_inf = _mock_inflection(raw_tags=["n", "sg", "nom"])
    nay_inf = _mock_inflection()
    b1 = _mock_gen_backend("nayiri", {"word": [("w-nay", nay_inf)]})
    b2 = _mock_gen_backend("apertium", {"word": [("w-apt", apt_inf)]})
    engine = _engine_with(b1, b2)

    all_gen = engine.generate_all("word", tags=["n", "sg", "nom"])
    assert "nayiri" in all_gen
    assert "apertium" in all_gen


def test_generate_all_no_tags_skips_apertium():
    nay_inf = _mock_inflection()
    b1 = _mock_gen_backend("nayiri", {"word": [("w-nay", nay_inf)]})
    b2 = _mock_gen_backend("apertium", {"word": [("w-apt", nay_inf)]})
    engine = _engine_with(b1, b2)

    all_gen = engine.generate_all("word")
    assert "nayiri" in all_gen
    assert "apertium" not in all_gen
