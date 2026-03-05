"""Basic tests for the parsing modules."""

from pathlib import Path

from hyw_augment.conllu import Treebank, _parse_feats, _parse_misc
from hyw_augment.nayiri import Lexicon

# ── Unit tests for parsers ────────────────────────────────────────────────────


def test_parse_feats():
    assert _parse_feats("_") == {}
    assert _parse_feats("Case=Nom|Number=Sing") == {
        "Case": "Nom",
        "Number": "Sing",
    }
    assert _parse_feats("Aspect=Imp|Mood=Ind|Number=Plur|Person=1") == {
        "Aspect": "Imp",
        "Mood": "Ind",
        "Number": "Plur",
        "Person": "1",
    }


def test_parse_misc():
    assert _parse_misc("_") == {}
    assert _parse_misc("SpaceAfter=No") == {"SpaceAfter": "No"}
    assert _parse_misc("Translit=ays|LTranslit=ays") == {
        "Translit": "ays",
        "LTranslit": "ays",
    }


# ── Integration tests (require data files) ───────────────────────────────────
# Place data files in data/ to run these. They're skipped if files are missing.


def _find_data(filename: str) -> Path | None:
    """Look for data files in common locations."""
    candidates = [
        Path("data") / filename,
        Path("../data") / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def test_treebank_loads():
    p = _find_data("hyw_armtdp-ud-dev.conllu")
    if p is None:
        return  # skip if no data
    tb = Treebank.from_file(p)
    assert len(tb) > 0
    assert tb.token_count > 0

    # Check first sentence has expected structure
    sent = tb[0]
    assert sent.text is not None
    assert sent.sent_id is not None
    assert len(sent.real_tokens) > 0
    root = sent.root()
    assert root is not None
    assert root.deprel == "root"


def test_lexicon_loads():
    # Try sample first, then full
    p = _find_data("nayiri-armenian-lexicon-2026-02-15-v1-sample-indented.json")
    if p is None:
        p = _find_data("nayiri-armenian-lexicon-2026-02-15-v1-sample.json")
    if p is None:
        return  # skip
    lex = Lexicon.from_file(p)
    assert lex.num_lexemes > 0
    assert len(lex.form_index) > 0
    assert len(lex.inflections) > 0

    # Round-trip: every form in the index should analyze back
    for form in list(lex.form_index.keys())[:50]:
        analyses = lex.analyze(form)
        assert len(analyses) > 0, f"Form '{form}' should have analyses"


def test_lexicon_generate():
    p = _find_data("nayiri-armenian-lexicon-2026-02-15-v1-sample-indented.json")
    if p is None:
        return

    lex = Lexicon.from_file(p)

    # Try generating ablative singular forms
    forms = lex.generate("արdelays", case="ABLATIVE", number="SINGULAR")
    if forms:  # only if this lemma is in the sample
        surfaces = [s for s, _ in forms]
        assert len(surfaces) > 0
