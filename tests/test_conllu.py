"""Tests for the CoNLL-U parser (conllu.py)."""

from pathlib import Path

import pytest

from hyw_augment.conllu import (
    Sentence,
    Token,
    Treebank,
    _parse_conllu,
    _parse_feats,
    _parse_misc,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

# Minimal two-sentence CoNLL-U snippet (uses ASCII tokens to keep it portable)
SAMPLE_CONLLU = """\
# sent_id = s1
# text = The cat sat.
1\tThe\tthe\tDET\t_\t_\t2\tdet\t_\tTranslit=The|LTranslit=the
2\tcat\tcat\tNOUN\t_\tNumber=Sing\t3\tnsubj\t_\tTranslit=cat
3\tsat\tsit\tVERB\t_\tTense=Past\t0\troot\t_\tSpaceAfter=No
4\t.\t.\tPUNCT\t_\t_\t3\tpunct\t_\t_

# sent_id = s2
# text = Dogs run.
1\tDogs\tdog\tNOUN\t_\tNumber=Plur\t2\tnsubj\t_\t_
2\trun\trun\tVERB\t_\tTense=Pres\t0\troot\t_\tSpaceAfter=No
3\t.\t.\tPUNCT\t_\t_\t2\tpunct\t_\t_

"""

# Sentence with a multiword token (MWT)
MWT_CONLLU = """\
# sent_id = mwt1
# text = don't worry
1-2\tdon't\t_\t_\t_\t_\t_\t_\t_\t_
1\tdo\tdo\tAUX\t_\t_\t3\taux\t_\t_
2\t't\tnot\tPART\t_\t_\t3\tadvmod\t_\t_
3\tworry\tworry\tVERB\t_\t_\t0\troot\t_\t_

"""


def _find_data(filename: str) -> Path | None:
    for base in [Path("data"), Path("../data")]:
        p = base / filename
        if p.exists():
            return p
    return None


# ── _parse_feats ──────────────────────────────────────────────────────────────

def test_parse_feats_underscore():
    assert _parse_feats("_") == {}


def test_parse_feats_single():
    assert _parse_feats("Number=Sing") == {"Number": "Sing"}


def test_parse_feats_multiple():
    assert _parse_feats("Case=Nom|Number=Sing") == {"Case": "Nom", "Number": "Sing"}


def test_parse_feats_many():
    result = _parse_feats("Aspect=Imp|Mood=Ind|Number=Plur|Person=1")
    assert result == {"Aspect": "Imp", "Mood": "Ind", "Number": "Plur", "Person": "1"}


# ── _parse_misc ───────────────────────────────────────────────────────────────

def test_parse_misc_underscore():
    assert _parse_misc("_") == {}


def test_parse_misc_space_after_no():
    assert _parse_misc("SpaceAfter=No") == {"SpaceAfter": "No"}


def test_parse_misc_multiple():
    result = _parse_misc("Translit=ays|LTranslit=ays")
    assert result == {"Translit": "ays", "LTranslit": "ays"}


def test_parse_misc_bare_flag():
    # A bare flag (no =) should be stored with empty string value
    result = _parse_misc("SpaceAfter")
    assert result == {"SpaceAfter": ""}


# ── Token properties ──────────────────────────────────────────────────────────

def _make_token(**kwargs) -> Token:
    defaults = dict(
        id="1", form="word", lemma="word", upos="NOUN", xpos="_",
        feats={}, head="0", deprel="root", deps="_", misc={}
    )
    defaults.update(kwargs)
    return Token(**defaults)


def test_token_is_multiword_false():
    t = _make_token(id="1")
    assert not t.is_multiword


def test_token_is_multiword_true():
    t = _make_token(id="1-2")
    assert t.is_multiword


def test_token_is_empty_false():
    t = _make_token(id="1")
    assert not t.is_empty


def test_token_is_empty_true():
    t = _make_token(id="1.1")
    assert t.is_empty


def test_token_translit_present():
    t = _make_token(misc={"Translit": "abc"})
    assert t.translit == "abc"


def test_token_translit_absent():
    t = _make_token(misc={})
    assert t.translit is None


def test_token_space_after_default():
    t = _make_token(misc={})
    assert t.space_after is True


def test_token_space_after_no():
    t = _make_token(misc={"SpaceAfter": "No"})
    assert t.space_after is False


def test_token_feat():
    t = _make_token(feats={"Number": "Sing", "Case": "Nom"})
    assert t.feat("Number") == "Sing"
    assert t.feat("Case") == "Nom"
    assert t.feat("Tense") is None


# ── _parse_conllu ─────────────────────────────────────────────────────────────

def test_parse_conllu_sentence_count():
    sentences = _parse_conllu(SAMPLE_CONLLU)
    assert len(sentences) == 2


def test_parse_conllu_metadata():
    sentences = _parse_conllu(SAMPLE_CONLLU)
    assert sentences[0].sent_id == "s1"
    assert sentences[0].text == "The cat sat."
    assert sentences[1].sent_id == "s2"


def test_parse_conllu_token_count():
    sentences = _parse_conllu(SAMPLE_CONLLU)
    # Both sentences have 4 tokens (including punctuation)
    assert len(sentences[0].tokens) == 4
    assert len(sentences[1].tokens) == 3


def test_parse_conllu_token_fields():
    sentences = _parse_conllu(SAMPLE_CONLLU)
    cat_tok = sentences[0].tokens[1]  # "cat"
    assert cat_tok.form == "cat"
    assert cat_tok.lemma == "cat"
    assert cat_tok.upos == "NOUN"
    assert cat_tok.feats == {"Number": "Sing"}
    assert cat_tok.head == "3"
    assert cat_tok.deprel == "nsubj"


def test_parse_conllu_misc_parsed():
    sentences = _parse_conllu(SAMPLE_CONLLU)
    first_tok = sentences[0].tokens[0]  # "The"
    assert first_tok.translit == "The"
    assert first_tok.lemma_translit == "the"


def test_parse_conllu_multiword_token():
    sentences = _parse_conllu(MWT_CONLLU)
    assert len(sentences) == 1
    sent = sentences[0]
    # Four token rows: 1-2, 1, 2, 3
    assert len(sent.tokens) == 4
    assert sent.tokens[0].is_multiword
    assert not sent.tokens[1].is_multiword


# ── Sentence methods ──────────────────────────────────────────────────────────

def test_sentence_real_tokens_excludes_mwt():
    sentences = _parse_conllu(MWT_CONLLU)
    sent = sentences[0]
    real = sent.real_tokens
    assert all(not t.is_multiword for t in real)
    assert len(real) == 3  # tokens 1, 2, 3


def test_sentence_words():
    sentences = _parse_conllu(SAMPLE_CONLLU)
    assert sentences[0].words == ["The", "cat", "sat", "."]


def test_sentence_lemmas():
    sentences = _parse_conllu(SAMPLE_CONLLU)
    assert sentences[0].lemmas == ["the", "cat", "sit", "."]


def test_sentence_by_upos():
    sentences = _parse_conllu(SAMPLE_CONLLU)
    nouns = sentences[0].by_upos("NOUN")
    assert len(nouns) == 1
    assert nouns[0].form == "cat"


def test_sentence_by_upos_multiple():
    sentences = _parse_conllu(SAMPLE_CONLLU)
    nouns_verbs = sentences[0].by_upos("NOUN", "VERB")
    assert {t.form for t in nouns_verbs} == {"cat", "sat"}


def test_sentence_root():
    sentences = _parse_conllu(SAMPLE_CONLLU)
    root = sentences[0].root()
    assert root is not None
    assert root.form == "sat"
    assert root.deprel == "root"


def test_sentence_root_none_if_missing():
    sent = Sentence()
    assert sent.root() is None


# ── Treebank methods ──────────────────────────────────────────────────────────

def test_treebank_from_text_via_parse():
    sentences = _parse_conllu(SAMPLE_CONLLU)
    tb = Treebank(sentences)
    assert len(tb) == 2


def test_treebank_token_count():
    sentences = _parse_conllu(SAMPLE_CONLLU)
    tb = Treebank(sentences)
    # s1: 4 real tokens (no MWT), s2: 3
    assert tb.token_count == 7


def test_treebank_iteration():
    sentences = _parse_conllu(SAMPLE_CONLLU)
    tb = Treebank(sentences)
    ids = [s.sent_id for s in tb]
    assert ids == ["s1", "s2"]


def test_treebank_getitem():
    sentences = _parse_conllu(SAMPLE_CONLLU)
    tb = Treebank(sentences)
    assert tb[0].sent_id == "s1"
    assert tb[1].sent_id == "s2"


def test_treebank_unique_forms():
    sentences = _parse_conllu(SAMPLE_CONLLU)
    tb = Treebank(sentences)
    forms = tb.unique_forms()
    assert "cat" in forms
    assert "Dogs" in forms


def test_treebank_unique_lemmas():
    sentences = _parse_conllu(SAMPLE_CONLLU)
    tb = Treebank(sentences)
    lemmas = tb.unique_lemmas()
    assert "cat" in lemmas
    assert "dog" in lemmas


def test_treebank_vocab():
    sentences = _parse_conllu(SAMPLE_CONLLU)
    tb = Treebank(sentences)
    vocab = tb.vocab()
    # "dog" lemma appears as surface form "Dogs"
    assert "dog" in vocab
    assert "Dogs" in vocab["dog"]


def test_treebank_pos_distribution():
    sentences = _parse_conllu(SAMPLE_CONLLU)
    tb = Treebank(sentences)
    dist = tb.pos_distribution()
    assert "NOUN" in dist
    assert "VERB" in dist
    assert dist["NOUN"] == 2  # cat (s1) + Dogs (s2)


def test_treebank_deprel_distribution():
    sentences = _parse_conllu(SAMPLE_CONLLU)
    tb = Treebank(sentences)
    dist = tb.deprel_distribution()
    assert "root" in dist
    assert "nsubj" in dist


# ── Integration test (requires real data file) ────────────────────────────────

def test_treebank_from_real_file():
    p = _find_data("hyw_armtdp-ud-dev.conllu")
    if p is None:
        pytest.skip("dev treebank not found")
    tb = Treebank.from_file(p)
    assert len(tb) > 100
    assert tb.token_count > 1000
    sent = tb[0]
    assert sent.sent_id is not None
    assert sent.root() is not None
