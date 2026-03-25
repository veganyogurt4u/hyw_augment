"""Tests for spelling.py — pipe protocol parsing and real hunspell integration.

Unit tests (top half) use io.StringIO to fake the hunspell process's stdout,
verifying the pipe protocol parser without needing a hunspell binary.

Integration tests (bottom half) run real hunspell against the HySpell hy-c
dictionary.  They are skipped when HySpell data is absent.
"""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hyw_augment.spelling import _SUGGEST_RE, SpellChecker

# ── _SUGGEST_RE regex ─────────────────────────────────────────────────────────

def test_suggest_re_matches_suggestion_line():
    m = _SUGGEST_RE.match("& misspelled 2 0: word1, word2")
    assert m is not None
    assert m.group(1) == "word1, word2"


def test_suggest_re_single_suggestion():
    m = _SUGGEST_RE.match("& word 1 0: onlyone")
    assert m is not None
    assert m.group(1) == "onlyone"


def test_suggest_re_no_match_asterisk():
    assert _SUGGEST_RE.match("* ") is None


def test_suggest_re_no_match_hash():
    assert _SUGGEST_RE.match("# misspelled 0") is None


def test_suggest_re_no_match_plus():
    assert _SUGGEST_RE.match("+ root") is None


# ── Fake-process helper ───────────────────────────────────────────────────────

def _make_spellchecker(result_line: str) -> SpellChecker:
    """Create a SpellChecker with a fake hunspell process.

    stdout is a StringIO containing the result line followed by the blank
    separator that hunspell pipe mode emits after each word.
    poll() returns None so _get_proc() treats the fake proc as still running
    and reuses it instead of spawning a real process.
    """
    sc = SpellChecker.__new__(SpellChecker)
    sc.available = True
    sc._hunspell = "/fake/hunspell"
    sc._dict_base = "/fake/hy-c"
    sc.dict_dir = Path("/fake")

    stdout = io.StringIO(f"{result_line}\n\n")

    # Include "wait" and "kill" in spec so close() / __del__ don't raise
    proc = MagicMock(spec=["poll", "stdin", "stdout", "wait", "kill"])
    proc.poll.return_value = None   # "process is running"
    proc.stdout = stdout
    proc.stdin = MagicMock()

    sc._proc = proc
    return sc


# ── SpellChecker.check ────────────────────────────────────────────────────────

def test_check_asterisk_is_valid():
    assert _make_spellchecker("* ").check("word") is True


def test_check_plus_is_valid():
    # + root means correct via compound/root analysis
    assert _make_spellchecker("+ root").check("word") is True


def test_check_minus_is_valid():
    # - root means correct via affix rules
    assert _make_spellchecker("- root").check("word") is True


def test_check_ampersand_is_invalid():
    assert _make_spellchecker("& word 2 0: sug1, sug2").check("word") is False


def test_check_hash_is_invalid():
    assert _make_spellchecker("# word 0").check("word") is False


def test_check_unavailable_returns_false():
    assert _unavailable_spellchecker().check("word") is False


# ── SpellChecker.suggest ──────────────────────────────────────────────────────

def test_suggest_returns_list_from_ampersand_line():
    assert _make_spellchecker("& word 2 0: sug1, sug2").suggest("word") == ["sug1", "sug2"]


def test_suggest_single_suggestion():
    assert _make_spellchecker("& word 1 0: onlyone").suggest("word") == ["onlyone"]


def test_suggest_hash_returns_empty():
    assert _make_spellchecker("# word 0").suggest("word") == []


def test_suggest_asterisk_returns_empty():
    # Correct word — no suggestions expected
    assert _make_spellchecker("* ").suggest("word") == []


def test_suggest_strips_whitespace_around_each():
    # Hunspell may pad suggestions with spaces
    result = _make_spellchecker("& word 3 0:  one , two , three ").suggest("word")
    assert result == ["one", "two", "three"]


def test_suggest_unavailable_returns_empty():
    assert _unavailable_spellchecker().suggest("word") == []


# ── SpellChecker.check_and_suggest ───────────────────────────────────────────

def test_check_and_suggest_valid_word():
    valid, suggestions = _make_spellchecker("* ").check_and_suggest("word")
    assert valid is True
    assert suggestions == []


def test_check_and_suggest_invalid_with_suggestions():
    valid, suggestions = _make_spellchecker("& word 2 0: sug1, sug2").check_and_suggest("word")
    assert valid is False
    assert suggestions == ["sug1", "sug2"]


def test_check_and_suggest_invalid_no_suggestions():
    valid, suggestions = _make_spellchecker("# word 0").check_and_suggest("word")
    assert valid is False
    assert suggestions == []


def test_check_and_suggest_unavailable():
    valid, suggestions = _unavailable_spellchecker().check_and_suggest("word")
    assert valid is False
    assert suggestions == []


# ── SpellChecker.check_batch (mocks subprocess.run) ──────────────────────────

def _batch_stdout(*results: str) -> str:
    """Build a fake hunspell -a batch stdout string.

    Structure: banner line, then for each word: result line + blank line.
    """
    lines = ["Hunspell 1.7.0\n"]
    for r in results:
        lines.append(r + "\n")
        lines.append("\n")
    return "".join(lines)


def _unavailable_spellchecker() -> SpellChecker:
    """SpellChecker with available=False and _proc initialised to avoid __del__ errors."""
    sc = SpellChecker.__new__(SpellChecker)
    sc.available = False
    sc._proc = None
    return sc


def _batch_spellchecker() -> SpellChecker:
    sc = SpellChecker.__new__(SpellChecker)
    sc.available = True
    sc._hunspell = "/fake/hunspell"
    sc._dict_base = "/fake/hy-c"
    sc._proc = None  # close() / __del__ check this before accessing _proc
    return sc


def test_check_batch_all_valid():
    sc = _batch_spellchecker()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout=_batch_stdout("* ", "* "))
        result = sc.check_batch(["w1", "w2"])
    assert result == {"w1": True, "w2": True}


def test_check_batch_mixed_valid_invalid():
    sc = _batch_spellchecker()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout=_batch_stdout("* ", "# w2 0"))
        result = sc.check_batch(["w1", "w2"])
    assert result["w1"] is True
    assert result["w2"] is False


def test_check_batch_unavailable_returns_empty():
    assert _unavailable_spellchecker().check_batch(["word"]) == {}


def test_check_batch_empty_input_returns_empty():
    sc = _batch_spellchecker()
    assert sc.check_batch([]) == {}


# ── SpellChecker.suggest_batch (mocks subprocess.run) ────────────────────────

def test_suggest_batch_returns_suggestions():
    sc = _batch_spellchecker()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            stdout=_batch_stdout("& w1 2 0: sug1, sug2", "* ")
        )
        result = sc.suggest_batch(["w1", "w2"])
    assert result["w1"] == ["sug1", "sug2"]
    assert result["w2"] == []


def test_suggest_batch_no_suggestions():
    sc = _batch_spellchecker()
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout=_batch_stdout("# w1 0"))
        result = sc.suggest_batch(["w1"])
    assert result["w1"] == []


def test_suggest_batch_unavailable_returns_empty():
    sc = _unavailable_spellchecker()
    assert sc.suggest_batch(["word"]) == {}


def test_suggest_batch_empty_input_returns_empty():
    sc = _batch_spellchecker()
    assert sc.suggest_batch([]) == {}


# ── Integration tests (real hunspell + hy-c dictionary) ──────────────────────
# These exercise the full SpellChecker against the real HySpell hy-c dictionary.
# Skipped when HySpell data is not configured.

# Armenian test word: MART = "man/person"
_MART = "\u0574\u0561\u0580\u0564"


class TestSpellCheckerIntegration:
    """Real hunspell integration — verifies actual spell-checking behavior."""

    @pytest.fixture(autouse=True)
    def _setup(self, spellcheck_dir):
        """Build a real SpellChecker from the HySpell Dictc directory."""
        self.sc = SpellChecker(spellcheck_dir)
        assert self.sc.available, (
            f"hunspell binary or hy-c dictionary not usable from {spellcheck_dir}"
        )
        yield
        self.sc.close()

    def test_valid_armenian_word(self):
        """A common Armenian word should be accepted by hunspell."""
        assert self.sc.check(_MART) is True

    def test_invalid_nonsense(self):
        """A nonsense ASCII string should be rejected."""
        assert self.sc.check("xyznonexistent") is False

    def test_suggest_returns_list(self):
        """Suggestions for a nonsense word should be a (possibly empty) list."""
        result = self.sc.suggest("xyznonexistent")
        assert isinstance(result, list)

    def test_check_and_suggest_valid(self):
        """A valid word: check=True, suggestions=[]."""
        valid, suggestions = self.sc.check_and_suggest(_MART)
        assert valid is True
        assert suggestions == []

    def test_check_batch_mixed(self):
        """Batch with one valid and one invalid word."""
        results = self.sc.check_batch([_MART, "xyznonexistent"])
        assert results[_MART] is True
        assert results["xyznonexistent"] is False

    def test_suggest_batch(self):
        """Batch suggest returns a dict mapping words to suggestion lists."""
        results = self.sc.suggest_batch(["xyznonexistent"])
        assert isinstance(results["xyznonexistent"], list)
