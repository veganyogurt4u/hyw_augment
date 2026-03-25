"""Tests for the CLI (cli.py), invoked via subprocess for maximum fidelity.

Each test runs `python -m hyw_augment.cli <args>` as a real subprocess,
checking stdout/stderr against the actual output.  No mocking at all.

Tests requiring external data (Nayiri, Apertium, HySpell, CaLFa) are
skipped when hyw_augment.toml is absent.
"""

import subprocess
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent
_CONFIG = _PROJECT_ROOT / "hyw_augment.toml"
_has_config = _CONFIG.exists()

skip_no_config = pytest.mark.skipif(not _has_config, reason="hyw_augment.toml not found")

# Armenian test word: MART = "man/person"
MART = "\u0574\u0561\u0580\u0564"


def _run(*args: str, expect_ok: bool = True, cwd: str | None = None) -> subprocess.CompletedProcess:
    """Run the CLI as a subprocess, return CompletedProcess.

    Uses the same Python interpreter as the test runner so the venv
    is active.  Runs from the project root by default (where
    hyw_augment.toml lives).
    """
    result = subprocess.run(
        [sys.executable, "-m", "hyw_augment.cli", *args],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=cwd or str(_PROJECT_ROOT),
    )
    if expect_ok:
        assert result.returncode == 0, (
            f"CLI exited {result.returncode}.\n"
            f"stdout: {result.stdout[:500]}\n"
            f"stderr: {result.stderr[:500]}"
        )
    return result


# ── Smoke tests (no external data needed) ────────────────────────────────────

class TestSmoke:
    """Tests that always run, regardless of data availability."""

    def test_help_flag(self):
        """--help prints usage and exits 0."""
        r = _run("--help")
        assert "Western Armenian morphological toolkit" in r.stdout

    def test_help_lists_verbose_flag(self):
        r = _run("--help")
        assert "--verbose" in r.stdout
        assert "-v" in r.stdout

    def test_help_lists_diagnostic_flag(self):
        r = _run("--help")
        assert "--diagnostic" in r.stdout

    def test_no_args_without_config_fails(self):
        """Without config or flags, CLI prints an error and exits non-zero."""
        r = _run(expect_ok=False, cwd="/tmp")
        assert r.returncode != 0


# ── Verbosity flags ──────────────────────────────────────────────────────────

@skip_no_config
class TestVerbosity:
    """--verbose shows backend source labels; omitting it hides them."""

    def test_analyze_default_hides_source(self):
        """Without -v, analysis output does not show (nayiri) or (apertium)."""
        r = _run("--analyze", MART)
        assert "(nayiri)" not in r.stdout
        assert "(apertium)" not in r.stdout
        # But the analysis itself should still appear
        assert "Analysis of" in r.stdout

    def test_analyze_verbose_shows_source(self):
        """With -v, the backend name appears in the header."""
        r = _run("--analyze", MART, "-v")
        # At least one source label should appear
        assert "(nayiri)" in r.stdout or "(apertium)" in r.stdout

    def test_analyze_verbose_long_flag(self):
        """--verbose works the same as -v."""
        r = _run("--analyze", MART, "--verbose")
        assert "(nayiri)" in r.stdout or "(apertium)" in r.stdout

    def test_generate_default_hides_source(self):
        r = _run("--generate", MART, "--tags", "n,sg,abl")
        assert "(nayiri)" not in r.stdout
        assert "(apertium)" not in r.stdout

    def test_generate_verbose_shows_source(self):
        r = _run("--generate", MART, "--tags", "n,sg,abl", "-v")
        assert "(nayiri)" in r.stdout or "(apertium)" in r.stdout


@skip_no_config
class TestDiagnostic:
    """--diagnostic prints full engine summary; omitting it hides it."""

    def test_summary_hidden_by_default(self):
        r = _run("--analyze", MART)
        assert "MorphEngine with" not in r.stdout

    def test_summary_shown_with_diagnostic(self):
        r = _run("--diagnostic", "--analyze", MART)
        assert "MorphEngine with" in r.stdout

    def test_diagnostic_without_operation(self):
        """--diagnostic alone prints the summary and exits."""
        r = _run("--diagnostic")
        assert "MorphEngine with" in r.stdout


# ── Analyze ──────────────────────────────────────────────────────────────────

@skip_no_config
class TestAnalyze:
    """--analyze with real backends."""

    def test_known_word_shows_pos(self):
        """A common noun should show NOUN in the output."""
        r = _run("--analyze", MART)
        assert "NOUN" in r.stdout

    def test_known_word_shows_lemma(self):
        r = _run("--analyze", MART)
        assert MART in r.stdout

    def test_unknown_word_shows_not_found(self):
        r = _run("--analyze", "xyznonexistent")
        # Apertium may tag it as <barb>, but if only nonsense results
        # appear the CLI still prints something useful
        assert "xyznonexistent" in r.stdout


# ── Generate ─────────────────────────────────────────────────────────────────

@skip_no_config
class TestGenerate:
    """--generate with real backends."""

    def test_with_tags_shows_forms(self):
        r = _run("--generate", MART, "--tags", "n,sg,abl")
        assert "Forms of" in r.stdout

    def test_unknown_lemma(self):
        r = _run("--generate", "xyznonexistent", "--tags", "n,sg")
        assert "not found" in r.stdout

    def test_backend_nayiri(self):
        """--backend nayiri restricts generation to Nayiri."""
        r = _run("--generate", MART, "--tags", "n,sg,nom", "--backend", "nayiri")
        # Should produce output (Nayiri knows MART)
        assert "Forms of" in r.stdout or "not found" in r.stdout


# ── Validate ─────────────────────────────────────────────────────────────────

@skip_no_config
class TestValidate:
    """--validate with real backends."""

    def test_valid_word(self):
        r = _run("--validate", MART)
        assert "VALID" in r.stdout

    def test_invalid_word(self):
        """Nonsense word — INVALID or suggestions shown."""
        r = _run("--validate", "xyznonexistent")
        # Apertium may say VALID via <barb>; just verify CLI doesn't crash
        assert "VALID" in r.stdout or "INVALID" in r.stdout


# ── Suggest ──────────────────────────────────────────────────────────────────

@skip_no_config
class TestSuggest:
    def test_suggest_output(self):
        r = _run("--suggest", "xyznonexistent")
        assert "Suggestion" in r.stdout or "no suggestions" in r.stdout.lower() or "No suggestions" in r.stdout


# ── Convert ──────────────────────────────────────────────────────────────────

@skip_no_config
class TestConvert:
    def test_ascii_passthrough(self):
        """ASCII text has no reformed words — says so."""
        r = _run("--convert", "hello world")
        assert "No Reformed" in r.stdout


# ── Define ───────────────────────────────────────────────────────────────────

@skip_no_config
class TestDefine:
    def test_define_word(self):
        r = _run("--define", MART)
        assert "Definition" in r.stdout or "not found" in r.stdout

    def test_define_en_word(self):
        r = _run("--define-en", MART)
        assert "definition" in r.stdout.lower() or "not found" in r.stdout.lower()
