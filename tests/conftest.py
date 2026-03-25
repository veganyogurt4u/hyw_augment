"""Shared test fixtures.

Provides helpers and pytest fixtures used across the test suite.

Session-scoped fixtures (full_engine, nayiri_fixture) are loaded once
and reused across all tests in a run for speed.  Every fixture that
depends on external data skips gracefully when the data is absent.
"""

import tomllib
from pathlib import Path

import pytest


# ── Path helpers ──────────────────────────────────────────────────────────────

def _find_config() -> Path | None:
    """Find hyw_augment.toml from the project root."""
    for base in [Path("."), Path("..")]:
        p = base / "hyw_augment.toml"
        if p.exists():
            return p.resolve()
    return None


def find_data(filename: str) -> Path | None:
    """Find a data file relative to the project root."""
    for base in [Path("data"), Path("../data")]:
        p = base / filename
        if p.exists():
            return p
    return None


# ── Lightweight fixtures (no external data) ───────────────────────────────────

@pytest.fixture(scope="session")
def nayiri_fixture():
    """A real Nayiri Lexicon built from the in-memory FIXTURE dict.

    Uses ASCII placeholder forms (noun-alpha, verb-beta, etc.) so it
    works everywhere without data files.  Returns a real Lexicon object
    — not a mock — that exercises real analyze/generate/is_valid_form paths.
    """
    from hyw_augment.nayiri import Lexicon
    from tests.test_nayiri import FIXTURE

    return Lexicon.from_dict(FIXTURE)


# ── Config-dependent fixtures ─────────────────────────────────────────────────

@pytest.fixture(scope="session")
def config_path() -> Path:
    """Resolve hyw_augment.toml; skip the entire session fixture chain if absent."""
    p = _find_config()
    if p is None:
        pytest.skip("hyw_augment.toml not found")
    return p


@pytest.fixture(scope="session")
def full_engine(config_path):
    """A fully-loaded MorphEngine from hyw_augment.toml.

    Session-scoped so the 66 MB Nayiri JSON and Apertium pipes are
    loaded only once.  Yields the engine and closes it after the session.
    """
    from hyw_augment.engine import MorphEngine

    engine = MorphEngine.from_config(config_path)
    yield engine
    engine.close()


@pytest.fixture(scope="session")
def hyspell_dir(config_path) -> Path:
    """Resolve the HySpell Dictionaries path from hyw_augment.toml.

    Skips the test if the config or directory is not found.
    """
    with config_path.open("rb") as f:
        cfg = tomllib.load(f)

    hs_dir = cfg.get("hyspell", {}).get("dir")
    if not hs_dir:
        pytest.skip("[hyspell] dir not configured in hyw_augment.toml")

    hp = Path(hs_dir)
    if not hp.is_absolute():
        hp = config_path.parent / hp

    if not hp.exists():
        pytest.skip(f"HySpell directory not found: {hp}")

    return hp


@pytest.fixture(scope="session")
def spellcheck_dir(hyspell_dir) -> Path:
    """The Dictc subdirectory containing the hy-c Hunspell dictionary."""
    p = hyspell_dir / "Dictc"
    if not p.exists():
        pytest.skip(f"Dictc not found: {p}")
    return p
