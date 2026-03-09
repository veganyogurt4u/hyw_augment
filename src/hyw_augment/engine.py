"""
Unified morphological analysis engine with fallback chain.

Orchestrates multiple backends (Nayiri lexicon, Apertium transducer, etc.)
behind a single interface, with TOML-based configuration.

Usage:
    from hyw_augment.engine import MorphEngine

    engine = MorphEngine.from_config()          # loads hyw_augment.toml
    results = engine.analyze("some_form")       # tries backends in order
    all_results = engine.analyze_all("form")    # returns results from every backend

    # Or build manually:
    engine = MorphEngine()
    engine.add_nayiri("data/nayiri.json", "data/words.json")
    engine.add_apertium("/path/to/apertium-hyw")
"""

from __future__ import annotations

import glob
import logging
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AnalysisResult:
    """Wrapper that tags an analysis with its source backend.

    Delegates .lemma, .pos, .description_en etc. to the inner analysis
    object (MorphAnalysis or ApertiumAnalysis — both have these attributes).
    """

    source: str  # "nayiri", "apertium", etc.
    analysis: Any  # MorphAnalysis | ApertiumAnalysis

    @property
    def lemma(self) -> str:
        return self.analysis.lemma

    @property
    def pos(self) -> str:
        return self.analysis.pos

    @property
    def description_en(self) -> str:
        return self.analysis.description_en

    @property
    def case(self) -> str | None:
        return self.analysis.case

    @property
    def number(self) -> str | None:
        return self.analysis.number

    @property
    def person(self) -> str | None:
        return self.analysis.person

    @property
    def article(self) -> str | None:
        return getattr(self.analysis, "article", None)

    def __repr__(self) -> str:
        return f"AnalysisResult({self.source}: {self.analysis!r})"


class MorphEngine:
    """Orchestrates multiple morphological backends with ordered fallback.

    Backends are tried in the order they were added.  The first backend
    that returns results wins for .analyze(); .analyze_all() queries
    every backend.
    """

    def __init__(self):
        self.backends: list[tuple[str, Any]] = []  # (name, backend)
        self.treebank = None  # Treebank | None
        self.spellchecker = None  # SpellChecker | None
        self.orthography = None   # OrthographyConverter | None
        self.glossary = None      # Glossary | None
        self.calfa = None         # CaLFALexicon | None

    # ── Construction helpers ─────────────────────────────────────────────

    def add_nayiri(self, *paths: str | Path) -> None:
        """Add a Nayiri lexicon backend (one or more JSON files, merged)."""
        from hyw_augment.nayiri import Lexicon

        resolved = _expand_paths(paths)
        if not resolved:
            return
        lex = Lexicon.from_files(*resolved)
        self.backends.append(("nayiri", lex))

    def add_apertium(self, apertium_dir: str | Path) -> None:
        """Add an Apertium transducer backend."""
        from hyw_augment.apertium import ApertiumAnalyzer

        apt = ApertiumAnalyzer(apertium_dir)
        if apt.available:
            self.backends.append(("apertium", apt))

    def add_spellcheck(self, dict_dir: str | Path) -> None:
        """Add a Hunspell spell checker (HySpell hy-c dictionary)."""
        from hyw_augment.spelling import SpellChecker

        sc = SpellChecker(dict_dir)
        if sc.available:
            self.spellchecker = sc

    def add_orthography(self, dict_dir: str | Path) -> None:
        """Add an orthography converter (Reformed -> Classical)."""
        from hyw_augment.orthography import OrthographyConverter

        self.orthography = OrthographyConverter(dict_dir)

    def add_glossary(self, path: str | Path) -> None:
        """Add a glossary (SmallArmDic definitions)."""
        from hyw_augment.glossary import Glossary

        self.glossary = Glossary.from_file(path)

    def add_calfa(self, calfa_dir: str | Path) -> None:
        """Add the Calfa English-definitions lexicon."""
        from hyw_augment.calfa import CaLFALexicon

        self.calfa = CaLFALexicon.from_dir(calfa_dir)

    def load_treebank(self, *paths: str | Path) -> None:
        """Load UD treebank files."""
        from hyw_augment.conllu import Treebank

        resolved = _expand_paths(paths)
        if not resolved:
            return
        self.treebank = Treebank.from_files(*resolved)

    @classmethod
    def from_config(cls, config_path: str | Path = "hyw_augment.toml") -> MorphEngine:
        """Build a MorphEngine from a TOML config file.

        Paths in the config are resolved relative to the config file's
        directory.  Glob patterns in paths are expanded.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with config_path.open("rb") as f:
            cfg = tomllib.load(f)

        base_dir = config_path.parent
        engine = cls()

        # Nayiri
        nayiri_cfg = cfg.get("nayiri", {})
        nayiri_paths = nayiri_cfg.get("paths", [])
        if nayiri_paths:
            resolved = _resolve_config_paths(nayiri_paths, base_dir)
            if resolved:
                engine.add_nayiri(*resolved)

        # Apertium
        apt_cfg = cfg.get("apertium", {})
        apt_dir = apt_cfg.get("dir")
        if apt_dir:
            apt_path = Path(apt_dir)
            if not apt_path.is_absolute():
                apt_path = base_dir / apt_path
            engine.add_apertium(apt_path)

        # Treebank
        tb_cfg = cfg.get("treebank", {})
        tb_paths = tb_cfg.get("paths", [])
        if tb_paths:
            resolved = _resolve_config_paths(tb_paths, base_dir)
            if resolved:
                engine.load_treebank(*resolved)

        # HySpell (spell checker, orthography converter, glossary)
        hyspell_cfg = cfg.get("hyspell", {})
        hyspell_dir = hyspell_cfg.get("dir")
        if hyspell_dir:
            hp = Path(hyspell_dir)
            if not hp.is_absolute():
                hp = base_dir / hp
            engine.add_spellcheck(hp / "Dictc")
            engine.add_orthography(hp)
            glossary_path = hp / "SmallArmDic.txt"
            if glossary_path.exists():
                engine.add_glossary(glossary_path)

        # Calfa English-definitions lexicon
        calfa_cfg = cfg.get("calfa", {})
        calfa_dir = calfa_cfg.get("dir")
        if calfa_dir:
            cp = Path(calfa_dir)
            if not cp.is_absolute():
                cp = base_dir / cp
            engine.add_calfa(cp)

        return engine

    # ── Analysis ─────────────────────────────────────────────────────────

    def analyze(self, form: str) -> list[AnalysisResult]:
        """Analyze a form using the first backend that returns results."""
        for name, backend in self.backends:
            raw = backend.analyze(form)
            if not raw and hasattr(backend, "analyze_insensitive"):
                raw = backend.analyze_insensitive(form)
            if raw:
                return [AnalysisResult(source=name, analysis=a) for a in raw]
        return []

    def analyze_all(self, form: str) -> dict[str, list[AnalysisResult]]:
        """Analyze a form against every backend, returning all results."""
        results: dict[str, list[AnalysisResult]] = {}
        for name, backend in self.backends:
            raw = backend.analyze(form)
            if not raw and hasattr(backend, "analyze_insensitive"):
                raw = backend.analyze_insensitive(form)
            if raw:
                results[name] = [AnalysisResult(source=name, analysis=a) for a in raw]
        return results

    def analyze_batch(
        self, forms: list[str],
    ) -> dict[str, list[AnalysisResult]]:
        """Batch-analyze forms: try Nayiri first (dict lookup), then batch
        the remaining misses through Apertium in one subprocess call."""
        results: dict[str, list[AnalysisResult]] = {}
        remaining = list(forms)

        for name, backend in self.backends:
            if not remaining:
                break

            if hasattr(backend, "analyze_batch"):
                batch = backend.analyze_batch(remaining)
                still_missing = []
                for form in remaining:
                    hits = batch.get(form, [])
                    if hits:
                        results[form] = [AnalysisResult(source=name, analysis=a) for a in hits]
                    else:
                        still_missing.append(form)
                remaining = still_missing
            else:
                still_missing = []
                for form in remaining:
                    raw = backend.analyze(form)
                    if not raw and hasattr(backend, "analyze_insensitive"):
                        raw = backend.analyze_insensitive(form)
                    if raw:
                        results[form] = [AnalysisResult(source=name, analysis=a) for a in raw]
                    else:
                        still_missing.append(form)
                remaining = still_missing

        return results

    # ── Generation ─────────────────────────────────────────────────────────

    def generate(self, lemma: str, tags: list[str] | None = None):
        """Generate surface forms for a lemma.

        When *tags* are provided (Apertium-style, e.g. ["n", "pl", "abl"]),
        Apertium is tried first; on miss the tags are translated to Nayiri
        keyword filters (unknown tags silently ignored).  When *tags* is
        None/empty, only Nayiri is queried (returns all forms for the lemma).

        Returns list[tuple[str, Inflection]].
        """
        if tags:
            return self._generate_with_tags(lemma, tags)
        return self._generate_wildcard(lemma)

    @staticmethod
    def _tags_to_nayiri_kwargs(tags: list[str]) -> dict[str, str]:
        """Translate Apertium tags to Nayiri generate() keyword filters."""
        from hyw_augment.apertium import (
            _APT_POS_MAP, _APT_CASE_MAP, _APT_NUMBER_MAP,
            _APT_PERSON_MAP, _APT_ARTICLE_MAP,
        )

        kwargs: dict[str, str] = {}
        for t in tags:
            if t in _APT_POS_MAP:
                kwargs["pos"] = _APT_POS_MAP[t]
            elif t in _APT_CASE_MAP:
                val = _APT_CASE_MAP[t]
                if "/" not in val:  # collapsed tags like "DATIVE/GENITIVE" → wildcard
                    kwargs["case"] = val
            elif t in _APT_NUMBER_MAP:
                kwargs["number"] = _APT_NUMBER_MAP[t]
            elif t in _APT_PERSON_MAP:
                kwargs["person"] = _APT_PERSON_MAP[t]
            elif t in _APT_ARTICLE_MAP:
                kwargs["article"] = _APT_ARTICLE_MAP[t]
            # Tags without a Nayiri equivalent are silently ignored
        return kwargs

    def _generate_with_tags(self, lemma: str, tags: list[str]) -> list:
        """Try Apertium first, fall back to Nayiri with tag translation."""
        # Apertium first
        for name, backend in self.backends:
            if name == "apertium" and hasattr(backend, "generate"):
                results = backend.generate(lemma, tags)
                if results:
                    return results

        # Nayiri fallback
        kwargs = self._tags_to_nayiri_kwargs(tags)
        for name, backend in self.backends:
            if name == "nayiri" and hasattr(backend, "generate"):
                results = backend.generate(lemma, **kwargs)
                if results:
                    return results

        return []

    def _generate_wildcard(self, lemma: str) -> list:
        """No tags: query Nayiri only (all forms for the lemma)."""
        for name, backend in self.backends:
            if name == "nayiri" and hasattr(backend, "generate"):
                results = backend.generate(lemma)
                if results:
                    return results
        return []

    def generate_all(
        self, lemma: str, tags: list[str] | None = None,
    ) -> dict[str, list]:
        """Generate forms from every backend, keyed by backend name."""
        nayiri_kwargs = self._tags_to_nayiri_kwargs(tags) if tags else {}
        results: dict[str, list] = {}

        for name, backend in self.backends:
            if not hasattr(backend, "generate"):
                continue
            if name == "apertium" and tags:
                hits = backend.generate(lemma, tags)
            elif name == "nayiri":
                hits = backend.generate(lemma, **nayiri_kwargs)
            else:
                continue
            if hits:
                results[name] = hits

        return results

    # ── Validation & spelling ─────────────────────────────────────────────

    def validate(self, form: str) -> bool:
        """Check if a word is valid in any backend or the spell checker.

        Tries (in order): Nayiri form_index, Apertium is_known, Hunspell check.
        """
        for name, backend in self.backends:
            if name == "nayiri":
                if backend.is_valid_form(form):
                    return True
            elif name == "apertium":
                if backend.is_known(form):
                    return True
        if self.spellchecker is not None:
            return self.spellchecker.check(form)
        return False

    def suggest(self, form: str) -> list[str]:
        """Get spelling suggestions for an invalid word."""
        if self.spellchecker is not None:
            return self.spellchecker.suggest(form)
        return []

    def convert_reformed(self, text: str) -> str:
        """Convert Reformed-orthography text to Classical."""
        if self.orthography is not None:
            return self.orthography.convert_text(text)
        return text

    def detect_reformed(self, text: str) -> list[tuple[str, str]]:
        """Find Reformed-orthography words in text.

        Returns list of (reformed, classical) pairs.
        """
        if self.orthography is not None:
            return self.orthography.detect_reformed_words(text)
        return []

    def lookup_definition(self, word: str):
        """Look up a word's definition in the glossary.

        Returns list of GlossaryEntry or None.
        """
        if self.glossary is not None:
            return self.glossary.lookup(word)
        return None

    def lookup_calfa(self, word: str):
        """Look up a word's English definition in the Calfa lexicon.

        Returns list of CaLFAEntry or None.
        """
        if self.calfa is not None:
            return self.calfa.lookup(word)
        return None

    def synonyms_calfa(self, word: str) -> list[str]:
        """Return Armenian synonyms from the Calfa synonyms lexicon."""
        if self.calfa is not None:
            return self.calfa.synonyms_for(word)
        return []

    # ── Introspection ────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [f"MorphEngine with {len(self.backends)} backend(s):"]
        for name, backend in self.backends:
            lines.append(f"  [{name}]")
            for sub_line in backend.summary().split("\n"):
                lines.append(f"    {sub_line}")
        if self.treebank:
            lines.append("  [treebank]")
            for sub_line in self.treebank.summary().split("\n"):
                lines.append(f"    {sub_line}")
        if self.spellchecker:
            lines.append("  [spellcheck]")
            for sub_line in self.spellchecker.summary().split("\n"):
                lines.append(f"    {sub_line}")
        if self.orthography:
            lines.append("  [orthography]")
            for sub_line in self.orthography.summary().split("\n"):
                lines.append(f"    {sub_line}")
        if self.glossary:
            lines.append("  [glossary]")
            for sub_line in self.glossary.summary().split("\n"):
                lines.append(f"    {sub_line}")
        if self.calfa:
            lines.append("  [calfa]")
            for sub_line in self.calfa.summary().split("\n"):
                lines.append(f"    {sub_line}")
        return "\n".join(lines)

    def close(self) -> None:
        """Clean up backends that hold resources (e.g. subprocesses)."""
        for _name, backend in self.backends:
            if hasattr(backend, "close"):
                backend.close()
        if self.spellchecker is not None:
            self.spellchecker.close()

    def __enter__(self) -> MorphEngine:
        return self

    def __exit__(self, *args) -> None:
        self.close()


# ── Path helpers ─────────────────────────────────────────────────────────

def _expand_paths(paths: tuple[str | Path, ...]) -> list[Path]:
    """Expand globs and return sorted list of existing Paths."""
    result = []
    for p in paths:
        p_str = str(p)
        if "*" in p_str or "?" in p_str:
            result.extend(Path(m) for m in sorted(glob.glob(p_str)))
        else:
            result.append(Path(p))
    return result


def _resolve_config_paths(raw_paths: list[str], base_dir: Path) -> list[Path]:
    """Resolve config paths relative to base_dir, expanding globs.

    Non-glob paths that do not exist are skipped with a warning rather than
    passed through — this lets generated/optional files (e.g. function-words.json)
    be listed in the config without crashing on a fresh clone.
    """
    result = []
    for p in raw_paths:
        full = base_dir / p if not Path(p).is_absolute() else Path(p)
        full_str = str(full)
        if "*" in full_str or "?" in full_str:
            result.extend(Path(m) for m in sorted(glob.glob(full_str)))
        elif full.exists():
            result.append(full)
        else:
            logger.warning("config path not found, skipping: %s", full)
    return result
