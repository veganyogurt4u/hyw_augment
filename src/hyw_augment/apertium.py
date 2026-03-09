"""
Apertium Western Armenian transducer integration.

Wraps hfst-lookup to provide morphological analysis and generation
using the apertium-hyw transducer as a complement to the
Nayiri lexicon.

Usage:
    from hyw_augment.apertium import ApertiumAnalyzer

    with ApertiumAnalyzer("/path/to/apertium-hyw") as apt:
        analyses = apt.analyze("some_armenian_form")
        for a in analyses:
            print(a.lemma, a.pos, a.tags, a.description_en)
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

# ── Tag labels: Apertium tag → human-readable English ────────────────────

_TAG_LABELS: dict[str, str] = {
    # POS
    "n": "noun", "v": "verb", "adj": "adjective", "adv": "adverb",
    "np": "proper noun", "prn": "pronoun", "det": "determiner",
    "post": "postposition", "pr": "preposition",
    "cnjcoo": "coordinating conjunction", "cnjsub": "subordinating conjunction",
    "part": "particle", "ij": "interjection", "num": "numeral",
    "abbr": "abbreviation",
    # Proper noun subtypes
    "ant": "anthroponym", "top": "toponym", "cog": "cognomen",
    "al": "other", "m": "masculine", "f": "feminine", "mf": "masc/fem",
    # Number
    "sg": "singular", "pl": "plural",
    # Case
    "nom": "nominative", "acc": "accusative", "gen": "genitive",
    "dat": "dative", "dat_gen": "dative/genitive",
    "abl": "ablative", "ins": "instrumental", "loc": "locative",
    # Definiteness
    "def": "definite", "indef": "indefinite",
    # Person
    "p1": "1st person", "p2": "2nd person", "p3": "3rd person",
    # Tense
    "pres": "present", "past": "past", "fut": "future",
    "aor": "aorist", "impf": "imperfect",
    # Aspect
    "pret": "preterite", "perf": "perfect",
    # Mood
    "indc": "indicative", "sbjv": "subjunctive", "imp": "imperative",
    "cond": "conditional", "opt": "optative", "proh": "prohibitive",
    # Polarity
    "neg": "negative",
    # Verbal
    "tv": "transitive", "iv": "intransitive",
    "inch": "inchoative", "caus": "causative", "pass": "passive",
    # Non-finite
    "inf": "infinitive", "ger": "gerund",
    "pp": "past participle", "pprs": "present participle",
    "cvb": "converb",
    # Punctuation
    "punct": "punctuation", "sent": "sentence-final",
    "lquot": "left quote", "rquot": "right quote",
    "guio": "hyphen", "cm": "comma",
}


# ── Apertium POS → normalized POS (matching Nayiri/UD conventions) ───────

_APT_POS_MAP: dict[str, str] = {
    "n": "NOUN", "np": "NOUN",
    "v": "VERB",
    "adj": "ADJECTIVE",
    "adv": "ADVERB",
    "prn": "PRONOUN",
    "det": "DETERMINER",
    "post": "ADPOSITION", "pr": "ADPOSITION",
    "cnjcoo": "CONJUNCTION", "cnjsub": "CONJUNCTION",
    "part": "PARTICLE",
    "ij": "INTERJECTION",
    "num": "NUMERAL",
}

# Case map: WArm collapses dative/genitive for nouns (dat_gen);
# pronouns keep separate dat, gen tags.
_APT_CASE_MAP: dict[str, str] = {
    "nom": "NOMINATIVE", "acc": "ACCUSATIVE", "gen": "GENITIVE",
    "dat": "DATIVE", "dat_gen": "DATIVE/GENITIVE",
    "abl": "ABLATIVE", "ins": "INSTRUMENTAL", "loc": "LOCATIVE",
}

_APT_NUMBER_MAP: dict[str, str] = {"sg": "SINGULAR", "pl": "PLURAL"}
_APT_PERSON_MAP: dict[str, str] = {"p1": "FIRST", "p2": "SECOND", "p3": "THIRD"}
_APT_ARTICLE_MAP: dict[str, str] = {"def": "DEFINITE", "indef": "INDEFINITE"}

# Reverse maps: normalized label → list of Apertium tags.
# Callers should use POS context to pick the right candidate
# (e.g. pronouns use "dat"/"gen", nouns use "dat_gen").
_REVERSE_POS: dict[str, list[str]] = {}
for _k, _v in _APT_POS_MAP.items():
    _REVERSE_POS.setdefault(_v, []).append(_k)

_REVERSE_CASE: dict[str, list[str]] = {}
for _k, _v in _APT_CASE_MAP.items():
    _REVERSE_CASE.setdefault(_v, []).append(_k)
# dat_gen also serves as both dative and genitive individually
_REVERSE_CASE.setdefault("DATIVE", []).append("dat_gen")
_REVERSE_CASE.setdefault("GENITIVE", []).append("dat_gen")

_REVERSE_NUMBER: dict[str, str] = {v: k for k, v in _APT_NUMBER_MAP.items()}
_REVERSE_PERSON: dict[str, str] = {v: k for k, v in _APT_PERSON_MAP.items()}
_REVERSE_ARTICLE: dict[str, str] = {v: k for k, v in _APT_ARTICLE_MAP.items()}

# ── Parsing helpers ──────────────────────────────────────────────────────

_ANALYSIS_RE = re.compile(r"^([^<]+)((?:<[^>]+>)*)$")
_TAG_RE = re.compile(r"<([^>]+)>")


# ── Tag → Inflection builder ─────────────────────────────────────────────

def _inflection_from_tags(tags: list[str]) -> "Inflection":
    """Build a nayiri.Inflection from a list of Apertium tags."""
    from hyw_augment.nayiri import Inflection

    # Determine POS / lemma_type
    pos_tags = set(tags) & set(_APT_POS_MAP)
    if pos_tags & {"v"}:
        lemma_type = "VERBAL"
    elif pos_tags & {"n", "np", "adj", "num", "prn", "det"}:
        lemma_type = "NOMINAL"
    else:
        lemma_type = "UNINFLECTED"

    # Map structured fields
    case = next((_APT_CASE_MAP[t] for t in tags if t in _APT_CASE_MAP), None)
    number = next((_APT_NUMBER_MAP[t] for t in tags if t in _APT_NUMBER_MAP), None)
    person = next((_APT_PERSON_MAP[t] for t in tags if t in _APT_PERSON_MAP), None)
    article = next((_APT_ARTICLE_MAP[t] for t in tags if t in _APT_ARTICLE_MAP), None)

    # Build human-readable description from tag labels
    display_en = ", ".join(_TAG_LABELS.get(t, t) for t in tags)

    return Inflection(
        inflection_id="apt:" + "|".join(tags),
        lemma_type=lemma_type,
        display_name_hy="",
        display_name_en=display_en,
        case=case,
        number=number,
        person=person,
        article=article,
        raw_tags=tags,
    )


# ── Data classes ─────────────────────────────────────────────────────────

@dataclass(slots=True)
class ApertiumAnalysis:
    """Result of analyzing a surface form via the Apertium transducer.

    Duck-type compatible with nayiri.MorphAnalysis for the properties
    that the CLI and other consumers use: .lemma, .pos, .description_en,
    .case, .number, .person, .article.
    """

    form: str
    lemma: str
    tags: list[str]
    raw: str  # full analysis string as returned by hfst-lookup
    weight: float = 0.0

    @property
    def pos(self) -> str:
        for tag in self.tags:
            if tag in _APT_POS_MAP:
                return _APT_POS_MAP[tag]
        return "UNKNOWN"

    @property
    def case(self) -> str | None:
        for tag in self.tags:
            if tag in _APT_CASE_MAP:
                return _APT_CASE_MAP[tag]
        return None

    @property
    def number(self) -> str | None:
        for tag in self.tags:
            if tag in _APT_NUMBER_MAP:
                return _APT_NUMBER_MAP[tag]
        return None

    @property
    def person(self) -> str | None:
        for tag in self.tags:
            if tag in _APT_PERSON_MAP:
                return _APT_PERSON_MAP[tag]
        return None

    @property
    def article(self) -> str | None:
        for tag in self.tags:
            if tag in _APT_ARTICLE_MAP:
                return _APT_ARTICLE_MAP[tag]
        return None

    @property
    def is_proper_noun(self) -> bool:
        return "np" in self.tags

    @property
    def description_en(self) -> str:
        """Human-readable English description built from tags."""
        parts = [_TAG_LABELS.get(t, t) for t in self.tags]
        return ", ".join(parts)

    def __repr__(self) -> str:
        tag_str = "".join(f"<{t}>" for t in self.tags)
        return f"ApertiumAnalysis({self.form!r} <- {self.lemma}{tag_str})"


# ── Main analyzer class ─────────────────────────────────────────────────

class ApertiumAnalyzer:
    """
    Wrapper around the Apertium Western Armenian morphological transducer.

    Uses hfst-lookup for analysis (surface -> lemma+tags) and generation
    (lemma+tags -> surface).  Designed to be used alongside the Nayiri
    lexicon as a fallback analyzer.

    Keeps a persistent hfst-lookup process for fast repeated analysis.
    Use as a context manager or call .close() when done:

        with ApertiumAnalyzer("/path/to/apertium-hyw") as apt:
            apt.analyze(...)
    """

    def __init__(self, apertium_dir: str | Path):
        self.apertium_dir = Path(apertium_dir)
        self.automorf = self.apertium_dir / "hyx@hyw.automorf.hfst"
        self.autogen = self.apertium_dir / "hyx@hyw.autogen.hfst"
        self._hfst_lookup = shutil.which("hfst-lookup")
        self.available = self._detect()
        self._morf_proc: subprocess.Popen | None = None

    def _detect(self) -> bool:
        if self._hfst_lookup is None:
            return False
        if not self.automorf.exists():
            return False
        return True

    # ── Context manager ──────────────────────────────────────────────────

    def __enter__(self) -> ApertiumAnalyzer:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        """Shut down the persistent hfst-lookup process."""
        if self._morf_proc is not None:
            try:
                self._morf_proc.stdin.close()
                self._morf_proc.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                self._morf_proc.kill()
            self._morf_proc = None

    def __del__(self) -> None:
        self.close()

    # ── Persistent process management ────────────────────────────────────

    def _get_morf_proc(self) -> subprocess.Popen:
        """Get or lazily start the persistent hfst-lookup process."""
        if self._morf_proc is None or self._morf_proc.poll() is not None:
            self._morf_proc = subprocess.Popen(
                [self._hfst_lookup, "-q", str(self.automorf)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,  # line-buffered on Python side
            )
        return self._morf_proc

    def _query_one(self, form: str) -> list[str]:
        """Send a single form to the persistent process, read until blank line."""
        proc = self._get_morf_proc()
        try:
            proc.stdin.write(form + "\n")
            proc.stdin.flush()
        except OSError:
            # Process died before write (BrokenPipeError etc.); clear so next
            # call to _get_morf_proc() will restart it.
            self._morf_proc = None
            return []

        lines = []
        while True:
            line = proc.stdout.readline()
            if not line:
                # True EOF: process died mid-read.  Clear cached proc.
                self._morf_proc = None
                break
            if not line.strip():
                # Blank separator line: normal end of hfst-lookup output block.
                break
            lines.append(line.rstrip("\n"))
        return lines

    # ── Batch mode (subprocess.run, for large jobs) ──────────────────────

    def _run_batch(
        self, transducer: Path, inputs: list[str], *, timeout: float = 30.0,
    ) -> str:
        """Run hfst-lookup as a one-shot subprocess for batch input."""
        input_text = "\n".join(inputs) + "\n"
        result = subprocess.run(
            [self._hfst_lookup, "-q", str(transducer)],
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout

    # ── Parsing ──────────────────────────────────────────────────────────

    @staticmethod
    def _parse_analysis_string(raw: str) -> tuple[str, list[str]] | None:
        """Parse 'lemma<tag1><tag2>...' into (lemma, [tag1, tag2, ...])."""
        m = _ANALYSIS_RE.match(raw)
        if m is None:
            return None
        lemma = m.group(1)
        tag_str = m.group(2)
        tags = _TAG_RE.findall(tag_str) if tag_str else []
        return lemma, tags

    def _parse_lines(self, form: str, lines: list[str]) -> list[ApertiumAnalysis]:
        """Parse raw output lines for a single form into ApertiumAnalysis list."""
        results = []
        for line in lines:
            parts = line.split("\t")
            if len(parts) < 2:
                continue

            analysis_str = parts[1]
            weight = 0.0
            if len(parts) > 2:
                try:
                    weight = float(parts[2])
                except ValueError:
                    pass

            if "+?" in analysis_str:
                continue

            parsed = self._parse_analysis_string(analysis_str)
            if parsed is None:
                continue

            lemma, tags = parsed
            results.append(ApertiumAnalysis(
                form=form,
                lemma=lemma,
                tags=tags,
                raw=analysis_str,
                weight=weight,
            ))
        return results

    def _parse_batch_output(self, output: str) -> dict[str, list[ApertiumAnalysis]]:
        """Parse multi-word hfst-lookup output into per-form analysis lists."""
        results: dict[str, list[ApertiumAnalysis]] = {}
        for line in output.split("\n"):
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                continue

            form = parts[0]
            analysis_str = parts[1]
            weight = 0.0
            if len(parts) > 2:
                try:
                    weight = float(parts[2])
                except ValueError:
                    pass

            if "+?" in analysis_str:
                results.setdefault(form, [])
                continue

            parsed = self._parse_analysis_string(analysis_str)
            if parsed is None:
                continue

            lemma, tags = parsed
            analysis = ApertiumAnalysis(
                form=form, lemma=lemma, tags=tags,
                raw=analysis_str, weight=weight,
            )
            results.setdefault(form, []).append(analysis)

        return results

    # ── Public API ───────────────────────────────────────────────────────

    def analyze(self, form: str) -> list[ApertiumAnalysis]:
        """Analyze a single surface form.  Uses a persistent process for speed."""
        if not self.available:
            return []
        raw_lines = self._query_one(form)
        return self._parse_lines(form, raw_lines)

    def analyze_insensitive(self, form: str) -> list[ApertiumAnalysis]:
        """Analyze with case fallback: try original, then lowercase."""
        results = self.analyze(form)
        if not results and form != form.lower():
            results = self.analyze(form.lower())
        return results

    def analyze_batch(self, forms: list[str]) -> dict[str, list[ApertiumAnalysis]]:
        """Analyze multiple surface forms in a single hfst-lookup call.

        Includes case fallback: forms that get no results are retried
        lowercased in a second batch call, with results keyed under
        the original form.
        """
        if not self.available or not forms:
            return {}
        unique = list(dict.fromkeys(forms))
        output = self._run_batch(self.automorf, unique)
        results = self._parse_batch_output(output)

        # Retry misses with lowercase
        retry = {f: f.lower() for f in unique
                 if not results.get(f) and f != f.lower()}
        if retry:
            lower_output = self._run_batch(self.automorf, list(dict.fromkeys(retry.values())))
            lower_results = self._parse_batch_output(lower_output)
            for orig, low in retry.items():
                if lower_results.get(low):
                    results[orig] = lower_results[low]

        return results

    def generate(self, lemma: str, tags: list[str]) -> list[tuple[str, "Inflection"]]:
        """Generate surface forms from a lemma + apertium tag list.

        Returns list of (surface_form, Inflection) tuples, matching the
        Nayiri generate() return type for unified engine use.

        Example:
            apt.generate("some_lemma", ["n", "pl", "abl", "def"])
        """
        if not self.available or not self.autogen.exists():
            return []
        tag_str = "".join(f"<{t}>" for t in tags)
        query = f"{lemma}{tag_str}"
        output = self._run_batch(self.autogen, [query])

        surfaces = []
        for line in output.split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2 and "+?" not in parts[1]:
                surfaces.append(parts[1])

        if not surfaces:
            return []

        inf = _inflection_from_tags(tags)
        return [(s, inf) for s in surfaces]

    def is_known(self, form: str) -> bool:
        """Check if a surface form is recognized by the transducer."""
        return len(self.analyze(form)) > 0

    def summary(self) -> str:
        lines = ["Apertium Western Armenian Transducer"]
        lines.append(f"  Directory:  {self.apertium_dir}")
        lines.append(f"  Available:  {self.available}")
        if self.available:
            lines.append(f"  Analyzer:   {self.automorf.name}")
            gen_status = "found" if self.autogen.exists() else "not found"
            lines.append(f"  Generator:  {self.autogen.name} ({gen_status})")
        else:
            if self._hfst_lookup is None:
                lines.append("  Problem:    hfst-lookup not found in PATH")
            if not self.automorf.exists():
                lines.append(f"  Problem:    {self.automorf} not found")
        return "\n".join(lines)
