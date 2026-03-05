"""
Parse the Nayiri Armenian Lexicon JSON and build morphological lookup tables.

Usage:
    from hyw_augment.nayiri import Lexicon

    lex = Lexicon.from_file("data/nayiri-armenian-lexicon-2026-02-15-v1.json")
    print(f"{lex.num_lexemes} lexemes, {lex.num_word_forms} word forms")

    # Look up a surface form
    analyses = lex.analyze("արշաdelays")
    for a in analyses:
        print(a.lemma, a.pos, a.case, a.number, a.article)

    # Generate forms from a lemma
    forms = lex.generate("արdelays", case="ABLATIVE", number="SINGULAR")
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(slots=True)
class Inflection:
    """A single inflection pattern from the Nayiri inflection catalog."""

    inflection_id: str
    lemma_type: str  # VERBAL, NOMINAL, UNINFLECTED
    display_name_hy: str
    display_name_en: str

    # Shared
    number: str | None = None  # SINGULAR, PLURAL
    person: str | None = None  # FIRST, SECOND, THIRD

    # Nominal
    case: str | None = None  # NOMINATIVE, ACCUSATIVE, GENITIVE, DATIVE, ABLATIVE, INSTRUMENTAL, LOCATIVE
    article: str | None = None

    # Verbal
    tense: str | None = None
    mood: str | None = None
    polarity: str | None = None  # POSITIVE, NEGATIVE
    verb_class: str | None = None


@dataclass(slots=True)
class MorphAnalysis:
    """Result of analyzing a surface form: its lemma + inflection details."""

    form: str  # the surface form that was looked up
    lemma: str  # dictionary form
    lexeme_id: str
    lemma_id: str
    pos: str  # NOUN, VERB, ADJECTIVE, ADVERB
    inflection: Inflection

    # Convenience accessors
    @property
    def case(self) -> str | None:
        return self.inflection.case

    @property
    def number(self) -> str | None:
        return self.inflection.number

    @property
    def person(self) -> str | None:
        return self.inflection.person

    @property
    def tense(self) -> str | None:
        return self.inflection.tense

    @property
    def mood(self) -> str | None:
        return self.inflection.mood

    @property
    def polarity(self) -> str | None:
        return self.inflection.polarity

    @property
    def article(self) -> str | None:
        return self.inflection.article

    @property
    def description_en(self) -> str:
        return self.inflection.display_name_en

    @property
    def description_hy(self) -> str:
        return self.inflection.display_name_hy

    def __repr__(self) -> str:
        return (
            f"MorphAnalysis({self.form!r} ← {self.lemma!r} [{self.pos}] "
            f"{self.inflection.display_name_en})"
        )


@dataclass(slots=True)
class LemmaEntry:
    """A lemma with all its word forms."""

    lemma_id: str
    lexeme_id: str
    lemma_string: str
    pos: str
    word_forms: list[tuple[str, str]]  # (surface_form, inflection_id)


class Lexicon:
    """
    The Nayiri Armenian Lexicon, indexed for fast morphological lookup.

    Two main indexes:
    - form_index:  surface_form → list[MorphAnalysis]  (analysis / parsing)
    - lemma_index: lemma_string → list[LemmaEntry]     (generation)
    """

    def __init__(self):
        self.inflections: dict[str, Inflection] = {}
        self.lemma_entries: list[LemmaEntry] = []

        # Lookup indexes (built by _build_indexes)
        self.form_index: dict[str, list[MorphAnalysis]] = {}
        self.lemma_index: dict[str, list[LemmaEntry]] = {}

        # Stats
        self.num_lexemes: int = 0
        self.num_word_forms: int = 0

    @classmethod
    def from_file(cls, path: str | Path) -> Lexicon:
        """Load from a Nayiri Lexicon JSON file (sample or full)."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls._from_raw(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> Lexicon:
        """Load from an already-parsed JSON dict."""
        return cls._from_raw(raw)

    @classmethod
    def _from_raw(cls, raw: dict) -> Lexicon:
        lex = cls()

        # 1. Parse inflection catalog
        for inf_raw in raw.get("inflections", []):
            inf = Inflection(
                inflection_id=inf_raw["inflectionId"],
                lemma_type=inf_raw.get("lemmaType", ""),
                display_name_hy=inf_raw.get("displayName", {}).get("hy", ""),
                display_name_en=inf_raw.get("displayName", {}).get("en", ""),
                number=inf_raw.get("grammaticalNumber"),
                person=inf_raw.get("grammaticalPerson"),
                case=inf_raw.get("grammaticalCase"),
                article=inf_raw.get("grammaticalArticle"),
                tense=inf_raw.get("verbTense"),
                mood=inf_raw.get("verbMood"),
                polarity=inf_raw.get("verbPolarity"),
                verb_class=inf_raw.get("verbalInflectionClass"),
            )
            lex.inflections[inf.inflection_id] = inf

        # 2. Parse lexemes → lemmas → word forms
        total_forms = 0
        for lexeme_raw in raw.get("lexemes", []):
            lexeme_id = lexeme_raw["lexemeId"]
            for lemma_raw in lexeme_raw.get("lemmas", []):
                lemma_id = lemma_raw["lemmaId"]
                lemma_str = lemma_raw["lemmaString"]
                pos = lemma_raw["partOfSpeech"]

                word_forms = []
                for wf in lemma_raw.get("wordForms", []):
                    surface = wf["s"]
                    inf_id = wf["i"]
                    word_forms.append((surface, inf_id))
                    total_forms += 1

                entry = LemmaEntry(
                    lemma_id=lemma_id,
                    lexeme_id=lexeme_id,
                    lemma_string=lemma_str,
                    pos=pos,
                    word_forms=word_forms,
                )
                lex.lemma_entries.append(entry)

        lex.num_lexemes = len(raw.get("lexemes", []))
        lex.num_word_forms = total_forms

        lex._build_indexes()
        return lex

    def _build_indexes(self) -> None:
        """Build the form→analysis and lemma→entry indexes."""
        self.form_index.clear()
        self.lemma_index.clear()

        for entry in self.lemma_entries:
            # lemma index
            self.lemma_index.setdefault(entry.lemma_string, []).append(entry)

            # form index
            for surface, inf_id in entry.word_forms:
                inf = self.inflections.get(inf_id)
                if inf is None:
                    continue
                analysis = MorphAnalysis(
                    form=surface,
                    lemma=entry.lemma_string,
                    lexeme_id=entry.lexeme_id,
                    lemma_id=entry.lemma_id,
                    pos=entry.pos,
                    inflection=inf,
                )
                self.form_index.setdefault(surface, []).append(analysis)

                # Also index lowercase for case-insensitive lookup
                lower = surface.lower()
                if lower != surface:
                    self.form_index.setdefault(lower, []).append(analysis)

    def merge(self, other: Lexicon) -> None:
        """
        Merge another Lexicon into this one (e.g. function words supplement).
        Inflections, lemma entries, and indexes are all combined.
        """
        self.inflections.update(other.inflections)
        self.lemma_entries.extend(other.lemma_entries)
        self.num_lexemes += other.num_lexemes
        self.num_word_forms += other.num_word_forms

        # Rebuild indexes from scratch (simpler than incremental merge)
        self._build_indexes()

    @classmethod
    def from_files(cls, *paths: str | Path) -> Lexicon:
        """Load and merge multiple lexicon JSON files."""
        result = None
        for p in paths:
            lex = cls.from_file(p)
            if result is None:
                result = lex
            else:
                result.merge(lex)
        if result is None:
            return cls()
        return result

    # ── Analysis (form → lemma + features) ────────────────────────────────

    def analyze(self, form: str) -> list[MorphAnalysis]:
        """
        Look up a surface form and return all possible analyses.
        Returns empty list if form is unknown.
        """
        return self.form_index.get(form, [])

    def analyze_insensitive(self, form: str) -> list[MorphAnalysis]:
        """Case-insensitive lookup."""
        return self.form_index.get(form.lower(), [])

    def is_valid_form(self, form: str) -> bool:
        """Check if a surface form exists in the lexicon."""
        return form in self.form_index or form.lower() in self.form_index

    # ── Generation (lemma + features → forms) ─────────────────────────────

    def generate(
        self,
        lemma: str,
        *,
        case: str | None = None,
        number: str | None = None,
        person: str | None = None,
        tense: str | None = None,
        mood: str | None = None,
        polarity: str | None = None,
        article: str | None = None,
    ) -> list[tuple[str, Inflection]]:
        """
        Given a lemma and desired features, return matching surface forms.
        Features that are None are treated as wildcards.
        """
        entries = self.lemma_index.get(lemma, [])
        results = []

        for entry in entries:
            for surface, inf_id in entry.word_forms:
                inf = self.inflections.get(inf_id)
                if inf is None:
                    continue
                if case is not None and inf.case != case:
                    continue
                if number is not None and inf.number != number:
                    continue
                if person is not None and inf.person != person:
                    continue
                if tense is not None and inf.tense != tense:
                    continue
                if mood is not None and inf.mood != mood:
                    continue
                if polarity is not None and inf.polarity != polarity:
                    continue
                if article is not None and inf.article != article:
                    continue
                results.append((surface, inf))

        return results

    def lemmas_for_pos(self, pos: str) -> list[str]:
        """List all lemma strings for a given POS."""
        return [e.lemma_string for e in self.lemma_entries if e.pos == pos]

    # ── Iteration / stats ─────────────────────────────────────────────────

    def all_forms(self) -> Iterator[str]:
        """Iterate over all unique surface forms."""
        yield from self.form_index.keys()

    def all_lemmas(self) -> Iterator[str]:
        """Iterate over all unique lemma strings."""
        yield from self.lemma_index.keys()

    def summary(self) -> str:
        lines = [
            f"Lexemes:        {self.num_lexemes}",
            f"Lemma entries:  {len(self.lemma_entries)}",
            f"Word forms:     {self.num_word_forms}",
            f"Unique forms:   {len(self.form_index)}",
            f"Inflections:    {len(self.inflections)}",
            "",
            "POS breakdown:",
        ]
        from collections import Counter
        pos_counts = Counter(e.pos for e in self.lemma_entries)
        for pos, count in pos_counts.most_common():
            forms = sum(
                len(e.word_forms) for e in self.lemma_entries if e.pos == pos
            )
            lines.append(f"  {pos:12s} {count:5d} lemmas, {forms:7d} forms")
        return "\n".join(lines)
