"""
Calfa lexical-databases integration.

Loads English-language definitions and Armenian synonyms from the Calfa
lexical-databases repository (https://github.com/calfa-co/lexical-databases).

Data is licensed CC BY-NC 4.0.  Cite:
  Vidal-Gorene C. & Decours-Perez A. (2020). Languages Resources for Poorly
  Endowed Languages: The Case Study of Classical Armenian. LREC 2020.
  https://aclanthology.org/2020.lrec-1.385

Usage:
    from hyw_augment.calfa import CaLFALexicon

    lex = CaLFALexicon.from_dir("/path/to/lexical-databases")
    entries = lex.lookup("word")    # Armenian or transliterated headword
    for e in entries:
        print(e.pos, e.definition_en)

    synonyms = lex.synonyms_for("word")
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

# Latin POS abbreviations used in en-definitions01.tsv -> normalized English POS
_POS_MAP: dict[str, str] = {
    "s.":    "NOUN",
    "adj.":  "ADJECTIVE",
    "adv.":  "ADVERB",
    "v.":    "VERB",
    "vn.":   "VERB_NOUN",
    "np.":   "PROPER_NOUN",
    "int.":  "INTERJECTION",
    "pron.": "PRONOUN",
    "prep.": "PREPOSITION",
    "conj.": "CONJUNCTION",
    "part.": "PARTICLE",
}


def _primary_pos(pos_raw: str) -> str:
    """Extract the primary (first recognized) POS from a raw abbreviation.

    Some entries have compound POS strings like "s. adv."; the first token
    that matches _POS_MAP is used as the normalized POS.  Falls through to
    the raw string if nothing matches.
    """
    for token in pos_raw.split():
        if token in _POS_MAP:
            return _POS_MAP[token]
    return pos_raw


@dataclass(slots=True)
class CaLFAEntry:
    """A single entry from the Calfa English-definitions lexicon."""
    headword:      str   # Title column (Armenian headword)
    complement:    str   # Complement column (inflection suffix hint, may be "")
    pos:           str   # normalized English POS (NOUN, VERB, ...)
    pos_raw:       str   # raw abbreviation as it appears in the TSV, e.g. "s." or "s. adv."
    definition_en: str   # English definition text


class CaLFALexicon:
    """
    English-language Armenian dictionary loaded from Calfa lexical-databases.

    Provides headword lookup with English definitions and Armenian synonyms.
    """

    def __init__(self):
        self.entries: dict[str, list[CaLFAEntry]] = {}
        self._synonyms: dict[str, list[str]] = {}  # uppercase key → synonym list
        self._total: int = 0

    @classmethod
    def from_dir(cls, path: str | Path) -> CaLFALexicon:
        """Load definitions and synonyms from a lexical-databases directory."""
        lex = cls()
        path = Path(path)

        defs_path = path / "definitions" / "en-definitions01.tsv"
        if defs_path.exists():
            lex._load_definitions(defs_path)

        syns_path = path / "synonyms" / "synonyms01.tsv"
        if syns_path.exists():
            lex._load_synonyms(syns_path)

        return lex

    def _load_definitions(self, path: Path) -> None:
        """Parse definitions/en-definitions01.tsv."""
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)  # skip header row
            for row in reader:
                if not row or row[0].startswith("<letter>"):
                    continue
                self._add_definition_row(row)

    def _add_definition_row(self, row: list[str]) -> None:
        """Process one TSV row from the definitions file.

        Columns: Title, Complement, POS_1, Def_1, ..., POS_6, Def_6
        """
        headword = row[0].strip()
        complement = row[1].strip() if len(row) > 1 else ""

        # Iterate over up to 6 POS/definition pairs (columns 2-3, 4-5, ... 12-13)
        for i in range(6):
            pos_col = 2 + i * 2
            def_col = pos_col + 1
            if pos_col >= len(row):
                break
            pos_raw = row[pos_col].strip()
            definition = row[def_col].strip() if def_col < len(row) else ""
            if not pos_raw:
                break

            pos = _primary_pos(pos_raw)
            entry = CaLFAEntry(
                headword=headword,
                complement=complement,
                pos=pos,
                pos_raw=pos_raw,
                definition_en=definition,
            )
            self.entries.setdefault(headword, []).append(entry)
            self._total += 1

    def _load_synonyms(self, path: Path) -> None:
        """Parse synonyms/synonyms01.tsv.

        Headwords are stored uppercase.  The Definition_n columns contain
        Armenian synonyms separated by "; ".
        """
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)  # skip header
            for row in reader:
                if not row or row[0].startswith("<letter>"):
                    continue
                headword = row[0].strip().upper()
                synonyms: list[str] = []
                # Definition columns are at indices 3, 5, 7, ... (every 2 starting at 3)
                for i in range(16):
                    def_col = 3 + i * 2
                    if def_col >= len(row):
                        break
                    cell = row[def_col].strip()
                    if cell:
                        synonyms.extend(s.strip() for s in cell.split("; ") if s.strip())
                if synonyms:
                    self._synonyms[headword] = synonyms

    def lookup(self, word: str) -> list[CaLFAEntry] | None:
        """Look up a word's English definitions.

        Tries exact match, then lowercase fallback.
        Returns a list of entries or None if not found.
        """
        entries = self.entries.get(word)
        if entries:
            return entries
        entries = self.entries.get(word.lower())
        return entries or None

    def synonyms_for(self, word: str) -> list[str]:
        """Return Armenian synonyms for a headword.

        Tries exact key, then uppercase (synonyms are stored uppercase).
        Returns an empty list if not found.
        """
        syns = self._synonyms.get(word)
        if syns is not None:
            return syns
        return self._synonyms.get(word.upper(), [])

    def summary(self) -> str:
        lines = ["Calfa Lexical-Databases (en-definitions)"]
        lines.append(f"  Headwords:    {len(self.entries):,}")
        lines.append(f"  Total entries: {self._total:,}")
        lines.append(f"  Synonym sets:  {len(self._synonyms):,}")
        return "\n".join(lines)
