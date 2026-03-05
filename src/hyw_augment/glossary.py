"""
Armenian glossary with definitions and POS information.

Loads the HySpell SmallArmDic.txt file — a 19K-entry Armenian explanatory
dictionary with headwords, POS tags, and Armenian-language definitions.

Usage:
    from hyw_augment.glossary import Glossary

    glossary = Glossary.from_file("/path/to/SmallArmDic.txt")
    entries = glossary.lookup("some_word")
    for e in entries:
        print(e.headword, e.pos, e.definition)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Armenian POS abbreviations -> normalized English POS labels
_POS_MAP: dict[str, str] = {
    "\u0563.":       "NOUN",           # գ.
    "\u0561\u056e.": "ADJECTIVE",      # ած.
    "\u0576\u0580\u0563.": "VERB_TR",  # նրգ. (transitive verb)
    "\u0579\u0566.": "VERB_INTR",      # չզ. (intransitive verb)
    "\u0585\u057f\u0580.": "LOANWORD", # օտր. (foreign/loanword)
    "\u0574\u056f.": "PARTICLE",       # մկ.
    "\u0576\u056d.": "PREPOSITION",    # նխ.
    "\u0577\u0572.": "CONJUNCTION",    # շղ.
    "\u0571\u0575\u0576.": "INTERJECTION",  # ձայն.
    "\u0564\u0565\u0580.": "PRONOUN",  # դեր.
    "\u057d.":       "PRONOUN_POSS",   # ս. (possessive)
    "\u057d\u0582\u056e.": "VERB_REFL",  # սուն. (reflexive)
    "\u0576\u0580\u0566.": "VERB_MID",  # նրզ. (middle voice)
}


@dataclass(slots=True)
class GlossaryEntry:
    """A single glossary entry with headword, POS, and definition."""
    headword: str
    pos: str          # normalized English POS (NOUN, VERB_TR, etc.)
    pos_raw: str      # original Armenian abbreviation
    definition: str   # Armenian-language definition text

    @property
    def is_transitive(self) -> bool | None:
        """For verbs, whether the verb is transitive."""
        if self.pos == "VERB_TR":
            return True
        if self.pos in ("VERB_INTR", "VERB_REFL", "VERB_MID"):
            return False
        return None


class Glossary:
    """
    Armenian explanatory dictionary loaded from SmallArmDic.txt.

    Provides headword lookup with POS and Armenian definitions.
    """

    def __init__(self):
        self.entries: dict[str, list[GlossaryEntry]] = {}
        self._total: int = 0

    @classmethod
    def from_file(cls, path: str | Path) -> Glossary:
        """Load a glossary from SmallArmDic.txt."""
        glossary = cls()
        path = Path(path)

        with path.open(encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                glossary._parse_line(line)

        return glossary

    def _parse_line(self, line: str) -> None:
        """Parse a single SmallArmDic line into one or more GlossaryEntry."""
        # Format: headword POS. [POS2.] definition text
        # Multi-POS: "headword ats. def1; g. def2"

        parts = line.split(None, 1)
        if len(parts) < 2:
            return

        headword = parts[0]
        rest = parts[1]

        # Handle semicolon-separated multi-POS entries
        # e.g., "ats. independent, self-governing; g. noble class member"
        segments = rest.split("; ")

        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue

            # Extract POS tag(s) from the beginning of the segment
            pos_raw, definition = self._extract_pos(segment)
            if not pos_raw:
                # No recognized POS tag — attach to previous POS or skip
                continue

            pos = _POS_MAP.get(pos_raw, pos_raw)

            # Clean up definition (strip trailing Armenian full stop)
            definition = definition.strip()
            if definition.endswith("\u0589"):  # Armenian full stop
                definition = definition[:-1].strip()

            entry = GlossaryEntry(
                headword=headword,
                pos=pos,
                pos_raw=pos_raw,
                definition=definition,
            )
            self.entries.setdefault(headword, []).append(entry)
            self._total += 1

    @staticmethod
    def _extract_pos(text: str) -> tuple[str, str]:
        """Extract the POS abbreviation from the start of text.

        Returns (pos_raw, remaining_text). If no POS found, returns ('', text).
        """
        # POS tags are short Armenian abbreviations ending with '.'
        # They appear at the start of the text, possibly multiple
        words = text.split()
        pos_tags = []
        consumed = 0

        for word in words:
            # Check if this looks like a POS tag
            clean = word.lstrip("(")  # handle "(s." etc.
            if (clean.endswith(".")
                    and len(clean) <= 6
                    and any("\u0530" <= c <= "\u058f" for c in clean)):
                pos_tags.append(clean)
                consumed += 1
            else:
                break

        if not pos_tags:
            return "", text

        # Use the first recognized POS tag
        pos_raw = pos_tags[0]
        remaining = " ".join(words[consumed:])
        return pos_raw, remaining

    def lookup(self, word: str) -> list[GlossaryEntry] | None:
        """Look up a word in the glossary.

        Returns a list of entries (a word may have multiple POS/definitions),
        or None if not found.
        """
        entries = self.entries.get(word)
        if entries:
            return entries
        # Try lowercase
        entries = self.entries.get(word.lower())
        return entries or None

    def summary(self) -> str:
        lines = ["Glossary (SmallArmDic)"]
        lines.append(f"  Headwords:    {len(self.entries):,}")
        lines.append(f"  Total entries: {self._total:,}")

        # POS distribution
        from collections import Counter
        pos_counts = Counter()
        for entry_list in self.entries.values():
            for entry in entry_list:
                pos_counts[entry.pos] += 1
        if pos_counts:
            lines.append("  POS breakdown:")
            for pos, count in pos_counts.most_common():
                lines.append(f"    {pos:15s} {count:,}")

        return "\n".join(lines)
