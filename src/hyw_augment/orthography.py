"""
Orthography conversion between Reformed and Classical Armenian.

Loads HySpell's lexicon mapping tables and suffix transformation rules
to convert Reformed-orthography text (Eastern Armenian / Soviet spelling)
to Classical orthography (Western Armenian / traditional spelling).

Usage:
    from hyw_augment.orthography import OrthographyConverter

    conv = OrthographyConverter("/path/to/HySpell/Dictionaries")
    classical = conv.convert_text("reformed armenian text here")
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# ── FlexMap rule parsing ────────────────────────────────────────────────────

# Strip trailing digits (Hunspell flag references like +ա125 → +ա)
_STRIP_FLAGS_RE = re.compile(r"\d+$")


@dataclass(slots=True)
class _FlexRule:
    """A single suffix transformation rule from RCFlexMap."""
    ref_suffix: str    # what the reformed word ends with
    ref_restore: str   # what to append after stripping ref_suffix to get base
    cls_strip: str     # what to strip from classical base
    cls_suffix: str    # what to append to get classical inflected form


def _parse_flex_side(side: str) -> tuple[str, str]:
    """Parse one side of a flex rule into (strip, add).

    Formats:
        +ABC        -> strip='', add='ABC'
        -XY+ABC     -> strip='XY', add='ABC'
        -XY|+ABC    -> strip='XY', add='ABC'
        [X]+        -> strip='', add='' (char replacement, handled separately)
    """
    side = _STRIP_FLAGS_RE.sub("", side)

    if "|" in side:
        parts = side.split("|")
        strip_part = parts[0].lstrip("-")
        add_part = parts[1].lstrip("+")
        return strip_part, add_part

    if side.startswith("-"):
        # -XY+ABC
        rest = side[1:]
        if "+" in rest:
            strip_part, add_part = rest.split("+", 1)
            return strip_part, add_part
        else:
            return rest, ""

    if side.startswith("+"):
        return "", side[1:]

    # Bracket notation [X]+ — character class, not a suffix rule
    return "", ""


def _parse_flex_rules(lines: list[str]) -> list[_FlexRule]:
    """Parse RCFlexMap.dic lines into FlexRule objects."""
    rules = []
    for line in lines:
        line = line.strip()
        if not line or ":" not in line:
            continue

        # Skip bracket notation (character class rules like [հ]+:[յ]+)
        if line.startswith("["):
            continue

        left, right = line.split(":", 1)
        ref_restore, ref_suffix = _parse_flex_side(left)
        cls_strip, cls_suffix = _parse_flex_side(right)

        if not ref_suffix and not ref_restore:
            continue

        rules.append(_FlexRule(
            ref_suffix=ref_suffix,
            ref_restore=ref_restore,
            cls_strip=cls_strip,
            cls_suffix=cls_suffix,
        ))
    return rules


# ── Character-class rules (line 1: [հ]+:[յ]+) ──────────────────────────────

_CHAR_REPLACE_RE = re.compile(r"^\[(.+?)\]\+:\[(.+?)\]\+$")


def _parse_char_rules(lines: list[str]) -> list[tuple[str, str]]:
    """Parse character-class replacement rules."""
    rules = []
    for line in lines:
        line = line.strip()
        m = _CHAR_REPLACE_RE.match(line)
        if m:
            rules.append((m.group(1), m.group(2)))
    return rules


# ── Main converter class ───────────────────────────────────────────────────

class OrthographyConverter:
    """
    Converts between Reformed and Classical Armenian orthography.

    Primary use: Reformed-to-Classical conversion for cleaning up LLM output
    that has Eastern Armenian / Reformed orthography contamination.

    Uses HySpell's mapping data:
    - RCLexMap.dic: word-level Reformed -> Classical mapping (160K entries)
    - RCFlexMap.dic: suffix transformation rules for inflected forms
    - RCExceptions.dic: words that should not be converted
    """

    def __init__(self, dict_dir: str | Path):
        self.dict_dir = Path(dict_dir)

        # Reformed -> Classical
        self.rc_lex_map: dict[str, str] = {}
        self.rc_flex_rules: list[_FlexRule] = []
        self.rc_char_rules: list[tuple[str, str]] = []
        self.rc_exceptions: set[str] = set()

        self._load()

    def _load(self) -> None:
        # RCLexMap: reformed:classical per line
        rc_lex_path = self.dict_dir / "Dictr" / "RCLexMap.dic"
        if rc_lex_path.exists():
            with rc_lex_path.open(encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if ":" in line:
                        parts = line.split(":", 1)
                        self.rc_lex_map[parts[0]] = parts[1]

        # RCFlexMap: suffix transformation rules
        rc_flex_path = self.dict_dir / "Dictr" / "RCFlexMap.dic"
        if rc_flex_path.exists():
            with rc_flex_path.open(encoding="utf-8-sig") as f:
                lines = f.readlines()
            self.rc_flex_rules = _parse_flex_rules(lines)
            self.rc_char_rules = _parse_char_rules(lines)

        # RCExceptions: words to skip
        rc_exc_path = self.dict_dir / "Dictr" / "RCExceptions.dic"
        if rc_exc_path.exists():
            with rc_exc_path.open(encoding="utf-8-sig") as f:
                for line in f:
                    word = line.strip()
                    if word:
                        self.rc_exceptions.add(word)

    def convert_word(self, form: str) -> str:
        """Convert a single word from Reformed to Classical orthography.

        Returns the Classical form if a mapping exists, otherwise returns
        the original word unchanged.
        """
        if not form or form in self.rc_exceptions:
            return form

        # 1. Direct lexicon lookup (handles base forms)
        if form in self.rc_lex_map:
            return self.rc_lex_map[form]

        # 2. Case-insensitive: try lowercase
        lower = form.lower()
        if lower in self.rc_lex_map:
            classical = self.rc_lex_map[lower]
            # Preserve original capitalization
            if form[0].isupper() and classical:
                classical = classical[0].upper() + classical[1:]
            return classical

        # 3. Try suffix rules (for inflected forms not in lex map)
        result = self._try_flex_rules(lower)
        if result is not None:
            if form[0].isupper():
                result = result[0].upper() + result[1:]
            return result

        # 4. Apply character-class replacements as last resort
        result = form
        for ref_char, cls_char in self.rc_char_rules:
            result = result.replace(ref_char, cls_char)
        if result != form:
            return result

        return form

    def _try_flex_rules(self, form: str) -> str | None:
        """Try suffix transformation rules to convert an inflected form.

        Algorithm:
        1. Check if word ends with the Reformed suffix
        2. Strip the suffix and restore the stem to get the base form
        3. Look up the base in the lex map to get the Classical base
        4. Apply the Classical suffix transformation
        """
        for rule in self.rc_flex_rules:
            if not rule.ref_suffix or not form.endswith(rule.ref_suffix):
                continue

            # Restore the reformed base form
            base = form[:-len(rule.ref_suffix)] + rule.ref_restore
            if not base:
                continue

            # Look up the base in the lex map
            cls_base = self.rc_lex_map.get(base)
            if cls_base is None:
                continue

            # Apply classical suffix transformation
            if rule.cls_strip:
                if cls_base.endswith(rule.cls_strip):
                    cls_stem = cls_base[:-len(rule.cls_strip)]
                else:
                    continue
            else:
                cls_stem = cls_base

            return cls_stem + rule.cls_suffix

        return None

    def convert_text(self, text: str) -> str:
        """Convert Reformed Armenian text to Classical orthography.

        Splits on word boundaries, converts each word, and reassembles
        preserving original whitespace and punctuation.
        """
        if not text:
            return text

        # Split into tokens (words and non-words) preserving everything
        tokens = re.findall(r"[\w]+|[^\w]+", text, re.UNICODE)
        return "".join(
            self.convert_word(tok) if tok[0].isalpha() else tok
            for tok in tokens
        )

    def is_reformed(self, form: str) -> bool:
        """Check if a word appears to be in Reformed orthography.

        Returns True if the word has a *different* Classical equivalent.
        Words that are spelled the same in both orthographies return False.
        """
        classical = self.convert_word(form)
        return classical != form

    def detect_reformed_words(self, text: str) -> list[tuple[str, str]]:
        """Find all Reformed-orthography words in text.

        Returns a list of (reformed, classical) pairs for words that
        have a different Classical spelling.
        """
        tokens = re.findall(r"[\w]+", text, re.UNICODE)
        results = []
        seen = set()
        for tok in tokens:
            if tok in seen or not tok[0].isalpha():
                continue
            seen.add(tok)
            classical = self.convert_word(tok)
            if classical != tok:
                results.append((tok, classical))
        return results

    def summary(self) -> str:
        lines = ["Orthography Converter (Reformed -> Classical)"]
        lines.append(f"  Data dir:     {self.dict_dir}")
        lines.append(f"  Lexicon map:  {len(self.rc_lex_map):,} entries")
        lines.append(f"  Flex rules:   {len(self.rc_flex_rules)} rules")
        lines.append(f"  Char rules:   {len(self.rc_char_rules)} rules")
        lines.append(f"  Exceptions:   {len(self.rc_exceptions)} words")
        return "\n".join(lines)
