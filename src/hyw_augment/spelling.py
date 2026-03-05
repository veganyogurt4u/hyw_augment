"""
Hunspell-based spell checking for Western Armenian (Classical orthography).

Wraps the hunspell CLI in pipe mode to provide word validation and
spelling suggestions, using the HySpell hy-c dictionary.

Usage:
    from hyw_augment.spelling import SpellChecker

    with SpellChecker("/path/to/HySpell/Dictionaries/Dictc") as sc:
        sc.check("some_form")          # True if valid
        sc.suggest("misspeled_form")   # list of suggestions
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

# Hunspell pipe mode output patterns:
#   *                         -> word is correct
#   + root                    -> word is correct (root compound)
#   - root                    -> word is correct (from affix rules)
#   & word N offset: s1, s2   -> misspelled, N suggestions at offset
#   # word offset             -> misspelled, no suggestions
_SUGGEST_RE = re.compile(r"^& \S+ \d+ \d+: (.+)$")


class SpellChecker:
    """
    Hunspell-based spell checker for Western Armenian.

    Uses the hunspell CLI in pipe mode (-a flag) with a persistent process
    for fast repeated checks.  Requires the 'hunspell' binary in PATH and
    a HySpell dictionary directory containing hy-c.aff and hy-c.dic.

    Use as a context manager or call .close() when done:

        with SpellChecker("/path/to/Dictc") as sc:
            if not sc.check(word):
                print(sc.suggest(word))
    """

    def __init__(self, dict_dir: str | Path):
        self.dict_dir = Path(dict_dir)
        self.aff_path = self.dict_dir / "hy-c.aff"
        self.dic_path = self.dict_dir / "hy-c.dic"
        # hunspell -d takes the path without extension
        self._dict_base = str(self.dict_dir / "hy-c")
        self._hunspell = shutil.which("hunspell")
        self.available = self._detect()
        self._proc: subprocess.Popen | None = None

    def _detect(self) -> bool:
        if self._hunspell is None:
            return False
        if not self.aff_path.exists() or not self.dic_path.exists():
            return False
        return True

    # -- Context manager -----------------------------------------------------

    def __enter__(self) -> SpellChecker:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        """Shut down the persistent hunspell process."""
        if self._proc is not None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                self._proc.kill()
            self._proc = None

    def __del__(self) -> None:
        self.close()

    # -- Persistent process management ---------------------------------------

    def _get_proc(self) -> subprocess.Popen:
        """Get or lazily start the persistent hunspell process."""
        if self._proc is None or self._proc.poll() is not None:
            self._proc = subprocess.Popen(
                [self._hunspell, "-a", "-i", "UTF-8", "-d", self._dict_base],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,  # line-buffered
            )
            # Read and discard the version banner line
            self._proc.stdout.readline()
        return self._proc

    def _query_one(self, form: str) -> str:
        """Send a single word, return the result line (*, &, or #)."""
        proc = self._get_proc()
        proc.stdin.write(form + "\n")
        proc.stdin.flush()

        result_line = proc.stdout.readline().rstrip("\n")
        # Read the trailing blank line
        proc.stdout.readline()
        return result_line

    # -- Public API ----------------------------------------------------------

    def check(self, form: str) -> bool:
        """Check if a word is valid according to the Hunspell dictionary."""
        if not self.available:
            return False
        result = self._query_one(form)
        # *, +, - all indicate a correct word
        return result.startswith(("*", "+", "-"))

    def suggest(self, form: str) -> list[str]:
        """Get spelling suggestions for a word.

        Returns an empty list if the word is correct or if no suggestions
        are available.
        """
        if not self.available:
            return []
        result = self._query_one(form)
        m = _SUGGEST_RE.match(result)
        if m:
            return [s.strip() for s in m.group(1).split(",")]
        return []

    def check_and_suggest(self, form: str) -> tuple[bool, list[str]]:
        """Check a word and return suggestions if invalid.

        Returns (is_valid, suggestions).
        """
        if not self.available:
            return False, []
        result = self._query_one(form)
        if result.startswith(("*", "+", "-")):
            return True, []
        m = _SUGGEST_RE.match(result)
        suggestions = [s.strip() for s in m.group(1).split(",")] if m else []
        return False, suggestions

    def check_batch(self, forms: list[str]) -> dict[str, bool]:
        """Check multiple words in a single hunspell call."""
        if not self.available or not forms:
            return {}
        unique = list(dict.fromkeys(forms))
        input_text = "\n".join(unique) + "\n"
        result = subprocess.run(
            [self._hunspell, "-a", "-i", "UTF-8", "-d", self._dict_base],
            input=input_text,
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Parse output: skip banner, one result + blank line per word
        lines = result.stdout.split("\n")
        results: dict[str, bool] = {}
        word_idx = 0
        for line in lines[1:]:  # skip banner
            if not line:
                continue
            if word_idx < len(unique):
                results[unique[word_idx]] = line.startswith(("*", "+", "-"))
                word_idx += 1
        return results

    def suggest_batch(self, forms: list[str]) -> dict[str, list[str]]:
        """Get spelling suggestions for multiple words."""
        if not self.available or not forms:
            return {}
        unique = list(dict.fromkeys(forms))
        input_text = "\n".join(unique) + "\n"
        result = subprocess.run(
            [self._hunspell, "-a", "-i", "UTF-8", "-d", self._dict_base],
            input=input_text,
            capture_output=True,
            text=True,
            timeout=30,
        )
        lines = result.stdout.split("\n")
        results: dict[str, list[str]] = {}
        word_idx = 0
        for line in lines[1:]:  # skip banner
            if not line:
                continue
            if word_idx < len(unique):
                m = _SUGGEST_RE.match(line)
                if m:
                    results[unique[word_idx]] = [
                        s.strip() for s in m.group(1).split(",")
                    ]
                else:
                    results[unique[word_idx]] = []
                word_idx += 1
        return results

    def summary(self) -> str:
        lines = ["HySpell Spell Checker (Classical Armenian)"]
        lines.append(f"  Dictionary: {self.dict_dir}")
        lines.append(f"  Available:  {self.available}")
        if self.available:
            lines.append(f"  Affix file: {self.aff_path.name}")
            lines.append(f"  Dict file:  {self.dic_path.name}")
        else:
            if self._hunspell is None:
                lines.append("  Problem:    hunspell not found in PATH")
            if not self.aff_path.exists():
                lines.append(f"  Problem:    {self.aff_path} not found")
            if not self.dic_path.exists():
                lines.append(f"  Problem:    {self.dic_path} not found")
        return "\n".join(lines)
