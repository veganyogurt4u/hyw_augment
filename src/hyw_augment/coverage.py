"""
Cross-reference the UD treebank against the Nayiri lexicon (and optionally
the Apertium transducer) to measure coverage.

This tells us:
- What % of treebank tokens can be found in the lexicon
- Which tokens are missing (gaps in lexicon coverage)
- How many Nayiri misses Apertium can rescue
- Where lemma mappings agree or disagree between the two resources
- POS-level breakdown of coverage

Usage:
    from hyw_augment import Treebank, Lexicon, check_coverage
    from hyw_augment.apertium import ApertiumAnalyzer

    tb = Treebank.from_file("data/hyw_armtdp-ud-dev.conllu")
    lex = Lexicon.from_file("data/nayiri-armenian-lexicon.json")
    apt = ApertiumAnalyzer("/path/to/apertium-hyw")
    report = check_coverage(tb, lex, apertium=apt)
    print(report.summary())
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from hyw_augment.conllu import Token, Treebank
from hyw_augment.nayiri import Lexicon

# POS tags we don't expect the lexicon to cover
SKIP_POS = {"PUNCT", "NUM", "SYM", "X"}


@dataclass
class TokenMatch:
    """Result of checking one UD token against the lexicon."""

    token: Token
    found: bool  # surface form found in lexicon
    lemma_match: bool  # at least one analysis shares the UD lemma
    analyses_count: int  # how many analyses the lexicon returned
    pos_match: bool  # at least one analysis shares the UD POS


@dataclass
class CoverageReport:
    """Aggregated coverage statistics."""

    total_tokens: int = 0
    skipped_tokens: int = 0  # PUNCT etc.
    checked_tokens: int = 0
    found_tokens: int = 0  # found by primary (Nayiri)
    lemma_matches: int = 0
    pos_matches: int = 0

    # Apertium fallback stats
    apertium_rescued: int = 0  # Nayiri missed, Apertium found
    apertium_lemma_matches: int = 0
    apertium_pos_matches: int = 0

    # Breakdown by UD POS
    by_pos: dict[str, dict[str, int]] = field(default_factory=dict)

    # Interesting cases
    missing_forms: Counter = field(default_factory=Counter)  # form → count
    missing_lemmas: Counter = field(default_factory=Counter)  # lemma → count
    lemma_mismatches: list[tuple[str, str, str, list[str]]] = field(
        default_factory=list
    )  # (form, ud_lemma, ud_pos, [analyzer_lemmas])
    pos_mismatches: list[tuple[str, str, str, list[str], list[str]]] = field(
        default_factory=list
    )  # (form, ud_lemma, ud_pos, [analyzer_lemmas], [analyzer_pos_ud])

    def summary(self) -> str:
        if self.checked_tokens == 0:
            return "No tokens checked."

        # percent formatting helper; note at end to avoid ruff complaining about named lambda
        pct = lambda n, d: f"{100*n/d:.1f}%" if d > 0 else "N/A" # noqa: E731

        total_found = self.found_tokens + self.apertium_rescued
        still_missing = self.checked_tokens - total_found

        lines = [
            "═══ Coverage Report ═══",
            "",
            f"Total tokens:   {self.total_tokens}",
            f"Skipped (PUNCT etc.): {self.skipped_tokens}",
            f"Checked:        {self.checked_tokens}",
            "",
            f"Nayiri found:        {self.found_tokens:5d}  ({pct(self.found_tokens, self.checked_tokens)})",
            f"  Lemma matches:       {self.lemma_matches:5d}  ({pct(self.lemma_matches, self.checked_tokens)})",
            f"  POS matches:         {self.pos_matches:5d}  ({pct(self.pos_matches, self.checked_tokens)})",
        ]

        if self.apertium_rescued > 0:
            lines.extend([
                "",
                f"Apertium rescued:    {self.apertium_rescued:5d}  ({pct(self.apertium_rescued, self.checked_tokens)})",
                f"  Lemma matches:       {self.apertium_lemma_matches:5d}  ({pct(self.apertium_lemma_matches, self.checked_tokens)})",
                f"  POS matches:         {self.apertium_pos_matches:5d}  ({pct(self.apertium_pos_matches, self.checked_tokens)})",
                "",
                f"Combined found:      {total_found:5d}  ({pct(total_found, self.checked_tokens)})",
            ])

        lines.extend([
            f"Not found:           {still_missing:5d}  ({pct(still_missing, self.checked_tokens)})",
            "",
            "─── By POS ───",
        ])

        for pos in sorted(self.by_pos.keys()):
            stats = self.by_pos[pos]
            checked = stats.get("checked", 0)
            nayiri = stats.get("found", 0)
            apt = stats.get("apertium", 0)
            if apt > 0:
                lines.append(
                    f"  {pos:12s}  {nayiri:4d}+{apt:<4d}/{checked:4d}  "
                    f"({pct(nayiri + apt, checked)}, Nayiri {pct(nayiri, checked)})"
                )
            else:
                lines.append(
                    f"  {pos:12s}  {nayiri:4d}/{checked:4d}  ({pct(nayiri, checked)})"
                )

        lines.append("")
        lines.append("─── Top 20 missing forms (after all backends) ───")
        for form, count in self.missing_forms.most_common(20):
            lines.append(f"  {form:25s}  x{count}")

        lines.append("")
        lines.append("─── Top 20 missing lemmas (after all backends) ───")
        for lemma, count in self.missing_lemmas.most_common(20):
            lines.append(f"  {lemma:25s}  x{count}")

        if self.lemma_mismatches:
            lines.append("")
            lines.append("─── Sample lemma mismatches (form found, lemma disagrees) ───")
            for form, ud_lemma, ud_pos, analyzer_lemmas in self.lemma_mismatches[:15]:
                nl = ", ".join(analyzer_lemmas[:3])
                lines.append(
                    f"  {form:20s}  UD: {ud_lemma:15s} ({ud_pos})  Analyzer: {nl}"
                )

        return "\n".join(lines)

    def write_mismatches(self, path: Path) -> None:
        """Write full mismatch lists to a TSV file for manual review.

        Columns: mismatch_type, form, ud_lemma, ud_pos, nayiri_lemmas, nayiri_pos, count
        Rows are deduplicated; count reflects how many times that combination appeared.
        """
        # Deduplicate lemma mismatches: key = (form, ud_lemma, ud_pos, nayiri_lemmas_tuple)
        lemma_counts: Counter = Counter()
        lemma_rows: dict = {}
        for form, ud_lemma, ud_pos, nayiri_lemmas in self.lemma_mismatches:
            key = (form, ud_lemma, ud_pos, tuple(sorted(nayiri_lemmas)))
            lemma_counts[key] += 1
            lemma_rows[key] = nayiri_lemmas

        # Deduplicate POS mismatches: key = (form, ud_lemma, ud_pos, nayiri_pos_tuple)
        pos_counts: Counter = Counter()
        pos_rows: dict = {}
        for form, ud_lemma, ud_pos, nayiri_lemmas, nayiri_pos in self.pos_mismatches:
            key = (form, ud_lemma, ud_pos, tuple(sorted(nayiri_pos)))
            pos_counts[key] += 1
            pos_rows[key] = (nayiri_lemmas, nayiri_pos)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(
                ["mismatch_type", "form", "ud_lemma", "ud_pos",
                 "nayiri_lemmas", "nayiri_pos", "count"]
            )

            for key, count in lemma_counts.most_common():
                form, ud_lemma, ud_pos, _ = key
                nayiri_lemmas = lemma_rows[key]
                writer.writerow([
                    "lemma", form, ud_lemma, ud_pos,
                    ", ".join(nayiri_lemmas), "", count,
                ])

            for key, count in pos_counts.most_common():
                form, ud_lemma, ud_pos, _ = key
                nayiri_lemmas, nayiri_pos = pos_rows[key]
                writer.writerow([
                    "pos", form, ud_lemma, ud_pos,
                    ", ".join(nayiri_lemmas), ", ".join(nayiri_pos), count,
                ])


# Map Nayiri POS names to UD POS tags for comparison
_NAYIRI_TO_UD_POS = {
    "NOUN": {"NOUN", "PROPN"},      # NOUN entries can match PROPN tokens
    "CONJUNCTION": {"CCONJ", "SCONJ"},  # CONJUNCTION entries can match either
    "VERB": {"VERB"},
    "ADJECTIVE": {"ADJ"},
    "ADVERB": {"ADV"},
    "ADPOSITION": {"ADP"},
    "PRONOUN": {"PRON"},
    "DETERMINER": {"DET"},
    "PARTICLE": {"PART"},
    "INTERJECTION": {"INTJ"},
    "AUX": {"AUX"},
}


def check_coverage(
    treebank: Treebank,
    lexicon: Lexicon,
    *,
    apertium=None,
    skip_pos: set[str] | None = None,
) -> CoverageReport:
    """
    Check how many treebank tokens the lexicon can analyze.

    Args:
        treebank: Parsed UD treebank
        lexicon: Loaded Nayiri lexicon
        apertium: Optional ApertiumAnalyzer for fallback coverage
        skip_pos: POS tags to skip (default: PUNCT, NUM, SYM, X)
    """
    if skip_pos is None:
        skip_pos = SKIP_POS

    report = CoverageReport()

    # First pass: Nayiri.  Collect misses for Apertium batch.
    # Each miss is (token, pos_stats_ref) so we can update in place.
    nayiri_misses: list[tuple[Token, dict]] = []

    for sent in treebank:
        for tok in sent.real_tokens:
            report.total_tokens += 1

            if tok.upos in skip_pos:
                report.skipped_tokens += 1
                continue

            report.checked_tokens += 1

            pos_stats = report.by_pos.setdefault(
                tok.upos, {"checked": 0, "found": 0, "apertium": 0}
            )
            pos_stats["checked"] += 1

            analyses = lexicon.analyze(tok.form)
            if not analyses:
                analyses = lexicon.analyze_insensitive(tok.form)

            if analyses:
                report.found_tokens += 1
                pos_stats["found"] += 1

                analyzer_lemmas = list({a.lemma for a in analyses})
                if tok.lemma in analyzer_lemmas:
                    report.lemma_matches += 1
                else:
                    report.lemma_mismatches.append(
                        (tok.form, tok.lemma, tok.upos, analyzer_lemmas)
                    )

                analyzer_pos_ud = set()
                for a in analyses:
                    analyzer_pos_ud |= _NAYIRI_TO_UD_POS.get(a.pos, {a.pos})

                if tok.upos in analyzer_pos_ud:
                    report.pos_matches += 1
                else:
                    report.pos_mismatches.append(
                        (tok.form, tok.lemma, tok.upos, analyzer_lemmas, sorted(analyzer_pos_ud))
                    )
            else:
                nayiri_misses.append((tok, pos_stats))

    # Second pass: Apertium batch on all Nayiri misses
    if apertium and apertium.available and nayiri_misses:
        missed_forms = list({tok.form for tok, _ in nayiri_misses})
        apt_results = apertium.analyze_batch(missed_forms)

        for tok, pos_stats in nayiri_misses:
            analyses = apt_results.get(tok.form, [])

            if analyses:
                report.apertium_rescued += 1
                pos_stats["apertium"] = pos_stats.get("apertium", 0) + 1

                analyzer_lemmas = list({a.lemma for a in analyses})
                if tok.lemma in analyzer_lemmas:
                    report.apertium_lemma_matches += 1
                else:
                    report.lemma_mismatches.append(
                        (tok.form, tok.lemma, tok.upos, analyzer_lemmas)
                    )

                analyzer_pos_ud = set()
                for a in analyses:
                    analyzer_pos_ud |= _NAYIRI_TO_UD_POS.get(a.pos, {a.pos})

                if tok.upos in analyzer_pos_ud:
                    report.apertium_pos_matches += 1
                else:
                    report.pos_mismatches.append(
                        (tok.form, tok.lemma, tok.upos, analyzer_lemmas, sorted(analyzer_pos_ud))
                    )
            else:
                report.missing_forms[tok.form] += 1
                report.missing_lemmas[tok.lemma] += 1
    else:
        # No apertium — all Nayiri misses are truly missing
        for tok, _pos_stats in nayiri_misses:
            report.missing_forms[tok.form] += 1
            report.missing_lemmas[tok.lemma] += 1

    return report
