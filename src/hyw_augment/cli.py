#!/usr/bin/env python3
"""
Western Armenian morphological toolkit CLI.

Loads backends from hyw_augment.toml by default, or override with flags:

    python -m hyw_augment.cli --analyze "WORD"
    python -m hyw_augment.cli --analyze "WORD" --config hyw_augment.toml
    python -m hyw_augment.cli --nayiri data/*.json --analyze "WORD"
    python -m hyw_augment.cli --coverage
    python -m hyw_augment.cli --coverage --mismatches data/mismatches.tsv
"""

import argparse
import sys
from pathlib import Path


def _find_default_config() -> Path | None:
    """Look for hyw_augment.toml in CWD or parent dirs."""
    candidate = Path("hyw_augment.toml")
    if candidate.exists():
        return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Western Armenian morphological toolkit"
    )
    parser.add_argument(
        "--config",
        metavar="FILE",
        help="Path to TOML config file (default: auto-detect hyw_augment.toml)",
    )
    parser.add_argument(
        "--conllu",
        nargs="+",
        help="Path(s) to .conllu treebank files (overrides config)",
    )
    parser.add_argument(
        "--nayiri",
        nargs="+",
        help="Path(s) to Nayiri lexicon JSON (overrides config)",
    )
    parser.add_argument(
        "--apertium",
        metavar="DIR",
        help="Path to apertium-hyw build directory (overrides config)",
    )
    parser.add_argument(
        "--analyze",
        help="Analyze a single word form",
    )
    parser.add_argument(
        "--generate",
        help="Generate forms for a lemma",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run coverage check (requires treebank + at least one analyzer)",
    )
    parser.add_argument(
        "--mismatches",
        metavar="FILE",
        help="Write full mismatch list to a TSV file (use with --coverage)",
    )
    parser.add_argument(
        "--hyspell",
        metavar="DIR",
        help="Path to HySpell Dictionaries directory (overrides config)",
    )
    parser.add_argument(
        "--validate",
        help="Check if a word is valid",
    )
    parser.add_argument(
        "--suggest",
        help="Get spelling suggestions for a word",
    )
    parser.add_argument(
        "--convert",
        help="Convert Reformed-orthography text to Classical",
    )
    parser.add_argument(
        "--define",
        help="Look up a word's definition",
    )
    parser.add_argument(
        "--calfa",
        metavar="DIR",
        help="Path to calfa lexical-databases directory (overrides config)",
    )
    parser.add_argument(
        "--define-en",
        help="Look up a word's English definition (Calfa lexicon)",
    )
    args = parser.parse_args()

    # ── Build engine ─────────────────────────────────────────────────────

    from hyw_augment.engine import MorphEngine

    has_explicit_flags = args.nayiri or args.apertium or args.conllu or args.hyspell or args.calfa

    if has_explicit_flags:
        # Explicit flags: build engine manually (flags override config)
        engine = MorphEngine()
        if args.nayiri:
            engine.add_nayiri(*args.nayiri)
        if args.apertium:
            engine.add_apertium(args.apertium)
        if args.conllu:
            engine.load_treebank(*args.conllu)
        if args.hyspell:
            hp = Path(args.hyspell)
            engine.add_spellcheck(hp / "Dictc")
            engine.add_orthography(hp)
            glossary_path = hp / "SmallArmDic.txt"
            if glossary_path.exists():
                engine.add_glossary(glossary_path)
        if args.calfa:
            engine.add_calfa(args.calfa)
    else:
        # Use config file
        config_path = Path(args.config) if args.config else _find_default_config()
        if config_path is None:
            parser.error(
                "No hyw_augment.toml found and no --nayiri/--apertium/--conllu flags given.\n"
                "  Either create a config file or pass flags explicitly."
            )
        engine = MorphEngine.from_config(config_path)

    with engine:
        print(engine.summary())
        print()

        # ── Analyze ──────────────────────────────────────────────────────

        if args.analyze:
            all_results = engine.analyze_all(args.analyze)
            if all_results:
                for source, results in all_results.items():
                    print(f"═══ Analysis of '{args.analyze}' ({source}) ═══")
                    for r in results:
                        print(f"  {r.lemma} [{r.pos}] — {r.description_en}")
            else:
                backends = ", ".join(name for name, _ in engine.backends) or "none loaded"
                print(f"'{args.analyze}' not found (backends: {backends})")
            print()

        # ── Generate ─────────────────────────────────────────────────────

        if args.generate:
            # Generation is Nayiri-specific for now (uses its inflection system)
            nayiri_backend = None
            for name, backend in engine.backends:
                if name == "nayiri":
                    nayiri_backend = backend
                    break

            if nayiri_backend:
                forms = nayiri_backend.generate(args.generate)
                if forms:
                    print(f"═══ Forms of '{args.generate}' (Nayiri) ═══")
                    seen = set()
                    for surface, inf in forms:
                        key = (surface, inf.display_name_en)
                        if key not in seen:
                            seen.add(key)
                            print(f"  {surface:30s}  {inf.display_name_en}")
                else:
                    print(f"Lemma '{args.generate}' not found in Nayiri lexicon.")
            else:
                print("Generation requires Nayiri lexicon (not loaded).")
            print()

        # ── Coverage ─────────────────────────────────────────────────────

        if args.coverage:
            nayiri_backend = None
            apertium_backend = None
            for name, backend in engine.backends:
                if name == "nayiri":
                    nayiri_backend = backend
                elif name == "apertium":
                    apertium_backend = backend

            if engine.treebank is None or nayiri_backend is None:
                print(
                    "ERROR: --coverage requires treebank and Nayiri lexicon",
                    file=sys.stderr,
                )
                sys.exit(1)

            from hyw_augment.coverage import check_coverage

            report = check_coverage(
                engine.treebank, nayiri_backend, apertium=apertium_backend,
            )
            print(report.summary())
            if args.mismatches:
                report.write_mismatches(Path(args.mismatches))
                print(f"\nMismatches written to {args.mismatches}")

        # ── Validate ────────────────────────────────────────────────────

        if args.validate:
            valid = engine.validate(args.validate)
            if valid:
                # Report which source confirmed it
                sources = []
                for name, backend in engine.backends:
                    if name == "nayiri" and backend.is_valid_form(args.validate):
                        sources.append("nayiri")
                    elif name == "apertium" and backend.is_known(args.validate):
                        sources.append("apertium")
                if engine.spellchecker and engine.spellchecker.check(args.validate):
                    sources.append("hunspell")
                print(f"'{args.validate}' is VALID (confirmed by: {', '.join(sources)})")
            else:
                suggestions = engine.suggest(args.validate)
                if suggestions:
                    print(f"'{args.validate}' is INVALID. Suggestions: {', '.join(suggestions)}")
                else:
                    print(f"'{args.validate}' is INVALID (no suggestions available)")
            print()

        # ── Suggest ─────────────────────────────────────────────────────

        if args.suggest:
            suggestions = engine.suggest(args.suggest)
            if suggestions:
                print(f"Suggestions for '{args.suggest}': {', '.join(suggestions)}")
            else:
                if engine.spellchecker is None:
                    print("Spell checker not available (hunspell not configured).")
                else:
                    print(f"No suggestions for '{args.suggest}'.")
            print()

        # ── Convert ─────────────────────────────────────────────────────

        if args.convert:
            result = engine.convert_reformed(args.convert)
            if result != args.convert:
                print(f"Converted: {result}")
                reformed = engine.detect_reformed(args.convert)
                if reformed:
                    print("Changed words:")
                    for ref, cls in reformed:
                        print(f"  {ref} -> {cls}")
            else:
                print(f"No Reformed-orthography words detected.")
            print()

        # ── Define ──────────────────────────────────────────────────────

        if args.define:
            entries = engine.lookup_definition(args.define)
            if entries:
                print(f"Definition of '{args.define}':")
                for e in entries:
                    print(f"  [{e.pos}] {e.definition}")
            else:
                if engine.glossary is None:
                    print("Glossary not available (HySpell not configured).")
                else:
                    print(f"'{args.define}' not found in glossary.")
            print()

        # ── Define (English) ────────────────────────────────────────────

        if args.define_en:
            entries = engine.lookup_calfa(args.define_en)
            if entries:
                print(f"English definition of '{args.define_en}':")
                for e in entries:
                    print(f"  [{e.pos}] {e.definition_en}")
            else:
                if engine.calfa is None:
                    print("Calfa lexicon not available (--calfa not configured).")
                else:
                    print(f"'{args.define_en}' not found in Calfa lexicon.")
            print()


if __name__ == "__main__":
    main()
