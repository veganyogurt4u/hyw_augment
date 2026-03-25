# Test Suite

372 tests across 11 files. Run with:

```
pytest tests/ -v
```

## Architecture

Tests are organized in two tiers:

- **Unit tests** use lightweight in-memory fixtures (no file I/O, no subprocesses).
  They always run, even in CI without external data.
- **Integration tests** use real backends loaded from `hyw_augment.toml`.
  They skip gracefully when data is absent via `pytest.skip()`.

Shared fixtures live in `conftest.py`. Session-scoped fixtures (`full_engine`,
`nayiri_fixture`, `hyspell_dir`) are loaded once per test run for speed.

## Files

| File | Tests | Type | What it covers |
|------|------:|------|----------------|
| `test_apertium.py` | 71 | Unit + Integration | Apertium tag parsing, `_parse_lines`, `_parse_batch_output`, `Inflection.from_tags`; integration tests analyze/generate real words via hfst-lookup |
| `test_engine.py` | 45 | Unit (real Nayiri fixture) | `AnalysisResult` delegation, `MorphEngine` routing: analyze, analyze_all, analyze_batch, generate, generate_all, validate; fallthrough logic; component delegation (suggest, convert, detect, lookup); context manager/close |
| `test_conllu.py` | 40 | Unit + Integration | CoNLL-U parsing: sentences, tokens, multiword tokens, empty nodes; `Treebank` loading and stats |
| `test_orthography.py` | 39 | Unit | `_parse_flex_side`, `_parse_char_rules`, `OrthographyConverter` with injected maps; word/text conversion, reformed detection |
| `test_spelling.py` | 35 | Unit + Integration | Hunspell pipe protocol parsing (`_SUGGEST_RE`, `*`/`+`/`-`/`&`/`#` responses); batch via mocked `subprocess.run`; integration class with real hunspell + hy-c dictionary |
| `test_nayiri.py` | 34 | Unit + Integration | Nayiri lexicon: `from_dict` fixture, form/lemma indexing, analyze, analyze_insensitive, generate with filters, merge, POS listing; integration with real 66 MB JSON |
| `test_integration.py` | 28 | Integration | Full `MorphEngine.from_config()` end-to-end: analyze, analyze_all, analyze_batch, generate, validate, spelling, orthography, glossary, CaLFa, engine construction |
| `test_calfa.py` | 28 | Unit + Integration | CaLFa TSV parsing, `CaLFAEntry` fields, lookup, synonyms; integration with real lexical-databases directory |
| `test_cli.py` | 24 | Integration (subprocess) | CLI via `subprocess.run`: `--help`, `--analyze`, `--generate`, `--validate`, `--suggest`, `--convert`, `--define`, `--define-en`, `--verbose`/`-v`, `--diagnostic` |
| `test_glossary.py` | 23 | Unit + Integration | Glossary line parsing, POS detection, lookup, definition formatting; integration with real SmallArmDic.txt |
| `test_basics.py` | 5 | Smoke | Quick parser smoke tests |

## Fixture patterns

**In-memory Nayiri fixture** (`test_nayiri.FIXTURE`): A small dict with 3
lexemes and 5 inflections using ASCII placeholder forms (`noun-alpha`,
`verb-beta`, etc.). Used by `test_engine.py` to test engine routing with
real `Lexicon` objects instead of mocks.

**Session-scoped `full_engine`** (`conftest.py`): Builds a complete
`MorphEngine` from `hyw_augment.toml` once per session. Used by
`test_integration.py` and any test that needs the full backend chain.

**`spellcheck_dir`** (`conftest.py`): Resolves the HySpell Dictc path from
config. Used by `test_spelling.py::TestSpellCheckerIntegration`.

## Running subsets

```bash
# Fast unit tests only (no data dependency, <1s)
pytest tests/test_engine.py tests/test_conllu.py tests/test_orthography.py tests/test_spelling.py -k "not Integration"

# Integration tests only
pytest tests/test_integration.py tests/test_cli.py -v

# Single module
pytest tests/test_apertium.py -v
```

## External data dependencies

Integration tests require these paths (configured in `hyw_augment.toml`):

| Resource | Path | Used by |
|----------|------|---------|
| Nayiri lexicon JSON | `data/nayiri-armenian-lexicon-*.json` | test_nayiri, test_integration |
| Apertium transducers | `/home/van/.../apertium-hyw` | test_apertium, test_integration, test_cli |
| HySpell dictionaries | `data/Dictionaries/` | test_spelling, test_orthography, test_glossary, test_integration, test_cli |
| CaLFa lexical-databases | `/home/van/.../lexical-databases` | test_calfa, test_integration, test_cli |
| UD treebank | `data/hyw_armtdp-ud-*.conllu` | test_conllu |

All integration tests skip with a clear message when their data is absent.
