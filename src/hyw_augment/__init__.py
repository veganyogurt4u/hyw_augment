"""hyw-augment: Western Armenian morphological augmentation layer for LLMs."""

from hyw_augment.apertium import ApertiumAnalysis, ApertiumAnalyzer
from hyw_augment.calfa import CaLFAEntry, CaLFALexicon
from hyw_augment.conllu import Sentence, Token, Treebank
from hyw_augment.coverage import check_coverage
from hyw_augment.engine import AnalysisResult, MorphEngine
from hyw_augment.glossary import Glossary, GlossaryEntry
from hyw_augment.nayiri import Lexicon, MorphAnalysis
from hyw_augment.orthography import OrthographyConverter
from hyw_augment.spelling import SpellChecker

__all__ = [
    "Treebank", "Sentence", "Token",
    "Lexicon", "MorphAnalysis",
    "ApertiumAnalyzer", "ApertiumAnalysis",
    "MorphEngine", "AnalysisResult",
    "SpellChecker", "OrthographyConverter",
    "Glossary", "GlossaryEntry",
    "CaLFALexicon", "CaLFAEntry",
    "check_coverage",
]
