from spacy.language import Language, BaseDefaults
from spacy.util import registry, load_config_from_str

from .tokenizer import MecabTokenizer
from .keyword_pattern import complex_korean_keyword_patterns

MECAB_CONFIG = """
[nlp]
[nlp.tokenizer]
@tokenizers = "spacy.ko.MecabTokenizer"
"""

@registry.tokenizers("spacy.ko.MecabTokenizer")
def create_tokenizer():
    def korean_tokenizer_factory(nlp):
        return MecabTokenizer(nlp.vocab)
    return korean_tokenizer_factory

class KoreanImprovedDefaults(BaseDefaults):
    """Improved Korean Language Defaults for Spacy.

    What's improved:
    - Use `python-mecab-ko` not `natto-py` as python binding for mecab-ko.
    - Added pre/post-processing logic to make it robust to web text processing.
    """
    config = load_config_from_str(MECAB_CONFIG)
    writing_system = {"direction": "ltr", "has_case": False, "has_letters": False}

class Korean(Language):
    """Mecab-based Korean Language Pipeline.
    
    CAUTION: This pipeline is partially destructive, meaning that it will
    not preserve the original whitespace in the text. This is because Mecab
    tokenizes the text and Spacy does not have a way to preserve the original
    whitespace.
    """
    lang = "ko"
    Defaults = KoreanImprovedDefaults


__all__ = ["Korean"]