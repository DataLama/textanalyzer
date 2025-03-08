from typing import Iterator, Any, Dict, List, Tuple, Set

import emoji
import regex
from itertools import chain
from mecab.types import Feature
from regex import Pattern

from spacy.tokens import Doc, Token
from spacy.symbols import POS, X
from spacy.util import DummyTokenizer
from spacy.vocab import Vocab

from .tag_map import TAG_MAP

Token.set_extension("tag_detail_t", default=None)

class MecabTokenizer(DummyTokenizer):
    def __init__(self, vocab: Vocab):
        self.vocab = vocab
        self._mecab = try_mecab_import()  # type: ignore[func-returns-value]
        self._mecab_tokenizer = None
        
        emojis = list({emo for emo_map in emoji.UNICODE_EMOJI.values() for emo in emo_map.keys()})
        cp_list = list(chain(*[[*emo] for emo in emojis]))
        self.emoji_set = set([cp for cp in cp_list if ord(cp)>127])
        self.grapheme = regex.compile('\X')

    @property
    def mecab_tokenizer(self):
        # This is a property so that initializing a pipeline with blank:ko is
        # possible without actually requiring mecab-ko, e.g. to run
        # `spacy init vectors ko` for a pipeline that will have a different
        # tokenizer in the end. The languages need to match for the vectors
        # to be imported and there's no way to pass a custom config to
        # `init vectors`.
        if self._mecab_tokenizer is None:
            self._mecab_tokenizer = self._mecab()
        return self._mecab_tokenizer

    def __reduce__(self):
        return MecabTokenizer, (self.vocab,)

    def __call__(self, text: str) -> Doc:
        if '\x00' in text:
            text = text.replace('\x00',' ') # replace \x00 to space
            
        dtokens = list(self.detailed_tokens(text))
        surfaces = [dt["surface"] for dt in dtokens]
        doc = Doc(self.vocab, words=surfaces, spaces=list(check_spaces(text, surfaces)))
        for token, dtoken in zip(doc, dtokens):
            first_tag, sep, eomi_tags = dtoken["tag"].partition("+")
            token.tag_ = first_tag  # stem(어간) or pre-final(선어말 어미)
            if token.tag_ in TAG_MAP:
                token.pos = TAG_MAP[token.tag_][POS]
            else:
                token.pos = X
            token.lemma_ = dtoken["lemma"]
            token._.tag_detail_t = eomi_tags
        
        return doc

    def detailed_tokens(self, text: str) -> Iterator[Dict[str, Any]]:
        ## Mecab tokenization
        # 품사 태그(POS)[0], 의미 부류(semantic class)[1],	종성 유무(jongseong)[2], 읽기(reading)[3],
        # 타입(type)[4], 첫번째 품사(start pos)[5],	마지막 품사(end pos)[6], 표현(expression)[7], *
        parsed = list(map(lambda x: x[1:], self.mecab_tokenizer.parse(text))) # compatible for legacy code. (span, surface, feature)
        
        ## emoji process
        emoji_in = self.emoji_set.intersection([*text])
        if emoji_in:
            grapheme_set = set(self.grapheme.findall(text))
            parsed = self._get_grapheme_aware_parsed(parsed, grapheme_set)
        
        ## Transform mecab to spaCy.
        for surface, feature in parsed:
            tag = feature.pos
            expr = feature.expression if feature.expression else '' # if expression is None replace None to empty string.
            lemma, _, remainder = expr.partition("/")
            if lemma == "*":
                lemma = surface
            yield {"surface": surface, "lemma": lemma, "tag": tag}
    
    def _get_grapheme_aware_parsed(self, parsed:List[Tuple], grapheme_set:Set) -> List[Tuple]:
        return get_grapheme_aware_parsed(parsed, grapheme_set, self.grapheme)

def get_grapheme_aware_parsed(parsed:List[Tuple], grapheme_set:Set, grapheme:Pattern) -> List[Tuple]:
        """Retokenize the mecab tokens for grapheme-aware form."""
        parsed_with_grapheme = []
        is_before_short = False
        is_joiner_exist = False
        
        for surface, feature in parsed:
            # 이모지는 SY(기타기호)로 매칭됨. 
            # SF(마침표, 물음표, 느낌표)와 같이 쓰이는 특수기호 있음.
            if feature.pos == 'SY' or feature.pos == 'SF':
                # check 'zero with joiner'
                if '\u200d' in surface:
                    is_joiner_exist = True
                
                human_char_list = grapheme.findall(surface)
                if len(human_char_list) > 1:
                    # too long
                    parsed_with_grapheme += [(c, Feature(*feature)) for c in human_char_list]
                    
                    if parsed_with_grapheme[-1][0] in grapheme_set:
                        is_before_short = False
                    else:
                        is_before_short = True

                elif (len(human_char_list) == 1) & (surface in grapheme_set):
                    # fit
                    parsed_with_grapheme.append((surface, feature))
                    is_before_short = False

                elif (len(human_char_list) == 1) & (surface not in grapheme_set):
                    # too short (코드포인트의 길이가 2 이상인 이모지의 조각)
                    if is_before_short:
                        # suffix parts
                        last_surface, last_feature = parsed_with_grapheme[-1]
                        parsed_with_grapheme[-1] = (f"{last_surface}{surface}", last_feature)
                        if f"{last_surface}{surface}" in grapheme_set:
                            is_before_short = False
                        else:
                            is_before_short = True
                    else:
                        # prefix parts
                        parsed_with_grapheme.append((surface, feature))
                        is_before_short = True
                else:
                    mecab_tokens = [s for s, _ in parsed]
                    raise NotImplementedError(f"This is the new edge case when processing emoji. \n tokens -> {mecab_tokens} \n error token -> {surface}")
            else:
                parsed_with_grapheme.append((surface, feature))
                is_before_short = False
        
        if is_joiner_exist:
            new_parsed_with_grapheme = []
            skip=False
            for i, (surface, feature) in enumerate(parsed_with_grapheme):
                if skip:
                    skip=False
                    continue
                if (surface == '\u200d') & (i != 0) & (i != len(parsed_with_grapheme)-1):
                    if new_parsed_with_grapheme[-1][-1].pos not in {'SY', 'SF'}:
                        continue
                    last_surface, last_feature = new_parsed_with_grapheme[-1]
                    if parsed_with_grapheme[i+1][-1].pos not in  {'SY', 'SF'}:
                        new_parsed_with_grapheme[-1] = (f"{last_surface}{surface}", feature)
                    else:
                        new_parsed_with_grapheme[-1] = (f"{last_surface}{surface}{parsed_with_grapheme[i+1][0]}", feature)
                        skip=True
                    continue
                new_parsed_with_grapheme.append((surface, feature))
            parsed_with_grapheme = new_parsed_with_grapheme

        return parsed_with_grapheme

def try_mecab_import() -> None:
    try:
        from mecab import MeCab

        return MeCab
    except ImportError:
        raise ImportError(
            'The Korean tokenizer ("spacy.ko.MecabTokenizer") requires, install following packages.'
            "$ pip install python-mecab-ko"
        ) from None

def check_spaces(text, tokens):
    prev_end = -1
    start = 0
    for token in tokens:
        idx = text.find(token, start)
        if prev_end > 0:
            yield prev_end != idx
        prev_end = idx + len(token)
        start = prev_end
    if start > 0:
        yield False