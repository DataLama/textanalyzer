# MIT License

# Copyright (c) 2020 Kim DongWook

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import re
import unicodedata
import bisect
import pickle
import pandas as pd
from typing import List, Union, Tuple, Iterable, Iterator, Dict
from itertools import accumulate
from pathlib import Path

import emoji
from bs4 import BeautifulSoup
from mecab import MeCab
from flashtext import KeywordProcessor
from soynlp.normalizer import repeat_normalize

from .data_utils import Doc, Token
from .process_utils import Tokenization, TokenCandidateGeneration, KoPreprocessing

class MecabTokenization(Tokenization):
    def __init__(self, custom_dir:str = None):
        self.preprocessor = KoPreprocessing()
        path = Path(custom_dir)
        str_match_dict = path.glob('*.txt')
        
        # init mecab custom dictionary
        self.mecab = MeCab()
            
        # init kpe custom dictionary
        self.kpe = KeywordProcessor()
        for fn in str_match_dict:
            self.kpe.add_keyword_from_file(fn)
        
        # features
        self._hashtag = []
            
    def __call__(self, doc: Union[str, Dict], use_rtk:bool=True) -> Doc:
        """Mecab++ Tokenizer
    
        - Mecab with flashtext's keyphrase extraction.
        - You can parse keypharse with use_rtk.
        
        Args:
            doc: text 
            use_rtk: retokenization with extracted keyphrase.
        
        Returns:
            doc: Textanalyzers.Doc class
        """
        
        doc = super().__call__(doc)
        if use_rtk:
            doc = self._retokenize(doc)
        return doc
    
        
    def _normalize(self, text: str) -> str:
        # eradicate html script
        html = re.compile("<(\"[^\"]*\"|'[^']*'|[^'\">])*>")
        if html.search(text) != None: # html js 처리
            soup = BeautifulSoup(text, "lxml")
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            
        # korean preprocessing format
        text = self.preprocessor.normalize_korean_pattern(text)
        
        # normalize repeated pattern
        text = repeat_normalize(text, num_repeats=3).strip()
        return text
    
    def _preprocess(self, text: str) -> str:
        text = self.preprocessor.rm_url(text)
        text = self.preprocessor.rm_email(text)
        self._hashtag = [ht.strip(self.preprocessor._re_meta) for ht in self.preprocessor.hashtag.findall(text)]
        text = self.preprocessor.rm_emoji(text)
        text = self.preprocessor.replace_hashtag(text)
        text = self.preprocessor.rm_mention(text)
        text = self.preprocessor.rm_image(text)
        return text.strip()
    
    def _tokenize(self, text: str) -> List:
        return self.mecab.parse(text)
    
    def _postprocess(self, doc: List) -> List:
        tokens = []
        for token, parsed in doc:
            feature = dict(parsed._asdict())
            feature['_pos'] = feature.pop('pos')
            tokens.append((token, feature))
        return tokens

    def _retokenize(self, doc:Doc) -> Doc:
        """토크나이즈된 Textanalyzers.Doc을 매칭된 문자열로 retokenize하는 프로세스
        
        - self.kpe로 사전에 등록된 keyphrase를 index와 함께 추출함.
            - case 1) 기존 여러개로 쪼개진 토큰이 keyphrase로 하나의 큰 토큰으로 추출되는 경우
        - add hashtag
        """
        keyphrases = self.kpe.extract_keywords(doc.text, span_info=True) # list of tuple
        # keyphrases[0] ->  (keyphrase, start_index, end_index)
        if keyphrases:
            kpe_idx = 0
            Tokens = []
            for token in doc.Tokens:
                if token.start == keyphrases[kpe_idx][1]:
                    end = sum(map(lambda tok: tok.end if tok.end == keyphrases[kpe_idx][2] else 0, doc.Tokens[token.offset:]))
                    if end != 0:
                        tok = Token(
                            DocId = token.DocId,
                            offset = token.offset,
                            start = token.start,
                            end = end,
                            text = keyphrases[kpe_idx][0]
                        )
                        feature = dict.fromkeys(token.features.keys())
                        feature['_pos'] = 'KEYPHRASE'
                        tok.update_feature(feature)
                        Tokens.append(tok)
                        if len(keyphrases) > kpe_idx+1:
                            kpe_idx += 1
                        continue
                if (len(Tokens)==0) or (Tokens[-1].end < token.end):
                    Tokens.append(token)
            doc.Tokens = Tokens
        # add hashtag
        doc.update_feature({'hashtag': self._hashtag})
        return doc