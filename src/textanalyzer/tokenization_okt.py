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
from typing import List, Union, Tuple, Iterable, Iterator
from itertools import accumulate

import emoji
from bs4 import BeautifulSoup
from konlpy.tag import Okt

from .data_utils import Doc, Token
from .process_utils import Tokenization, TokenCandidateGeneration, KoPreprocessing

class OktTokenization(Tokenization):
    def __init__(self,tokenizer_dir, custom_dict:str = None):
        # okt tokenizer
        self.okt = Okt(max_heap_size=1024*4)
        self.preprocessor = KoPreprocessing()
        
        if custom_dict:
            pass
        
    def _normalize(self, text: str) -> str:
        # unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # eradicate html script
        html = re.compile("<(\"[^\"]*\"|'[^']*'|[^'\">])*>")
        if html.search(text) != None: # html js 처리
            soup = BeautifulSoup(text, "lxml")
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
        
        # chinese preprocessing format
        text = self.preprocessor.normalize_korean_pattern(text)
        
        # normalize repeated pattern
        text = repeat_normalize(text, num_repeats=3)
        
        return text.strip()
    
    def _preprocess(self, text: str) -> str:
        text = self.preprocessor.rm_url(text)
        text = self.preprocessor.rm_email(text)
        text = self.preprocessor.rm_emoji(text)
        text = self.preprocessor.rm_hashtag(text)
        text = self.preprocessor.rm_mention(text)
        text = self.preprocessor.rm_image(text)
        return text.strip()
    
    def _tokenize(self, text: str) -> Tuple[List[str]]:
        return self.soy.tokenize(text, flatten=self._is_flatten, remove_r = self._is_remove_r)
    
    def _postprocess(self, doc: Tuple[List[str]]) -> Tuple[List[str]]:
        processed_doc = []
        
        # L, R part 후처리
        for l_part, r_part in doc:
            
            ## l_part
            l_part = repeat_normalize(l_part, num_repeats=3) # normalization
            sub_l_part = re.findall(r"[\w]+|[\W]+", l_part) # 하나의 토큰안에 단어, punct, 품사가 같이 있는 경우 분리
            if len(sub_l_part)==2:
                processed_doc += [(sub, 'L') for sub in sub_l_part] 
            else:
                processed_doc.append((l_part, 'L'))      
            
            ## r_part
            if r_part !='':
                r_part = repeat_normalize(r_part, num_repeats=3) # normalization
                sub_r_part = re.findall(r"[\w]+|[\W]+", r_part) # 하나의 토큰안에 단어, punct, 품사가 같이 있는 경우 분리
                if len(sub_r_part)==2:
                    processed_doc += [(sub, 'R') for sub in sub_r_part] 
                else:
                    processed_doc.append((r_part, 'R'))
        
        return map(list, zip(*processed_doc))
    

    # class SoynlpTCG(TokenCandidateGeneration):
#     def __init__(self, ngram):
#         self.pos = {'L'}
#         self.N = ngram
    
#     def get_candidate(self, doc: Doc) -> Doc:
#         if doc.tokenizable:
#             unigram = [token for token in doc.tokens 
#                        if (token.pos in self.pos) and (token.text.strip() != '')  and (re.search(r'[a-zA-Z0-9ㄱ-힣]', token.text) != None)]
#             candidates = [unigram]
#             if self.N > 1:
#                 for i in range(1, self.N):
#                     candidates.append(self._ngram(unigram, (i+1)))
#             doc.candidates = candidates
#             return doc
#         else:
#             doc.candidates = [[]]
#             return doc
    
#     def _ngram(self, unigram: List[Token], n: int) -> List[Token]:
#         return [ngram for ngram in zip(*[unigram[i:] for i in range(n)])]