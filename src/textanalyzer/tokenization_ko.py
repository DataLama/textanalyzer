import re
import unicodedata
import bisect
import pickle
import pandas as pd
from typing import List, Union, Tuple, Iterable, Iterator
from itertools import accumulate

import emoji
from bs4 import BeautifulSoup
from soynlp.tokenizer import LTokenizer
from soynlp.normalizer import repeat_normalize

from .data_utils import Doc, Token
from .process_utils import Tokenization, TokenCandidateGeneration, KoPreprocessing

class SoynlpTokenization(Tokenization):
    def __init__(self,tokenizer_dir, custom_dict:str = None):
        # 보존 단어 리스트
        if custom_dict:
            add_scores = pd.read_csv(custom_dict, encoding='utf-8', header=None)
        else:
            add_scores = pd.read_csv(f'{tokenizer_dir}/token_dict.csv', encoding='utf-8', header=None)
        add_scores = dict(zip(add_scores[0], add_scores[1]))
        
        with open(f'{tokenizer_dir}/words.p', 'rb') as rf:
            words = pickle.load(rf)    
            cohesion_score = {word: score.cohesion_forward for word, score in words.items()}
            cohesion_score.update(add_scores)
            cohesion_score = {k: v for k, v in sorted(cohesion_score.items(), key=lambda item: item[1], reverse=True) if v > 0}
            
        with open(f'{tokenizer_dir}/nouns.p', 'rb') as rf:
            nouns = pickle.load(rf)
            noun_score = {noun: score.score for noun, score in nouns.items()}
            noun_score.update(add_scores)
            noun_cohesion_score = {noun: score + cohesion_score.get(noun, 0) for noun, score in noun_score.items()} 
            self._noun_cohesion_score = {k: v for k, v in sorted(noun_cohesion_score.items(), key=lambda item: item[1], reverse=True) if v > 0}
            
        self.soy = LTokenizer(scores=self._noun_cohesion_score)
        self._is_flatten = False # no_flatten
        self._is_remove_r = False # no_remove
        self.preprocessor = KoPreprocessing()
        
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
        
    
    def _alignment(self, doc: Tuple[List[str]]) -> List:
        """Only use token to alignment."""
        i = 0
        offsets = []
        tokens = doc[0]
        re_meta = {char:f'\\{char}' for char in '$()*+.?[]\^{}|-'}
        
        for token in tokens:
            sliced_text = self.original_text[i:]
            
            # 정규표현식 사용할 때, 메타문자가 토큰에 포함되어있을 경우 에러 발생함.
            pattern = token
            if set(pattern).intersection(re_meta.keys()) != set():
                pattern = ''.join(map(lambda char:re_meta[char] if char in re_meta else char, pattern))
            search = re.search(pattern, sliced_text)
            
            if search:
                start = search.start()
                offsets.append((i+start))
                i += (start + len(token))
            else: # 어떤 토큰이 original text에 없을 경우 => 전처리 과정에서 text가 변함.
                l_offset = []
                for tok in token:
                    if tok in re_meta:
                        start = re.search(re_meta[tok], sliced_text).start()
                    else:
                        start = re.search(tok, sliced_text).start()
                    l_offset.append((i+start))
                offsets.append(l_offset)
                i += start
        return offsets
    
class SoynlpTCG(TokenCandidateGeneration):
    def __init__(self, ngram):
        self.pos = {'L'}
        self.N = ngram
    
    def get_candidate(self, doc: Doc) -> Doc:
        if doc.tokenizable:
            unigram = [token for token in doc.tokens 
                       if (token.pos in self.pos) and (token.text.strip() != '')  and (re.search(r'[a-zA-Z0-9ㄱ-힣]', token.text) != None)]
            candidates = [unigram]
            if self.N > 1:
                for i in range(1, self.N):
                    candidates.append(self._ngram(unigram, (i+1)))
            doc.candidates = candidates
            return doc
        else:
            doc.candidates = [[]]
            return doc
    
    def _ngram(self, unigram: List[Token], n: int) -> List[Token]:
        return [ngram for ngram in zip(*[unigram[i:] for i in range(n)])]