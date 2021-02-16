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
from typing import List, Union, Tuple, Iterable, Iterator
from itertools import accumulate

import emoji
from bs4 import BeautifulSoup
from LAC import LAC
from soynlp.normalizer import repeat_normalize

from .data_utils import Doc, Token
from .process_utils import Tokenization, TokenCandidateGeneration, ZhPreprocessing


class LACTokenization(Tokenization):
    def __init__(self, custom_dict:str = None):        
        # lac tokenizer https://github.com/baidu/lac
        self.lac = LAC(mode='lac')
        self.preprocessor = ZhPreprocessing()
        
        # Custom Dict https://github.com/baidu/lac/tree/master/python
        if custom_dict:
            self.lac.load_customization(customization_file=custom_dict)
    
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
        text = self.preprocessor.normalize_chinese_pattern(text)
        
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
        token, pos = self.lac.run(text)
        return token, pos
    
    def _postprocess(self, doc: Tuple[List[str]]) -> Tuple[List[str]]:
        # rule 1 : {'没', '否', '不', '未', '难'} + (동사 or 형용사)
        doc = self._postproc_rule_1(doc)
        # rule 2: 동사 + 的
        doc = self._postproc_rule_2(doc)
        # rule 3 : preserving pattern with custom dict (브랜드, 상품과 같이 붙여쓰기가 필요한 단어들)
        return doc[0], doc[1]
    
    def _postproc_rule_1(self, doc: Tuple[List[str]]) -> Tuple[List[str]]:
        """
        [*] Rule 1 {'没', '否', '不', '未', '难'} + (동사 or 형용사)
            중국어에서 위와 같은 부정사와 동사 또는 형용사가 연속으로 출현할 경우, 함께 봐야지 키워드의 의미가 있음.
        """
        # define variables
        token = doc[0]
        pos = doc[1]
        new_token = []
        new_pos = []
        neg_tok = set('没否不未难')
        sub_pos = {'v', 'vd', 'vn', 'a', 'an', 'ad'}
        skip = False
        
        # process
        for i, (t_now, t_next) in enumerate(zip(token, token[1:])):
            # if skip==True this token is already read. So skip now
            if skip:
                skip = False
                continue
            
            ## if true the rule concat t_now and t_next
            if (t_now in neg_tok) and (pos[i+1] in sub_pos):
                new_token.append(f"{t_now}{t_next}")
                new_pos.append(pos[i+1])
                skip = True    
            else:
                new_token.append(t_now)
                new_pos.append(pos[i])

        # if the final token is not in the rule the last token and pos should be added.
        if not skip:
            new_token.append(token[-1])
            new_pos.append(pos[-1])
            
        return new_token, new_pos
    
    def _postproc_rule_2(self, doc: Tuple[List[str]]) -> Tuple[List[str]]:
        """
        [*] Rule 2 동사 + 的
            중국어에서 동사 + 的은 동명사 같은 느낌으로 파생적인 의미를 만들어냄.
        """
        # define variables
        token = doc[0]
        pos = doc[1]
        new_token = []
        new_pos = []
        sub_pos = {'v', 'vd', 'vn'}
        skip = False
        
        # process
        for i, (t_now, t_next) in enumerate(zip(token, token[1:])):
            # if skip==True this token is already read. So skip now
            if skip:
                skip = False
                continue
            
            ## if true the rule concat t_now and t_next
            if (pos[i] in sub_pos) and (t_next == '的'):
                new_token.append(f"{t_now}{t_next}")
                new_pos.append(pos[i+1])
                skip = True    
            else:
                new_token.append(t_now)
                new_pos.append(pos[i])

        # if the final token is not in the rule the last token and pos should be added.
        if not skip:
            new_token.append(token[-1])
            new_pos.append(pos[-1])
            
        return new_token, new_pos
    
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

class LacTCG(TokenCandidateGeneration):
    def __init__(self, ngram):
        self.pos = {'LOC', 'ORG', 'PER', 'TIME', 'a', 'an', 'm', 'n', 'nr', 'ns', 'nt', 'nz', 'q', 's', 't', 'v', 'vn'}
        self.N = ngram
    
    def get_candidate(self, doc: Doc) -> Doc:
        if doc.tokenizable:
            unigram = [token for token in doc.tokens if (token.pos in self.pos) and (token.text.strip() != '')]
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