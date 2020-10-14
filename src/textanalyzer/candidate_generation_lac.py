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
import bisect
from typing import List, Union, Tuple, Iterable
from itertools import accumulate

import emoji
from bs4 import BeautifulSoup
from LAC import LAC

from .data_utils import InputText, TextFeature
from .candidate_generation_utils import Tokenization, TokenCandidateGeneration


class LACTokenization(Tokenization):
    def __init__(self, custom_dict : Iterable[str] = None):
        ## preprocess
        self.emojis = ''.join(emoji.UNICODE_EMOJI.keys())
        self.pattern = re.compile(f'[^ .,-/+?!/@$%~％·∼()。、，《 》“”：0-9a-zA-Z\u4e00-\u9fff{self.emojis}]+')
        self.html = re.compile("<(\"[^\"]*\"|'[^']*'|[^'\">])*>")
        self.url = re.compile(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
        self.email = re.compile('([0-9a-zA-Z_]|[^\s^\w])+(@)[a-zA-Z]+.[a-zA-Z)]+')
        
        ## tokenizer
        # lac tokenizer https://github.com/baidu/lac
        
        self.lac = LAC(mode='lac')
        
        ## preserving pattern
        self.custom_dict = custom_dict
        if self.custom_dict:
            self.p_pattern = self._preserve_pattern_gen(custom_dict)
        
    def _preserve_pattern_gen(self, custom_dict) -> dict:
        """우선 regular expression으로 구현. (to.do. flashtext)"""
        return {word:re.compile('\s*'.join(word)) for word in custom_dict}
    
    def _preprocess(self, text: str) -> str:
        # html과 JS 제거하기.
        if self.html.search(text) != None: # html js 처리
            soup = BeautifulSoup(text, "lxml")
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
        # 숫자, 문자, 공백, 이모지, 특수문자를 제외한 모든 유니코드 제거
        text = text.strip()
        text = self.pattern.sub(' ', text)
        # URL을 URL태그로 변환
        text = self.url.sub(' [URL] ', text) # url
        # Email을 email 태그로 변환 
        text = self.email.sub(' [EMAIL] ', text)
        # 해시태그 사이에 spacing
        for d in re.findall(r'#(\w+)', text):
            p = re.compile(f'#{d}')
            text = p.sub(f' #{d} ', text)
        return text
        
    def tokenize(self, text: str) -> Tuple[List[str]]:
        token, pos = self.lac.run(text)
        return token, pos
    
    def _postprocess(self, doc: Tuple[List[str]]) -> Tuple[List[str]]:
        # rule 1 : {'没', '否', '不', '未', '难'} + (동사 or 형용사)
        doc = self._postproc_rule_1(doc)
        # rule 2: 동사 + 的
        doc = self._postproc_rule_2(doc)
        # rule 3 : preserving pattern with custom dict (브랜드, 상품과 같이 붙여쓰기가 필요한 단어들)
        if self.custom_dict:
            doc = self._postproc_preserve_pattern(doc)
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
        
    def _postproc_preserve_pattern(self, doc: Tuple[List[str]]) -> Tuple[List[str]]:
        """우선 regular expression으로 구현. (to.do. flashtext)"""
        token = doc[0]
        pos = doc[1]
        new_token = []
        new_pos = []
        
        # regular expression 기반의 pattern matching
        text = " ".join(token)
        char_len = list(accumulate([len(x)+1 if (i!=0) and ((i+1)!=len(token)) else len(x) 
                                    for i, x in enumerate(token)]))

        for word, p in self.p_pattern.items(): # p: pattern, s: search, b: begin, e: end
            concat_idx = []
            for s in p.finditer(text): # text에 속한 pattern들 search
                b, e = s.span()
                b = bisect.bisect_left(char_len, b)
                e = bisect.bisect_right(char_len, e)
                concat_idx.append((b, e)) # text에서 인덱스위치와 token, pos의 인덱스를 매핑
            
            # concatenation pattern이 존재할 경우 
            if concat_idx != []:
                before = 0
                for b, e in concat_idx:
                    if before < b:
                        new_token += token[before:b]
                        new_pos += pos[before:b]
                        
                    new_token.append(''.join(token[b:e]))
                    new_pos.append('n') # 명사로 취급
                    before = e
                
                # 마지막 패턴 뒤의 token과 Pos들 붙여주기
                if (e+1) < len(text):
                    new_token += token[e:]
                    new_pos += pos[e:]
                    
        return new_token, new_pos



class LacTCG(TokenCandidateGeneration):
    def __init__(self, ngram):
        self.pos = {'LOC', 'ORG', 'PER', 'TIME', 'a', 'an', 'm', 'n', 'nr', 'ns', 'nt', 'nz', 'q', 's', 't', 'v', 'vn'}
        self.ngram = ngram
        
    def get_candidate(self, doc: InputText) -> TextFeature:
        if doc.title:
            # unigram -> filter the token with pos
            title_unigram = [tok for tok, p in zip(doc.title_tokens, doc.title_pos) 
                                         if (p in self.pos) and (tok.strip() != '')]
            # build ngram
            title_candidate = [[] for _ in range(self.ngram)]
            title_candidate[0] += title_unigram
            
            title_bigram = None
            title_trigram = None
            title_4gram = None
            
            if self.ngram >= 2:
                title_bigram = self._make_ngram(title_unigram, 2)
                title_candidate[1] += title_bigram
            if self.ngram >= 3:
                title_trigram = self._make_ngram(title_unigram, 3)
                title_candidate[2] += title_trigram
            if self.ngram >= 4:
                title_4gram = self._make_ngram(title_unigram, 4)
                title_candidate[3] += title_4gram
                
        if doc.content:
            # unigram -> filter the token with pos
            content_unigram = [tok for tok, p in zip(doc.content_tokens, doc.content_pos) 
                                         if (p in self.pos) and (tok.strip() != '')]
            # build ngram
            content_candidate = [[] for _ in range(self.ngram)]
            content_candidate[0] += content_unigram
            
            content_bigram = None
            content_trigram = None
            content_4gram = None
            
            if self.ngram >= 2:
                content_bigram = self._make_ngram(content_unigram, 2)
                content_candidate[1] += content_bigram
            if self.ngram >= 3:
                content_trigram = self._make_ngram(content_unigram, 3)
                content_candidate[2] += content_trigram
            if self.ngram >= 4:
                content_4gram = self._make_ngram(content_unigram, 4)
                content_candidate[3] += content_4gram
                
        if doc.title:
            return TextFeature(
                guid = doc.guid,
                title = doc.title,
                title_pos = doc.title_pos,
                title_unigram = title_unigram,
                title_bigram = title_bigram,
                title_trigram = title_trigram,
                title_4gram = title_4gram,
                title_candidate = title_candidate,
                content = doc.content,
                content_pos = doc.content_pos,
                content_unigram = content_unigram,
                content_bigram = content_bigram,
                content_trigram = content_trigram,
                content_4gram = content_4gram,
                content_candidate = content_candidate,
        )
        else:
            return TextFeature(
                guid = doc.guid,
                content = doc.content,
                content_pos = doc.content_pos,
                content_unigram = content_unigram,
                content_bigram = content_bigram,
                content_trigram = content_trigram,
                content_4gram = content_4gram,
                content_candidate = content_candidate,
            )
            
    def _make_ngram(self, text: List[str], n: int) -> List[Tuple[str]]:
        return [''.join(ngram) for ngram in zip(*[text[i:] for i in range(n)])] # '' 으로 join