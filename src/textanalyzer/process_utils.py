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
import emoji
import unicodedata
from typing import List, Union, Tuple, Dict, Iterator

from .data_utils import Doc, Token


class Tokenization:
    """Base Class for Tokenization. This can be applied line by line."""
    def __call__(self, doc: Union[str, Dict], use_alignment:bool=False) -> Doc:
        """Tokenize the data with callable format."""        
        if type(doc) == dict:
            Id, text = list(doc.items())[0]
        elif type(doc) == str:
            Id = None
            text = doc
        else:
            raise TypeError(f'The type of doc should be {dict} rather than {type(doc)}.')
        # normalize text before preprocessing
        preprocessed_text = self._normalize(text)
        
        # if text is not None
        if text:
#             self.original_text = text # for alignment
            preprocessed_text = self._preprocess(preprocessed_text)
            if preprocessed_text:
                tokens, pos = self.tokenize(preprocessed_text)
                tokens = [Token(DocId=Id, text=tok, pos=p) for tok, p in zip(tokens, pos)]
                tokenizable = True
            else:
                tokens = []
                tokenizable = False

#             offsets = self._alignment((tokens, pos)) # offset of each
#             tokens = [Token(DocId=Id, text=tok, pos=p) 
#                       for tok, p, offset in zip(tokens, pos, offsets)]
        else:
            tokens = []
            tokenizable = False

        # Return Doc
        return Doc(
            Id = Id,
            text = text,
            preprocessed_text = preprocessed_text,
            tokenizable = tokenizable,
            tokens = tokens
        )
    
    def _normalize(self, text: str) -> str:
        """[Overwrite] Normalize string of input data.(Default: NFKC)"""
        return unicodedata.normalize('NFKC', text)
        
    def _preprocess(self, text: str) -> str:
        """[OverWrite] Preprocessing stinrgs of input text data."""
        raise NotImplementedError()
    
    def _tokenize(self, text: str) -> Tuple[List[str]]:
        """[OverWrite] Tokenize the String to List of String.
        * input : string
        * return : first list is token list and second list is pos list. 
            e.g.) (['쿠팡', '에서', '후기', '가', '좋길래', '닦토용', '으로', '샀는데', '별로에요', '....'],  ['L', 'R', 'L', 'R', 'L', 'L', 'R', 'L', 'L', 'R'])
            return token, pos
        
        * We're highly recommend you to use detokenizable tokenizer.
        """
        raise NotImplementedError()
  
    def _postprocess(self, doc: Tuple[List[str]]) -> Tuple[List[str]]:
        """[OverWrite] Postprocessing for tokenized data."""
        raise NotImplementedError()
        
    def _alignment(self, doc: Tuple[List[str]]) -> List:
        """[OverWrite] Calculate the offset of token in the document. 
        * highly recommend to return iterator
        """
        raise NotImplementedError()
        
    def tokenize(self, text: str) -> Tuple[List[str]]:
        """Tokenize the single string with pre, post processing."""
        return self._postprocess(self._tokenize(text))
    

        
class TokenCandidateGeneration:
    """From tokenized token candidates, filter the candidates for keyphrase extraction"""
    def __call__(self, doc: Doc) -> Doc:
        return self.get_candidate(doc)
    
    def get_candidate(self, doc: Doc) -> Doc:
        """[OverWrite] Candidates extraction from token list"""
        raise NotImplementedError()
                

class ZhPreprocessing:
    """All about chinese text preprocessing function in NLP"""
    def __init__(self):
        emojis = "".join(emoji.UNICODE_EMOJI.keys())
        self.emoji = emoji.get_emoji_regexp()
        self.pattern = re.compile(f'[^ ％·∼。、，《 》“”：\x00-\x7F\u4e00-\u9fff{emojis}]+') # 기호, 영어, 중국어, 이모티콘
        self.url = re.compile(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
        self.email = re.compile('([0-9a-zA-Z_]|[^\s^\w])+(@)[a-zA-Z]+.[a-zA-Z)]+')
        self.hashtag = re.compile(f'#([{emojis}\w-]+)')
        self.mention = re.compile(f'@([\w-]+)')
        self.image = re.compile(r'(\[image#0\d\])')
    
    def normalize_chinese_pattern(self, text:str) -> str:
        """영어, 중국어, 이모지, 특수기호를 제외한 모든 것을 제거함."""
        return self.pattern.sub('', text)
    
    def rm_url(self, text:str) -> str:
        return self.url.sub('', text)
    
    def rm_email(self, text:str) -> str:
        return self.email.sub('', text)
    
    def rm_emoji(self, text:str) -> str:
        return self.emoji.sub('', text)
    
    def rm_hashtag(self, text:str) -> str:
        return self.hashtag.sub('', text)
    
    def rm_mention(self, text:str) -> str:
        return self.mention.sub('', text)

    def rm_image(self, text:str) -> str:
        return self.image.sub('', text)

class KoPreprocessing:
    """All about chinese text preprocessing function in NLP"""
    def __init__(self):
        emojis = "".join(emoji.UNICODE_EMOJI.keys())
        self.emoji = emoji.get_emoji_regexp()
        self.pattern = re.compile(f'[^ ％·∼\x00-\x7Fㄱ-ㅣ가-힣{emojis}]+') # 기호, 영어, 한글, 이모티콘
        self.url = re.compile(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
        self.email = re.compile('([0-9a-zA-Z_]|[^\s^\w])+(@)[a-zA-Z]+.[a-zA-Z)]+')
        self.hashtag = re.compile(f'#([{emojis}\w-]+)')
        self.mention = re.compile(f'@([\w-]+)')
        self.image = re.compile(r'(\[image#0\d\])')
        self.white_space_character = re.compile(r'\s+')
    
    def normalize_korean_pattern(self, text:str) -> str:
        """영어, 한글, 이모지, 특수기호를 제외한 모든 것을 제거함."""
        return self.pattern.sub('', text)
    
    def rm_url(self, text:str) -> str:
        return self.url.sub(' ', text)
    
    def rm_email(self, text:str) -> str:
        return self.email.sub(' ', text)
    
    def rm_emoji(self, text:str) -> str:
        return self.emoji.sub(' ', text)
    
    def rm_mention(self, text:str) -> str:
        return self.mention.sub(' ', text)

    def rm_image(self, text:str) -> str:
        return self.image.sub(' ', text)
    
    def rm_hashtag(self, text:str) -> str:
        return self.hashtag.sub(' ', text)
    
    def spacing_hashtag(self, text:str) -> str:
        for ht in self.hashtag.findall(text):
            p = re.compile(f'#{ht}')
            text = p.sub(f' #{ht} ', text)
        return text
    
    def normalize_space(self, text:str) -> str:
        return self.white_space_character.sub(' ', text)

    