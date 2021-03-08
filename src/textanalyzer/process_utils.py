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
from pytz import timezone, utc
from datetime import datetime

from .data_utils import Doc, Token


class Tokenization:
    """Base Class for Tokenization. This can be applied line by line."""
    
    def __call__(self, doc: Union[str, Dict]) -> Doc:
        Id, text = self._input_handler(doc)
        normalized_text = self._normalize(text)
        
        if text:
            preprocessed_text = self._preprocess(normalized_text)
            if preprocessed_text:
                # Token
                tokens = self.tokenize(preprocessed_text)
                token_indices = self._alignment(normalized_text, tokens)
                processed_tokens = []
                for offset, ((tok, feature), (start, end))  in enumerate(zip(tokens, token_indices)):
                    token = Token(
                        DocId = Id,
                        offset = offset,
                        start = start,
                        end = end,
                        text = tok
                    )
                    token.update_feature(feature)
                    if (end < len(tok)) and (normalized_text[end] == ' '):
                        token.update_feature({'_space': True})
                    else:
                        token.update_feature({'_space': False})                    
                    processed_tokens.append(token)
                    
                Tokens = processed_tokens
                tokenizable = True
            else:
                # No token after preprocess
                Tokens = []
                tokenizable = False
        else:
            # empty string
            Tokens = []
            tokenizable = False
            
        # Doc
        doc = Doc(
            Id = Id,
            text = normalized_text,
            Tokens = Tokens
        )
        doc.update_feature({"_tokenizable" : tokenizable})
        return doc
    
    
    def _input_handler(self, doc: Union[str, Dict]) -> Tuple:
        if type(doc) == dict:
            Id, text = list(doc.items())[0]
        elif type(doc) == str:
            KST = timezone('Asia/Seoul')
            now = datetime.utcnow()
            today = utc.localize(now).astimezone(KST)
            Id = today.strftime('%Y%m%d-%H%M%S-%f')
            text = doc
        else:
            raise TypeError(f'The type of doc should be {dict} rather than {type(doc)}.')
        return Id, text
    
    def _normalize(self, text: str) -> str:
        """[Overwrite] Normalize string of input data.(Default: NFKC)"""
        return unicodedata.normalize('NFKC', text)
        
    def _preprocess(self, text: str) -> str:
        """[OverWrite] Preprocessing stinrgs of input text data."""
        raise NotImplementedError()
    
    def _tokenize(self, text: str) -> List:
        """[OverWrite] Tokenize the String to List of String.
        * input : string
        * return : list of tuple
                    - first index is token.
                    - second index is feature which contains pos.
        
        * We're highly recommend you to use detokenizable tokenizer.
        """
        raise NotImplementedError()
  
    def _postprocess(self, doc: List) -> List:
        """[OverWrite] Postprocessing for tokenized data."""
        raise NotImplementedError()
        
    def _alignment(self, text:str, tokens:List) -> List:
        """Calculate the alignment of tokens in the normalized text."""
        token_indices = []
        offset = 0
        
        for token in tokens:
            tok = token[0] # token[0] is text of token
            start = text.index(tok)
            token_indices.append((
                offset + start,
                offset + start + len(tok)
            ))
            offset += start + len(tok)
            text = text[start + len(tok):]
        return token_indices

        
    def tokenize(self, text: str) -> List:
        """Tokenize the single string with pre, post processing."""
        return self._postprocess(self._tokenize(text))
    

        
class TokenCandidateGeneration:
    """From tokenized token candidates, filter the candidates for keyphrase extraction"""
    def __call__(self, doc: Doc) -> Doc:
        return self.get_candidate(doc)
    
    def get_candidate(self, doc: Doc) -> Doc:
        """[OverWrite] Candidates extraction from token list"""
        raise NotImplementedError()

        
class KoPreprocessing:
    """All about chinese text preprocessing function in NLP"""
    def __init__(self):
        emojis = "".join(emoji.UNICODE_EMOJI.keys())
        self.emoji = emoji.get_emoji_regexp()
        self.pattern = re.compile(f'[^ ％·∼\x00-\x7Fᄀ-ᅵ가-힣{emojis}]+') # 기호, 영어, 한글, 이모티콘
        self.url = re.compile(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
        self.email = re.compile('([0-9a-zA-Z_]|[^\s^\w])+(@)[a-zA-Z]+.[a-zA-Z)]+')
        self.hashtag = re.compile(f'#([{emojis}\w-]+)')
        self.mention = re.compile(f'@([\w-]+)')
        self.image = re.compile(r'(\[image#0\d\])')
        self.white_space_character = re.compile(r'\s+')
        self._re_meta = "$()*+.?[\^{|"
    
    def normalize_korean_pattern(self, text:str) -> str:
        """영어, 한글, 이모지, 특수기호를 제외한 모든 것을 제거함."""
        text = unicodedata.normalize('NFKC', text)
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
            ht = ht.strip(self._re_meta)
            p = re.compile(f'#{ht}')
            text = p.sub(f' #{ht} ', text)
        return text
    
    def replace_hashtag(self, text:str) -> str:
        for ht in self.hashtag.findall(text):
            ht = ht.strip(self._re_meta)
            p = re.compile(f'#{ht}')
            text = p.sub(f' {ht} ', text)
        return text
    
    def normalize_space(self, text:str) -> str:
        return self.white_space_character.sub(' ', text)

            
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
        self._re_meta = "$()*+.?[\^{|"
    
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
    
    def replace_hashtag(self, text:str) -> str:
        for ht in self.hashtag.findall(text):
            ht = ht.strip(self._re_meta)
            p = re.compile(f'#{ht}')
            text = p.sub(f'{ht}', text)
        return text

    def rm_image(self, text:str) -> str:
        return self.image.sub('', text)


    