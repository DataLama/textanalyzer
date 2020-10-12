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


import unicodedata
from typing import List, Union, Tuple

from .data_utils import InputText, TextFeature


class Tokenization:
    """Base Class for Tokenization. This can be applied line by line."""
    def __call__(self, doc: InputText) -> TextFeature:
        """Tokenize the data with callable format."""
        # Initialize the variables
        title_tokens = []
        title_pos = []
        content_tokens = []
        content_pos = []
        
        # if the value is empty, tokenize and return values
        if doc.title:
            title_tokens, title_pos = self._postprocess(self.tokenize(self._preprocess(self._normalize(doc.title))))
        if doc.content:
            content_tokens, content_pos = self._postprocess(self.tokenize(self._preprocess(self._normalize(doc.title))))
        
        return TextFeature(
            title_tokens = title_tokens,
            content_tokens = content_tokens,
            title_pos = title_pos,
            content_pos = content_pos
            )
    
    def _normalize(self, text: str) -> str:
        """[Overwrite] Normalize string of input data.(Default: NFKC)"""
        return unicodedata.normalize('NFKC', text)    
        
    def _preprocess(self, text: str) -> str:
        """[OverWrite] Preprocessing stinrgs of input text data."""
        raise NotImplementedError()
    
    def tokenize(self, doc: str) -> Tuple[List[str]]:
        """[OverWrite] Tokenize the String to List of String. Return Dataclass is TextFeature.
        * input : string
        * return : first list is token list and second list is pos list. 
            e.g.) (['쿠팡', '에서', '후기', '가', '좋길래', '닦토용', '으로', '샀는데', '별로에요', '....'],  ['L', 'R', 'L', 'R', 'L', 'L', 'R', 'L', 'L', 'R'])
            return token, pos
        """
        raise NotImplementedError()
    
    def _postprocess(self, doc: Tuple[List[str]]) -> Tuple[List[str]]:
        """[OverWrite] Postprocessing for List of TextFeature"""
        raise NotImplementedError()

        
class TokenCandidateGeneration:
    """From tokenized token candidates, filter the candidates for keyphrase extraction"""
    def get_candidate(self):
        """[OverWrite] Candidates extraction from token list"""
        raise NotImplementedError()
                