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

import json
import dataclasses
from dataclasses import dataclass,field
from typing import List, Optional, Union

@dataclass
class Token:
    """
    Token-level textual-features. I recommend subclassing this dataclass for your needs.
    
    Args:
        DocId: Unique id for the example. 
        offset: Char-level position of the token in parent Document.
                If offset is integer use offset and size to indexing the token in original document.
                If offset is list use list to indexing the token int original document.
        text: The raw text of Token. 
        pos: Part-of-speeches after tokenizing.
    """
    DocId: str
    text: str
    pos: str
    
    @property
    def size(self):
        """offset과 size를 활용하여 이 토큰이 원래 Document에 어디에 위치하는지 알 수 있음."""
        return len(self.text)
    
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"

@dataclass
class Doc:
    """
    Document-level textual-features. I recommend subclassing this dataclass for your needs.
    
    Args:
        Id: Unique id for the example. 
        text: The raw text of Doc.
        tokens: List of Tokens.
    """
    Id: str
    text: str
    preprocessed_text: str
    tokenizable: bool
    tokens: List[Optional[Token]]
        
        
    @property
    def size(self):
        """Size of Document"""
        return len(self.text)
        
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"
