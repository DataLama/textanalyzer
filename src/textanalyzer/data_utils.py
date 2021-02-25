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
from typing import Dict, List, Optional, Union

@dataclass
class Token:
    """Analyzing Token-Level Textual Features
    
    - Contains Id and raw text
    - Contains Token-Level textual features
    - Recommend subclassing this dataclass for your needs.
    
    Args:
        DocId: Unique id for the example.
        offset: index of the token in the Doc.Tokens
        start: begin index of the token in the Doc.text
        end: end index of the token in the Doc.text
        text: The raw text of Doc.
        
    """
    DocId: str
    offset: int
    start: int
    end: int
    text: str
    
    def __post_init__(self):
        self._features = dict()

    def _update(self, features:Dict) -> None:
        """Update features for parsed by attributes."""
        key_set = []
        for k, v in features.items():
            setattr(self, k, v)
            key_set.append(k)
        print(f"New features {', '.join(key_set)} is updated.")
    
    def update_feature(self, features:Dict) -> None:
        self._update(features)
        self._features.update(**features)
    
    @property
    def features(self) -> Dict:
        """Get all features"""
        return self._features
        
    @property
    def size(self) -> int:
        """Size of Document"""
        return len(self.text)
        
    def to_dict(self):
        """Serializes this instance to dictionary."""
        return dataclasses.asdict(self)

@dataclass
class Doc:
    """Analyzing Document-Level Textual Features
    
    - Contains Id and raw text
    - Contains Document-Level textual features
    - Contains Chunked Result (Sents) and Tokenized Result (Tokens)
    - Recommend subclassing this dataclass for your needs.
    
    Args:
        Id: Unique id for the example. 
        text: The raw text of Doc.
        Tokens: List of Tokens. Tokens are the result of tokenization.
    """
    Id: str
    text: str
    Tokens: List[Optional[Token]]
    
    def __post_init__(self):
        self._features = dict()

    def _update(self, features:Dict) -> None:
        """Update features for parsed by attributes."""
        key_set = []
        for k, v in features.items():
            setattr(self, k, v)
            key_set.append(k)
        print(f"New features {', '.join(key_set)} is updated.")
    
    def update_feature(self, features:Dict) -> None:
        self._update(features)
        self._features.update(**features)
    
    def __getitem__(self, idx):
        """Doc class is basically the container of tokens."""
        return self.Tokens[idx]
        
    def __len__(self):
        """Length of Doc is token-level length"""
        return len(self.Tokens)
    
    @property
    def features(self) -> Dict:
        """Get all features"""
        return self._features
    
    @property
    def size(self):
        """Size of Document as a char-level length"""
        return len(self.text)
        
    def to_dict(self):
        """Serializes this instance to dictionary."""
        return dataclasses.asdict(self)

    

# To-Do
# * Add feature name checker
# ```
#     def _check_feature(self, features:Dict) -> None:
#         """Check the name of features"""
#         for k, _ in features.items():
#             prefix = k.split('-')[0]
#             condition = (prefix == 'tag') or (prefix == 'vec')
#             if not condition:
#                 raise ValueError(f"{k}'s prefix must be 'tag' or 'vec'.")
                
                
#         self._check_feature(features)
# ```
