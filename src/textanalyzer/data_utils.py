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
class InputText:
    """
    A single Document data for Text Data analysis.
    
    Args:
        guid: Unique id for the example.
        title: (Optional) string. The raw text for document's title.
        content: string. The raw text for document's contents.
    """

    guid: str
    title: Optional[str] = None
    content: str = None
    

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


@dataclass(frozen=True)
class TextFeature:
    """
    The textual features of single Document.
    
    Args:
        {title, content}_tokens: List of String. Tokenized token list for raw text.
        {title, content}_pos: (Optional) List of String. Part-of-speeches for each token.
    """
    
    title_tokens: List[str]
    content_tokens: List[str]
    
    title_pos: Optional[List[str]] = None
    content_pos: Optional[List[str]] = None
    #### more features with the development
    
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"