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

from itertools import chain
from typing import List, Tuple
from collections import Counter, ChainMap

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer

from .ranking_utils import TokenRanking


class LacTokenRanking(TokenRanking):
    def __init__(self, top_k = None):
        self.top_k = top_k 

    def get_score(self, candidates: List[List[List[str]]]) -> List[Tuple[str, float]]:
        """
        input style : docs[doc[unigram, bigram, trigram]]
        """
        # build DTM
        dtm, idx2token, token2idx = self._build_document_term_matrix(candidates)
        
        # fit_transform with tfidf vector without normalization - 튜닝
        tfidf = TfidfTransformer()
        dtm = tfidf.fit_transform(dtm)
        scores = np.squeeze(np.asarray(dtm.sum(axis=0))) # document-wise sum
        
        # top_k를 특별히 지정하지 않으면 상위 20%를 노출함.
        if not self.top_k:
            self.top_k = int(len(scores) * 0.2)
        
        # Get keyphrase
        keyphrase_args = np.argsort(-scores)[:self.top_k]
        keyphrase = [(idx2token[i], scores[i]) for i in keyphrase_args]
        
        return keyphrase
    
    def _build_document_term_matrix(self, candidates: List[List[List[str]]]) -> Tuple[csr_matrix, List, dict]:
        ## apply custom weight map(func, candidates)        
        bunch_of_bow = list(map(lambda x: Counter(chain(*x)), candidates)) # List[Counter]
        idx2token = sorted(dict(ChainMap(*bunch_of_bow)).keys())
        token2idx = {tok:i for i, tok in enumerate(idx2token)}
        
        # Transform list-of-dict to document-term-matrix using sparse matrix
        rows = list(chain(*[[doc_idx] * len(doc) for doc_idx, doc in enumerate(bunch_of_bow)])) # for (i,j)~DTM row-wise index position
        cols, data = list(zip(*chain(*[doc.items() for doc in bunch_of_bow]))) # term keywords, data is frequence
        cols = [token2idx[c] for c in cols] # transform term keyword to for (i,j)~DTM column-wise index position
        dtm = csr_matrix((data, (rows, cols)))
        
        return dtm, idx2token, token2idx