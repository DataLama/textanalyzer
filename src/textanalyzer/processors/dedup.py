# MinHash + LSH style deduplication
from typing import List, Dict

import xxhash
import tiktoken
import numpy as np
from datasketch import MinHash, MinHashLSH
from datasets import load_dataset

from textanalyzer.processors.base import BaseProcessor

ENC = tiktoken.get_encoding("o200k_base")
SEED = 921110
N_GRAM_SIZE = 5 # n-gram size
N_PERM = 800 # signature size
THRESHOLD = 0.91 # -> b=20, r=40 by (1/b)^(1/r) = 0.91
SIGNATURE_COL = "signature"

def byte_tokenize(text:str) -> List[bytes]:
    """Tokenize text into byte tokens (by tiktoken)"""
    return ENC.decode_tokens_bytes(ENC.encode(text))

def get_byte_ngrams(byte_tokens: List[bytes], n: int) -> List[bytes]:
    return [b''.join(byte_tokens[i:i+n]) for i in range(len(byte_tokens) - n + 1)]

def xxhash_func(b:bytes) -> int:
    """Based on datasketch's benchmark(https://ekzhu.com/datasketch/minhash.html#use-different-hash-functions)"""
    return xxhash.xxh3_64(b).intdigest()


class MinHashProcessor(BaseProcessor):
    def __init__(self, num_perm:int=N_PERM, ngram_size:int=N_GRAM_SIZE):
        self.num_perm = num_perm
        self.ngram_size = ngram_size
    
    def __call__(self, examples: Dict[str, List], text_column: str = 'text') -> Dict[str, List[List[int]]]:
        texts = examples[text_column]
        byte_tokens = [byte_tokenize(text) for text in texts]
        byte_ngrams = [get_byte_ngrams(tokens, self.ngram_size) for tokens in byte_tokens]

        signatures = []
        for ngrams in byte_ngrams:
            minhash = MinHash(num_perm=self.num_perm, seed=SEED, hashfunc=xxhash_func)
            for ngram in ngrams:
                minhash.update(ngram)
            signatures.append(minhash.digest().tolist())

        return {SIGNATURE_COL: signatures}
    
class StreamingDedupProcessor(BaseProcessor):
    def __init__(self, num_perm:int=N_PERM, threshold:float=THRESHOLD):
        self.num_perm = num_perm
        self.threshold = threshold
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    
    def __call__(self, examples: Dict[str, List], indices:List[int]) -> Dict[str, List]:
        signatures = examples[SIGNATURE_COL]
        is_dups = []

        for idx, signature in zip(indices, signatures):
            minhash = MinHash(num_perm=self.num_perm)
            minhash.hashvalues = np.array(signature)

            if self.lsh.query(minhash):
                is_dups.append(True)
            else:
                self.lsh.insert(idx, minhash)
                is_dups.append(False)

        return {"is_dups": is_dups}