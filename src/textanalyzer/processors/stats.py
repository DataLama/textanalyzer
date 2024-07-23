# Here processor is aimed to basic statistics of the text
from typing import Callable, Union, Dict, List
import math
import statistics
from collections import Counter

from textanalyzer.processors.base import BaseProcessor, TokenizerFactory

class TokenCountProcessor(BaseProcessor):
    def __init__(self, tokenizer: Union[str, Callable] = 'tiktoken'):
        self.tokenizer = self._get_tokenizer(tokenizer)
    
    def _get_tokenizer(self, tokenizer):
        if isinstance(tokenizer, Callable):
            return tokenizer
        return TokenizerFactory.get_tokenizer(tokenizer)
    
    def __call__(self, examples: Dict[str, List], text_column: str = 'text') -> Dict[str, List]:
        token_counts = []
        for text in examples[text_column]:
            tokens = self.tokenizer(text)
            token_counts.append(len(tokens))
        return {'token_count': token_counts}
    
    def compute_stats(self, token_counts:List[int], with_sentence_length_distribution:bool=False) -> Dict[str, Union[int, float, Dict[int, int]]]:
        """Compute statistics of token counts
        Args:
            token_counts: List of token counts
        
        Returns:
            Dict[str, Any]: Dictionary of statistics
        """

        # 총 토큰 수
        total_tokens = sum(token_counts)

        # 평균 문장 길이
        avg_sentence_length = total_tokens / len(token_counts) if len(token_counts) > 0 else 0

        # 문장 길이의 표준편차
        sentence_length_std = statistics.stdev(token_counts) if len(token_counts) > 1 else 0

        # 최장 문장과 최단 문장의 토큰 수
        max_sentence_length = max(token_counts)
        min_sentence_length = min(token_counts)
        
        # 문장 길이 분포
        sentence_length_distribution = dict(Counter(token_counts))
        
        # 중앙값 문장 길이
        median_sentence_length = statistics.median(token_counts)

        outputs = {
            "total_tokens": total_tokens,
            "avg_sentence_length": avg_sentence_length,
            "sentence_length_std": sentence_length_std,
            "max_sentence_length": max_sentence_length,
            "min_sentence_length": min_sentence_length,
            "median_sentence_length": median_sentence_length
        }

        if with_sentence_length_distribution:
            outputs["sentence_length_distribution"] = sentence_length_distribution
        
        return outputs

    def format_token_counts(self, num: int) -> str:
        """Format token counts
        Args:
            num: Number to format

        Returns:
            str: Formatted number
        """
        if num < 1000:
            return str(num)
        elif num < 1000000:
            return f"{num / 1000:.1f}K"
        elif num < 1000000000:
            return f"{num / 1000000:.1f}M"
        elif num < 1000000000000:
            return f"{num / 1000000000:.1f}B"
        else:
            return f"{num / 1000000000000:.1f}T"