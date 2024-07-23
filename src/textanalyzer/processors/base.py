from abc import ABC, abstractmethod

from typing import Dict, List, Union, Any

class BaseProcessor(ABC):
    """Processor is compatible with huggingface datasets' map mathod.
    
    * Now this processor is designed to process the batch.map processing task.
    """
    @abstractmethod
    def __call__(
        self,
        examples: Dict[str, List],
        *extra_args
    ) -> Dict[str, List]:
        """This call must be used with ds.map with `batched=True`.
        
        Reference: https://huggingface.co/docs/datasets/v2.20.0/en/package_reference/main_classes#datasets.Dataset.map
        """
        pass

class TokenizerFactory:
    """Basic tokenizers factory class."""

    @staticmethod
    def get_mecab_tokenizer():
        try:
            from mecab import MeCab
            mecab = MeCab()
        except ImportError:
            raise ImportError("MeCab is not installed. Please install it to use the MeCab tokenizer.(`pip install python-mecab-ko`)")
        return mecab.morphs

    @staticmethod
    def get_tiktoken_tokenizer():
        """Get TikToken tokenizer (current version is `o200k_base`)"""
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        return enc.encode
    
    @staticmethod
    def get_huggingface_tokenizer(model_name):
        from transformers import AutoTokenizer
        hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
        return hf_tokenizer.tokenize

    @classmethod
    def get_tokenizer(cls, tokenizer_type):
        if tokenizer_type == 'mecab':
            return cls.get_mecab_tokenizer()
        if tokenizer_type == 'tiktoken':
            return cls.get_tiktoken_tokenizer()
        elif tokenizer_type.startswith('hf_'):
            return cls.get_huggingface_tokenizer(tokenizer_type[3:])
        else:
            raise ValueError("Unsupported tokenizer type")