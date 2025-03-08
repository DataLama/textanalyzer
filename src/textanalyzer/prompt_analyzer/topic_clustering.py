import json
import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
from enum import Enum

import numpy as np
import torch
import datasets
import langchain_core
import spacy
import openai
import tiktoken
from tqdm import tqdm
from  langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from bertopic import BERTopic
from bertopic.representation import PartOfSpeech, OpenAI
from bertopic.backend import OpenAIBackend

from textanalyzer import Korean
from textanalyzer.core.ko import Korean, complex_korean_keyword_patterns
from textanalyzer.processors.base import BaseProcessor


class Language(str, Enum):
    KO = "ko"
    EN = "en"

@dataclass
class ModelConfig:
    min_topic_size: int = 32
    batch_size: int = 2000
    max_conv_length: int = 10000
    embedding_model: str = "jinaai/jina-embeddings-v3"
    gpt_model: str = "gpt-4o-mini"
    language: Language = Language.KO

class ConversationProcessor(BaseProcessor):
    def __init__(self, config: ModelConfig):
        self.config = config
        
    def _process_single_conversation(self, conv_data: Dict[str, Any]) -> Optional[str]:
        if conv_data["language"] != self.config.language:
            return None
            
        processed_text = []
        for message in conv_data["messages"]:
            if message["role"] != "user":
                continue
                
            content = message["content"]
            if isinstance(content, list):
                processed_text.append(content[0])
            elif isinstance(content, str):
                processed_text.append(content)
            else:
                raise ValueError(f"Unknown content type: {type(content)}")
                
        conv = "\n".join(processed_text)
        
        # Filter out short conversations
        if len(conv) <= 32:
            return None

        return conv[:self.config.max_conv_length]
    
    def __call__(self, examples: Dict[str, List], *extra_args) -> Dict[str, List]:
        """Process a batch of messages."""
        processed_convs = []
        for i in range(len(examples["messages"])):
            # Construct single example
            example = {key: examples[key][i] for key in examples.keys()}
            processed = self._process_single_messages(example)
            processed_convs.append(processed if processed else "")
            
        return {"processed_text": processed_convs}
    
class EmbeddingGenerator:
    def __init__(self, embedding_model: Embeddings, config: ModelConfig):
        self.embedding_model = embedding_model
        self.config = config
        
    def generate_embeddings(self, dataset: datasets.Dataset) -> np.ndarray:
        all_embeddings = []
        
        for i in tqdm(range(0, len(dataset), self.config.batch_size)):
            batch = dataset[i:i + self.config.batch_size]["processed_text"]
            # Filter out empty strings
            batch = [text for text in batch if text]
            
            if not batch:
                continue
                
            embeddings = self.embedding_model.embed_documents(batch)
            all_embeddings.extend(embeddings)
            
        return np.array(all_embeddings)

def load_embedding_model(model_name:str) -> langchain_core.embeddings.Embeddings:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name == "jinaai/jina-embeddings-v3":
        model_kwargs = {"device": device, "trust_remote_code": True}
        encode_kwargs = {
            "task": "text-matching",
            "prompt_name": "text-matching",
        }
        embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        return embedding_model
    
    raise ValueError(f"Unknown model name: {model_name}")


class TopicModelBuilder:
    def __init__(
            self,
            embedding_model: Embeddings,
            config: ModelConfig,
        ):
        self.embedding_model = embedding_model
        self.config = config
        
    def _initialize_pos_model(self) -> PartOfSpeech:
        if self.config.language == "ko":
            return PartOfSpeech(Korean(), pos_patterns=complex_korean_keyword_patterns)
        elif self.config.language == "en":
            try:
                return PartOfSpeech("en_core_web_sm")
            except:
                spacy.cli.download("en_core_web_sm")
                return PartOfSpeech("en_core_web_sm")
            
    def _create_openai_model(self) -> OpenAI:
        prompt = """
        I have a topic that contains the following documents:
        [DOCUMENTS]
        The topic is described by the following keywords: [KEYWORDS]

        Based on the information above, extract a short but highly descriptive topic label of at most 5 words. Make sure it is in the following format:
        topic: <topic label>
        """
        client = openai.OpenAI()
        tokenizer = tiktoken.encoding_for_model(self.config.gpt_model)
        if self.config.gpt_model not in set(map(lambda x: x.id, client.models.list().data)):
            client = openai.AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY_EASTUS"),  
                api_version=os.getenv("OPENAI_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_EASTUS")
            )
            tokenizer = tiktoken.get_encoding("o200k_base")
        
        return OpenAI(
            self.client,
            model=self.config.gpt_model,
            exponential_backoff=True,
            chat=True,
            prompt=prompt,
            nr_docs=50,
            doc_length=500,
            tokenizer=tokenizer
        )
        
    def build_model(self) -> BERTopic:
        pos_model = self._initialize_pos_model()
        openai_model = self._create_openai_model()
        
        representation_model = {
            "OpenAI": openai_model,
            "POS": pos_model
        }
            
        return BERTopic(
            verbose=True,
            embedding_model=self.embedding_model,
            representation_model=representation_model,
            min_topic_size=self.config.min_topic_size,
        )

class TopicModelRunner:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.embedding_model = load_embedding_model(config.embedding_model)
        self.processor = ConversationProcessor(config)
        self.embedding_generator = EmbeddingGenerator(self.embedding_model, config)
        self.model_builder = TopicModelBuilder(self.embedding_model, config)
    
    def _load_dataset(self, input_path: str) -> datasets.Dataset:
        """Load dataset from JSON file."""
        if input_path.endswith('.json'):
            dataset = datasets.load_dataset('json', data_files=input_path)['train']
        else:
            dataset = datasets.load_dataset(input_path)['train']
        return dataset
        
    def run(self, args: argparse.Namespace):
        # Load embeddings if provided
        if args.embedding_file:
            embeddings = np.load(args.embedding_file)
            if not args.post_process_conv:
                raise ValueError("Please provide post process conv file")
            dataset = self._load_dataset(args.post_process_conv)
        else:
            # Load and process dataset
            dataset = self._load_dataset(args.conv_file)
            if args.first_n is not None:
                dataset = dataset.select(range(min(len(dataset), args.first_n)))
            
            # Process conversations using HuggingFace datasets
            processed_dataset = dataset.map(
                self.processor,
                batched=True,
                remove_columns=dataset.column_names,
                desc="Processing conversations"
            )
            
            # Filter out empty processed texts
            processed_dataset = processed_dataset.filter(
                lambda x: bool(x['processed_text']),
                desc="Filtering empty texts"
            )
            
            # Generate embeddings
            embeddings = self.embedding_generator.generate_embeddings(processed_dataset)
            
            # Save processed data and embeddings
            os.makedirs(args.output_dir, exist_ok=True)
            processed_dataset.save_to_disk(f"{args.output_dir}/processed_dataset")
            np.save(f"{args.output_dir}/embeddings.npy", embeddings)
        
        print("#convos:", len(processed_dataset))
        
        # Get processed texts for topic modeling
        texts = processed_dataset["processed_text"]
        
        # Build and train topic model
        topic_model = self.model_builder.build_model()
        topics, _ = topic_model.fit_transform(texts, embeddings)
        
        # Process results
        new_topics = topic_model.reduce_outliers(texts, topics)
        with open(f"{args.output_dir}/conv_topics.json", "w") as f:
            json.dump(new_topics, f, default=str)
        
        # Save model and results
        topic_model.save(
            f"{args.output_dir}/model_dir",
            serialization="pytorch",
            save_ctfidf=True,
            save_embedding_model=topic_model.embedding_model
        )
        
        df = topic_model.get_topic_info()
        df.to_csv(f"{args.output_dir}/topics.csv", index=False)