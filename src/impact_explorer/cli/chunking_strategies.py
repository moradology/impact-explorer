import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


@dataclass
class Chunk:
    """
    Represents a chunk of text with start/end indices and metadata.
    """
    text: str
    start_index: int
    end_index: int
    metadata: Optional[dict] = None


class ChunkingStrategy(ABC):
    """
    Abstract Base Class for text chunking strategies.
    """

    @abstractmethod
    def chunk(self, text: str, **kwargs) -> List[Chunk]:
        """
        Chunk the input text into smaller pieces with metadata.
        """
        pass

    @classmethod
    @abstractmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """
        Add required arguments for this strategy.
        """
        pass

    @classmethod
    @abstractmethod
    def from_args(cls, args: argparse.Namespace) -> "ChunkingStrategy":
        """
        Build an instance of this class with supplied arguments.
        """
        pass


class SlidingWindowStrategy(ChunkingStrategy):
    """
    Implements sliding window chunking with overlapping text segments.
    """
    def __init__(self, chunk_size: int, overlap: int, embedding_model: str):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)

    def chunk(self, text: str) -> List[Chunk]:
        """
        Chunk text using sliding window strategy.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            chunks.append(
                Chunk(
                    text=chunk_text,
                    start_index=start,
                    end_index=end,
                    metadata={
                        "length": len(chunk_text),
                        "tokens": self.tokenizer.tokenize(chunk_text),
                    },
                )
            )
            start += self.chunk_size - self.overlap
        return chunks

    def __str__(self) -> str:
        """
        Return a string representation suitable for filenames.
        """
        return f"sliding-window_size-{self.chunk_size}_overlap-{self.overlap}"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """
        Add arguments for sliding window strategy.
        """
        parser.add_argument(
            "--chunk_size",
            type=int,
            required=True,
            help="Size of each chunk for sliding window",
        )
        parser.add_argument(
            "--overlap",
            type=int,
            required=True,
            help="Size of sliding window overlap chunks",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ChunkingStrategy":
        """
        Build an instance with supplied arguments.
        """
        return cls(chunk_size=args.chunk_size,
                   overlap=args.overlap,
                   embedding_model=args.embedding_model)


class SentenceStrategy(ChunkingStrategy):
    """
    Implements sentence-based chunking strategy.
    """
    def __init__(self, max_sentences: int, embedding_model: str):
        self.max_sentences = max_sentences
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModelForTokenClassification.from_pretrained(embedding_model)
        self.nlp = pipeline('token-classification', model=self.model, tokenizer=self.tokenizer)

    def chunk(self, text: str, **kwargs) -> List[Chunk]:
        """
        Chunk text by grouping sentences.
        """
        max_sentences = kwargs.get("max_sentences", self.max_sentences)
        sentences = self.nlp(text, aggregation_strategy="simple")
        sentences = [sentence['word'] for sentence in sentences]

        chunks = []
        current_position = 0

        for i in range(0, len(sentences), max_sentences):
            chunk_sentences = sentences[i: i + max_sentences]
            chunk_text = " ".join(chunk_sentences)
            start_index = current_position
            end_index = start_index + len(chunk_text)
            current_position = end_index + 1

            tokens = self.tokenizer.tokenize(chunk_text)

            chunks.append(
                Chunk(
                    text=chunk_text,
                    start_index=start_index,
                    end_index=end_index,
                    metadata={
                        "num_sentences": len(chunk_sentences),
                        "tokens": tokens,
                    },
                )
            )
        return chunks

    def __str__(self) -> str:
        """
        Return a string representation suitable for filenames.
        """
        return f"sentence_max-{self.max_sentences}"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """
        Add arguments for sentence-based chunking strategy.
        """
        parser.add_argument(
            "--max_sentences",
            type=int,
            required=True,
            help="Maximum number of sentences per chunk",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ChunkingStrategy":
        """
        Build an instance with supplied arguments.
        """
        return cls(max_sentences=args.max_sentences,
                   embedding_model=args.embedding_model)


strategies = {"sentence": SentenceStrategy, "sliding-window": SlidingWindowStrategy}
