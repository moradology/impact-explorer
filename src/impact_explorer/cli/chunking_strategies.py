import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from transformers import AutoTokenizer
from utils import load_spacy_model


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

    def __init__(self, chunk_size: int, overlap: int):
        self.chunk_size = chunk_size
        self.overlap = overlap

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
                    metadata={"window_length": len(chunk_text)},
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
            "--chunk-size",
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
        return cls(chunk_size=args.chunk_size, overlap=args.overlap)


class SentenceStrategy(ChunkingStrategy):
    """
    Implements sentence-based chunking strategy.
    """

    def __init__(self, max_sentences: int, overlap: int):
        self.max_sentences = max_sentences
        self.overlap = overlap
        self.nlp = load_spacy_model()

    def chunk(self, text: str) -> List[Chunk]:
        """
        Chunk text by grouping sentences.
        """
        max_sentences = self.max_sentences
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]

        chunks = []
        current_position = 0

        i = 0
        while i < len(sentences):
            chunk_sentences = sentences[i : i + max_sentences]
            # Combine sentences into chunk text
            chunk_text = " ".join(chunk_sentences)
            start_index = current_position
            end_index = start_index + len(chunk_text)
            current_position = end_index + 1

            chunks.append(
                Chunk(
                    text=chunk_text,
                    start_index=start_index,
                    end_index=end_index,
                    metadata={"num_sentences": len(chunk_sentences)},
                )
            )

            # next chunk with overlap
            i += max_sentences - self.overlap

        return chunks

    def __str__(self) -> str:
        """
        Return a string representation suitable for filenames.
        """
        return f"sentence_max{self.max_sentences}_overlap{self.overlap}"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """
        Add arguments for sentence-based chunking strategy.
        """
        parser.add_argument(
            "--max-sentences",
            type=int,
            required=True,
            help="Maximum number of sentences per chunk",
        )
        parser.add_argument(
            "--overlap",
            type=int,
            required=True,
            help="Sentence overlap",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ChunkingStrategy":
        """
        Build an instance with supplied arguments.
        """
        return cls(max_sentences=args.max_sentences, overlap=args.overlap)


class TokenTextSplitterStrategy:
    """
    Implements chunking by splitting text into chunks based on tokenization.

    Currently just busted as hell. TODO: fix it
    """

    def __init__(self, max_tokens: int, overlap: int, embedding_model: str):
        """
        Initializes TokenTextSplitter with max token count and overlap.

        :param max_tokens: Maximum number of tokens per chunk.
        :param overlap: Number of overlapping tokens between chunks.
        :param embedding_model: Pre-trained model to use for tokenization.
        """
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)

    def chunk(self, text: str) -> List[Chunk]:
        """
        Chunk text using token-based strategy with overlap.

        :param text: The input text to be chunked.
        :return: A list of Chunk objects.
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []

        start = 0
        while start < len(tokens):
            end = min(start + self.max_tokens, len(tokens))
            token_chunk = tokens[start:end]
            chunk_text = self.tokenizer.decode(
                token_chunk, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

            chunks.append(
                Chunk(
                    text=chunk_text,
                    start_index=start,
                    end_index=end,
                    metadata={"length": len(chunk_text), "tokens": token_chunk},
                )
            )
            # Move start forward by max_tokens minus overlap to create overlap
            start += self.max_tokens - self.overlap

        return chunks

    def __str__(self) -> str:
        """
        Return a string representation suitable for filenames.
        """
        return f"tokentext_max{self.max_tokens}_overlap{self.overlap}"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """
        Add arguments for sentence-based chunking strategy.
        """
        parser.add_argument(
            "--max-tokens",
            type=int,
            required=True,
            help="Maximum number of tokens per chunk",
        )
        parser.add_argument(
            "--overlap",
            type=int,
            required=True,
            help="Token overlap",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ChunkingStrategy":
        """
        Build an instance with supplied arguments.
        """
        return cls(
            max_tokens=args.max_tokens,
            overlap=args.overlap,
            embedding_model=args.embedding_model,
        )


strategies = {
    "sentence": SentenceStrategy,
    "sliding-window": SlidingWindowStrategy,
    "token-text": TokenTextSplitterStrategy,
}
