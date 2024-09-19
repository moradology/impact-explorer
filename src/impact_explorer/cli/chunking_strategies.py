from abc import ABC, abstractmethod
import argparse
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class Chunk:
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

        Args:
            text (str): The input text to be chunked.
            **kwargs: Strategy-specific arguments.

        Returns:
            List[Chunk]: A list of Chunk objects containing text and metadata.
        """
        pass

    @classmethod
    @abstractmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """
        Get the list of required arguments for this strategy.
        """
        pass

    @classmethod
    @abstractmethod
    def from_args(cls, args: argparse.Namespace) -> 'ChunkingStrategy':
        """
        Build an instance of this class with supplied arguments
        """
        pass

class SlidingWindowStrategy(ChunkingStrategy):

    def __init__(self, chunk_size: int, overlap: int):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[Chunk]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            chunks.append(Chunk(
                text=chunk_text,
                start_index=start,
                end_index=end,
                metadata={
                    "length": len(chunk_text),
                    "tokens": chunk_text.split()  # Simple tokenization
                }
            ))
            start += self.chunk_size - self.overlap
        return chunks

    def __str__(self) -> str:
        """Return a string representation suitable for filenames."""
        return f"sliding-window_size-{self.chunk_size}_overlap-{self.overlap}"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--chunk_size", type=int, required=True, help="Size of each chunk for sliding window")
        parser.add_argument("--overlap", type=int, required=True, help="Size of sliding window overlap chunks")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'ChunkingStrategy':
        """
        Build an instance of this class with supplied arguments
        """
        return cls(chunk_size=args.chunk_size, overlap=args.overlap)


class SentenceStrategy(ChunkingStrategy):

    def __init__(self, max_sentences: int):
        self.max_sentences = max_sentences

    def chunk(self, text: str, **kwargs) -> List[Chunk]:
        max_sentences = kwargs.get('max_sentences', 5)
        import nltk
        nltk.download('punkt', quiet=True)
        sentences = nltk.sent_tokenize(text)
        
        chunks = []
        for i in range(0, len(sentences), max_sentences):
            chunk_sentences = sentences[i:i+max_sentences]
            chunk_text = ' '.join(chunk_sentences)
            start_index = text.index(chunk_sentences[0])
            end_index = start_index + len(chunk_text)
            chunks.append(Chunk(
                text=chunk_text,
                start_index=start_index,
                end_index=end_index,
                metadata={
                    "num_sentences": len(chunk_sentences),
                    "tokens": chunk_text.split()
                }
            ))
        return chunks


    def __str__(self) -> str:
        """Return a string representation suitable for filenames."""
        return f"sentence_max-{self.max_sentences}"

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--max_sentences", type=int, required=True, help="Maximum number of sentences per chunk")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'ChunkingStrategy':
        """
        Build an instance of this class with supplied arguments
        """
        return cls(max_sentences=args.max_sentences)


strategies = {
    "sentence": SentenceStrategy,
    "sliding-window": SlidingWindowStrategy
}