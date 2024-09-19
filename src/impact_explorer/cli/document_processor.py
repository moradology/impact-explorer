import argparse
import os
from typing import List, Tuple
from chunking_strategies import ChunkingStrategy, strategies, Chunk
from utils import print_chroma_stats
import chromadb
import re
from sentence_transformers import SentenceTransformer

def sanitize_filename(name: str) -> str:
    """Sanitize the input string to be used as a filename."""
    return re.sub(r'[^\w\-_\.]', '_', name)

def generate_chroma_db_name(chunking_strategy: ChunkingStrategy, embedding_model: str) -> str:
    """Generate a unique ChromaDB name based on configuration."""
    embedding_model_short = embedding_model.split('/')[-1]
    db_name = f"{chunking_strategy}_{embedding_model_short}"
    return sanitize_filename(db_name)

def parse_arguments() -> argparse.Namespace:
    print("Debug: Starting parse_arguments()")

    parser = argparse.ArgumentParser(description="Document Processor and Embedder")
    parser.add_argument("--input", required=True, help="Input file path or string")
    parser.add_argument("--output-dir", default="./", help="Output directory for ChromaDB")
    parser.add_argument("--chunking-strategy", required=True, choices=list(strategies.keys()), help="Chunking strategy to use")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Name of the sentence-transformers model to use for embeddings")
    
    print("Debug: Parsing known arguments")
    args, remaining = parser.parse_known_args()
    print(f"Debug: args = {args}")
    print(f"Debug: remaining = {remaining}")
    
    strategy_class = strategies[args.chunking_strategy]
    
    # Add strategy-specific arguments; build chunking strategy
    strategy_parser = argparse.ArgumentParser()
    strategy_class.add_arguments(strategy_parser)
    strategy_args = strategy_parser.parse_args(remaining)
    chunking_strategy = strategy_class.from_args(strategy_args)

    chroma_db_name = generate_chroma_db_name(chunking_strategy, args.embedding_model)
    
    final_args = argparse.Namespace(
        input=args.input,
        output_dir=args.output_dir,
        chroma_db_name=chroma_db_name,
        chunking_strategy=chunking_strategy,
        embedding_model=args.embedding_model,
    )
    
    return final_args

def load_document(input_path: str) -> str:
    file_extension = os.path.splitext(input_path)[1].lower()
    
    if file_extension == '.txt':
        with open(input_path, 'r', encoding='utf-8') as file:
            return file.read()
    elif file_extension in ['.pdf', '.docx']:
        # TODO: Implement PDF and DOCX loading
        raise NotImplementedError(f"Loading {file_extension} files is not yet implemented")
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

def generate_embeddings(chunks: List[Chunk], model_name: str) -> List[List[float]]:
    model = SentenceTransformer(model_name)
    return model.encode([chunk.text for chunk in chunks]).tolist()

def initialize_chroma(output_dir: str, chroma_db_name: str) -> chromadb.Client:
    os.makedirs(output_dir, exist_ok=True)
    return chromadb.PersistentClient(path="/".join([x.strip('/') for x in [output_dir, chroma_db_name]]))

def add_to_chroma(client: chromadb.Client, collection_name: str, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
    collection = client.get_or_create_collection(collection_name)
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        # Convert metadata to appropriate types
        metadata = {
            **{k: str(v) if isinstance(v, list) else v for k, v in chunk.metadata.items()},
            "start_index": chunk.start_index,
            "end_index": chunk.end_index
        }
        collection.add(
            documents=[chunk.text],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[f"doc_{i}"]
        )

def process_document(args: argparse.Namespace) -> None:
    # Load document
    text = load_document(args.input)
    
    # Get chunking strategy
    chunking_strategy = args.chunking_strategy
    
    # Chunk text
    chunks = chunking_strategy.chunk(text)
    
    # Generate embeddings
    embeddings = generate_embeddings(chunks, args.embedding_model)
    
    # Initialize ChromaDB
    client = initialize_chroma(args.output_dir, args.chroma_db_name)
    
    # Add to ChromaDB
    add_to_chroma(client, "documents", chunks, embeddings)
    print_chroma_stats(client, "documents")
    
    print(f"Processed document using {chunking_strategy} and stored in {args.output_dir} with name {args.chroma_db_name}")


def main():
    args = parse_arguments()
    process_document(args)

if __name__ == "__main__":
    main()