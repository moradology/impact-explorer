import argparse
import os
import re
import sys
from typing import List

import chromadb
import docx
import markdown2
import PyPDF2
from chunking_strategies import Chunk, ChunkingStrategy, strategies
from sentence_transformers import SentenceTransformer
from utils import generate_doc_id, print_chroma_stats, print_model_help


def sanitize_filename(name: str) -> str:
    """Sanitize the input string to be used as a filename."""
    return re.sub(r"[^\w\-_\.]", "_", name)


def generate_chroma_db_name(
    chunking_strategy: ChunkingStrategy, embedding_model: str
) -> str:
    """Generate a unique ChromaDB name based on configuration."""
    embedding_model_short = embedding_model.split("/")[-1]
    db_name = f"{chunking_strategy}_{embedding_model_short}"
    return sanitize_filename(db_name)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Document Processor and Embedder")
    parser.add_argument(
        "--model-help",
        action="store_true",
        help="Display a summary of pretrained models and their trade-offs",
    )
    temp_args, _ = parser.parse_known_args()

    if temp_args.model_help:
        print_model_help()
        sys.exit(0)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-file", type=str, help="Path to the input file.")
    group.add_argument("--input-dir", type=str, help="Path to the input directory.")

    parser.add_argument(
        "--output-dir", default="./", help="Output directory for ChromaDB"
    )
    parser.add_argument(
        "--chunking-strategy",
        required=True,
        choices=list(strategies.keys()),
        help="Chunking strategy to use",
    )
    parser.add_argument(
        "--embedding-model",
        default="all-distilroberta-v1",
        help="Name of the sentence-transformers model to use for embeddings",
    )

    args, remaining = parser.parse_known_args()

    strategy_class = strategies[args.chunking_strategy]

    # Add strategy-specific arguments; build chunking strategy
    strategy_parser = argparse.ArgumentParser()
    strategy_class.add_arguments(strategy_parser)
    strategy_args = strategy_parser.parse_args(remaining)
    strategy_args.embedding_model = args.embedding_model
    chunking_strategy = strategy_class.from_args(strategy_args)

    chroma_db_name = generate_chroma_db_name(chunking_strategy, args.embedding_model)

    final_args = argparse.Namespace(
        output_dir=args.output_dir,
        chroma_db_name=chroma_db_name,
        chunking_strategy=chunking_strategy,
        embedding_model=args.embedding_model,
    )
    if args.input_file:
        input_file = args.input_file
        input_dir = None
    elif args.input_dir:
        input_file = None
        input_dir = args.input_dir

    final_args = argparse.Namespace(
        output_dir=args.output_dir,
        chroma_db_name=chroma_db_name,
        chunking_strategy=chunking_strategy,
        embedding_model=args.embedding_model,
        input_file=input_file,
        input_dir=input_dir,
    )

    return final_args


def load_document(input_path: str) -> str:
    file_extension = os.path.splitext(input_path)[1].lower()

    if file_extension == ".txt":
        with open(input_path, "r", encoding="utf-8") as file:
            return file.read()

    if file_extension == ".md":
        with open(input_path, "r", encoding="utf-8") as file:
            markdown = file.read()
        return markdown2.markdown(markdown)

    elif file_extension == ".pdf":
        try:
            with open(input_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            raise ValueError(f"Error reading PDF file: {e}")

    elif file_extension == ".docx":
        try:
            doc = docx.Document(input_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        except Exception as e:
            raise ValueError(f"Error reading DOCX file: {e}")

    else:
        raise ValueError(f"Unsupported file type: {file_extension}")


def generate_embeddings(chunks: List[Chunk], model_name: str) -> List[List[float]]:
    model = SentenceTransformer(model_name)

    texts = [chunk.text for chunk in chunks if chunk.text.strip()]
    if not texts:
        raise ValueError("No valid texts to generate embeddings for.")

    embeddings = model.encode(texts).tolist()
    if len(embeddings) != len(texts):
        raise ValueError("Mismatch between number of texts and embeddings.")

    return embeddings


def initialize_chroma(output_dir: str, chroma_db_name: str) -> chromadb.Client:
    os.makedirs(output_dir, exist_ok=True)
    return chromadb.PersistentClient(
        path="/".join([x.strip("/") for x in [output_dir, chroma_db_name]])
    )


def add_to_chroma(
    client: chromadb.Client,
    collection_name: str,
    chunks: List[Chunk],
    embeddings: List[List[float]],
    doc_id: str,
    batch_size: int = 200,
) -> None:
    collection = client.get_or_create_collection(collection_name)

    valid_chunks_embeddings = [
        (chunk, embedding)
        for chunk, embedding in zip(chunks, embeddings)
        if len(embedding) > 0
    ]
    if not valid_chunks_embeddings:
        raise ValueError("No valid chunks or embeddings to add to database.")

    for i in range(0, len(valid_chunks_embeddings), batch_size):
        batch = valid_chunks_embeddings[i : i + batch_size]
        chunk_batch = [chunk.text for chunk, _ in batch]
        embedding_batch = [embedding for _, embedding in batch]
        metadata_batch = [
            {
                **{
                    k: str(v) if isinstance(v, list) else v
                    for k, v in chunk.metadata.items()
                },
                "start_index": chunk.start_index,
                "end_index": chunk.end_index,
            }
            for chunk, _ in batch
        ]
        ids_batch = [f"{doc_id}_chunk{j}" for j in range(i, i + len(batch))]

        collection.add(
            documents=chunk_batch,
            embeddings=embedding_batch,
            metadatas=metadata_batch,
            ids=ids_batch,
        )

        print(f"Added batch {i//batch_size + 1}: {len(batch)} documents to ChromaDB")


def process_document(
    input_file: str,
    chunking_strategy: str,
    embedding_model: str,
    chroma_db_name: str,
    output_dir: str,
) -> None:
    text = load_document(input_file)
    chunking_strategy = chunking_strategy
    chunks = chunking_strategy.chunk(text)
    embeddings = generate_embeddings(chunks, embedding_model)

    client = initialize_chroma(output_dir, chroma_db_name)
    doc_id = generate_doc_id(input_file)
    add_to_chroma(client, "documents", chunks, embeddings, doc_id)
    print_chroma_stats(client, "documents")

    print(
        f"Processed document using chunking={chunking_strategy}; "
        f"model={embedding_model}, and stored in {output_dir} "
        f"with name {chroma_db_name}"
    )


def main():
    args = parse_arguments()
    chunking_strategy = args.chunking_strategy
    embedding_model = args.embedding_model
    chroma_db_name = args.chroma_db_name
    output_dir = args.output_dir

    files = []
    if args.input_file:
        files.append(args.input_file)
    elif args.input_dir:
        for file in os.listdir(args.input_dir):
            file_path = os.path.join(args.input_dir, file)
            if os.path.isfile(file_path):
                files.append(file_path)

    for file in files:
        print(f"Processing file: {file}")
        process_document(
            file, chunking_strategy, embedding_model, chroma_db_name, output_dir
        )


if __name__ == "__main__":
    main()
