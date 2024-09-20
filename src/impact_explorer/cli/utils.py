import os
import subprocess

import chromadb
import spacy


def print_model_help():
    """
    Prints a high-level description of various SentenceTransformer models and their trade-offs.
    """
    help_text = """
Available Pretrained Models and Trade-offs:
-------------------------------------------
1. all-MiniLM-L6-v2
   - Size: 66M params | Max tokens: 256 | Speed: Very fast | Accuracy: Moderate
   - Use Case: Large-scale, real-time search with moderate accuracy.

2. all-MiniLM-L12-v2
   - Size: 110M params | Max tokens: 256 | Speed: Fast | Accuracy: Better than L6
   - Use Case: Fast general-purpose search with good accuracy.

3. paraphrase-MiniLM-L6-v2
   - Size: 66M params | Max tokens: 256 | Speed: Very fast | Accuracy: Moderate
   - Use Case: Paraphrase detection, sentence similarity.

4. paraphrase-MpNet-base-v2
   - Size: 110M params | Max tokens: 256 | Speed: Fast | Accuracy: High
   - Use Case: High-accuracy sentence similarity, semantic search.

5. multi-qa-MiniLM-L6-cos-v1
   - Size: 66M params | Max tokens: 256 | Speed: Very fast | Accuracy: QA-optimized
   - Use Case: Question-answering, fast multi-lingual search.

6. all-distilroberta-v1
   - Size: 82M params | Max tokens: 512 | Speed: Moderate | Accuracy: High
   - Use Case: Longer text, nuanced comparisons.

7. msmarco-distilbert-base-v4
   - Size: 66M params | Max tokens: 512 | Speed: Fast | Accuracy: Retrieval-optimized
   - Use Case: Passage retrieval, document search.

8. roberta-large-nli-stsb-mean-tokens
   - Size: 355M params | Max tokens: 512 | Speed: Slow | Accuracy: Very high
   - Use Case: High-accuracy semantic search, small-scale.
"""
    print(help_text)


def print_chroma_stats(client: chromadb.Client, collection_name: str) -> None:
    collection = client.get_collection(collection_name)

    print("\n" + "=" * 60)
    print(" ChromaDB Collection Statistics ".center(60, "="))
    print("=" * 60 + "\n")

    # Get the count of items
    count = collection.count()
    print(f"📊 Total documents: {count}")

    if count > 0:
        # Peek at the first item
        first_item = collection.peek(limit=1)

        # Embedding dimensionality
        embedding_dim = len(first_item["embeddings"][0])
        print(f"🧠 Embedding dimensions: {embedding_dim}")

        # Metadata keys
        metadata_keys = set(first_item["metadatas"][0].keys())
        print("\n📋 Metadata fields:")
        for key in metadata_keys:
            print(f"  • {key}")

        # Sample some items
        sample_size = min(3, count)
        sample = collection.get(ids=[f"doc_{i}" for i in range(sample_size)])

        print(f"\n🔍 Peek at {sample_size} documents:")
        for i in range(sample_size):
            print(f"\n  Document {i+1}:")
            print(f"  {'ID:':<10} {sample['ids'][i]}")
            print(f"  {'Preview:':<10} {sample['documents'][i][:100]}...")
            print("  Metadata:")
            for k, v in sample["metadatas"][i].items():
                print(f"    {k:<15} {v}")

        # Some interesting stats
        doc_lengths = [len(doc) for doc in sample["documents"]]
        avg_length = sum(doc_lengths) / len(doc_lengths)
        print("\n📏 Document Statistics:")
        print(f"  {'Average length:':<20} {avg_length:.2f} characters")
        print(f"  {'Shortest document:':<20} {min(doc_lengths)} characters")
        print(f"  {'Longest document:':<20} {max(doc_lengths)} characters")


def load_spacy_model():
    """Load en_core_web_sm, installing as necessary"""
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Model 'en_core_web_sm' not found. Downloading now...")
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
    # this means up to 2gb in mem. or like 1.5 moby dicks
    nlp.max_length = 2000000
    return nlp


def generate_doc_id(input_path: str) -> str:
    document_name = os.path.basename(input_path).replace(".", "-")
    sanitized_name = "".join(
        c if c.isalnum() or c in ("-", "_") else "_" for c in document_name
    )

    return sanitized_name
