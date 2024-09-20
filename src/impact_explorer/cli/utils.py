import chromadb
import subprocess
import spacy


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