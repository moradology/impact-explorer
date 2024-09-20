# Impact Explorer

This project implements a FastAPI server that integrates with Anthropic's Claude API to provide a chat interface enhanced with Retrieval-Augmented Generation (RAG) capabilities. It includes both a backend server and a simple frontend for interacting with Claude. Additionally, it features a powerful CLI tool for document processing and embedding generation using ChromaDB.

<img width="1572" alt="image" src="https://github.com/user-attachments/assets/d883726b-6515-4382-8e8b-e9b845660c2b">
 

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Configuration](#configuration)
- [Running the Server](#running-the-server)
- [Document Processing CLI](#document-processing-cli)
  - [Features](#features)
  - [Usage](#usage)
  - [Output](#output)
- [Important Notes](#important-notes)
- [Development](#development)
- [License](#license)

## Prerequisites

- Python 3.7+
- An Anthropic API key
- ChromaDB installed and configured
- SentenceTransformers for embedding generation

## Setup

1. **Clone this repository:**

   ```bash
   git clone https://github.com/moradology/impact-explorer.git
   cd impact-explorer
   ```

2. **Setup Python virtual environment**

   The following will set up a virtual environment and the development dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   make install-dev
   ```

3. **Install dependencies from `pyproject.toml`:**

   ```bash
   make install
   ```

   If you're developing the project, install with editable mode:

   ```bash
   make install-dev
   ```

## Configuration

Before running the server or using the CLI, you need to set several environment variables.

1. **Anthropic API Key**

   Set your Anthropic API key as an environment variable:

   - On Unix or macOS:

     ```bash
     export ANTHROPIC_API_KEY=your_api_key_here
     ```

   - On Windows (Command Prompt):

     ```bash
     set ANTHROPIC_API_KEY=your_api_key_here
     ```

   - On Windows (PowerShell):

     ```powershell
     $env:ANTHROPIC_API_KEY = "your_api_key_here"
     ```

2. **ChromaDB Path**

   Set the path where your ChromaDB database is stored:

   - On Unix or macOS:

     ```bash
     export CHROMA_PATH=/path/to/your/chromadb/
     ```

   - On Windows (Command Prompt):

     ```bash
     set CHROMA_PATH=C:\path\to\your\chromadb
     ```

   - On Windows (PowerShell):

     ```powershell
     $env:CHROMA_PATH = "C:\path\to\your\chromadb"
     ```

3. **Embedding Model Name**

   Set the name of the embedding model you are using:

   - On Unix or macOS:

     ```bash
     export EMBEDDING_MODEL=all-distilroberta-v1
     ```

   - On Windows (Command Prompt):

     ```bash
     set EMBEDDING_MODEL=all-distilroberta-v1
     ```

   - On Windows (PowerShell):

     ```powershell
     $env:EMBEDDING_MODEL = "all-distilroberta-v1"
     ```

   **Note:** The embedding model specified here must match the one used during the document embedding process for ChromaDB.

## Running the Server

1. **Start the FastAPI server:**

   ```bash
   make run
   ```

   The server will start running on `http://localhost:8000`.

2. **Navigate to the Frontend:**

   Open your browser and go to `http://localhost:8000`. The `templates/index.html` will be hosted at the application root (`/`).

3. **Chat Interface:**

   Use the chat interface to interact with Claude. The server uses Retrieval-Augmented Generation (RAG) to enhance responses with relevant information from your documents.

## Document Processing CLI

The project includes a Command Line Interface (CLI) tool for processing documents, generating embeddings, and storing them in a ChromaDB database. This is essential for preparing data for Retrieval-Augmented Generation (RAG) tasks.

### Features

- Support for various chunking strategies (e.g., sliding window)
- Flexible embedding model selection
- Integration with ChromaDB for efficient storage and retrieval
- Detailed output statistics for processed documents

### Usage

To use the document processing CLI:

```bash
python src/impact_explorer/cli/document_processor.py --input <input_file_or_directory> --output <output_dir> --chunking-strategy <strategy> [strategy-specific-args] --embedding-model <model_name>
```

**Example command:**

```bash
python src/impact_explorer/cli/document_processor.py \
  --input documents/mobydick.txt \
  --output ./moby_db/ \
  --chunking-strategy sliding-window \
  --chunk_size 100 \
  --overlap 20 \
  --embedding-model all-distilroberta-v1
```

This command will:

- Process `documents/mobydick.txt`
- Use a sliding window strategy with a window size of 100 and overlap of 20
- Generate embeddings using the `all-distilroberta-v1` model
- Store the results in ChromaDB in the `./moby_db/` directory

**Important:** Ensure that the `--embedding-model` parameter matches the `EMBEDDING_MODEL` environment variable used by the server.

### Output

After processing, the CLI will display statistics about the generated ChromaDB collection, including:

- Total number of documents
- Embedding dimensions
- Metadata fields
- Sample documents
- Document length statistics

The processed data is stored in ChromaDB and can be loaded for RAG queries on FastAPI startup.

## Important Notes

- **Embedding Model Consistency:** The embedding model used in both the CLI and the server must be the same. This ensures that the embeddings generated for the user's queries are compatible with those stored in ChromaDB.

  - **Example:**

    If you used `all-distilroberta-v1` for generating embeddings in the CLI, set:

    ```bash
    export EMBEDDING_MODEL=all-distilroberta-v1
    ```

- **Retrieval-Augmented Generation (RAG):**

  The server uses the user's query to retrieve the most relevant documents from ChromaDB and includes them in the prompt sent to the LLM. This enhances the response with specific information from your document collection.


## Development

To run the server in development mode with auto-reload:

```bash
make run
```

## License

[MIT License](LICENSE)
