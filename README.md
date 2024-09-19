# Impact Explorer

This project implements a FastAPI server that integrates with Anthropic's Claude API to provide a chat interface. It includes both a backend server and a simple frontend for interacting with Claude.
Additionally, it features a powerful CLI tool for document processing and embedding generation.

## Prerequisites

- Python 3.7+
- An Anthropic API key

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/moradology/impact-explorer.git
   cd impact-explorer
   ```

2. Setup Python virtual environment
The following will set up a virtual environment and the development dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
make install-dev
```

3. Install dependencies from `pyproject.toml`:
   ```
   make install
   ```

   If you're developing the project, install with editable mode:
   ```
   make install-dev
   ```

## Document Processing CLI

The project includes a Command Line Interface (CLI) tool for processing documents, generating embeddings, and storing them in a ChromaDB database. This is useful for preparing data for Retrieval-Augmented Generation (RAG) tasks.

### Features

- Support for various chunking strategies (e.g., sliding window)
- Flexible embedding model selection
- Integration with ChromaDB for efficient storage and retrieval
- Detailed output statistics for processed documents

### Usage

To use the document processing CLI:

```bash
python src/impact_explorer/cli/document_processor.py --input <input_file> --output <output_dir> --chunking-strategy <strategy> [strategy-specific-args] --embedding-model <model_name>
```

Example command:

```bash
python src/impact_explorer/cli/document_processor.py --input documents/mobydick.txt --output ./moby_db/ --chunking-strategy sliding-window --chunk_size 100 --overlap 20 --embedding-model distilbert-base-uncased
```

This command will:
- Process documents/mobydick.txt
- Use a sliding window strategy with a window size of 100 and overlap of 20
- Generate embeddings using the distilled bert model
- Store the results in ChromaDB in the ./moby_db/ directory

### Output

After processing, the CLI will display statistics about the generated ChromaDB collection, including:
- Total number of documents
- Embedding dimensions
- Metadata fields
- A look at sample documents
- Document length statistics

The processed data can be loaded for RAG queries on FastAPI startup.

## Configuration

Set your Anthropic API key as an environment variable:

- On Unix or MacOS:
  ```
  export ANTHROPIC_API_KEY=your_api_key_here
  ```

- On Windows (Command Prompt):
  ```
  set ANTHROPIC_API_KEY=your_api_key_here
  ```

- On Windows (PowerShell):
  ```
  $env:ANTHROPIC_API_KEY = "your_api_key_here"
  ```

## Running the Server

1. Start the FastAPI server:
   ```bash
   make run
   ```

   The server will start running on `http://localhost:8000`.

   Navigate to the server in a browser and `templates/index.html` will be hosted on application root (`/`).

## Development

To run the server in development mode with auto-reload:

```python
make run
```

## License

[MIT License](LICENSE)
