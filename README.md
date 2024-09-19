# Impact Explorer

This project implements a FastAPI server that integrates with Anthropic's Claude API to provide a chat interface. It includes both a backend server and a simple frontend for interacting with Claude.

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